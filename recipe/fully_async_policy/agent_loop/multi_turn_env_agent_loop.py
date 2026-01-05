# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import copy
import logging
import os
import uuid
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState
from verl.protocol import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def torch_to_numpy(tensor, is_object=False):
    """Convert torch tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError(f"Unsupported type: {type(tensor)})")
    if is_object:
        tensor = tensor.astype(object)
    return tensor


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 256 * 256):
    """Process image for multimodal inputs."""
    from PIL import Image
    import math

    if isinstance(image, torch.Tensor):
        image = torch_to_numpy(image)
    if image.max() < 1:
        image = image * 255.0
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = Image.fromarray(image)

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


@register("async_partial_multi_turn_env_agent")
class AsyncPartialMultiTurnEnvAgentLoop(AgentLoopBase):
    """
    Multi-turn agent loop with environment interaction support for fully async training.
    Supports partial rollout with multiple environment steps.
    Migrated from synchronize version's TrajectoryCollector.
    """

    def __init__(self, trainer_config, **kwargs):
        super().__init__(trainer_config, **kwargs)
        self.config = trainer_config.config
        self.enable_partial_rollout = self.config.async_training.get("partial_rollout", False)
        self.max_steps = self.config.env.max_steps
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.max_prompt_length = self.config.data.max_prompt_length

        # Initialize environment manager
        self.envs = None
        self._init_envs()

    def _init_envs(self):
        """Initialize environment manager from config."""
        try:
            from agent_system.environments import make_envs

            env_config = self.config.env
            self.envs = make_envs(env_config)
            logger.info(f"[MultiTurnEnvAgent] Environment manager initialized: {type(self.envs).__name__}")
        except ImportError:
            logger.warning(
                "[MultiTurnEnvAgent] Could not import agent_system.environments. "
                "Environment interaction will not be available."
            )
        except Exception as e:
            logger.error(f"[MultiTurnEnvAgent] Failed to initialize environments: {e}")
            raise

    async def run(
        self, sampling_params: dict[str, Any], *, cancellation_event: asyncio.Event = None, **kwargs
    ) -> AgentLoopOutput:
        """
        Main entrance, supports interruption/recovery for multi-turn environment interaction.

        Args:
            sampling_params: Sampling parameters
            cancellation_event: Cancellation signal
            **kwargs: Contains output (for recovery), raw_prompt, param_version, etc.

        Returns:
            AgentLoopOutput: Include the is_cancel flag
        """
        param_version = kwargs.get("param_version", 0)
        agent_data = None
        state = None

        # Check whether this is a partial task
        output: Optional[AgentLoopOutput] = kwargs.get("output", None)
        if output and output.extra_fields.get("is_cancel", False):
            agent_data, state = self._restore_from_output(output)
            logger.info(f"[MultiTurnEnvAgent] Resuming from {state.value}")
        else:
            if output and not output.extra_fields.get("is_cancel", False):
                # Completed, return directly
                return output

            agent_data = await self._init_agent_data(kwargs, param_version)
            state = AgentState.PENDING
            logger.info("[MultiTurnEnvAgent] Start from scratch")

        # Run state machine
        state = await self._run_state_machine(agent_data, state, sampling_params, cancellation_event)

        # Build output
        if state == AgentState.TERMINATED:
            return self._build_completed_output(agent_data, param_version)
        else:
            # Build cancelled output
            return self._build_cancelled_output(agent_data, state)

    async def _init_agent_data(self, kwargs: dict, param_version: int) -> AgentData:
        """Initialize agent data from kwargs."""
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize environment if needed
        # Try to get gen_batch from kwargs, or construct from available data
        gen_batch = kwargs.get("gen_batch", None)
        if gen_batch is None:
            # Create a gen_batch for single sample from available kwargs
            data_source = kwargs.get("data_source", "unknown")
            if isinstance(data_source, (list, np.ndarray)) and len(data_source) > 0:
                data_source = data_source[0]
            gen_batch = DataProto.from_single_dict(
                data={
                    "raw_prompt": [messages],
                    "data_source": [data_source],
                },
                meta_info=kwargs.get("meta_info", {}),
            )

        # Create AgentData instance
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=None,
            interaction_kwargs={},
        )

        # Store environment-related data
        agent_data.extra_fields["gen_batch"] = gen_batch
        agent_data.extra_fields["param_version_start"] = param_version
        agent_data.extra_fields["param_version_end"] = param_version
        agent_data.extra_fields["trajectory_data"] = []  # Store trajectory steps
        agent_data.extra_fields["is_done"] = False
        agent_data.extra_fields["step"] = 0
        agent_data.extra_fields["obs"] = None
        agent_data.extra_fields["traj_uid"] = str(uuid.uuid4())
        
        # Initialize episode-level statistics (similar to original version)
        agent_data.extra_fields["episode_rewards"] = 0.0
        agent_data.extra_fields["episode_lengths"] = 0
        agent_data.extra_fields["tool_callings"] = 0.0
        agent_data.extra_fields["uid"] = str(uuid.uuid4())  # For env grouping if needed

        return agent_data

    def _restore_from_output(self, output: AgentLoopOutput) -> tuple[AgentData, AgentState]:
        """Restore AgentState and AgentData from output."""
        agent_data = output.extra_fields.get("agent_data", None)
        agent_state = output.extra_fields.get("agent_state", None)
        if agent_data is None or agent_state is None:
            raise ValueError(f"Unexpected situation: agent_data is {agent_data}, agent_state is {agent_state}")
        return agent_data, agent_state

    async def _run_state_machine(
        self,
        agent_data: AgentData,
        state: AgentState,
        sampling_params: dict[str, Any],
        cancellation_event: asyncio.Event = None,
    ) -> AgentState:
        """
        State machine for multi-turn environment interaction.
        Currently, interruptions are only supported to occur in the GENERATING state.
        """
        # State machine loop
        while state != AgentState.TERMINATED:
            if cancellation_event and cancellation_event.is_set():
                logger.info(f"[MultiTurnEnvAgent] Cancellation detected. Interrupted before/at state: {state.value}")
                return state

            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state_partial(agent_data, sampling_params, cancellation_event)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"[MultiTurnEnvAgent] Invalid state: {state}")
                return AgentState.TERMINATED

        return AgentState.TERMINATED

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: use initial observation from feed_sample or reset environment."""
        if self.envs is None:
            raise RuntimeError("Environment manager not initialized")

        gen_batch = agent_data.extra_fields["gen_batch"]
        step = agent_data.extra_fields["step"]

        # On first step, check if initial_obs is provided from feed_sample
        if step == 0:
            initial_obs = gen_batch.non_tensor_batch.get("initial_obs", None)
            initial_infos = gen_batch.non_tensor_batch.get("initial_infos", None)
            
            if initial_obs is not None and len(initial_obs) > 0:
                # Use initial observation from feed_sample
                obs_dict = initial_obs[0] if isinstance(initial_obs, (list, np.ndarray)) else initial_obs
                # Convert back to environment format (list of observations)
                obs = {}
                if "text" in obs_dict:
                    obs["text"] = [obs_dict["text"]]
                if "image" in obs_dict:
                    obs["image"] = [obs_dict["image"]]
                if "anchor" in obs_dict:
                    obs["anchor"] = [obs_dict["anchor"]]
                
                info_dict = initial_infos[0] if initial_infos is not None and len(initial_infos) > 0 else {}
                infos = [info_dict] if info_dict else [{}]
                
                agent_data.extra_fields["obs"] = obs
                agent_data.extra_fields["infos"] = infos
                logger.info("[MultiTurnEnvAgent] Using initial observation from feed_sample")
            else:
                # Fallback: reset environment if initial_obs not provided
                env_kwargs = gen_batch.non_tensor_batch.get("env_kwargs", None)
                if env_kwargs is not None and len(env_kwargs) > 0:
                    env_kwargs = env_kwargs[0] if isinstance(env_kwargs, (list, np.ndarray)) else env_kwargs
                else:
                    env_kwargs = None

                # Reset environment (synchronous call, run in executor)
                obs, infos = await self.loop.run_in_executor(None, lambda: self.envs.reset(kwargs=env_kwargs))
                agent_data.extra_fields["obs"] = obs
                agent_data.extra_fields["infos"] = infos
                logger.info("[MultiTurnEnvAgent] Environment reset in agent loop")

        # Preprocess observation
        batch = await self._preprocess_observation(gen_batch, agent_data.extra_fields["obs"])
        agent_data.extra_fields["current_batch"] = batch

        # Prepare prompt from observation
        agent_data.messages = await self._build_messages_from_obs(batch, agent_data.extra_fields["obs"])

        # Tokenize prompt
        agent_data.prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                agent_data.messages, add_generation_prompt=True, tokenize=True
            ),
        )

        return AgentState.GENERATING

    async def _preprocess_observation(self, gen_batch: DataProto, obs: dict) -> DataProto:
        """Preprocess observation into model-processable format."""
        # For single sample, batch_size is 1
        batch_size = 1
        processed_samples = []

        for item in range(batch_size):
            processed = await self._preprocess_single_sample(item, gen_batch, obs)
            processed_samples.append(processed)

        # Aggregate batch data
        batch = collate_fn(processed_samples)

        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(data=batch, meta_info=gen_batch.meta_info)
        return new_batch

    async def _preprocess_single_sample(self, item: int, gen_batch: DataProto, obs: dict) -> dict:
        """Process a single observation sample."""
        raw_prompt = gen_batch.non_tensor_batch.get("raw_prompt", [None])[item]
        data_source = gen_batch.non_tensor_batch.get("data_source", ["unknown"])[item]

        # Get observation components
        obs_texts = obs.get("text", None)
        obs_images = obs.get("image", None)
        obs_anchors = obs.get("anchor", None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        obs_content = ""
        if obs_text is not None:
            obs_content += obs_text
        else:
            logger.warning(f"Warning: No text observation found for item {item}!")

        chat = np.array([{"content": obs_content, "role": "user"}])

        # Apply chat template
        prompt_with_chat_template = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False),
        )

        # Initialize return dict
        row_dict = {}

        # Process multimodal data
        if is_multi_modal and self.processor is not None:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace(
                "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
            )
            processed_image = await self.loop.run_in_executor(None, lambda: process_image(obs_image))
            row_dict["multi_modal_data"] = {"image": [processed_image]}
            image_inputs = self.processor.image_processor(row_dict["multi_modal_data"]["image"], return_tensors="pt")
            image_grid_thw = image_inputs.get("image_grid_thw", None)
            row_dict["multi_modal_inputs"] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size ** 2
                index = 0
                while "<image>" in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        "<image>",
                        "<|vision_start|>"
                        + "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length)
                        + "<|vision_end|>",
                        1,
                    )
                    index += 1
                prompt_with_chat_template = prompt_with_chat_template.replace(
                    "<|placeholder|>", self.processor.image_token
                )
        else:
            raw_prompt = prompt_with_chat_template

        # Tokenize
        input_ids, attention_mask = await self.loop.run_in_executor(
            None,
            lambda: verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=self.tokenizer,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            ),
        )

        # Compute position ids
        if is_multi_modal and self.processor is not None and image_grid_thw is not None:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        )
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.config.data.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )

        # Build final output dict
        row_dict.update(
            {
                "input_ids": input_ids[0],
                "attention_mask": attention_mask[0],
                "position_ids": position_ids[0],
                "raw_prompt_ids": raw_prompt_ids,
                "anchor_obs": _obs_anchor,
                "index": item,
                "data_source": data_source,
            }
        )

        if self.config.data.get("return_raw_chat", False):
            row_dict["raw_prompt"] = chat.tolist()

        return row_dict

    async def _build_messages_from_obs(self, batch: DataProto, obs: dict) -> list[dict]:
        """Build messages from observation."""
        if self.config.data.get("return_raw_chat", False):
            raw_prompts = batch.non_tensor_batch.get("raw_prompt", [])
            if raw_prompts and len(raw_prompts) > 0:
                return raw_prompts[0] if isinstance(raw_prompts[0], list) else raw_prompts
        # Fallback: build from observation text
        obs_texts = obs.get("text", None)
        if obs_texts is not None and len(obs_texts) > 0:
            return [{"role": "user", "content": obs_texts[0]}]
        return [{"role": "user", "content": ""}]

    async def _handle_generating_state_partial(
        self, agent_data: AgentData, sampling_params: dict[str, Any], cancellation_event: asyncio.Event = None
    ) -> AgentState:
        """Handle GENERATING state, support partial rollout."""
        with simple_timer("generate_sequences", agent_data.metrics):
            # Partial interface
            if self.enable_partial_rollout:
                response_ids, log_probs, is_cancel = await self.server_manager.generate_for_partial(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=sampling_params,
                    image_data=agent_data.image_data,
                )

                if is_cancel:
                    # Save the generated parts
                    agent_data.response_ids = response_ids
                    agent_data.prompt_ids += agent_data.response_ids
                    agent_data.response_mask += [1] * len(response_ids)
                    if log_probs:
                        agent_data.response_logprobs += log_probs
                    if len(agent_data.response_mask) >= self.response_length:
                        # If response_length has reached the limit, terminate
                        return AgentState.TERMINATED
                    return AgentState.GENERATING
            else:
                # Original generate interface
                output = await self.server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=sampling_params,
                    image_data=agent_data.image_data,
                )
                response_ids = output.token_ids
                log_probs = output.log_probs

        # Decode response
        text_action = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
        )

        # Step environment
        agent_data.assistant_turns += 1
        agent_data.response_ids = response_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if log_probs:
            agent_data.response_logprobs += log_probs

        # Create batch_output with responses (similar to original batch.union(batch_output))
        # This is needed to match the original data structure
        step = agent_data.extra_fields["step"]
        current_batch = agent_data.extra_fields["current_batch"]
        
        # Create a batch_output DataProto with responses
        # Convert response_ids to tensor format to match original structure
        response_tensor = torch.tensor([response_ids], dtype=torch.long)
        batch_output = DataProto.from_single_dict(
            data={"responses": response_tensor},
            meta_info=current_batch.meta_info,
        )
        
        # Union current_batch with batch_output (similar to original: batch = batch.union(batch_output))
        # This ensures batch.batch['responses'] is available
        current_batch = current_batch.union(batch_output)
        
        # Add uid and traj_uid to batch (similar to original)
        current_batch.non_tensor_batch["uid"] = np.array([agent_data.extra_fields["uid"]], dtype=object)
        current_batch.non_tensor_batch["traj_uid"] = np.array([agent_data.extra_fields["traj_uid"]], dtype=object)

        # Step environment (synchronous call, run in executor)
        next_obs, rewards, dones, infos = await self.loop.run_in_executor(
            None, lambda: self.envs.step([text_action])
        )

        # Process rewards and dones
        if len(rewards.shape) == 2:
            rewards = rewards.squeeze(1)
        if len(dones.shape) == 2:
            dones = dones.squeeze(1)

        rewards = torch_to_numpy(rewards)
        dones = torch_to_numpy(dones)

        # Extract reward and done for single sample
        reward = rewards[0] if len(rewards) > 0 else 0.0
        is_done = dones[0] if len(dones) > 0 else False
        info = infos[0] if len(infos) > 0 else {}

        # Add fields to batch (similar to original version)
        current_batch.non_tensor_batch["rewards"] = np.array([reward], dtype=object)
        current_batch.non_tensor_batch["active_masks"] = np.array([not is_done], dtype=object)
        
        # Extract is_action_valid from info
        if "is_action_valid" in info:
            current_batch.non_tensor_batch["is_action_valid"] = np.array([info["is_action_valid"]], dtype=bool)
        else:
            current_batch.non_tensor_batch["is_action_valid"] = np.array([True], dtype=bool)
        
        # Extract tool_calling if available
        if "tool_calling" in info:
            agent_data.extra_fields["tool_callings"] += float(info["tool_calling"])
        
        # Extract additional metrics from info (similar to original)
        metrics_dict = {}
        for key in ["title_score", "r_type", "r_att", "r_option", "w_att", "w_option", "w_price"]:
            if key in info:
                metrics_dict[key] = info[key]
        
        # Update episode-level statistics (similar to original)
        if not is_done:
            agent_data.extra_fields["episode_rewards"] += float(reward)
            agent_data.extra_fields["episode_lengths"] += 1

        # Store trajectory step with complete batch data
        trajectory_step = {
            "step": step,
            "response_ids": response_ids,
            "text_action": text_action,
            "batch": current_batch,  # Now includes responses, rewards, active_masks, etc.
            "rewards": reward,
            "dones": is_done,
            "infos": info,
            "metrics": metrics_dict,
        }
        agent_data.extra_fields["trajectory_data"].append(trajectory_step)

        # Check termination conditions
        agent_data.extra_fields["is_done"] = is_done
        agent_data.extra_fields["step"] = step + 1

        if is_done or step + 1 >= self.max_steps:
            return AgentState.TERMINATED

        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED

        # Update observation for next step
        agent_data.extra_fields["obs"] = next_obs
        agent_data.extra_fields["infos"] = infos

        # Continue to next turn
        return AgentState.PENDING

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle processing tools state (not used in environment interaction, but required by interface)."""
        return AgentState.TERMINATED

    def _build_completed_output(self, agent_data: AgentData, param_version: int) -> AgentLoopOutput:
        """Build completed output."""
        # Extract final response from trajectory
        trajectory_data = agent_data.extra_fields.get("trajectory_data", [])
        if trajectory_data:
            # Use the last step's response
            last_step = trajectory_data[-1]
            response_ids = last_step.get("response_ids", agent_data.response_ids)
        else:
            response_ids = agent_data.response_ids

        # Calculate total response from all steps
        all_response_ids = []
        all_response_mask = []
        all_response_logprobs = []
        for step_data in trajectory_data:
            step_response = step_data.get("response_ids", [])
            all_response_ids.extend(step_response)
            all_response_mask.extend([1] * len(step_response))
            if agent_data.response_logprobs:
                # Approximate: use same logprobs pattern
                all_response_logprobs.extend([0.0] * len(step_response))

        # Use accumulated response or single response
        if all_response_ids:
            final_response_ids = all_response_ids[: self.response_length]
            final_response_mask = all_response_mask[: self.response_length]
            final_logprobs = (
                all_response_logprobs[: self.response_length] if all_response_logprobs else None
            )
        else:
            final_response_ids = response_ids[: self.response_length]
            final_response_mask = agent_data.response_mask[: self.response_length]
            final_logprobs = (
                agent_data.response_logprobs[: self.response_length] if agent_data.response_logprobs else None
            )

        # Calculate prompt_ids (initial prompt)
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]

        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=final_response_ids,
            response_mask=final_response_mask,
            multi_modal_data=multi_modal_data,
            response_logprobs=final_logprobs,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )

        # Add trajectory information (similar to original version)
        output.extra_fields.update(
            {
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                "is_cancel": False,
                "param_version_start": agent_data.extra_fields["param_version_start"],
                "param_version_end": param_version,
                "trajectory_data": trajectory_data,
                "traj_uid": agent_data.extra_fields["traj_uid"],
                # Episode-level statistics (similar to original)
                "episode_rewards": agent_data.extra_fields.get("episode_rewards", 0.0),
                "episode_lengths": agent_data.extra_fields.get("episode_lengths", 0),
                "tool_callings": agent_data.extra_fields.get("tool_callings", 0.0),
                "uid": agent_data.extra_fields.get("uid", ""),
            }
        )

        return output

    def _build_cancelled_output(self, agent_data: AgentData, state: AgentState) -> AgentLoopOutput:
        """Build cancelled output."""
        return AgentLoopOutput(
            prompt_ids=[],
            response_ids=[],
            response_mask=[],
            multi_modal_data={},
            response_logprobs=None,
            num_turns=0,
            metrics=agent_data.metrics,
            extra_fields={
                "is_cancel": True,
                "agent_data": agent_data,
                "agent_state": state,
            },
        )


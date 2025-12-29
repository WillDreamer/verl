#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'


log() { printf "\n\033[1;32m[+] %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m[!] %s\033[0m\n" "$*"; }
die() { printf "\033[1;31m[x] %s\033[0m\n" "$*"; exit 1; }
trap 'die "Error is in Line $LINENO (exit=$?)。"' ERR

as_root() {
  if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    sudo -H bash -lc "$*"
  else
    bash -lc "$*"
  fi
}


# log "Run setup game environment "
# CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
# if [[ -z "${CONDA_BASE}" ]]; then
#   for p in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/anaconda3"; do
#     [[ -d "$p" ]] && CONDA_BASE="$p" && break
#   done
# fi
# if [[ -z "${CONDA_BASE}" ]]; then
#   echo "Conda not found, please confirm it is installed (miniconda or anaconda)." >&2
#   exit 1
# fi
# CONDA_BASE=/data1/xw27/miniconda3
# source "${CONDA_BASE}/etc/profile.d/conda.sh"
source ~/anaconda3/etc/profile.d/conda.sh


conda create -n agentrl_science_async python==3.11.5 -y
conda activate agentrl_science_async
python3 -m pip install uv


# python3 -m uv pip install -e ".[sglang]"
python3 -m uv pip install -e ".[vllm]"
pip install --no-deps -e .
python3 -m uv pip install flash-attn==2.8.3 --no-build-isolation --no-deps
python3 -m uv pip install -r ./requirements.txt
python3 -m uv pip install qwen_vl_utils
python3 -m uv pip install word2number
# sudo apt-get update && sudo apt-get install -y firejail
python3 -m uv pip install "fastapi[all]" uvicorn
### sudo 替换为 ->
# cd $HOME
# git clone https://github.com/netblue30/firejail.git
# cd firejail
# ./configure --prefix=$HOME/.local --disable-suid
# make -j
# make install
# # 添加到 PATH
# echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
# source ~/.bashrc


if command -v conda >/dev/null 2>&1; then
  conda activate agentrl_science_async || true
fi

log "hf auth whoami"
if command -v hf >/dev/null 2>&1; then
  hf auth whoami || die "whoami failed "
else
  huggingface-cli whoami || die "whoami failed"
fi

# cd sandbox
# uvicorn sandbox_api:app --host 127.0.0.1 --port 12345 --workers 4

log "全部完成 🎉"
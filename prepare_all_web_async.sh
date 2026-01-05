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


source ~/anaconda3/etc/profile.d/conda.sh
conda create -n agentrl_web_async python==3.11.5 -y
conda activate agentrl_web_async
python3 -m pip install uv


# python3 -m uv pip install -e ".[sglang]"
python3 -m uv pip install -e ".[vllm]"
pip install --no-deps -e .
python3 -m uv pip install flash-attn==2.8.3 --no-build-isolation --no-deps
log "运行 setup_webshop.sh（若存在）"
cd ./agent_system/environments/env_package/webshop/webshop && bash setup.sh || warn "未找到 setup webshop，跳过"


if command -v conda >/dev/null 2>&1; then
  conda activate agentrl_web_async || true
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
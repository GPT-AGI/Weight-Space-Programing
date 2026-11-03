#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${WSP_SANDBOX_ENV:-/opt/wsp-sandbox}"
PYTHON="${WSP_SANDBOX_PYTHON:-3.12}"
PYPI_MIRROR="${WSP_PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_TIMEOUT="${WSP_TORCH_TIMEOUT:-600}"
UV="${UV_BIN:-$(command -v uv || true)}"

if [[ -z "$UV" ]]; then
  echo "uv is required; install it before building the sandbox environment" >&2
  exit 1
fi

echo "Creating isolated environment at $VENV"
if [[ -z "$VENV" || "$VENV" == "/" ]]; then
  echo "refusing to replace unsafe environment path: $VENV" >&2
  exit 1
fi
rm -rf -- "$VENV"
"$UV" venv --relocatable --python "$PYTHON" "$VENV"

echo "Installing tokenizer and tensor dependencies from $PYPI_MIRROR"
"$UV" pip install \
  --python "$VENV/bin/python" \
  --index-url "$PYPI_MIRROR" \
  numpy safetensors tokenizers transformers

install_torch() {
  local index="$1"
  timeout "$TORCH_TIMEOUT" "$UV" pip install \
    --python "$VENV/bin/python" \
    --index-url "$index" \
    --extra-index-url "$PYPI_MIRROR" \
    --index-strategy unsafe-best-match \
    torch
}

echo "Installing CPU-only PyTorch (time limit: ${TORCH_TIMEOUT}s per source)"
if ! install_torch "https://download.pytorch.org/whl/cpu"; then
  echo "Official CPU wheel source failed; trying the Aliyun CPU mirror" >&2
  if ! install_torch "https://mirrors.aliyun.com/pytorch-wheels/cpu"; then
    echo "Aliyun failed; trying the Tsinghua CPU wheel mirror" >&2
    install_torch "https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cpu"
  fi
fi

if [[ -f "$ROOT/sdk/pyproject.toml" ]]; then
  echo "Installing the sandbox-only wsp_sdk package"
  "$UV" pip install --python "$VENV/bin/python" --no-deps "$ROOT/sdk"
else
  echo "sdk/ is not present yet; rerun this script after W1 lands it" >&2
fi

"$VENV/bin/python" - <<'PY'
import numpy
import safetensors
import tokenizers
import torch
import transformers

if torch.version.cuda is not None:
    raise SystemExit(f"expected a CPU-only torch wheel, got CUDA {torch.version.cuda}")
print("sandbox environment ready")
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
PY

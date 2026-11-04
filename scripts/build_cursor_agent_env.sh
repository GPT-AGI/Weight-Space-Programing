#!/usr/bin/env bash
set -euo pipefail

VENV="${WSP_CURSOR_SDK_ENV:-/opt/wsp-cursor-agent}"
PYTHON="${WSP_CURSOR_PYTHON:-3.12}"
PYPI_MIRROR="${WSP_PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
UV="${UV_BIN:-$(command -v uv || true)}"

if [[ -z "$UV" ]]; then
  echo "uv is required to build the Cursor Agent runtime" >&2
  exit 1
fi
if [[ -z "$VENV" || "$VENV" == "/" ]]; then
  echo "refusing to replace unsafe environment path: $VENV" >&2
  exit 1
fi

rm -rf -- "$VENV"
"$UV" venv --relocatable --python "$PYTHON" "$VENV"
"$UV" pip install \
  --python "$VENV/bin/python" \
  --index-url "$PYPI_MIRROR" \
  cursor-sdk

"$VENV/bin/python" - <<'PY'
import importlib.metadata
import cursor_sdk

print("cursor agent runtime ready:", importlib.metadata.version("cursor-sdk"))
PY

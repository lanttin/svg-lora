#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-train_config.yaml}"
PYTHON_BIN="${PYTHON:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Could not find python or python3 to read $CONFIG_PATH" >&2
    exit 1
  fi
fi

mapfile -t SWIFT_ARGS < <("$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import sys

import yaml

config_path = sys.argv[1]
with open(config_path, encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

if not isinstance(config, dict):
    raise SystemExit(f"{config_path} must contain a YAML mapping at top level")

for key, value in config.items():
    if value is None:
        continue
    print(f"--{key}")
    if isinstance(value, bool):
        print(str(value).lower())
    elif isinstance(value, (list, tuple)):
        for item in value:
            print(item)
    else:
        print(value)
PY
)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
swift sft "${SWIFT_ARGS[@]}"

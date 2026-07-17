#!/usr/bin/env bash
set -euo pipefail

python student_kit/select_best_adapter.py \
  --output-root output/swift-svg-lora \
  --adapter-output adapter \
  --report adapter_selection.json

#!/usr/bin/env bash
# run_all.sh
set -euo pipefail

DATA_PATH="datasets/gsm8k/main/test-00000-of-00001.parquet"
SCRIPT="batch_inf.py"

# まとめて回すモデルたち（3.7B / 13B）
MODELS=(
  "llm-jp/llm-jp-3-3.7b-instruct"
  "llm-jp/llm-jp-3-13b-instruct"
)

for M in "${MODELS[@]}"; do
  echo "===== Running: $M ====="
  uv run "$SCRIPT" \
    --model "$M" \
    --data  "$DATA_PATH"
done


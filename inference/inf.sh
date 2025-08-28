 uv run batch_inf.py \
  --dataset mgsm --lang ja \
  --data datasets/mgsm/mgsm_ja.tsv \
  --model llm-jp/llm-jp-3-1.8b-instruct \
  --adapter ../train/outputs/llmjp1p8b-ja-cot-lora \
  --batch-size 64 --max-new-tokens 384 \

uv run train_lora_sft.py \
  --data ../datasets/gsm8k_ja/train.jsonl \
  --model llm-jp/llm-jp-3-1.8b-instruct \
  --out outputs/llmjp1p8b-ja-cot-lora \
  --seq-len 1536 --mbsz 16 --grad-accum 8 --epochs 1 --r 16

# train_lora_sft.py  (TRL 0.21.0)
import re
import argparse
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM

PROMPT_HEAD = (
    "### 指示:\n"
    "あなたは算数問題を解くアシスタントです。段階的に考え、最後に必ず 1 行だけ "
    "`Answer: 数値` として最終解を出力してください。\n\n"
    "### 問題:\n{q}\n\n### 応答:\n"
)

def to_answer_line(ans_ja: str) -> str:
    """GSM8K系の '#### 123' を 'Answer: 123' に正規化"""
    ans_ja = ans_ja or ""
    m = re.search(r"####\s*([-\d,\.]+)", ans_ja)
    return re.sub(r"####\s*([-\d,\.]+)", r"Answer: \1", ans_ja).strip() if m else ans_ja.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../datasets/gsm8k_ja/train.jsonl")
    ap.add_argument("--model", default="llm-jp/llm-jp-3-1.8b-instruct")
    ap.add_argument("--out", default="outputs/llmjp1p8b-ja-cot-lora")
    ap.add_argument("--seq-len", type=int, default=1536)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--mbsz", type=int, default=16)        # per-device micro batch
    ap.add_argument("--grad-accum", type=int, default=8)   # 勾配蓄積
    ap.add_argument("--r", type=int, default=16)
    args = ap.parse_args()

    # 1) 入力を読み込み
    ds = load_dataset("json", data_files=args.data, split="train")

    # 2) TRLの「prompt-completion 形式」にマップ（completionのみ損失になる）:contentReference[oaicite:3]{index=3}
    def to_pc(example):
        prompt = PROMPT_HEAD.format(q=example["question_ja"])
        completion = to_answer_line(example["answer_ja"])
        return {"prompt": prompt, "completion": completion}
    ds_pc = ds.map(to_pc, remove_columns=ds.column_names, num_proc=4)

    # 3) LoRA設定（PEFT）
    peft_cfg = LoraConfig(
        r=args.r, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    # 4) モデル本体（bf16などは SFTConfig の model_init_kwargs からも渡せます）:contentReference[oaicite:4]{index=4}
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")

    # 5) SFTTrainer（completion_only_loss=True が prompt-completion ではデフォルト）:contentReference[oaicite:5]{index=5}
    trainer = SFTTrainer(
        model=model,
        peft_config=peft_cfg,
        train_dataset=ds_pc,
        args=SFTConfig(
            output_dir=args.out,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.mbsz,
            gradient_accumulation_steps=args.grad_accum,
            bf16=True,
            logging_steps=50,
            save_steps=1000,
            save_total_limit=2,
            packing=True,                 # 例の詰め込み最適化（有効なら速い）:contentReference[oaicite:6]{index=6}
            max_length=args.seq_len,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            dataloader_num_workers=4,
            report_to="none",
            # from_pretrainedの追加引数を渡したいときは:
            # model_init_kwargs={"torch_dtype": "bfloat16"},
        ),
    )

    trainer.train()
    trainer.model.save_pretrained(args.out)

if __name__ == "__main__":
    main()


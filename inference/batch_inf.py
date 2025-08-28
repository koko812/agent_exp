# run_instruct_cli.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from prompt_builder import make_prompt_builder
import torch, re, argparse
from pathlib import Path
from tqdm.auto import tqdm
import json, pandas as pd
import unicodedata

# ===== existing helpers =====
def check_prompt(prompt_builder, model_name):
    print("\n---- check prompt ----")
    exp_sentence = "My name is LLM! What is your name?"
    print(f"[{model_name}]")
    print(prompt_builder(exp_sentence))
    print("----------------------\n")


def extract_pred(text: str):
    m = re.search(r"Answer:\s*(-?\d+)\s*$", text, flags=re.IGNORECASE|re.MULTILINE)
    if m: return int(m.group(1))
    m2 = re.search(r"(?s)(-?\d+)(?!.*\d)", text)
    return int(m2.group(1)) if m2 else None

def _norm_text(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s)).strip()

def extract_final_numeric(text: str):
    """
    GSM8K/MGSM の正解表記から最終数値を抽出。
    - '#### 42' を優先、それが無ければ文中の最後の数値
    - 見つからなければ None
    """
    if text is None:
        return None
    s = _norm_text(text)
    m = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", s)
    if m:
        return int(m.group(1).replace(",", "")) if re.fullmatch(r"-?\d+(?:,\d{3})*", m.group(1)) or re.fullmatch(r"-?\d+(?:\.\d+)?", m.group(1)) else None
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", s)
    if not nums:
        return None
    last = nums[-1].replace(",", "")
    try:
        return int(float(last))
    except:
        return None

def load_mgsm_tsv(path: str) -> pd.DataFrame:
    """
    ヘッダ無しTSV（問題\t答）の MGSM を読む → DataFrame[question, answer]
    """
    df = pd.read_csv(path, sep="\t", header=None, names=["question", "answer"], dtype=str, encoding="utf-8")
    df = df.dropna(subset=["question", "answer"])
    df["question"] = df["question"].map(_norm_text)
    df["answer"]   = df["answer"].map(_norm_text)
    return df

@torch.no_grad()
def solve_batch_with_progress(questions, gold, prompt_builder, model, tok,
                              batch_size=32, max_new_tokens=256, save_path=None, desc="GSM8K"):
    preds, gens = [], []
    correct, seen = 0, 0
    model.eval()
    pbar = tqdm(total=len(questions), desc=desc, unit="q")

    fout = open(save_path, "w") if save_path else None

    for i in range(0, len(questions), batch_size):
        chunk = questions[i:i+batch_size]
        texts = [prompt_builder(q) for q in chunk]
        enc = tok(texts, return_tensors="pt", padding=True)
        enc.pop("token_type_ids", None)  # decoder-only safety
        enc = {k: v.to(model.device) for k, v in enc.items()}

        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
        )
        gen_ids = out[:, enc["input_ids"].shape[1]:]
        dec = tok.batch_decode(gen_ids, skip_special_tokens=True)

        batch_preds = [extract_pred(t) for t in dec]
        preds.extend(batch_preds); gens.extend(dec)

        for j, p in enumerate(batch_preds):
            g = gold[i+j]
            seen += 1
            if p == g: correct += 1
            if fout:
                rec = {"idx": i+j, "question": questions[i+j], "pred": p, "gold": g}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        pbar.update(len(chunk))
        pbar.set_postfix(acc=f"{correct/seen:.2%}")

    if fout: fout.close()
    pbar.close()
    return gens, preds, correct, seen

def default_out_from_input(input_path: str, model_name: str) -> str:
    """
    datasets/<NAME>/.../(train|test)-xxxx.parquet → preds_<NAME>_<split>_<model>.jsonl
    'datasets' の直後のディレクトリ名を使用し、ファイル名に train/test を含める。
    model_name は / や - を置換してファイル名に使える形にする。
    """
    p = Path(input_path)
    parts = p.parts

    # データセット名
    name = None
    if "datasets" in parts:
        i = parts.index("datasets")
        if i + 1 < len(parts):
            name = parts[i + 1]

    # train/test/data の判定
    split = "train" if "train" in p.name.lower() or "train" in str(p.parent).lower() else \
            "test"  if "test"  in p.name.lower() or "test"  in str(p.parent).lower() else "data"

    # モデル名をファイル名用にサニタイズ
    model_tag = model_name.replace("/", "__").replace("-", "_")

    base = f"preds_{name}_{split}_{model_tag}.jsonl" if name else f"preds_{split}_{model_tag}.jsonl"
    return str(Path("results") / base)

def main():
    parser = argparse.ArgumentParser()
    # argparse 追加
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="PEFT/LoRA adapter directory (e.g., ../train/outputs/llmjp1p8b-ja-cot-lora)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/gsm8k/main/test-00000-of-00001.parquet",
        help="Input dataset parquet path (default: GSM8K test parquet)",
    )
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "mgsm"],
        default="gsm8k",
        help="Dataset type: gsm8k (parquet) or mgsm (tsv)",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "ja", "zh"],
        default="en",
        help="Language for MGSM (used when --dataset mgsm)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSONL path (default: auto → preds_<name>_<split>.jsonl)",
    )
    parser.add_argument("--model", type=str, default="llm-jp/llm-jp-3-1.8b-instruct")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    LANG2FILE = {
        "en": "datasets/mgsm/mgsm_en.tsv",
        "ja": "datasets/mgsm/mgsm_ja.tsv",
        "zh": "datasets/mgsm/mgsm_zh.tsv",
    }

    model_name = args.model
    tok = AutoTokenizer.from_pretrained(model_name)

    if args.adapter:
        # まずベースをロードしてアダプタを差す（どのモデルでも動く安全ルート）
        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        model = PeftModel.from_pretrained(base, args.adapter)  # ★ ここでアダプタ装着
        # （任意）推論だけなら多少速く＆省メモリに： model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    from prompt_builder import make_prompt_builder
    prompt_builder = make_prompt_builder(model_name, tok)
    check_prompt(prompt_builder, model_name)

    # --- Load dataset ---
    if args.dataset == "mgsm":
        data_path = args.data if args.data is not None else LANG2FILE[args.lang]
        df = load_mgsm_tsv(data_path)
        questions = df["question"].tolist()
        gold = [extract_final_numeric(a) for a in df["answer"].tolist()]
    else:
        df = pd.read_parquet(args.data)
        questions = df["question"].tolist()
        gold = [extract_final_numeric(a) for a in df["answer"].tolist()]

    in_path = args.data if args.data is not None else (LANG2FILE[args.lang] if args.dataset=="mgsm" else None)
    out_path = args.out or default_out_from_input(in_path or "", args.model)
    desc = f"{Path(in_path).name if in_path else args.dataset} ({args.dataset.upper()})"
    gens, preds, correct, seen = solve_batch_with_progress(
        questions,
        gold,
        prompt_builder=prompt_builder,
        model=model,
        tok=tok,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        save_path=out_path,
        desc=desc,
    )
    print(f"Final accuracy: {correct}/{seen} = {correct/seen:.2%}")
    print(f"Saved predictions → {out_path}")

if __name__ == "__main__":
    main()


# translate_gsm8k_en_to_ja_jsonl.py
import argparse, json, re
from pathlib import Path
from glob import glob

import pandas as pd
from tqdm import tqdm
import vllm

PROMPT_TPL = """<|plamo:op|>dataset
translation
<|plamo:op|>input lang=English
{src}
<|plamo:op|>output lang=Japanese
"""

def build_prompt(text: str) -> str:
    return PROMPT_TPL.format(src=text)

def extract_gold(answer_en: str) -> str | None:
    """GSM8K の解説テキストから '#### 42' を抜き出して返す（カンマ除去）"""
    if answer_en is None:
        return None
    m = re.search(r"####\s*([-\d,]+(?:\.\d+)?)", str(answer_en))
    if not m:  # 念のため末尾の数値も許容
        m2 = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(answer_en))
        if not m2:
            return None
        val = m2[-1]
    else:
        val = m.group(1)
    return val.replace(",", "")

def batched_translate(llm, texts, batch_size=64):
    """英→日をバッチで翻訳して list[str] を返す"""
    sampling = vllm.SamplingParams(temperature=0.0, max_tokens=1024, stop=["<|plamo:op|>"])
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Translate en→ja", unit="ex"):
        batch = texts[i:i+batch_size]
        prompts = [build_prompt(t) for t in batch]
        resps = llm.generate(prompts, sampling)
        ja = [r.outputs[0].text.strip() if r.outputs else "" for r in resps]
        out.extend(ja)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help='入力 parquet のパス or グロブ（例: "datasets/gsm8k/main/train-*.parquet"）')
    ap.add_argument("--out", dest="out", required=True,
                    help="出力 JSONL（例: datasets/gsm8k_ja/train.jsonl）")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--model", default="pfnet/plamo-2-translate")
    ap.add_argument("--max-model-len", type=int, default=2000)
    ap.add_argument("--max-batched-tokens", type=int, default=2000)
    args = ap.parse_args()

    files = sorted(glob(args.inp)) or [args.inp]
    dfs = []
    for p in files:
        df = pd.read_parquet(p)[["question", "answer"]].copy()
        df["__source_file"] = Path(p).name
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # vLLM モデル
    llm = vllm.LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_batched_tokens,
    )

    # 翻訳（質問→日本語）
    qs_en = df["question"].tolist()
    qs_ja = batched_translate(llm, qs_en, batch_size=args.batch_size)

    # 翻訳（解説つき答え→日本語）
    ans_en = df["answer"].tolist()
    ans_ja = batched_translate(llm, ans_en, batch_size=args.batch_size)

    # gold 抽出
    golds = [extract_gold(a) for a in ans_en]

    # JSONL へ保存
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(len(df)):
            rec = {
                "id": i,
                "source_file": df["__source_file"].iloc[i],
                "question_en": qs_en[i],
                "question_ja": qs_ja[i],
                "answer_en": ans_en[i],   # 連鎖思考付きの英語解説
                "answer_ja": ans_ja[i],   # その日本語訳
                "gold": golds[i],         # 数値（文字列）。必要なら int(float(...)) で数値化
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(df)} examples → {out_path}")

if __name__ == "__main__":
    main()


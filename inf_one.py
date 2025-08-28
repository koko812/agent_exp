from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, re
import pandas as pd

df = pd.read_parquet("gsm8k/main/train-00000-of-00001.parquet")
print(df.head())

model_name = "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

def solve_one(question: str):
    # --- English prompt, no angle brackets, force final line format ---
    sys = (
        "You are a helpful math assistant. Solve the problem step by step. "
        "At the end, output exactly one final line in the format: Answer: number"
    )
    usr = (
        f"Problem: {question}\n"
        "Write intermediate reasoning first.\n"
        "Finally, output exactly one line starting with 'Answer:' followed by the final number only."
    )

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ]

    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # keep thinking OFF
    )
    inputs = tok([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,                    # baseline: deterministic
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )

    gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # --- Extraction: prefer explicit "Answer: <num>", else last integer anywhere (DOTALL) ---
    m = re.search(r"Answer:\s*(-?\d+)\s*$", gen, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        pred = int(m.group(1))
    else:
        # fallback: last integer in the whole text (handles newlines via DOTALL)
        m2 = re.search(r"(?s)(-?\d+)(?!.*\d)", gen)
        pred = int(m2.group(1)) if m2 else None

    return gen, pred


def gold_from_gsm8k(answer_field: str):
    m = re.search(r"####\s*(-?\d+)", answer_field)
    return int(m.group(1)) if m else None

row = df.iloc[0]
gen, pred = solve_one(row["question"])
gold = gold_from_gsm8k(row["answer"])
print("=== MODEL OUTPUT ===\n", gen)
print("\nPRED:", pred, " GOLD:", gold, "  CORRECT:", pred == gold)


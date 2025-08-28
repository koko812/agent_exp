# prompt_builder.py

def build_prompt_qwen(tok):
    SYS = (
        "You are a helpful math assistant. Solve the problem step by step. "
        "At the end, output exactly one final line in the format: Answer: number"
    )
    def f(q: str) -> str:
        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content":
             f"Problem: {q}\nWrite intermediate reasoning first.\n"
             "Finally, output exactly one line starting with 'Answer:' followed by the final number only."},
        ]
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    return f


def build_prompt_llmjp_base(_tok):
    SYS = (
        "You are a helpful math assistant. Solve the problem step by step. "
        "At the end, output exactly one final line in the format: Answer: number"
    )
    def f(q: str) -> str:
        return (
            f"{SYS}\n\n"
            f"Problem: {q}\n"
            "Write intermediate reasoning first.\n"
            "Finally, output exactly one line starting with 'Answer:' followed by the final number only."
        )
    return f


def _build_prompt_llmjp_instruct(tok):
    SYS = (
        "You are a helpful math assistant. Solve the problem step by step. "
        "At the end, output exactly one final line in the format: Answer: number"
    )
    def f(q: str) -> str:
        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content":
             f"Problem: {q}\nWrite intermediate reasoning first.\n"
             "Finally, output exactly one line starting with 'Answer:' followed by the final number only."},
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f

# prompt_builder.py に追加 or 置き換え

def __build_prompt_llmjp_instruct(tok, use_fewshot=False):
    SYS = (
        "あなたは算数問題を解くアシスタントです。推論過程を段階的に示し、"
        "最後に必ず 1 行だけ 'Answer: 数値' の形式で最終解を出力してください。"
        "角括弧などは使わず、数値のみを書いてください。"
    )

    # few-shot は (question, rationale, final_answer) で管理
    shots = []
    if use_fewshot:
        shots = [
            ("Ken has 7 apples and buys 5 more. How many apples does he have?",
             "7 + 5 = 12.",
             "12"),
            ("A box has 20 pencils. He gives away half. How many are left?",
             "Half of 20 is 10, so 20 - 10 = 10.",
             "10"),
        ]

    def f(q: str) -> str:
        msgs = [{"role": "user", "content": SYS}]
        for qs, rat, ans in shots:
            msgs.append({"role": "user", "content": f"問題: {qs}"})
            msgs.append({"role": "assistant", "content": f"考え方: {rat}\nAnswer: {ans}"})
        msgs.append({
            "role": "user",
            "content": (
                f"問題: {q}\n"
            )
        })
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    return f

def build_prompt_llmjp_instruct(tok, use_fewshot=False):
    SYS = (
        "あなたは算数問題を解くアシスタントです。推論過程を段階的に示し、"
        "最後に必ず 1 行だけ 'Answer: 数値' の形式で最終解を出力してください。"
        "角括弧などは使わず、数値のみを書いてください。"
    )

    # few-shot は (question, rationale, final_answer) で管理
    shots = []
    if use_fewshot:
        shots = [
            ("ケンは7個のリンゴを持っています，さらに5個買った場合合計何個リンゴを持っていますか？",
             "7 + 5 = 12.",
             "12"),
            ("箱には２０個ボールが入っています．彼はそのうち半分を持っていきました．箱には何個ボールが残っていますか？",
             "20 の半分は 10 です. 20 - 10 = 10.",
             "10"),
        ]

    def f(q: str) -> str:
        msgs = [{"role": "user", "content": SYS}]
        for qs, rat, ans in shots:
            msgs.append({"role": "user", "content": f"問題: {qs}"})
            msgs.append({"role": "assistant", "content": f"考え方: {rat}\nAnswer: {ans}"})
        msgs.append({
            "role": "user",
            "content": (
                f"問題: {q}\n"
            )
        })
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    return f


def make_prompt_builder(model_name, tok):
    name = model_name.lower()
    if "instruct" in name:
        return build_prompt_llmjp_instruct(tok, use_fewshot=True)
    if "qwen" in name:
        return build_prompt_qwen(tok)
    # default: base
    return build_prompt_llmjp_base(tok)


import json, gzip, re
from config import CORPUS_JSONL, CHUNKS_JSONL_GZ, MAX_TOK, OVERLAP

def simple_tokenize(t: str):
    return t.split()

def detok(tokens):
    return " ".join(tokens)

def section_split(text: str):
    # ざっくりな見出し検出（雑でOK→後で改善）
    pat = re.compile(r'\n\s*(\d+(\.\d+)*)\s+[A-Z][^\n]{3,}|^\s*[IVXLC]+\.\s+[A-Z]', re.M)
    idxs = [m.start() for m in pat.finditer(text)]
    if not idxs:
        return [text]
    idxs = sorted(set([0] + idxs))  # 先頭を含める
    cuts = idxs[1:] + [len(text)]
    return [text[s:e] for s, e in zip(idxs, cuts)]

def window(tokens, max_len=1000, overlap=150):
    step = max_len - overlap
    for i in range(0, max(len(tokens)-1, 0)+1, step):
        yield tokens[i:i+max_len]

def make_chunks(rec):
    base = {
        "doc_id": rec.get("arxiv_id") or rec.get("id"),
        "title": rec.get("title"),
        "year": rec.get("year"),
        "venue": rec.get("venue"),
        "citationCount": rec.get("citationCount"),
        "topic": rec.get("topic"),
        "source_url": rec.get("source_url") or rec.get("pdf"),
    }
    text = rec.get("text") or rec.get("abstract") or ""
    secs = section_split(text)
    out = []
    for sec in secs:
        toks = simple_tokenize(sec)
        for w in window(toks, MAX_TOK, OVERLAP):
            out.append({**base, "text": detok(w)})
    return out

def main():
    count_in, count_out = 0, 0
    with open(CORPUS_JSONL, "r") as fin, gzip.open(CHUNKS_JSONL_GZ, "wt", encoding="utf-8") as fout:
        for line in fin:
            count_in += 1
            rec = json.loads(line)
            chunks = make_chunks(rec)
            for c in chunks:
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")
                count_out += 1
    print(f"docs={count_in}, chunks={count_out}, saved -> {CHUNKS_JSONL_GZ}")

if __name__ == "__main__":
    main()


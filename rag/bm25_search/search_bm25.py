import json, gzip, pickle
from rank_bm25 import BM25Okapi
from config import CHUNKS_JSONL_GZ, BM25_PICKLE, BM25_TOK_PICKLE, DOCS_PICKLE

def yield_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    docs = []      # メタ＋テキスト
    corpus = []    # テキストのみ
    tokenized = [] # BM25用 token

    for rec in yield_jsonl_gz(CHUNKS_JSONL_GZ):
        docs.append(rec)
        text = rec["text"]
        corpus.append(text)
        tokenized.append(text.lower().split())

    bm25 = BM25Okapi(tokenized)
    with open(BM25_PICKLE, "wb") as f: pickle.dump(bm25, f)
    with open(BM25_TOK_PICKLE, "wb") as f: pickle.dump(tokenized, f)
    with open(DOCS_PICKLE, "wb") as f: pickle.dump(docs, f)

    print(f"BM25 saved: {BM25_PICKLE}")
    print(f"Tokenized saved: {BM25_TOK_PICKLE}")
    print(f"Docs saved: {DOCS_PICKLE}")
    print(f"chunks={len(docs)}")

if __name__ == "__main__":
    main()


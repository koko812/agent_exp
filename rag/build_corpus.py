# build_corpus.py
import json, re, time, pathlib, sys, datetime
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

IN_FILES = [
    "paper_urls/nerf_papers_topcited.json",
    "paper_urls/resnet_topcited.json",
    "paper_urls/transformer_topicted.json",
]
OUT_PATH = "corpus.jsonl"
LOG_DIR = pathlib.Path("log"); LOG_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "corpus-builder/0.1 (+research-use)"}

def latest_ar5iv(url: str) -> str:
    return re.sub(r"v\d+$", "", url.strip())

def html_to_text(ar5iv_url: str, timeout=60) -> str:
    r = requests.get(latest_ar5iv(ar5iv_url), headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    blocks = [p.get_text(" ", strip=True) for p in soup.select("section p, p") if p.get_text(strip=True)]
    text = "\n\n".join(blocks)
    text = re.split(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", text)[0]
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()

def iter_records(paths):
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            for rec in data:
                yield rec

def main():
    start_time = time.time()
    out = open(OUT_PATH, "w", encoding="utf-8")
    kept = skipped = 0

    records = list(iter_records(IN_FILES))
    for rec in tqdm(records, desc="Building corpus"):
        try:
            txt = html_to_text(rec["ar5iv_html"])
        except Exception:
            skipped += 1
            continue
        if len(txt) < 2000:
            skipped += 1
            continue

        item = {
            "arxiv_id": rec.get("arxiv_id"),
            "title": rec.get("title"),
            "year": rec.get("year"),
            "venue": rec.get("venue", "arXiv"),
            "citationCount": rec.get("citationCount"),
            "arxiv_published": rec.get("arxiv_published"),
            "source_url": rec.get("ar5iv_html"),
            "pdf": rec.get("pdf"),
            "text": txt,
        }
        out.write(json.dumps(item, ensure_ascii=False) + "\n")
        kept += 1
        time.sleep(0.2)  # polite

    out.close()
    elapsed = time.time() - start_time
    msg = f"[done] kept={kept}, skipped={skipped}, elapsed={elapsed:.1f}s, saved -> {OUT_PATH}"
    print(msg)

    # ログファイルに書き出し
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"build_corpus_{now}.log"
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(msg + "\n")
    print(f"log saved to {log_path}")

if __name__ == "__main__":
    main()


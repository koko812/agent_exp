# collect_topcited_arxiv.py
# Usage:
#   SEMANTIC_SCHOLAR_API_KEY=xxxx uv run collect_topcited_arxiv.py
# Optional env vars:
#   ARXIV_QUERY, MAX_RESULTS=1000, BATCH_SIZE=5, CITATION_THRESHOLD=50, OUTPATH=nerf_topcited.json

import os, re, time, json, requests
import feedparser, urllib.parse as up
from dateutil import parser
from tqdm import tqdm

# ===== Config =====
API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
QUERY = os.environ.get("ARXIV_QUERY") or \
        '( (ti:NeRF OR all:"neural radiance field" OR all:"neural radiance fields" OR ti:"novel view synthesis") AND (cat:cs.CV OR cat:eess.IV OR cat:cs.GR) )'
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "2000"))
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE",  "500"))
CITATION_THRESHOLD = int(os.environ.get("CITATION_THRESHOLD", "50"))
OUTPATH = os.environ.get("OUTPATH", "resnet_topcited.json")

ARXIV_API = "https://export.arxiv.org/api/query"
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "title,year,venue,citationCount,externalIds"

# ===== helpers =====
NEW_ID = re.compile(r"^\d{4}\.\d{5}$")
OLD_ID = re.compile(r"^[a-z\-]+(\.[A-Z]{2})?/\d+$")    # 旧式ID

def norm_id(aid: str) -> str:
    return re.sub(r'v\d+$', '', (aid or '').strip())

def valid_arxiv_id(aid: str) -> bool:
    return bool(NEW_ID.match(aid) or OLD_ID.match(aid))

# ===== arXiv: 自前ページング（空ページはスキップ/2回連続で打ち切り） =====
def fetch_ids_with_meta(query, total=2000, page_size=100, sleep=0.5, order="descending"):
    out, start, empty_hits = [], 0, 0
    while len(out) < total:
        params = {
            "search_query": query,
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": order,
        }
        url = f"{ARXIV_API}?{up.urlencode(params)}"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        feed = feedparser.parse(r.text)

        if not feed.entries:
            empty_hits += 1
            if empty_hits >= 2: break
            start += page_size
            continue

        empty_hits = 0
        for e in feed.entries:
            raw = e.id.rsplit('/', 1)[-1]          # 例: 2003.08934v1
            aid = norm_id(raw)
            if not valid_arxiv_id(aid): continue
            title = (e.title or "").strip()
            pub = parser.parse(e.published).date().isoformat() if getattr(e, "published", None) else None
            pdf = next((L["href"] for L in getattr(e, "links", []) if "pdf" in L.get("href","")), f"https://arxiv.org/pdf/{aid}.pdf")
            out.append({
                "arxiv_id": aid,
                "title": title,
                "arxiv_published": pub,
                "pdf": pdf,
                "ar5iv_html": f"https://ar5iv.org/html/{aid}",
            })
            if len(out) >= total: break
            if len(out) % 100 == 0:
                print(f"Fetched {len(out)} papers so far...")
        start += page_size
        time.sleep(sleep)

    return out

# ===== Semantic Scholar: batch（400は分割、429は待機） =====
def s2_batch(ids, api_key=API_KEY):
    clean = []
    for x in ids:
        z = norm_id(x)
        if z and valid_arxiv_id(z):
            clean.append(z)
        else:
            clean.append(None)

    # 全て不正ならNone列で返す
    if all(z is None for z in clean):
        return [None] * len(ids)

    payload_ids = [f"arXiv:{z}" if z else None for z in clean]
    # Noneは個別に置換するので、APIに渡す配列は有効IDのみ
    idx_map, api_ids = {}, []
    for i, z in enumerate(payload_ids):
        if z is not None:
            idx_map[len(api_ids)] = i
            api_ids.append(z)

    headers = {"Content-Type": "application/json"}
    if api_key: headers["x-api-key"] = api_key
    params = {"fields": S2_FIELDS}

    def call_api(sub_ids):
        if not sub_ids: return []
        r = requests.post(S2_BATCH_URL, headers=headers, params=params, json={"ids": sub_ids}, timeout=120)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 1))
            time.sleep(wait)
            return call_api(sub_ids)
        if r.status_code == 400 and len(sub_ids) > 1:
            mid = len(sub_ids)//2
            return call_api(sub_ids[:mid]) + call_api(sub_ids[mid:])
        r.raise_for_status()
        return r.json()

    api_res = call_api(api_ids)

    # 入力順復元（元のidsの長さに合わせる）
    out = [None] * len(ids)
    for j, item in enumerate(api_res):
        i = idx_map[j]
        out[i] = item if (item and isinstance(item, dict)) else None
    return out

# ===== Main =====
def main():
    print("Query:", QUERY)
    print("Fetching arXiv ids…")
    papers = fetch_ids_with_meta(QUERY, total=MAX_RESULTS, page_size=100, order="descending")
    print(f"got {len(papers)} ids")

    results = []
    print("Fetching from Semantic Scholar (batch)…")
    for i in tqdm(range(0, len(papers), BATCH_SIZE)):
        chunk = papers[i:i+BATCH_SIZE]
        ids = [p["arxiv_id"] for p in chunk]
        s2_list = s2_batch(ids)

        for p, rec in zip(chunk, s2_list):
            rec = rec if (rec and isinstance(rec, dict)) else {}
            title = rec.get("title", p["title"])
            year  = rec.get("year", int(p["arxiv_published"][:4]) if p.get("arxiv_published") else None)
            venue = rec.get("venue", "arXiv")
            cites = rec.get("citationCount")

            if cites is None or cites < CITATION_THRESHOLD:
                continue

            results.append({
                "arxiv_id": p["arxiv_id"],
                "title": title,
                "year": year,
                "venue": venue,
                "citationCount": cites,
                "arxiv_published": p.get("arxiv_published"),
                "ar5iv_html": p["ar5iv_html"],
                "pdf": p["pdf"],
            })
        time.sleep(1.1)  # 1 req/sec

    results.sort(key=lambda x: (x.get("citationCount", 0), x.get("year", 0)), reverse=True)
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} papers (citationCount ≥ {CITATION_THRESHOLD}) to {OUTPATH}")

if __name__ == "__main__":
    main()


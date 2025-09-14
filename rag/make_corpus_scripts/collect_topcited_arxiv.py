# collect_topcited_arxiv.py
# Usage:
#   SEMANTIC_SCHOLAR_API_KEY=xxxx uv run collect_topcited_arxiv.py
# Optional env vars:
#   MAX_RESULTS=1000, BATCH_SIZE=5, CITATION_THRESHOLD=50

import os, re, time, json, requests, arxiv
from dateutil import parser
from tqdm import tqdm

# ========= Config =========
API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

#QUERY = os.environ.get("ARXIV_QUERY") or \
        #        '(all:"neural radiance field" OR all:"neural radiance fields" OR ti:NeRF OR ti:"novel view synthesis") AND (cat:cs.CV OR cat:eess.IV OR cat:cs.GR)'
QUERY = '(all:"auto encoder" OR all:"autoencoder")'

MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "10000"))         # arXivからの最大取得数
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE",  "500"))            # S2への問い合わせ単位
CITATION_THRESHOLD = int(os.environ.get("CITATION_THRESHOLD", "1000"))
OUTPATH = os.environ.get("OUTPATH", "transformer_papers_topcited.json")

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "title,year,venue,citationCount,externalIds"

# ========= ID 正規化/検証 =========
NEW_ID = re.compile(r"^\d{4}\.\d{5}$")                 # 2017.12345
OLD_ID = re.compile(r"^[a-z\-]+(\.[A-Z]{2})?/\d+$")    # cs/0701234, math-ph/0601001 など

def norm_id(aid: str) -> str:
    aid = (aid or "").strip()
    return re.sub(r'v\d+$', '', aid)  # 2509.07242v1 -> 2509.07242

def valid_arxiv_id(aid: str) -> bool:
    return bool(NEW_ID.match(aid) or OLD_ID.match(aid))

# ========= Semantic Scholar Batch =========
def s2_batch(ids, api_key=API_KEY):
    """
    ids: arXiv短縮IDのリスト（例: ["1706.03762", "2402.09353"]）
    返り: 入力順に対応する結果のリスト（None混在あり得る）
    """
    # 正規化 + 検証 + 重複排除（入力順維持）
    clean, bad, seen = [], [], set()
    for x in ids:
        z = norm_id(x)
        if not z or not valid_arxiv_id(z):
            bad.append(x); continue
        if z in seen:
            continue
        seen.add(z); clean.append(z)
    if bad:
        print(f"[warn] skip invalid ids: {bad[:5]}{' ...' if len(bad)>5 else ''}")

    if not clean:
        return [None] * len(ids)  # 入力長に合わせて返す

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    params  = {"fields": S2_FIELDS}
    payload = {"ids": [f"arXiv:{z}" for z in clean]}

    r = requests.post(S2_BATCH_URL, headers=headers, params=params, json=payload, timeout=120)

    # レート制限
    if r.status_code == 429:
        wait = int(r.headers.get("Retry-After", 1))
        print(f"[rate] 429 Too Many Requests; wait {wait}s")
        time.sleep(wait)
        return s2_batch(ids, api_key)

    # 400なら半分割で原因切り分け（入力順で再構成）
    if r.status_code == 400:
        if len(clean) == 1:
            print(f"[error] 400 on single id: {clean[0]} -> {r.text[:160]}")
            # 元のidsに合わせてNoneを返す（位置合わせ）
            m = {clean[0]: None}
            return [m.get(norm_id(x), None) for x in ids]
        mid = len(clean) // 2
        left  = s2_batch(clean[:mid], api_key)
        right = s2_batch(clean[mid:], api_key)
        merged = (left or []) + (right or [])
        # cleanは重複排除後の列なので、入力idsに再マップ
        idx_map = {c:i for i, c in enumerate(clean)}
        out = []
        for x in ids:
            z = norm_id(x)
            if z in idx_map:
                out.append(merged[idx_map[z]])
            else:
                out.append(None)
        return out

    r.raise_for_status()
    res = r.json()  # list( dict | None )
    # cleanベースの配列 → idsに再マップ
    idx_map = {c:i for i, c in enumerate(clean)}
    out = []
    for x in ids:
        z = norm_id(x)
        if z in idx_map:
            out.append(res[idx_map[z]])
        else:
            out.append(None)
    return out

# ========= arXiv 取得 =========
def fetch_arxiv(query: str, max_results: int):
    client = arxiv.Client(num_retries=0, delay_seconds=0)  # 空ページで粘らない
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
        #sort_order=arxiv.SortOrder.Ascending,
    )
    papers = []
    try:
        for r in client.results(search):
            papers.append(r)
            if len(papers) % 100 == 0:
                print(f"Fetched {len(papers)} from arXiv…")
    except arxiv.UnexpectedEmptyPageError:
        print(f"[info] Reached empty page at {len(papers)}; stopping early.")
    print(f"Total fetched from arXiv: {len(papers)}")
    return papers

# ========= Main =========
def main():
    print("Query:", QUERY)
    papers = fetch_arxiv(QUERY, MAX_RESULTS)

    results = []
    print("Fetching paper infos from Semantic Scholar (batch)…")

    # バッチサイズごとに問い合わせ（1 req/sec に配慮）
    for i in tqdm(range(0, len(papers), BATCH_SIZE)):
        chunk = papers[i:i+BATCH_SIZE]
        ids = [p.get_short_id() for p in chunk]
        s2_list = s2_batch(ids)

        # 入力順に対応づけて処理（None耐性）
        for p, item in zip(chunk, s2_list):
            aid = p.get_short_id()
            rec = item if (item and isinstance(item, dict)) else {}
            title = rec.get("title", p.title.strip())
            year  = rec.get("year", parser.parse(p.published.isoformat()).year)
            venue = rec.get("venue", "arXiv")
            cites = rec.get("citationCount")
            arxiv_pub = parser.parse(p.published.isoformat()).date().isoformat()

            # フィルタ：被引用数閾値
            if cites is None or cites < CITATION_THRESHOLD:
                continue

            entry = {
                "arxiv_id": aid,
                "title": title,
                "year": year,                     # 出版年（S2基準、無ければ投稿年）
                "venue": venue,                   # 出版元（会議/ジャーナル or arXiv）
                "citationCount": cites,
                "arxiv_published": arxiv_pub,     # arXiv投稿日
                "ar5iv_html": f"https://ar5iv.org/html/{aid}",
                "pdf": p.pdf_url,
            }
            results.append(entry)
            print(f'- {title} ({year}, {venue}) Citations: {cites}')

        time.sleep(1.1)  # 1 req/sec 制限に配慮

    # ソート（引用数降順→年降順）
    results.sort(key=lambda x: (x.get("citationCount", 0), x.get("year", 0)), reverse=True)

    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} papers (citationCount ≥ {CITATION_THRESHOLD}) to {OUTPATH}")

if __name__ == "__main__":
    main()


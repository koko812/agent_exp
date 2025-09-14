# collect_rag_arxiv_with_citations_batch.py
import arxiv
import requests, json, time, os, re
from dateutil import parser

API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

#QUERY = '(all:"retrieval-augmented" OR ti:"retrieval augmented generation" OR ti:RAG OR ti:rerank OR ti:HyDE OR ti:RAPTOR) AND (cat:cs.CL OR cat:cs.IR)'
#QUERY = 'ti:"DoRA"'
QUERY = '(ti:NeRF OR ti:"neural radiance field" OR ti:"neural radiance fields")'


S2_BASE = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "title,year,venue,citationCount,externalIds"

def normalize_arxiv_id(aid: str) -> str:
    return re.sub(r'v\d+$', '', aid)

def enrich_batch(arxiv_ids, api_key=API_KEY):
    """Semantic Scholar の batch API でまとめて問い合わせ"""
    ids = [f"arXiv:{normalize_arxiv_id(aid)}" for aid in arxiv_ids]
    headers = {"x-api-key": api_key} if api_key else {}
    params = {"fields": FIELDS}
    resp = requests.post(S2_BASE, headers=headers, params=params, json={"ids": ids}, timeout=60)
    if resp.status_code == 200:
        return resp.json()
    elif resp.status_code == 429:
        wait = int(resp.headers.get("Retry-After", 2))
        print(f"Rate limit hit, waiting {wait}s…")
        time.sleep(wait)
        return enrich_batch(arxiv_ids, api_key)
    else:
        print(f"[Error] {resp.status_code}: {resp.text[:200]}")
        return []

results = []
client = arxiv.Client()

# === まず arXiv から 50 件取る ===
search = arxiv.Search(
    query=QUERY,
    max_results=50,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)

# === 5件ずつバッチ処理 ===
all_papers = list(client.results(search))
for i in range(0, len(all_papers), 5):
    batch = all_papers[i:i+5]
    arxiv_ids = [r.get_short_id() for r in batch]

    # Semantic Scholar batch API
    s2_results = enrich_batch(arxiv_ids)

    # マッピング: arXiv ID → S2情報
    # s2_results は list（ときどき None を含む）
    s2_map = {}
    for aid, item in zip(arxiv_ids, s2_results or []):
        norm = normalize_arxiv_id(aid)
        if not item or not isinstance(item, dict) or not item.get("paperId"):
            # 未収載/未統合など → 空dictでフォールバック
            s2_map[norm] = {}
            continue
        ext = item.get("externalIds") or {}
        sid = normalize_arxiv_id(ext.get("ArXiv") or ext.get("arXiv") or norm)
        s2_map[sid] = item


    # 出力組み立て
    for r in batch:
        short_id = r.get_short_id()
        arxiv_year = parser.parse(r.published.isoformat()).year
        s2 = s2_map.get(normalize_arxiv_id(short_id), {})

        entry = {
            "arxiv_id": short_id,
            "title": s2.get("title", r.title.strip()),
            "year":  s2.get("year", parser.parse(r.published.isoformat()).year),
            "date":  r.published.isoformat(),
            "venue": s2.get("venue", "arXiv"),
            "citationCount": s2.get("citationCount"),  # 未収載は None のままOK
            "ar5iv_html": f"https://ar5iv.org/html/{short_id}",
            "pdf": r.pdf_url,
        }


        results.append(entry)

        print(f'- {entry["title"]} ({entry["year"]}, {entry["date"]}, {entry["venue"]}) '
              f'Citations: {entry["citationCount"]}')

    time.sleep(1.1)  # レート制限対策（1リクエスト/秒）

# === JSON 保存 ===
with open("rag_papers.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nSaved 50 papers to rag_papers.json")


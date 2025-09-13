# collect_rag_arxiv_with_citations.py
import arxiv
import requests, json, time
from dateutil import parser
import re
import os

API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")



# === Step 1: arXiv 検索 ===
#QUERY = '(all:"retrieval-augmented" OR ti:"retrieval augmented generation" OR ti:RAG OR ti:rerank OR ti:HyDE OR ti:RAPTOR) AND (cat:cs.CL OR cat:cs.IR)'
QUERY = 'ti:"DoRA"'


search = arxiv.Search(
    query=QUERY,
    max_results=20,  # まずは少なめで
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)

# === Step 2: Semantic Scholar から enrich ===
S2_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,year,venue,citationCount"

def normalize_arxiv_id(aid: str) -> str:
    return re.sub(r'v\d+$', '', aid)  # 2509.07242v1 → 2509.07242



def enrich_with_semanticscholar(arxiv_id, api_key=API_KEY):
    url = f"{S2_BASE}/paper/arXiv:{normalize_arxiv_id(arxiv_id)}"
    headers = {"x-api-key": api_key} if api_key else {}
    r = requests.get(url, params={"fields": FIELDS}, headers=headers, timeout=30)
    if r.status_code == 200:
        print("get")
        return r.json()
    elif r.status_code == 429:
        time.sleep(2)  # レート制限 → 待つ
        return enrich_with_semanticscholar(arxiv_id, api_key)
    else:
        print("error")
        return {"error": r.text}

results = []
for r in arxiv.Client().results(search):
    short_id = r.get_short_id()
    arxiv_year = parser.parse(r.published.isoformat()).year

    # Semantic Scholar 情報を付与
    s2 = enrich_with_semanticscholar(short_id)

    entry = {
        "arxiv_id": short_id,
        "title": s2.get("title", r.title.strip()),
        "year": s2.get("year", arxiv_year),
        "venue": s2.get("venue", "arXiv"),   # 会議名/ジャーナル名、なければ arXiv
        "citationCount": s2.get("citationCount", None),
        "ar5iv_html": f"https://ar5iv.org/html/{short_id}",
        "pdf": r.pdf_url,
    }
    results.append(entry)

    # 確認用に print
    print(f'- {entry["title"]} ({entry["year"]}, {entry["venue"]}) '
          f'Citations: {entry["citationCount"]}')

# === Step 3: JSON 出力 ===
with open("rag_papers.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nSaved to rag_papers.json")


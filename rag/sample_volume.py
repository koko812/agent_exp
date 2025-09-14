# sample_volume.py
import json, re, requests
from bs4 import BeautifulSoup

def read_jsonl_or_json(path, limit=20):
    # 10行=1レコード前提の整形済みJSONなら json.load でOK
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit]

def html_to_text(url):
    html = requests.get(url, timeout=60).text
    soup = BeautifulSoup(html, "html.parser")
    # セクション本文中心に抽出（荒くてもOK）
    texts = [p.get_text(" ", strip=True) for p in soup.select("section p, p")]
    txt = "\n\n".join(texts)
    # 連続空白を詰める
    txt = re.sub(r"\s+\n", "\n", re.sub(r"[ \t]+", " ", txt))
    return txt

def rough_token_count(text):
    # 英文なら「4文字 ≒ 1トークン」くらいの目安
    return int(len(text) / 4)

if __name__ == "__main__":
    paths = [
        "paper_urls/nerf_papers_topcited.json",
        "paper_urls/resnet_topcited.json",
        "paper_urls/transformer_topicted.json",
    ]
    sample_n = 6  # 各ファイルから少数ずつ
    total_chars = total_tokens = 0
    sampled = 0

    for p in paths:
        items = read_jsonl_or_json(p, limit=sample_n)
        for it in items:
            url = it["ar5iv_html"]
            # バージョン付きURLを見かけたら最後の vN は落として最新版でもOK
            url = re.sub(r"v\d+$", "", url)
            txt = html_to_text(url)
            total_chars += len(txt)
            total_tokens += rough_token_count(txt)
            sampled += 1
            print(f"[{sampled}] {it['title'][:60]}… chars={len(txt):,} ~tokens≈{rough_token_count(txt):,}")

    avg_chars = total_chars // max(sampled,1)
    avg_tokens = total_tokens // max(sampled,1)
    print(f"\nAvg per paper: {avg_chars:,} chars, ~{avg_tokens:,} tokens")
    est_total_tokens = avg_tokens * 920  # 全体見込み
    est_chunks = est_total_tokens // 700
    print(f"Estimated total: ~{est_total_tokens:,} tokens → ~{est_chunks:,} chunks (800t chunk, 100t overlap)")


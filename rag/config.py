from pathlib import Path

# ルート（このスクリプトを置いたプロジェクト直下を想定）
PROJECT_DIR = Path(__file__).resolve().parent

# 入力コーパス（今ある corpus.jsonl を読む）
CORPUS_JSONL = PROJECT_DIR / "../datasets/arxiv_corpus/corpus.jsonl"

# 出力フォルダ（指定どおり）
DATA_DIR = (PROJECT_DIR / "../datasets/arxiv_corpus").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 生成物
CHUNKS_JSONL_GZ = DATA_DIR / "chunks.jsonl.gz"
BM25_PICKLE     = DATA_DIR / "bm25.pkl"
BM25_TOK_PICKLE = DATA_DIR / "bm25_tokenized.pkl"
DOCS_PICKLE     = DATA_DIR / "docs.pkl"

# チャンク設定（あとで変えてOK）
MAX_TOK = 1000
OVERLAP = 150


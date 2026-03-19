"""
All tunables for the RAG data preparation pipeline.
Every other module imports constants from here — never scatter magic values.
"""

SEED_URLS = [
    "https://eecs.berkeley.edu/",
    "https://www2.eecs.berkeley.edu/",
    "https://cs.berkeley.edu/",
]

ALLOWED_DOMAINS = {
    "eecs.berkeley.edu",
    "www2.eecs.berkeley.edu",
    "cs.berkeley.edu",
}

# Hard cap on total pages crawled
MAX_PAGES = 20000

# Minimum seconds between requests to the same domain
CRAWL_DELAY_SECONDS = 0.0

# Per-domain overrides (take precedence over both CRAWL_DELAY_SECONDS and robots.txt)
DOMAIN_CRAWL_DELAYS = {
    "www2.eecs.berkeley.edu": 0.5,
}

REQUEST_TIMEOUT_SECONDS = 10

MAX_RETRIES = 3

# Base multiplier for exponential backoff between retries
RETRY_BACKOFF_BASE = 2.0

USER_AGENT = "UCB-ANLP-RAG-Crawler/1.0 (educational; cs-course-project)"

# "resiliparse" (primary) or "bs4" (fallback / override)
EXTRACTOR = "resiliparse"

# Pages with fewer extracted characters than this are discarded
MIN_CONTENT_LENGTH = 20

# Chunker
CHUNK_SIZE = 100

# Retriever
BM25_TOP_K = 5

# Generator
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"
MAX_NEW_TOKENS = 32

RAW_PAGES_DIR = "data/raw_pages"
CORPUS_JSONL_PATH = "data/corpus.jsonl"
CORPUS_COMBINED_PATH = "data/corpus_combined.jsonl"

# --- Embedding (Dense Retrieval) ---
EMBEDDING_MODEL = "models/gte-modernbert-base-fp16"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_QUERY_PREFIX = ""
EMBEDDINGS_PATH = "data/embeddings.npy"
FAISS_INDEX_PATH = "data/faiss_index.bin"
DENSE_TOP_K = 5

# --- Chunking ---
CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 50
# Documents with fewer words than this are kept whole (not chunked)
CHUNK_MIN_DOC_WORDS = 200
CHUNKS_JSONL_PATH = "data/chunks.jsonl"

# --- Reranking ---
RERANKING_ENABLED = False        # Toggle without code changes
RERANK_RETRIEVE_K = 20           # Retrieve this many, then rerank down to RERANK_TOP_K
RERANK_TOP_K = 5                 # Keep this many after reranking (passed to generator)

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
MAX_PAGES = 5000

# Minimum seconds between requests to the same domain
CRAWL_DELAY_SECONDS = 1.0

REQUEST_TIMEOUT_SECONDS = 10

MAX_RETRIES = 3

# Base multiplier for exponential backoff between retries
RETRY_BACKOFF_BASE = 2.0

USER_AGENT = "UCB-ANLP-RAG-Crawler/1.0 (educational; cs-course-project)"

# "resiliparse" (primary) or "bs4" (fallback / override)
EXTRACTOR = "resiliparse"

# Pages with fewer extracted characters than this are discarded
MIN_CONTENT_LENGTH = 100

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

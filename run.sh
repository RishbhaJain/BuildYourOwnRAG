#!/usr/bin/env bash
# RAG pipeline entrypoint.
# Usage: bash run.sh <questions_path> <predictions_path>

set -euo pipefail

# Load .env if present
if [ -f .env ]; then
    set -a; source .env; set +a
fi

python3 download_model.py

# Rebuild FAISS index from embeddings if missing (index is not stored in git)
python3 -c "
import os, numpy as np, faiss, config
from pathlib import Path
from retriever.artifacts import write_index_manifest
if not os.path.exists(config.FAISS_INDEX_PATH) or not os.path.exists(config.FAISS_INDEX_MANIFEST_PATH):
    print('Rebuilding FAISS index from embeddings...')
    emb = np.load(config.EMBEDDINGS_PATH).astype('float32')
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    write_index_manifest(
        Path(config.FAISS_INDEX_MANIFEST_PATH),
        Path(config.FAISS_INDEX_PATH),
        Path(config.RETRIEVAL_MANIFEST_PATH),
        rows=index.ntotal,
        dimensions=index.d,
    )
    print(f'Index built: {index.ntotal} vectors -> {config.FAISS_INDEX_PATH}')
"

python3 run_pipeline.py "$1" "$2" --retriever hybrid --top-k 20

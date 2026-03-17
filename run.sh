#!/usr/bin/env bash
# RAG pipeline entrypoint.
# Usage: bash run.sh <questions_path> <predictions_path>

set -euo pipefail

python3 download_model.py

python3 run_pipeline.py "$1" "$2" --retriever hybrid --top-k 10

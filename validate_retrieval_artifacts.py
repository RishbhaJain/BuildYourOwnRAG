"""Validate that chunks and embeddings belong to the same retrieval build."""

import argparse
import json
from pathlib import Path

import config
from retriever.artifacts import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=Path(config.CHUNKS_JSONL_PATH))
    parser.add_argument("--embeddings", type=Path, default=Path(config.EMBEDDINGS_PATH))
    parser.add_argument(
        "--manifest", type=Path, default=Path(config.RETRIEVAL_MANIFEST_PATH)
    )
    args = parser.parse_args()
    report = validate_manifest(
        args.manifest,
        args.chunks,
        args.embeddings,
        embedding_model=config.EMBEDDING_MODEL,
        query_prefix=config.EMBEDDING_QUERY_PREFIX,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

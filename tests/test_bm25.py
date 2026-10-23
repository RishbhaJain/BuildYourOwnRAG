import json

from retriever.bm25_retriever import BM25Retriever


def _write_chunks(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = [
        {"chunk_id": "transformers", "text": "transformer attention architecture"},
        {"chunk_id": "databases", "text": "relational database indexing"},
        {"chunk_id": "networks", "text": "computer network routing"},
    ]
    chunks_path.write_text(
        "".join(f"{json.dumps(chunk)}\n" for chunk in chunks),
        encoding="utf-8",
    )
    return chunks_path


def test_retrieve_top_k_excludes_zero_score_filler_results(tmp_path):
    retriever = BM25Retriever(chunks_path=_write_chunks(tmp_path))

    results = retriever.retrieve_top_k("transformer", k=3)

    assert [result["chunk_id"] for result in results] == ["transformers"]
    assert results[0]["bm25_score"] > 0


def test_retrieve_top_k_returns_no_results_without_lexical_evidence(tmp_path):
    retriever = BM25Retriever(chunks_path=_write_chunks(tmp_path))

    assert retriever.retrieve_top_k("xylophone", k=3) == []

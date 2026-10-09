import json
import struct
from pathlib import Path

import pytest

from retriever.artifacts import (
    ArtifactIntegrityError,
    build_manifest,
    validate_manifest,
    write_manifest,
)


def write_chunks(path: Path, chunk_ids: list[str]) -> None:
    rows = [
        json.dumps({"chunk_id": chunk_id, "text": f"text for {chunk_id}"})
        for chunk_id in chunk_ids
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_npy(path: Path, rows: int, dimensions: int, *, truncate: int = 0) -> None:
    header_text = (
        f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {dimensions}), }}"
    )
    padding = (16 - ((10 + len(header_text) + 1) % 16)) % 16
    header = (header_text + " " * padding + "\n").encode("latin1")
    contents = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
    contents += b"\x00" * (rows * dimensions * 4)
    path.write_bytes(contents[:-truncate] if truncate else contents)


def artifact_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.npy"
    manifest = tmp_path / "retrieval_manifest.json"
    write_chunks(chunks, ["chunk-1", "chunk-2"])
    write_npy(embeddings, rows=2, dimensions=3)
    return chunks, embeddings, manifest


def test_manifest_captures_content_hashes_shape_and_model_lineage(tmp_path: Path):
    chunks, embeddings, manifest_path = artifact_paths(tmp_path)

    manifest = write_manifest(
        manifest_path,
        chunks,
        embeddings,
        embedding_model="model/revision",
        query_prefix="query: ",
    )

    assert manifest["schema_version"] == 1
    assert manifest["chunks"]["rows"] == 2
    assert len(manifest["chunks"]["sha256"]) == 64
    assert manifest["embeddings"] == {
        "path": str(embeddings),
        "sha256": manifest["embeddings"]["sha256"],
        "rows": 2,
        "dimensions": 3,
        "dtype": "<f4",
    }
    assert (
        validate_manifest(
            manifest_path,
            chunks,
            embeddings,
            embedding_model="model/revision",
            query_prefix="query: ",
        )["embeddings"]["dimensions"]
        == 3
    )


def test_manifest_rejects_modified_chunks(tmp_path: Path):
    chunks, embeddings, manifest = artifact_paths(tmp_path)
    write_manifest(
        manifest,
        chunks,
        embeddings,
        embedding_model="model/revision",
        query_prefix="",
    )
    write_chunks(chunks, ["chunk-1", "changed"])

    with pytest.raises(ArtifactIntegrityError, match="chunks.sha256"):
        validate_manifest(
            manifest,
            chunks,
            embeddings,
            embedding_model="model/revision",
            query_prefix="",
        )


def test_manifest_rejects_embedding_model_drift(tmp_path: Path):
    chunks, embeddings, manifest = artifact_paths(tmp_path)
    write_manifest(
        manifest,
        chunks,
        embeddings,
        embedding_model="model/revision",
        query_prefix="",
    )

    with pytest.raises(ArtifactIntegrityError, match="Embedding model"):
        validate_manifest(
            manifest,
            chunks,
            embeddings,
            embedding_model="different/model",
            query_prefix="",
        )


def test_build_rejects_chunk_embedding_row_mismatch(tmp_path: Path):
    chunks, embeddings, _ = artifact_paths(tmp_path)
    write_npy(embeddings, rows=3, dimensions=3)

    with pytest.raises(ArtifactIntegrityError, match="row counts differ"):
        build_manifest(
            chunks,
            embeddings,
            embedding_model="model/revision",
            query_prefix="",
        )


def test_chunk_validation_rejects_duplicate_ids(tmp_path: Path):
    chunks, embeddings, _ = artifact_paths(tmp_path)
    write_chunks(chunks, ["duplicate", "duplicate"])

    with pytest.raises(ArtifactIntegrityError, match="Duplicate chunk_id"):
        build_manifest(
            chunks,
            embeddings,
            embedding_model="model/revision",
            query_prefix="",
        )


def test_embedding_validation_rejects_truncated_array(tmp_path: Path):
    chunks, embeddings, _ = artifact_paths(tmp_path)
    write_npy(embeddings, rows=2, dimensions=3, truncate=4)

    with pytest.raises(ArtifactIntegrityError, match="byte size mismatch"):
        build_manifest(
            chunks,
            embeddings,
            embedding_model="model/revision",
            query_prefix="",
        )

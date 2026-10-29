"""Integrity manifests for coupled retrieval artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
INDEX_MANIFEST_SCHEMA_VERSION = 1
_NPY_MAGIC = b"\x93NUMPY"
_DTYPE_PATTERN = re.compile(r"^[<>=|]?[biufc](\d+)$")


class ArtifactIntegrityError(ValueError):
    """Raised when retrieval artifacts do not match their manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_chunks(path: Path) -> dict[str, int | str]:
    """Validate chunk JSONL and return content-addressed metadata."""

    rows = 0
    chunk_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactIntegrityError(
                    f"Invalid chunk JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(chunk, dict):
                raise ArtifactIntegrityError(
                    f"Chunk at {path}:{line_number} must be an object"
                )
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ArtifactIntegrityError(
                    f"Chunk at {path}:{line_number} needs a non-empty chunk_id"
                )
            if chunk_id in chunk_ids:
                raise ArtifactIntegrityError(
                    f"Duplicate chunk_id in {path}: {chunk_id}"
                )
            if not isinstance(text, str) or not text.strip():
                raise ArtifactIntegrityError(
                    f"Chunk {chunk_id} at {path}:{line_number} needs non-empty text"
                )
            chunk_ids.add(chunk_id)
            rows += 1
    if rows == 0:
        raise ArtifactIntegrityError(f"Chunk artifact is empty: {path}")
    return {"sha256": sha256_file(path), "rows": rows}


def inspect_embeddings(path: Path) -> dict[str, int | str]:
    """Read a two-dimensional NPY header without loading the array into memory."""

    with path.open("rb") as handle:
        if handle.read(len(_NPY_MAGIC)) != _NPY_MAGIC:
            raise ArtifactIntegrityError(
                f"Embedding artifact is not an NPY file: {path}"
            )
        version = tuple(handle.read(2))
        if version == (1, 0):
            header_length_bytes = handle.read(2)
            if len(header_length_bytes) != 2:
                raise ArtifactIntegrityError(f"Truncated NPY header: {path}")
            header_length = struct.unpack("<H", header_length_bytes)[0]
            encoding = "latin1"
        elif version in {(2, 0), (3, 0)}:
            header_length_bytes = handle.read(4)
            if len(header_length_bytes) != 4:
                raise ArtifactIntegrityError(f"Truncated NPY header: {path}")
            header_length = struct.unpack("<I", header_length_bytes)[0]
            encoding = "utf-8" if version == (3, 0) else "latin1"
        else:
            raise ArtifactIntegrityError(f"Unsupported NPY version {version}: {path}")
        header = handle.read(header_length)
        data_offset = handle.tell()
    if len(header) != header_length:
        raise ArtifactIntegrityError(f"Truncated NPY header: {path}")
    try:
        metadata = ast.literal_eval(header.decode(encoding).strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        raise ArtifactIntegrityError(f"Invalid NPY header: {path}") from exc
    if not isinstance(metadata, dict):
        raise ArtifactIntegrityError(f"Invalid NPY header metadata: {path}")

    shape = metadata.get("shape")
    dtype = metadata.get("descr")
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(type(value) is not int or value < 1 for value in shape)
    ):
        raise ArtifactIntegrityError(
            f"Embeddings must have a positive two-dimensional shape: {path}"
        )
    if metadata.get("fortran_order") is not False:
        raise ArtifactIntegrityError(
            f"Embeddings must use C-contiguous NPY storage: {path}"
        )
    if not isinstance(dtype, str) or not (match := _DTYPE_PATTERN.fullmatch(dtype)):
        raise ArtifactIntegrityError(f"Unsupported embedding dtype {dtype!r}: {path}")
    item_size = int(match.group(1))
    expected_size = data_offset + shape[0] * shape[1] * item_size
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ArtifactIntegrityError(
            f"Embedding byte size mismatch for {path}: expected {expected_size}, got {actual_size}"
        )
    return {
        "sha256": sha256_file(path),
        "rows": shape[0],
        "dimensions": shape[1],
        "dtype": dtype,
    }


def build_manifest(
    chunks_path: Path,
    embeddings_path: Path,
    *,
    embedding_model: str,
    query_prefix: str,
) -> dict[str, Any]:
    chunks = inspect_chunks(chunks_path)
    embeddings = inspect_embeddings(embeddings_path)
    if chunks["rows"] != embeddings["rows"]:
        raise ArtifactIntegrityError(
            "Chunk and embedding row counts differ: "
            f"{chunks['rows']} chunks vs {embeddings['rows']} embeddings"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": embedding_model,
        "query_prefix": query_prefix,
        "chunks": {"path": str(chunks_path), **chunks},
        "embeddings": {"path": str(embeddings_path), **embeddings},
    }


def write_manifest(
    manifest_path: Path,
    chunks_path: Path,
    embeddings_path: Path,
    *,
    embedding_model: str,
    query_prefix: str,
) -> dict[str, Any]:
    manifest = build_manifest(
        chunks_path,
        embeddings_path,
        embedding_model=embedding_model,
        query_prefix=query_prefix,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_manifest(
    manifest_path: Path,
    chunks_path: Path,
    embeddings_path: Path,
    *,
    embedding_model: str,
    query_prefix: str,
) -> dict[str, Any]:
    """Recompute artifact metadata and fail if it differs from the manifest."""

    try:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactIntegrityError(
            f"Missing retrieval artifact manifest: {manifest_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ArtifactIntegrityError(
            f"Invalid retrieval manifest JSON: {manifest_path}"
        ) from exc
    if (
        not isinstance(expected, dict)
        or expected.get("schema_version") != SCHEMA_VERSION
    ):
        raise ArtifactIntegrityError(
            f"Retrieval manifest schema_version must be {SCHEMA_VERSION}: {manifest_path}"
        )
    if expected.get("embedding_model") != embedding_model:
        raise ArtifactIntegrityError(
            "Embedding model does not match the retrieval manifest"
        )
    if expected.get("query_prefix") != query_prefix:
        raise ArtifactIntegrityError(
            "Embedding query prefix does not match the retrieval manifest"
        )

    actual = build_manifest(
        chunks_path,
        embeddings_path,
        embedding_model=embedding_model,
        query_prefix=query_prefix,
    )
    for artifact in ("chunks", "embeddings"):
        expected_metadata = expected.get(artifact)
        if not isinstance(expected_metadata, dict):
            raise ArtifactIntegrityError(
                f"Retrieval manifest is missing {artifact} metadata"
            )
        for field, actual_value in actual[artifact].items():
            if field == "path":
                continue
            if expected_metadata.get(field) != actual_value:
                raise ArtifactIntegrityError(
                    f"{artifact}.{field} does not match the retrieval manifest"
                )
    return actual


def build_index_manifest(
    index_path: Path,
    retrieval_manifest_path: Path,
    *,
    rows: int,
    dimensions: int,
) -> dict[str, Any]:
    """Bind a derived FAISS index to the exact retrieval artifacts it indexes."""
    if rows < 1 or dimensions < 1:
        raise ArtifactIntegrityError("FAISS index shape must be positive")
    return {
        "schema_version": INDEX_MANIFEST_SCHEMA_VERSION,
        "retrieval_manifest_sha256": sha256_file(retrieval_manifest_path),
        "index": {
            "path": str(index_path),
            "sha256": sha256_file(index_path),
            "rows": rows,
            "dimensions": dimensions,
        },
    }


def write_index_manifest(
    index_manifest_path: Path,
    index_path: Path,
    retrieval_manifest_path: Path,
    *,
    rows: int,
    dimensions: int,
) -> dict[str, Any]:
    manifest = build_index_manifest(
        index_path,
        retrieval_manifest_path,
        rows=rows,
        dimensions=dimensions,
    )
    index_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    index_manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def validate_index_manifest(
    index_manifest_path: Path,
    index_path: Path,
    retrieval_manifest_path: Path,
    *,
    expected_rows: int,
    expected_dimensions: int,
) -> dict[str, Any]:
    """Fail closed if an index is stale, modified, or built for different artifacts."""
    try:
        expected = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactIntegrityError(
            f"Missing FAISS index manifest: {index_manifest_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ArtifactIntegrityError(
            f"Invalid FAISS index manifest JSON: {index_manifest_path}"
        ) from exc
    if (
        not isinstance(expected, dict)
        or expected.get("schema_version") != INDEX_MANIFEST_SCHEMA_VERSION
    ):
        raise ArtifactIntegrityError(
            "FAISS index manifest schema_version must be "
            f"{INDEX_MANIFEST_SCHEMA_VERSION}: {index_manifest_path}"
        )

    actual_retrieval_sha256 = sha256_file(retrieval_manifest_path)
    if expected.get("retrieval_manifest_sha256") != actual_retrieval_sha256:
        raise ArtifactIntegrityError(
            "FAISS index was built from a different retrieval manifest"
        )
    index_metadata = expected.get("index")
    if not isinstance(index_metadata, dict):
        raise ArtifactIntegrityError("FAISS index manifest is missing index metadata")
    if index_metadata.get("rows") != expected_rows:
        raise ArtifactIntegrityError("FAISS index row count does not match embeddings")
    if index_metadata.get("dimensions") != expected_dimensions:
        raise ArtifactIntegrityError("FAISS index dimensions do not match embeddings")
    actual_index_sha256 = sha256_file(index_path)
    if index_metadata.get("sha256") != actual_index_sha256:
        raise ArtifactIntegrityError("FAISS index checksum does not match its manifest")

    return {
        "schema_version": INDEX_MANIFEST_SCHEMA_VERSION,
        "retrieval_manifest_sha256": actual_retrieval_sha256,
        "index": {
            "path": str(index_path),
            "sha256": actual_index_sha256,
            "rows": expected_rows,
            "dimensions": expected_dimensions,
        },
    }

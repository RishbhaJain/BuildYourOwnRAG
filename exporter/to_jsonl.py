"""
Exporter: reads raw_pages/ → cleans each page → writes corpus.jsonl.

JSONL schema (one record per line):
{
    "id":              str   — 16-char SHA-256 hex of canonical URL
    "url":             str   — canonical URL after redirects
    "title":           str   — page <title>, may be empty
    "text":            str   — cleaned plain text
    "crawl_timestamp": str   — ISO-8601 UTC timestamp from crawl
    "content_length":  int   — character count of text
    "extractor":       str   — "resiliparse" or "bs4"
}
"""

import json
import logging
import os

import config
from cleaner.cleaner import clean_page
from crawler.storage import load_all_meta, load_html

logger = logging.getLogger(__name__)


def export_to_jsonl(
    raw_dirs: list[str] | str = config.RAW_PAGES_DIR,
    output_path: str = config.CORPUS_JSONL_PATH,
) -> tuple[int, int]:
    """
    Clean all pages from one or more raw_dirs and write to output_path as JSONL.
    Pages with duplicate IDs (same URL crawled in multiple passes) are written
    only once — whichever copy is encountered first.

    Args:
        raw_dirs: A single directory path or a list of directory paths.

    Returns:
        (written, skipped) — counts of records written and skipped.
    """
    if isinstance(raw_dirs, str):
        raw_dirs = [raw_dirs]

    # Collect metadata from all directories; track which dir each page lives in
    all_meta: list[tuple[dict, str]] = []  # (meta, raw_dir)
    for raw_dir in raw_dirs:
        for meta in load_all_meta(raw_dir):
            all_meta.append((meta, raw_dir))

    if not all_meta:
        logger.warning(
            "No metadata files found in %s. Run the crawler first.", raw_dirs
        )
        return 0, 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    written = 0
    skipped = 0
    seen_ids: set[str] = set()

    with open(output_path, "w", encoding="utf-8") as out:
        for meta, raw_dir in all_meta:
            page_id = meta["page_id"]

            # Deduplicate across directories
            if page_id in seen_ids:
                skipped += 1
                continue

            html = load_html(page_id, raw_dir)

            if html is None:
                logger.warning("[%s] HTML file missing — skipping", page_id)
                skipped += 1
                continue

            cleaned = clean_page(html, page_id=page_id)
            if cleaned is None:
                logger.debug("[%s] Discarded (below MIN_CONTENT_LENGTH)", page_id)
                skipped += 1
                continue

            record = {
                "id": page_id,
                "url": meta["url"],
                "title": cleaned["title"],
                "text": cleaned["text"],
                "crawl_timestamp": meta["crawl_timestamp"],
                "content_length": len(cleaned["text"]),
                "extractor": cleaned["extractor"],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            seen_ids.add(page_id)
            written += 1

    logger.info(
        "Export complete: %d written, %d skipped → %s", written, skipped, output_path
    )
    return written, skipped

"""
Integration tests for exporter/to_jsonl.py.

Creates a temporary raw_pages/ directory with fixture HTML + meta files,
runs the exporter, and asserts the JSONL output is correct.
"""

import json
import os
from datetime import UTC, datetime

from exporter.to_jsonl import export_to_jsonl


def _write_fixture(raw_dir: str, page_id: str, html: str, url: str) -> None:
    """Write an HTML file and its metadata sidecar to raw_dir."""
    with open(os.path.join(raw_dir, f"{page_id}.html"), "w", encoding="utf-8") as f:
        f.write(html)

    meta = {
        "page_id": page_id,
        "url": url,
        "original_url": url,
        "status_code": 200,
        "content_type": "text/html; charset=utf-8",
        "crawl_timestamp": datetime.now(UTC).isoformat(),
    }
    with open(
        os.path.join(raw_dir, f"{page_id}.meta.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(meta, f)


GOOD_HTML = """
<html>
<head><title>Good Page</title></head>
<body>
  <main>
    <p>This page has a substantial amount of content that will pass the minimum length threshold
    and should appear in the exported corpus file without issue.</p>
  </main>
</body>
</html>
"""

EMPTY_HTML = "<html><body></body></html>"


class TestExportToJsonl:
    def test_good_page_exported(self, tmp_path):
        raw_dir = str(tmp_path / "raw_pages")
        os.makedirs(raw_dir)
        output = str(tmp_path / "corpus.jsonl")

        _write_fixture(
            raw_dir, "aabbccdd11223344", GOOD_HTML, "https://eecs.berkeley.edu/good"
        )

        written, skipped = export_to_jsonl(raw_dirs=raw_dir, output_path=output)

        assert written == 1
        assert skipped == 0

        with open(output, "r", encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert record["id"] == "aabbccdd11223344"
        assert record["url"] == "https://eecs.berkeley.edu/good"
        assert record["title"] == "Good Page"
        assert len(record["text"]) > 0
        assert record["content_length"] == len(record["text"])
        assert record["extractor"] in ("resiliparse", "bs4")

    def test_empty_page_skipped(self, tmp_path):
        raw_dir = str(tmp_path / "raw_pages")
        os.makedirs(raw_dir)
        output = str(tmp_path / "corpus.jsonl")

        _write_fixture(
            raw_dir, "0011223344556677", EMPTY_HTML, "https://eecs.berkeley.edu/empty"
        )

        written, skipped = export_to_jsonl(raw_dirs=raw_dir, output_path=output)

        assert written == 0
        assert skipped == 1

    def test_missing_html_skipped(self, tmp_path):
        """A meta.json with no corresponding .html file should be skipped gracefully."""
        raw_dir = str(tmp_path / "raw_pages")
        os.makedirs(raw_dir)
        output = str(tmp_path / "corpus.jsonl")

        # Write only the meta, no HTML file
        meta = {
            "page_id": "deadbeefdeadbeef",
            "url": "https://eecs.berkeley.edu/missing",
            "original_url": "https://eecs.berkeley.edu/missing",
            "status_code": 200,
            "content_type": "text/html",
            "crawl_timestamp": datetime.now(UTC).isoformat(),
        }
        with open(
            os.path.join(raw_dir, "deadbeefdeadbeef.meta.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(meta, f)

        written, skipped = export_to_jsonl(raw_dirs=raw_dir, output_path=output)

        assert written == 0
        assert skipped == 1

    def test_empty_raw_dir_returns_zeros(self, tmp_path):
        raw_dir = str(tmp_path / "raw_pages")
        os.makedirs(raw_dir)
        output = str(tmp_path / "corpus.jsonl")

        written, skipped = export_to_jsonl(raw_dirs=raw_dir, output_path=output)

        assert written == 0
        assert skipped == 0

    def test_multiple_pages_all_written(self, tmp_path):
        raw_dir = str(tmp_path / "raw_pages")
        os.makedirs(raw_dir)
        output = str(tmp_path / "corpus.jsonl")

        for i, pid in enumerate(["aaaa111100000001", "bbbb222200000002"]):
            _write_fixture(
                raw_dir,
                pid,
                GOOD_HTML,
                f"https://eecs.berkeley.edu/page{i}",
            )

        written, skipped = export_to_jsonl(raw_dirs=raw_dir, output_path=output)
        assert written == 2
        assert skipped == 0

        with open(output, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert "text" in record
            assert "url" in record

    def test_valid_duplicate_survives_missing_html_in_earlier_directory(self, tmp_path):
        first_raw_dir = str(tmp_path / "first_raw_pages")
        second_raw_dir = str(tmp_path / "second_raw_pages")
        os.makedirs(first_raw_dir)
        os.makedirs(second_raw_dir)
        output = str(tmp_path / "corpus.jsonl")
        page_id = "feedfacefeedface"

        _write_fixture(
            first_raw_dir,
            page_id,
            GOOD_HTML,
            "https://eecs.berkeley.edu/stale",
        )
        os.remove(os.path.join(first_raw_dir, f"{page_id}.html"))
        _write_fixture(
            second_raw_dir,
            page_id,
            GOOD_HTML,
            "https://eecs.berkeley.edu/complete",
        )

        written, skipped = export_to_jsonl(
            raw_dirs=[first_raw_dir, second_raw_dir],
            output_path=output,
        )

        assert written == 1
        assert skipped == 1
        with open(output, "r", encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["id"] == page_id
        assert record["url"] == "https://eecs.berkeley.edu/complete"

"""
Tests for crawler/fetcher.py — uses unittest.mock to mock urllib HTTP calls.
"""

import time
import http.client
import socket
import urllib.error
import pytest
from unittest.mock import MagicMock, patch

from crawler.fetcher import Fetcher, FetchResult
from crawler.robots import RobotsCache


def _make_robots(allowed: bool = True, crawl_delay: float | None = None) -> RobotsCache:
    """Return a RobotsCache mock that allows or disallows all URLs."""
    robots = MagicMock(spec=RobotsCache)
    robots.is_allowed.return_value = allowed
    robots.get_crawl_delay.return_value = crawl_delay
    return robots


def _make_response(url: str, status: int, content_type: str, body: str = "") -> MagicMock:
    """Create a mock urllib response context manager."""
    mock_resp = MagicMock()
    mock_resp.url = url
    mock_resp.status = status
    mock_resp.headers.get.side_effect = lambda key, default="": (
        content_type if key == "Content-Type" else default
    )
    mock_resp.headers.get_content_charset.return_value = "utf-8"
    mock_resp.read.return_value = body.encode("utf-8") if isinstance(body, str) else body
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_fetch_html_success():
    resp = _make_response(
        "https://eecs.berkeley.edu/page", 200,
        "text/html; charset=utf-8", "<html><body>Hello</body></html>",
    )
    with patch("urllib.request.urlopen", return_value=resp):
        fetcher = Fetcher(crawl_delay=0.0)
        result = fetcher.fetch("https://eecs.berkeley.edu/page", _make_robots())
    assert result.error is None
    assert result.html == "<html><body>Hello</body></html>"
    assert result.status_code == 200


def test_fetch_non_html_returns_error():
    resp = _make_response(
        "https://eecs.berkeley.edu/doc.pdf", 200, "application/pdf",
    )
    with patch("urllib.request.urlopen", return_value=resp):
        fetcher = Fetcher(crawl_delay=0.0)
        result = fetcher.fetch("https://eecs.berkeley.edu/doc.pdf", _make_robots())
    assert result.html is None
    assert "Non-HTML" in result.error


def test_fetch_disallowed_by_robots():
    fetcher = Fetcher(crawl_delay=0.0)
    result = fetcher.fetch("https://eecs.berkeley.edu/private", _make_robots(allowed=False))
    assert result.html is None
    assert "robots" in result.error.lower()


def test_fetch_404():
    hdrs = http.client.HTTPMessage()
    hdrs["Content-Type"] = "text/html"
    fp = MagicMock()
    fp.read.return_value = b"<html>Not Found</html>"
    e = urllib.error.HTTPError(
        "https://eecs.berkeley.edu/missing", 404, "Not Found", hdrs, fp
    )
    with patch("urllib.request.urlopen", side_effect=e):
        fetcher = Fetcher(crawl_delay=0.0)
        result = fetcher.fetch("https://eecs.berkeley.edu/missing", _make_robots())
    assert result.status_code == 404
    assert result.html is not None


def test_fetch_timeout_exhausts_retries():
    timeout_error = urllib.error.URLError(socket.timeout("timed out"))
    with patch("urllib.request.urlopen", side_effect=timeout_error):
        fetcher = Fetcher(crawl_delay=0.0, max_retries=3, backoff_base=0.0)
        result = fetcher.fetch("https://eecs.berkeley.edu/slow", _make_robots())
    assert result.html is None
    assert result.error is not None


def test_rate_limit_enforced_per_domain():
    """Two requests to the same domain should be separated by at least crawl_delay."""
    delay = 0.1
    fetcher = Fetcher(crawl_delay=delay)
    robots = _make_robots()

    resp1 = _make_response("https://eecs.berkeley.edu/page", 200, "text/html", "<html></html>")
    resp2 = _make_response("https://eecs.berkeley.edu/page", 200, "text/html", "<html></html>")

    t0 = time.monotonic()
    with patch("urllib.request.urlopen", side_effect=[resp1, resp2]):
        fetcher.fetch("https://eecs.berkeley.edu/page", robots)
        fetcher.fetch("https://eecs.berkeley.edu/page", robots)
    elapsed = time.monotonic() - t0

    assert elapsed >= delay, f"Rate limit not enforced: elapsed={elapsed:.3f}s < delay={delay}s"


def test_redirect_url_captured():
    """The final post-redirect URL should be stored in result.url."""
    resp = _make_response(
        "https://eecs.berkeley.edu/new", 200, "text/html", "<html>New page</html>",
    )
    with patch("urllib.request.urlopen", return_value=resp):
        fetcher = Fetcher(crawl_delay=0.0)
        result = fetcher.fetch("https://eecs.berkeley.edu/old", _make_robots())
    assert result.original_url == "https://eecs.berkeley.edu/old"
    assert "new" in result.url

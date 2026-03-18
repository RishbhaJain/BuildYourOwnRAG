"""
HTTP fetcher with per-domain rate limiting, retry, and redirect handling.
"""

import time
import logging
import socket
import urllib.request
import urllib.error
from dataclasses import dataclass
from urllib.parse import urlparse

from crawler.robots import RobotsCache
import config

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    url: str            # Final URL after redirects
    original_url: str   # URL that was originally requested
    status_code: int    # HTTP status code (-1 for network errors)
    content_type: str
    html: str | None    # None on error or non-HTML response
    error: str | None   # Human-readable error description, or None on success


class Fetcher:
    """
    Fetches URLs with:
    - Per-domain rate limiting (respects robots.txt Crawl-delay or config default)
    - Exponential backoff retries on transient errors
    - Content-Type enforcement (only stores text/html responses)
    - Redirect following (urllib built-in)
    """

    def __init__(
        self,
        user_agent: str = config.USER_AGENT,
        timeout: int = config.REQUEST_TIMEOUT_SECONDS,
        crawl_delay: float = config.CRAWL_DELAY_SECONDS,
        max_retries: int = config.MAX_RETRIES,
        backoff_base: float = config.RETRY_BACKOFF_BASE,
    ) -> None:
        self._user_agent = user_agent
        self._timeout = timeout
        self._crawl_delay = crawl_delay
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        # domain (netloc) -> last fetch timestamp
        self._last_fetch: dict[str, float] = {}

    def _domain(self, url: str) -> str:
        return urlparse(url).netloc.lower()

    def _enforce_rate_limit(self, domain: str, delay: float) -> None:
        last = self._last_fetch.get(domain, 0.0)
        wait = delay - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        self._last_fetch[domain] = time.monotonic()

    def fetch(self, url: str, robots_cache: RobotsCache) -> FetchResult:
        """
        Fetch a URL, enforcing robots.txt and rate limits.
        Returns a FetchResult; html is None on any error or non-HTML response.
        """
        if not robots_cache.is_allowed(url):
            logger.debug("Disallowed by robots.txt: %s", url)
            return FetchResult(
                url=url, original_url=url, status_code=-1,
                content_type="", html=None, error="Disallowed by robots.txt",
            )

        domain = self._domain(url)
        delay = config.DOMAIN_CRAWL_DELAYS.get(domain) or robots_cache.get_crawl_delay(url) or self._crawl_delay
        self._enforce_rate_limit(domain, delay)

        for attempt in range(self._max_retries):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": self._user_agent}
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as response:
                    final_url = response.url
                    status_code = response.status
                    content_type = response.headers.get("Content-Type", "")

                    if "text/html" not in content_type:
                        return FetchResult(
                            url=final_url, original_url=url,
                            status_code=status_code,
                            content_type=content_type,
                            html=None,
                            error=f"Non-HTML content-type: {content_type}",
                        )

                    charset = response.headers.get_content_charset() or "utf-8"
                    html = response.read().decode(charset, errors="replace")
                    return FetchResult(
                        url=final_url,
                        original_url=url,
                        status_code=status_code,
                        content_type=content_type,
                        html=html,
                        error=None,
                    )

            except urllib.error.HTTPError as e:
                # Handle 429 Too Many Requests
                if e.code == 429:
                    retry_after = float(
                        e.headers.get("Retry-After") or
                        self._crawl_delay * (self._backoff_base ** attempt)
                    )
                    logger.warning("429 on %s — waiting %.1fs", url, retry_after)
                    time.sleep(retry_after)
                    continue

                # Redirect loop exhausted
                if 300 <= e.code < 400:
                    return FetchResult(
                        url=url, original_url=url, status_code=-1,
                        content_type="", html=None, error="Too many redirects",
                    )

                # Try to return HTML error body (e.g. 404 page)
                content_type = e.headers.get("Content-Type", "") if e.headers else ""
                if "text/html" in content_type:
                    try:
                        charset = (
                            (e.headers.get_content_charset() or "utf-8")
                            if e.headers else "utf-8"
                        )
                        html = e.read().decode(charset, errors="replace")
                        return FetchResult(
                            url=url, original_url=url, status_code=e.code,
                            content_type=content_type, html=html, error=None,
                        )
                    except Exception:
                        pass

                return FetchResult(
                    url=url, original_url=url, status_code=e.code,
                    content_type=content_type, html=None,
                    error=f"HTTP error: {e.code}",
                )

            except urllib.error.URLError as e:
                reason = e.reason
                if isinstance(reason, (socket.timeout, TimeoutError)) or (
                    isinstance(reason, OSError) and "timed out" in str(reason).lower()
                ):
                    logger.warning(
                        "Timeout on %s (attempt %d/%d)", url, attempt + 1, self._max_retries
                    )
                    if attempt < self._max_retries - 1:
                        time.sleep(self._backoff_base ** attempt)
                else:
                    logger.warning("Request error on %s: %s", url, e)
                    if attempt < self._max_retries - 1:
                        time.sleep(self._backoff_base ** attempt)

            except Exception as exc:
                logger.warning("Unexpected error on %s: %s", url, exc)
                if attempt < self._max_retries - 1:
                    time.sleep(self._backoff_base ** attempt)

        return FetchResult(
            url=url, original_url=url, status_code=-1,
            content_type="", html=None, error="Max retries exceeded",
        )

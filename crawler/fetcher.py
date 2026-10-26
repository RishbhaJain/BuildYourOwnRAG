"""HTTP fetcher with per-hop crawl policy, rate limiting, and retries."""

import logging
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import config
from crawler.robots import RobotsCache
from crawler.url_filter import is_allowed_domain

logger = logging.getLogger(__name__)


class FetchPolicyError(RuntimeError):
    """Raised before a request that violates the configured crawl boundary."""


class _PolicyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Authorize every redirect target before urllib follows it."""

    def __init__(self, authorize: Callable[[str], None]) -> None:
        super().__init__()
        self._authorize = authorize

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self._authorize(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass
class FetchResult:
    url: str  # Final URL after redirects
    original_url: str  # URL that was originally requested
    status_code: int  # HTTP status code (-1 for network errors)
    content_type: str
    html: str | None  # None on error or non-HTML response
    error: str | None  # Human-readable error description, or None on success


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
        allowed_domains: set[str] = config.ALLOWED_DOMAINS,
    ) -> None:
        self._user_agent = user_agent
        self._timeout = timeout
        self._crawl_delay = crawl_delay
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._allowed_domains = set(allowed_domains)
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

    def _authorize_url(self, url: str, robots_cache: RobotsCache) -> None:
        """Enforce scheme, domain, robots, and rate limits before one network hop."""
        if urlparse(url).scheme.lower() not in {"http", "https"}:
            raise FetchPolicyError("Only HTTP and HTTPS crawl targets are allowed")
        if not is_allowed_domain(url, self._allowed_domains):
            raise FetchPolicyError("URL is outside the configured crawl domains")
        if not robots_cache.is_allowed(url):
            raise FetchPolicyError("Disallowed by robots.txt")

        domain = self._domain(url)
        delay = (
            config.DOMAIN_CRAWL_DELAYS.get(domain)
            or robots_cache.get_crawl_delay(url)
            or self._crawl_delay
        )
        self._enforce_rate_limit(domain, delay)

    def _open_request(self, request: urllib.request.Request, robots_cache: RobotsCache):
        redirect_handler = _PolicyRedirectHandler(
            lambda target: self._authorize_url(target, robots_cache)
        )
        return urllib.request.build_opener(redirect_handler).open(
            request, timeout=self._timeout
        )

    def fetch(self, url: str, robots_cache: RobotsCache) -> FetchResult:
        """
        Fetch a URL, enforcing robots.txt and rate limits.
        Returns a FetchResult; html is None on any error or non-HTML response.
        """
        try:
            self._authorize_url(url, robots_cache)
        except FetchPolicyError as exc:
            logger.debug("Blocked by crawl policy: %s", url)
            return FetchResult(
                url=url,
                original_url=url,
                status_code=-1,
                content_type="",
                html=None,
                error=str(exc),
            )

        for attempt in range(self._max_retries):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": self._user_agent}
                )
                with self._open_request(req, robots_cache) as response:
                    final_url = response.url
                    status_code = response.status
                    content_type = response.headers.get("Content-Type", "")

                    if "text/html" not in content_type:
                        return FetchResult(
                            url=final_url,
                            original_url=url,
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

            except FetchPolicyError as exc:
                logger.warning("Redirect blocked by crawl policy: %s", url)
                return FetchResult(
                    url=url,
                    original_url=url,
                    status_code=-1,
                    content_type="",
                    html=None,
                    error=str(exc),
                )

            except urllib.error.HTTPError as e:
                # Handle 429 Too Many Requests
                if e.code == 429:
                    retry_after = float(
                        e.headers.get("Retry-After")
                        or self._crawl_delay * (self._backoff_base**attempt)
                    )
                    logger.warning("429 on %s — waiting %.1fs", url, retry_after)
                    time.sleep(retry_after)
                    continue

                # Redirect loop exhausted
                if 300 <= e.code < 400:
                    return FetchResult(
                        url=url,
                        original_url=url,
                        status_code=-1,
                        content_type="",
                        html=None,
                        error="Too many redirects",
                    )

                # Try to return HTML error body (e.g. 404 page)
                content_type = e.headers.get("Content-Type", "") if e.headers else ""
                if "text/html" in content_type:
                    try:
                        charset = (
                            (e.headers.get_content_charset() or "utf-8")
                            if e.headers
                            else "utf-8"
                        )
                        html = e.read().decode(charset, errors="replace")
                        return FetchResult(
                            url=url,
                            original_url=url,
                            status_code=e.code,
                            content_type=content_type,
                            html=html,
                            error=None,
                        )
                    except (LookupError, OSError, UnicodeError) as exc:
                        logger.debug(
                            "Could not decode HTTP error body for %s: %s", url, exc
                        )

                return FetchResult(
                    url=url,
                    original_url=url,
                    status_code=e.code,
                    content_type=content_type,
                    html=None,
                    error=f"HTTP error: {e.code}",
                )

            except urllib.error.URLError as e:
                reason = e.reason
                if isinstance(reason, (socket.timeout, TimeoutError)) or (
                    isinstance(reason, OSError) and "timed out" in str(reason).lower()
                ):
                    logger.warning(
                        "Timeout on %s (attempt %d/%d)",
                        url,
                        attempt + 1,
                        self._max_retries,
                    )
                    if attempt < self._max_retries - 1:
                        time.sleep(self._backoff_base**attempt)
                else:
                    logger.warning("Request error on %s: %s", url, e)
                    if attempt < self._max_retries - 1:
                        time.sleep(self._backoff_base**attempt)

        return FetchResult(
            url=url,
            original_url=url,
            status_code=-1,
            content_type="",
            html=None,
            error="Max retries exceeded",
        )

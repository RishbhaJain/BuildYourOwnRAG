"""
robots.txt fetching and caching.
One robots.txt fetch per domain, cached for the lifetime of the crawl.
"""

from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
import urllib.request
import urllib.error
import logging

logger = logging.getLogger(__name__)


class RobotsCache:
    """
    Fetches and caches robots.txt per domain root (scheme + netloc).
    Thread-unsafe — designed for single-threaded crawl loop.
    """

    def __init__(self, user_agent: str) -> None:
        self._user_agent = user_agent
        # domain_root (e.g. "https://eecs.berkeley.edu") -> RobotFileParser
        self._cache: dict[str, RobotFileParser] = {}

    def _domain_root(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _fetch_and_normalize(self, robots_url: str) -> str | None:
        """
        Fetch robots.txt and strip any directives that appear before the first
        User-agent: line. Some sites (e.g. eecs.berkeley.edu) emit non-standard
        preamble lines (like a bare Crawl-delay) that confuse Python's
        RobotFileParser into treating the whole file as disallow-all.
        Returns the normalized content, or None on fetch failure.
        """
        try:
            req = urllib.request.Request(
                robots_url, headers={"User-Agent": self._user_agent}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                text = resp.read().decode(charset, errors="replace")
            lines = text.splitlines()
            # Drop all lines before the first "User-agent:" directive
            for i, line in enumerate(lines):
                if line.strip().lower().startswith("user-agent:"):
                    return "\n".join(lines[i:])
            # No User-agent line found — return as-is (parser will allow-all)
            return text
        except Exception:
            return None

    def _get_parser(self, url: str) -> RobotFileParser:
        root = self._domain_root(url)
        if root not in self._cache:
            robots_url = urljoin(root, "/robots.txt")
            parser = RobotFileParser(robots_url)
            content = self._fetch_and_normalize(robots_url)
            if content is None:
                logger.warning(
                    "Could not fetch robots.txt from %s — assuming allow-all", robots_url
                )
                parser.allow_all = True
            else:
                parser.parse(content.splitlines())
                logger.debug("Fetched robots.txt from %s", robots_url)
            self._cache[root] = parser
        return self._cache[root]

    def is_allowed(self, url: str) -> bool:
        """Return True if our user-agent is permitted to fetch this URL."""
        return self._get_parser(url).can_fetch(self._user_agent, url)

    def get_crawl_delay(self, url: str) -> float | None:
        """Return the Crawl-delay directive for our user-agent, or None."""
        delay = self._get_parser(url).crawl_delay(self._user_agent)
        return float(delay) if delay is not None else None

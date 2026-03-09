"""Web article fetcher using Playwright."""

import re
import urllib.robotparser
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright
from playwright.sync_api._generated import Browser, BrowserContext, Page

from personal_knowledge_base.fetchers.base import Fetcher, FetchResult


@dataclass
class RobotsCheckResult:
    """Result of checking robots.txt for a URL.

    Attributes:
        can_fetch: Whether the URL can be fetched according to robots.txt.
        crawl_delay: Optional crawl delay specified in robots.txt.
        message: Human-readable message about the robots.txt check.

    """

    can_fetch: bool
    crawl_delay: float | None = None
    message: str = ""


class WebFetcher(Fetcher):
    """Fetcher for web articles and general web content.

    Uses Playwright to fetch web pages, including JavaScript-rendered content.
    Extracts clean article text using readability heuristics.

    Features:
        - JavaScript rendering support
        - robots.txt respect
        - Paywall detection
        - Error handling for common issues (404, timeout, etc.)
        - Clean text extraction from HTML

    """

    # User agent for fetching
    _USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    # Timeout settings (in milliseconds)
    _DEFAULT_TIMEOUT = 30000  # 30 seconds
    _NAVIGATION_TIMEOUT = 30000  # 30 seconds

    # Paywall indicators in page content or selectors
    _PAYWALL_SELECTORS = [
        "[class*='paywall']",
        "[class*='subscription']",
        "[class*='subscribe']",
        "[id*='paywall']",
        "[id*='subscription']",
        ".article__paywall",
        ".paywall-container",
        ".subscription-required",
        "[data-testid*='paywall']",
        "[data-testid*='subscription']",
    ]

    _PAYWALL_KEYWORDS = [
        "subscribe to continue",
        "subscribe to read",
        "subscription required",
        "premium content",
        "exclusive content",
        "members only",
        "sign in to read",
        "log in to continue",
        "create an account to continue",
    ]

    def __init__(
        self,
        timeout: int = _DEFAULT_TIMEOUT,
        navigation_timeout: int = _NAVIGATION_TIMEOUT,
        respect_robots_txt: bool = True,
    ) -> None:
        """Initialize the web fetcher.

        Args:
            timeout: General timeout in milliseconds.
            navigation_timeout: Navigation timeout in milliseconds.
            respect_robots_txt: Whether to check and respect robots.txt.

        """
        self._timeout = timeout
        self._navigation_timeout = navigation_timeout
        self._respect_robots_txt = respect_robots_txt
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}

    def can_fetch(self, url: str) -> bool:
        """Check if this fetcher can handle the given URL.

        This fetcher can handle any HTTP or HTTPS URL that is not
        a YouTube URL (which has its own fetcher).

        Args:
            url: The URL to check.

        Returns:
            True if this fetcher can handle the URL, False otherwise.

        """
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False

            # Exclude YouTube URLs (handled by YouTubeFetcher)
            return not self._is_youtube_url(url)
        except (ValueError, TypeError):
            return False

    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL.

        Args:
            url: The URL to check.

        Returns:
            True if the URL is a YouTube URL.

        """
        youtube_patterns = [
            r"youtube\.com",
            r"youtu\.be",
        ]
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in youtube_patterns)

    def _check_robots_txt(self, url: str) -> RobotsCheckResult:
        """Check if URL can be fetched according to robots.txt.

        Args:
            url: The URL to check.

        Returns:
            RobotsCheckResult indicating if fetching is allowed.

        """
        if not self._respect_robots_txt:
            return RobotsCheckResult(can_fetch=True, message="robots.txt check disabled")

        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

            # Check cache first
            if robots_url not in self._robots_cache:
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(robots_url)
                try:
                    rp.read()
                except Exception as e:
                    return RobotsCheckResult(
                        can_fetch=True,
                        message=f"Could not fetch robots.txt: {e}",
                    )
                self._robots_cache[robots_url] = rp

            rp = self._robots_cache[robots_url]
            can_fetch = rp.can_fetch(self._USER_AGENT, url)
            crawl_delay_raw = rp.crawl_delay(self._USER_AGENT)
            crawl_delay = float(crawl_delay_raw) if crawl_delay_raw is not None else None

            if can_fetch:
                return RobotsCheckResult(
                    can_fetch=True,
                    crawl_delay=crawl_delay,
                    message="Fetching allowed by robots.txt",
                )
            else:
                return RobotsCheckResult(
                    can_fetch=False,
                    message="Fetching blocked by robots.txt",
                )

        except Exception as e:
            return RobotsCheckResult(
                can_fetch=True,
                message=f"Error checking robots.txt: {e}",
            )

    def fetch(self, url: str) -> FetchResult:
        """Fetch and extract content from a web page.

        Args:
            url: The URL to fetch.

        Returns:
            FetchResult containing the extracted content or error information.

        """
        # Validate URL
        if not self.can_fetch(url):
            return FetchResult(
                url=url,
                content_type="article",
                success=False,
                error_message="Invalid URL or URL type not supported by WebFetcher.",
            )

        # Check robots.txt
        robots_result = self._check_robots_txt(url)
        if not robots_result.can_fetch:
            return FetchResult(
                url=url,
                content_type="article",
                success=False,
                error_message=f"Blocked by robots.txt: {robots_result.message}",
            )

        try:
            return self._fetch_with_playwright(url, robots_result)
        except Exception as e:
            return FetchResult(
                url=url,
                content_type="article",
                success=False,
                error_message=f"Unexpected error: {str(e)}",
            )

    def _fetch_with_playwright(self, url: str, robots_result: RobotsCheckResult) -> FetchResult:
        """Fetch content using Playwright.

        Args:
            url: The URL to fetch.
            robots_result: Result of robots.txt check.

        Returns:
            FetchResult with extracted content.

        """
        with sync_playwright() as p:
            browser: Browser | None = None
            context: BrowserContext | None = None
            page: Page | None = None

            try:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=self._USER_AGENT,
                    viewport={"width": 1920, "height": 1080},
                )
                page = context.new_page()
                page.set_default_timeout(self._timeout)
                page.set_default_navigation_timeout(self._navigation_timeout)

                # Navigate to the page
                response = page.goto(url, wait_until="networkidle")

                if response is None:
                    return FetchResult(
                        url=url,
                        content_type="article",
                        success=False,
                        error_message="Failed to load page: No response received.",
                    )

                # Check HTTP status
                status = response.status
                if status == 404:
                    return FetchResult(
                        url=url,
                        content_type="article",
                        success=False,
                        error_message="Page not found (404).",
                    )
                elif status == 403:
                    return FetchResult(
                        url=url,
                        content_type="article",
                        success=False,
                        error_message="Access forbidden (403).",
                    )
                elif status == 500:
                    return FetchResult(
                        url=url,
                        content_type="article",
                        success=False,
                        error_message="Server error (500).",
                    )
                elif status >= 400:
                    return FetchResult(
                        url=url,
                        content_type="article",
                        success=False,
                        error_message=f"HTTP error {status}.",
                    )

                # Wait for content to load
                page.wait_for_load_state("domcontentloaded")

                # Check for paywall
                paywall_check = self._detect_paywall(page)
                if paywall_check["detected"]:
                    return FetchResult(
                        url=url,
                        content_type="article",
                        success=False,
                        error_message=f"Paywall detected: {paywall_check['reason']}",
                    )

                # Extract content
                content_data = self._extract_content(page)

                # Build metadata
                metadata: dict[str, Any] = {
                    "robots_txt_check": robots_result.message,
                    "http_status": status,
                    "final_url": page.url,
                }
                if robots_result.crawl_delay:
                    metadata["crawl_delay"] = robots_result.crawl_delay

                return FetchResult(
                    url=url,
                    title=content_data["title"],
                    content=content_data["content"],
                    content_type="article",
                    author=content_data.get("author"),
                    published_date=content_data.get("published_date"),
                    metadata=metadata,
                    success=True,
                    fetched_at=datetime.now(),
                )

            except Exception as e:
                error_message = str(e).lower()

                if "timeout" in error_message:
                    return FetchResult(
                        url=url,
                        content_type="article",
                        success=False,
                        error_message="Page load timeout. The site may be slow or unresponsive.",
                    )
                elif "net::" in error_message or "err_" in error_message:
                    return FetchResult(
                        url=url,
                        content_type="article",
                        success=False,
                        error_message=f"Network error: {str(e)}",
                    )
                else:
                    return FetchResult(
                        url=url,
                        content_type="article",
                        success=False,
                        error_message=f"Fetch error: {str(e)}",
                    )

            finally:
                if page:
                    page.close()
                if context:
                    context.close()
                if browser:
                    browser.close()

    def _detect_paywall(self, page: Page) -> dict[str, Any]:
        """Detect if the page has a paywall.

        Args:
            page: The Playwright page object.

        Returns:
            Dict with 'detected' boolean and 'reason' string.

        """
        # Check for paywall selectors
        for selector in self._PAYWALL_SELECTORS:
            try:
                element = page.locator(selector).first
                if element.is_visible(timeout=1000):
                    return {
                        "detected": True,
                        "reason": f"Paywall element found: {selector}",
                    }
            except Exception:
                continue

        # Check for paywall keywords in page text
        try:
            page_text = page.locator("body").inner_text(timeout=5000).lower()
            for keyword in self._PAYWALL_KEYWORDS:
                if keyword in page_text:
                    return {
                        "detected": True,
                        "reason": f"Paywall keyword found: '{keyword}'",
                    }
        except Exception:
            pass

        return {"detected": False, "reason": ""}

    def _extract_content(self, page: Page) -> dict[str, Any]:
        """Extract clean content from the page.

        Args:
            page: The Playwright page object.

        Returns:
            Dict with extracted content including title, body text, author, etc.

        """
        result: dict[str, Any] = {
            "title": "",
            "content": "",
            "author": None,
            "published_date": None,
        }

        # Extract title
        try:
            # Try article title first, then og:title, then regular title
            title_selectors = [
                'h1[data-testid="title"]',
                'h1[class*="title"]',
                'h1[class*="headline"]',
                "h1",
                'meta[property="og:title"]',
                "title",
            ]
            for selector in title_selectors:
                try:
                    if selector.startswith("meta"):
                        title = page.locator(selector).first.get_attribute("content")
                    else:
                        title = page.locator(selector).first.inner_text(timeout=2000)
                    if title and title.strip():
                        result["title"] = title.strip()
                        break
                except Exception:
                    continue
        except Exception:
            pass

        # Extract author
        try:
            author_selectors = [
                'meta[name="author"]',
                'meta[property="article:author"]',
                '[class*="author"]',
                '[class*="byline"]',
                '[rel="author"]',
            ]
            for selector in author_selectors:
                try:
                    if selector.startswith("meta"):
                        author = page.locator(selector).first.get_attribute("content")
                    else:
                        author = page.locator(selector).first.inner_text(timeout=1000)
                    if author and author.strip():
                        result["author"] = author.strip()
                        break
                except Exception:
                    continue
        except Exception:
            pass

        # Extract published date
        try:
            date_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="publishedDate"]',
                'meta[name="date"]',
                "time[datetime]",
                '[class*="published"]',
                '[class*="date"]',
            ]
            for selector in date_selectors:
                try:
                    if selector.startswith("meta"):
                        date_str = page.locator(selector).first.get_attribute("content")
                    elif "datetime" in selector:
                        date_str = page.locator(selector).first.get_attribute("datetime")
                    else:
                        date_str = page.locator(selector).first.inner_text(timeout=1000)
                    if date_str and date_str.strip():
                        result["published_date"] = self._parse_date(date_str.strip())
                        break
                except Exception:
                    continue
        except Exception:
            pass

        # Extract main content
        result["content"] = self._extract_main_content(page)

        return result

    def _extract_main_content(self, page: Page) -> str:
        """Extract the main article content from the page.

        Uses heuristics to find the main content area and extract clean text.

        Args:
            page: The Playwright page object.

        Returns:
            Clean extracted text content.

        """
        # Common content selectors in order of preference
        content_selectors = [
            # Article-specific selectors
            "article",
            '[role="main"]',
            "main",
            # Common article content classes
            '[class*="article-content"]',
            '[class*="article-body"]',
            '[class*="post-content"]',
            '[class*="entry-content"]',
            '[class*="content-body"]',
            '[class*="story-body"]',
            '[class*="prose"]',
            # Generic content areas
            ".content",
            "#content",
            # Fallback to body
            "body",
        ]

        for selector in content_selectors:
            try:
                element = page.locator(selector).first
                if element.is_visible(timeout=2000):
                    # Get text content
                    text = element.inner_text(timeout=5000)
                    if text and len(text.strip()) > 100:
                        return self._clean_text(text)
            except Exception:
                continue

        # Fallback: extract all text from body
        try:
            text = page.locator("body").inner_text(timeout=5000)
            return self._clean_text(text)
        except Exception:
            return ""

    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing extra whitespace and unwanted elements.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text.

        """
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove common navigation/footer text patterns
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # Skip very short lines that are likely navigation
            if len(line) < 3:
                continue
            # Skip common navigation/footer patterns
            if line.lower() in ("home", "about", "contact", "privacy", "terms", "copyright"):
                continue
            cleaned_lines.append(line)

        return "\n\n".join(cleaned_lines).strip()

    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse a date string into a datetime object.

        Args:
            date_str: Date string to parse.

        Returns:
            Parsed datetime or None if parsing fails.

        """
        if not date_str:
            return None

        # Common date formats
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
            "%m/%d/%Y",
            "%d/%m/%Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Try ISO format with fractional seconds
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            pass

        return None

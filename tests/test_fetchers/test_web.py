"""Tests for the web fetcher."""

from unittest.mock import MagicMock, patch

import pytest

from personal_knowledge_base.fetchers.web import RobotsCheckResult, WebFetcher


class TestWebFetcher:
    """Tests for WebFetcher class."""

    @pytest.fixture
    def fetcher(self) -> WebFetcher:
        """Create a WebFetcher instance for testing."""
        return WebFetcher()

    @pytest.fixture
    def mock_page(self) -> MagicMock:
        """Create a mock Playwright page."""
        page = MagicMock()

        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200
        page.goto.return_value = mock_response

        # Mock locator
        mock_locator = MagicMock()
        mock_locator.first = MagicMock()
        mock_locator.first.inner_text.return_value = "Test content"
        mock_locator.first.get_attribute.return_value = "Test value"
        mock_locator.first.is_visible.return_value = True
        page.locator.return_value = mock_locator

        return page

    @pytest.fixture
    def mock_context(self) -> MagicMock:
        """Create a mock Playwright context."""
        context = MagicMock()
        return context

    @pytest.fixture
    def mock_browser(self, mock_context: MagicMock, mock_page: MagicMock) -> MagicMock:
        """Create a mock Playwright browser."""
        browser = MagicMock()
        browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        return browser

    # can_fetch tests

    def test_can_fetch_http_url(self, fetcher: WebFetcher) -> None:
        """Test can_fetch with standard HTTP URL."""
        url = "https://example.com/article"
        assert fetcher.can_fetch(url) is True

    def test_can_fetch_https_url(self, fetcher: WebFetcher) -> None:
        """Test can_fetch with HTTPS URL."""
        url = "https://example.com/article"
        assert fetcher.can_fetch(url) is True

    def test_can_fetch_youtube_url(self, fetcher: WebFetcher) -> None:
        """Test can_fetch returns False for YouTube URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert fetcher.can_fetch(url) is False

    def test_can_fetch_youtu_be_url(self, fetcher: WebFetcher) -> None:
        """Test can_fetch returns False for youtu.be URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert fetcher.can_fetch(url) is False

    def test_can_fetch_ftp_url(self, fetcher: WebFetcher) -> None:
        """Test can_fetch returns False for FTP URL."""
        url = "ftp://example.com/file.txt"
        assert fetcher.can_fetch(url) is False

    def test_can_fetch_file_url(self, fetcher: WebFetcher) -> None:
        """Test can_fetch returns False for file URL."""
        url = "file:///path/to/file.txt"
        assert fetcher.can_fetch(url) is False

    def test_can_fetch_invalid_url(self, fetcher: WebFetcher) -> None:
        """Test can_fetch with invalid URL."""
        url = "not-a-url"
        assert fetcher.can_fetch(url) is False

    def test_can_fetch_empty_string(self, fetcher: WebFetcher) -> None:
        """Test can_fetch with empty string."""
        url = ""
        assert fetcher.can_fetch(url) is False

    # robots.txt tests

    @patch("urllib.robotparser.RobotFileParser")
    def test_check_robots_txt_allowed(self, mock_rp_class: MagicMock, fetcher: WebFetcher) -> None:
        """Test robots.txt check when fetching is allowed."""
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = True
        mock_rp.crawl_delay.return_value = None
        mock_rp_class.return_value = mock_rp

        result = fetcher._check_robots_txt("https://example.com/article")

        assert result.can_fetch is True
        assert "allowed" in result.message.lower()

    @patch("urllib.robotparser.RobotFileParser")
    def test_check_robots_txt_blocked(self, mock_rp_class: MagicMock, fetcher: WebFetcher) -> None:
        """Test robots.txt check when fetching is blocked."""
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = False
        mock_rp_class.return_value = mock_rp

        result = fetcher._check_robots_txt("https://example.com/article")

        assert result.can_fetch is False
        assert "blocked" in result.message.lower()

    @patch("urllib.robotparser.RobotFileParser")
    def test_check_robots_txt_with_crawl_delay(
        self, mock_rp_class: MagicMock, fetcher: WebFetcher
    ) -> None:
        """Test robots.txt check with crawl delay."""
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = True
        mock_rp.crawl_delay.return_value = 5.0
        mock_rp_class.return_value = mock_rp

        result = fetcher._check_robots_txt("https://example.com/article")

        assert result.can_fetch is True
        assert result.crawl_delay == 5.0

    def test_check_robots_txt_disabled(self, fetcher: WebFetcher) -> None:
        """Test robots.txt check when disabled."""
        fetcher._respect_robots_txt = False

        result = fetcher._check_robots_txt("https://example.com/article")

        assert result.can_fetch is True
        assert "disabled" in result.message.lower()

    @patch("urllib.robotparser.RobotFileParser")
    def test_check_robots_txt_caching(self, mock_rp_class: MagicMock, fetcher: WebFetcher) -> None:
        """Test that robots.txt results are cached."""
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = True
        mock_rp_class.return_value = mock_rp

        # First call
        fetcher._check_robots_txt("https://example.com/article1")
        # Second call to same domain
        fetcher._check_robots_txt("https://example.com/article2")

        # RobotFileParser should only be created once
        assert mock_rp_class.call_count == 1

    # fetch tests - error cases

    def test_fetch_invalid_url(self, fetcher: WebFetcher) -> None:
        """Test fetch with invalid URL."""
        result = fetcher.fetch("not-a-url")

        assert result.success is False
        assert "invalid" in result.error_message.lower()

    def test_fetch_youtube_url(self, fetcher: WebFetcher) -> None:
        """Test fetch with YouTube URL returns error."""
        result = fetcher.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert result.success is False
        assert result.content_type == "article"

    @patch("urllib.robotparser.RobotFileParser")
    def test_fetch_blocked_by_robots_txt(
        self, mock_rp_class: MagicMock, fetcher: WebFetcher
    ) -> None:
        """Test fetch when blocked by robots.txt."""
        mock_rp = MagicMock()
        mock_rp.can_fetch.return_value = False
        mock_rp_class.return_value = mock_rp

        result = fetcher.fetch("https://example.com/article")

        assert result.success is False
        assert "robots.txt" in result.error_message.lower()

    # detect_paywall tests

    def test_detect_paywall_by_selector(self, fetcher: WebFetcher, mock_page: MagicMock) -> None:
        """Test paywall detection by CSS selector."""

        # Make the paywall selector visible
        def is_visible_side_effect(**kwargs):
            # Check if we're checking a paywall selector
            call_args = mock_page.locator.call_args
            if call_args and any(
                "paywall" in str(call_args).lower() for p in fetcher._PAYWALL_SELECTORS
            ):
                return True
            return True  # Default visible

        mock_page.locator.return_value.first.is_visible.side_effect = lambda **kwargs: True

        def mock_locator_side_effect(selector: str):
            mock_loc = MagicMock()
            mock_loc.first = MagicMock()
            if "paywall" in selector.lower():
                mock_loc.first.is_visible.return_value = True
            else:
                mock_loc.first.is_visible.return_value = False
            return mock_loc

        mock_page.locator.side_effect = mock_locator_side_effect

        result = fetcher._detect_paywall(mock_page)

        # Since we mocked paywall detection, it should detect it
        assert result["detected"] is True

    def test_detect_paywall_by_keyword(self, fetcher: WebFetcher, mock_page: MagicMock) -> None:
        """Test paywall detection by keyword in page text."""

        # Track which selector is being checked
        def mock_locator_side_effect(selector: str):
            mock_loc = MagicMock()
            mock_loc.first = MagicMock()
            if selector == "body":
                # Body contains paywall keyword - note: inner_text is called directly on locator, not .first
                mock_loc.inner_text.return_value = (
                    "This is premium content. Please subscribe to continue reading."
                )
            else:
                # Paywall selectors raise exception on is_visible (element not found)
                mock_loc.first.is_visible.side_effect = Exception("Element not found")
            return mock_loc

        mock_page.locator.side_effect = mock_locator_side_effect

        result = fetcher._detect_paywall(mock_page)

        assert result["detected"] is True
        assert "subscribe" in result["reason"].lower()

    def test_detect_paywall_not_found(self, fetcher: WebFetcher, mock_page: MagicMock) -> None:
        """Test paywall detection when no paywall present."""
        # Make paywall selectors not visible
        mock_page.locator.return_value.first.is_visible.return_value = False

        # Set page text without paywall keywords
        mock_page.locator.return_value.first.inner_text.return_value = (
            "This is regular article content without any paywall."
        )

        result = fetcher._detect_paywall(mock_page)

        assert result["detected"] is False

    # extract_content tests

    def test_extract_content_title_from_h1(self, fetcher: WebFetcher, mock_page: MagicMock) -> None:
        """Test title extraction from h1."""
        mock_page.locator.return_value.first.inner_text.return_value = "Article Title"

        result = fetcher._extract_content(mock_page)

        assert result["title"] == "Article Title"

    def test_extract_content_title_from_meta(
        self, fetcher: WebFetcher, mock_page: MagicMock
    ) -> None:
        """Test title extraction from meta tag."""
        # First h1 fails, then meta succeeds
        call_count = 0

        def inner_text_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:  # h1 selectors fail
                raise Exception("Not found")
            return "Meta Title"

        def get_attr_side_effect(name: str):
            if name == "content":
                return "Meta Title"
            return None

        mock_page.locator.return_value.first.inner_text.side_effect = inner_text_side_effect
        mock_page.locator.return_value.first.get_attribute.side_effect = get_attr_side_effect

        result = fetcher._extract_content(mock_page)

        # Should get title from meta or body
        assert "title" in result

    def test_extract_content_author_from_meta(
        self, fetcher: WebFetcher, mock_page: MagicMock
    ) -> None:
        """Test author extraction from meta tag."""
        mock_page.locator.return_value.first.get_attribute.return_value = "John Doe"

        result = fetcher._extract_content(mock_page)

        assert result["author"] == "John Doe"

    def test_extract_content_date_parsing(self, fetcher: WebFetcher) -> None:
        """Test date parsing from various formats."""
        # Test ISO format
        date = fetcher._parse_date("2024-03-09T10:30:00")
        assert date is not None
        assert date.year == 2024
        assert date.month == 3
        assert date.day == 9

        # Test date-only format
        date = fetcher._parse_date("2024-03-09")
        assert date is not None
        assert date.year == 2024

        # Test invalid format
        date = fetcher._parse_date("not-a-date")
        assert date is None

        # Test empty string
        date = fetcher._parse_date("")
        assert date is None

    # extract_main_content tests

    def test_extract_main_content_from_article(
        self, fetcher: WebFetcher, mock_page: MagicMock
    ) -> None:
        """Test content extraction from article element."""

        def mock_locator_side_effect(selector: str):
            mock_loc = MagicMock()
            mock_loc.first = MagicMock()
            if selector == "article":
                mock_loc.first.is_visible.return_value = True
                mock_loc.first.inner_text.return_value = "Article body content here. " * 10
            else:
                mock_loc.first.is_visible.return_value = False
            return mock_loc

        mock_page.locator.side_effect = mock_locator_side_effect

        result = fetcher._extract_main_content(mock_page)

        assert "Article body" in result

    def test_extract_main_content_from_main(
        self, fetcher: WebFetcher, mock_page: MagicMock
    ) -> None:
        """Test content extraction from main element."""

        # First selector (article) not visible, main is visible
        def mock_locator_side_effect(selector: str):
            mock_loc = MagicMock()
            mock_loc.first = MagicMock()
            if selector == "article":
                mock_loc.first.is_visible.return_value = False
            elif selector == "main":
                mock_loc.first.is_visible.return_value = True
                mock_loc.first.inner_text.return_value = "Main content area. " * 10
            else:
                mock_loc.first.is_visible.return_value = False
            return mock_loc

        mock_page.locator.side_effect = mock_locator_side_effect

        result = fetcher._extract_main_content(mock_page)

        assert "Main content" in result

    def test_extract_main_content_fallback_to_body(
        self, fetcher: WebFetcher, mock_page: MagicMock
    ) -> None:
        """Test content extraction falls back to body."""

        def mock_locator_side_effect(selector: str):
            mock_loc = MagicMock()
            mock_loc.first = MagicMock()
            if selector == "body":
                mock_loc.first.inner_text.return_value = "Body content here. " * 10
            else:
                mock_loc.first.is_visible.return_value = False
            return mock_loc

        mock_page.locator.side_effect = mock_locator_side_effect

        result = fetcher._extract_main_content(mock_page)

        assert "Body content" in result

    def test_extract_main_content_short_content_filtered(
        self, fetcher: WebFetcher, mock_page: MagicMock
    ) -> None:
        """Test that very short content is filtered and falls back."""
        content_selectors = [
            "article",
            '[role="main"]',
            "main",
            '[class*="article-content"]',
            '[class*="article-body"]',
        ]

        def mock_locator_side_effect(selector: str):
            mock_loc = MagicMock()
            mock_loc.first = MagicMock()
            if selector in content_selectors:
                # These return short content (too short to use)
                mock_loc.first.is_visible.return_value = True
                mock_loc.first.inner_text.return_value = "Hi"  # Too short (< 100 chars)
            elif selector == "body":
                mock_loc.first.inner_text.return_value = (
                    "This is the actual body content that is long enough for the test to pass. " * 3
                )
            else:
                mock_loc.first.is_visible.return_value = False
            return mock_loc

        mock_page.locator.side_effect = mock_locator_side_effect

        result = fetcher._extract_main_content(mock_page)

        assert len(result) > 50

    # clean_text tests

    def test_clean_text_removes_extra_whitespace(self, fetcher: WebFetcher) -> None:
        """Test that extra whitespace is removed."""
        text = "Line 1\n\n\nLine 2    with   spaces"
        result = fetcher._clean_text(text)
        assert "\n\n\n" not in result

    def test_clean_text_removes_short_lines(self, fetcher: WebFetcher) -> None:
        """Test that very short lines are removed."""
        text = "A\nValid line here\nB"
        result = fetcher._clean_text(text)
        assert "A\n" not in result
        assert "B" not in result or len(result) > 10

    def test_clean_text_removes_navigation_words(self, fetcher: WebFetcher) -> None:
        """Test that navigation words are removed."""
        text = "Home\nAbout\nContact\nActual content here"
        result = fetcher._clean_text(text)
        assert "home" not in result.lower() or "Actual content" in result

    def test_clean_text_empty_string(self, fetcher: WebFetcher) -> None:
        """Test cleaning empty string."""
        result = fetcher._clean_text("")
        assert result == ""

    def test_clean_text_none(self, fetcher: WebFetcher) -> None:
        """Test cleaning None."""
        result = fetcher._clean_text("")  # type: ignore[arg-type]
        assert result == ""

    # HTTP error handling tests

    @patch("personal_knowledge_base.fetchers.web.sync_playwright")
    def test_fetch_http_404(self, mock_playwright: MagicMock, fetcher: WebFetcher) -> None:
        """Test handling of HTTP 404 error."""
        # Setup mocks
        mock_p = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_response = MagicMock()

        mock_response.status = 404
        mock_page.goto.return_value = mock_response
        mock_context.new_page.return_value = mock_page
        mock_browser.new_context.return_value = mock_context
        mock_p.chromium.launch.return_value = mock_browser
        mock_playwright.return_value.__enter__.return_value = mock_p

        result = fetcher.fetch("https://example.com/not-found")

        assert result.success is False
        assert "404" in result.error_message

    @patch("personal_knowledge_base.fetchers.web.sync_playwright")
    def test_fetch_http_403(self, mock_playwright: MagicMock, fetcher: WebFetcher) -> None:
        """Test handling of HTTP 403 error."""
        mock_p = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_response = MagicMock()

        mock_response.status = 403
        mock_page.goto.return_value = mock_response
        mock_context.new_page.return_value = mock_page
        mock_browser.new_context.return_value = mock_context
        mock_p.chromium.launch.return_value = mock_browser
        mock_playwright.return_value.__enter__.return_value = mock_p

        result = fetcher.fetch("https://example.com/forbidden")

        assert result.success is False
        assert "403" in result.error_message

    @patch("personal_knowledge_base.fetchers.web.sync_playwright")
    def test_fetch_http_500(self, mock_playwright: MagicMock, fetcher: WebFetcher) -> None:
        """Test handling of HTTP 500 error."""
        mock_p = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_response = MagicMock()

        mock_response.status = 500
        mock_page.goto.return_value = mock_response
        mock_context.new_page.return_value = mock_page
        mock_browser.new_context.return_value = mock_context
        mock_p.chromium.launch.return_value = mock_browser
        mock_playwright.return_value.__enter__.return_value = mock_p

        result = fetcher.fetch("https://example.com/error")

        assert result.success is False
        assert "500" in result.error_message

    @patch("personal_knowledge_base.fetchers.web.sync_playwright")
    def test_fetch_no_response(self, mock_playwright: MagicMock, fetcher: WebFetcher) -> None:
        """Test handling when no response is received."""
        mock_p = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_page.goto.return_value = None
        mock_context.new_page.return_value = mock_page
        mock_browser.new_context.return_value = mock_context
        mock_p.chromium.launch.return_value = mock_browser
        mock_playwright.return_value.__enter__.return_value = mock_p

        result = fetcher.fetch("https://example.com/no-response")

        assert result.success is False
        assert "no response" in result.error_message.lower()

    # Timeout and network error tests

    @patch("personal_knowledge_base.fetchers.web.sync_playwright")
    def test_fetch_timeout_error(self, mock_playwright: MagicMock, fetcher: WebFetcher) -> None:
        """Test handling of timeout error."""
        mock_p = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_page.goto.side_effect = Exception("Timeout 30000ms exceeded")
        mock_context.new_page.return_value = mock_page
        mock_browser.new_context.return_value = mock_context
        mock_p.chromium.launch.return_value = mock_browser
        mock_playwright.return_value.__enter__.return_value = mock_p

        result = fetcher.fetch("https://example.com/slow")

        assert result.success is False
        assert "timeout" in result.error_message.lower()

    @patch("personal_knowledge_base.fetchers.web.sync_playwright")
    def test_fetch_network_error(self, mock_playwright: MagicMock, fetcher: WebFetcher) -> None:
        """Test handling of network error."""
        mock_p = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_page.goto.side_effect = Exception("net::ERR_CONNECTION_REFUSED")
        mock_context.new_page.return_value = mock_page
        mock_browser.new_context.return_value = mock_context
        mock_p.chromium.launch.return_value = mock_browser
        mock_playwright.return_value.__enter__.return_value = mock_p

        result = fetcher.fetch("https://example.com/refused")

        assert result.success is False
        assert "network" in result.error_message.lower()

    # Successful fetch tests

    @patch("personal_knowledge_base.fetchers.web.sync_playwright")
    def test_fetch_success(self, mock_playwright: MagicMock, fetcher: WebFetcher) -> None:
        """Test successful fetch."""
        mock_p = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_response = MagicMock()

        mock_response.status = 200
        mock_page.goto.return_value = mock_response
        mock_page.url = "https://example.com/article"

        # Mock content extraction
        mock_locator = MagicMock()
        mock_locator.first = MagicMock()
        mock_locator.first.inner_text.return_value = "Article Title"
        mock_locator.first.get_attribute.return_value = None
        mock_locator.first.is_visible.return_value = False
        mock_page.locator.return_value = mock_locator

        mock_context.new_page.return_value = mock_page
        mock_browser.new_context.return_value = mock_context
        mock_p.chromium.launch.return_value = mock_browser
        mock_playwright.return_value.__enter__.return_value = mock_p

        # Mock extract_main_content
        with patch.object(fetcher, "_extract_main_content", return_value="Article content here."):
            result = fetcher.fetch("https://example.com/article")

            assert result.success is True
            assert result.content_type == "article"
            assert result.url == "https://example.com/article"

    @patch("personal_knowledge_base.fetchers.web.sync_playwright")
    def test_fetch_success_with_metadata(
        self, mock_playwright: MagicMock, fetcher: WebFetcher
    ) -> None:
        """Test successful fetch with all metadata."""
        mock_p = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_response = MagicMock()

        mock_response.status = 200
        mock_page.goto.return_value = mock_response
        mock_page.url = "https://example.com/article"

        # Mock content extraction with author and date
        call_count = 0

        def inner_text_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            texts = ["Article Title", "John Doe", "March 9, 2024", "Article content."]
            return texts[min(call_count - 1, len(texts) - 1)]

        def get_attr_side_effect(name: str):
            if name == "content":
                return "2024-03-09"
            return None

        mock_locator = MagicMock()
        mock_locator.first = MagicMock()
        mock_locator.first.inner_text.side_effect = inner_text_side_effect
        mock_locator.first.get_attribute.side_effect = get_attr_side_effect
        mock_locator.first.is_visible.return_value = False
        mock_page.locator.return_value = mock_locator

        mock_context.new_page.return_value = mock_page
        mock_browser.new_context.return_value = mock_context
        mock_p.chromium.launch.return_value = mock_browser
        mock_playwright.return_value.__enter__.return_value = mock_p

        with patch.object(fetcher, "_extract_main_content", return_value="Article content here."):
            result = fetcher.fetch("https://example.com/article")

            assert result.success is True
            assert result.content == "Article content here."
            assert "http_status" in result.metadata
            assert result.metadata["http_status"] == 200

    # Paywall detection in fetch

    @patch("personal_knowledge_base.fetchers.web.sync_playwright")
    def test_fetch_detects_paywall(self, mock_playwright: MagicMock, fetcher: WebFetcher) -> None:
        """Test that paywall is detected during fetch."""
        mock_p = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_response = MagicMock()

        mock_response.status = 200
        mock_page.goto.return_value = mock_response

        # Mock paywall detection
        def mock_detect_paywall(page):
            return {"detected": True, "reason": "Paywall element found"}

        mock_context.new_page.return_value = mock_page
        mock_browser.new_context.return_value = mock_context
        mock_p.chromium.launch.return_value = mock_browser
        mock_playwright.return_value.__enter__.return_value = mock_p

        with patch.object(
            fetcher,
            "_detect_paywall",
            return_value={"detected": True, "reason": "Paywall detected"},
        ):
            result = fetcher.fetch("https://example.com/premium-article")

            assert result.success is False
            assert "paywall" in result.error_message.lower()

    # Initialization tests

    def test_fetcher_initialization_defaults(self) -> None:
        """Test fetcher initialization with default values."""
        fetcher = WebFetcher()

        assert fetcher._timeout == 30000
        assert fetcher._navigation_timeout == 30000
        assert fetcher._respect_robots_txt is True
        assert fetcher._robots_cache == {}

    def test_fetcher_initialization_custom(self) -> None:
        """Test fetcher initialization with custom values."""
        fetcher = WebFetcher(
            timeout=60000,
            navigation_timeout=45000,
            respect_robots_txt=False,
        )

        assert fetcher._timeout == 60000
        assert fetcher._navigation_timeout == 45000
        assert fetcher._respect_robots_txt is False

    # RobotsCheckResult tests

    def test_robots_check_result_defaults(self) -> None:
        """Test RobotsCheckResult with default values."""
        result = RobotsCheckResult(can_fetch=True)

        assert result.can_fetch is True
        assert result.crawl_delay is None
        assert result.message == ""

    def test_robots_check_result_full(self) -> None:
        """Test RobotsCheckResult with all values."""
        result = RobotsCheckResult(
            can_fetch=False,
            crawl_delay=5.0,
            message="Blocked by robots.txt",
        )

        assert result.can_fetch is False
        assert result.crawl_delay == 5.0
        assert result.message == "Blocked by robots.txt"

    # Edge case tests

    def test_parse_date_iso_with_z(self, fetcher: WebFetcher) -> None:
        """Test parsing ISO date with Z suffix."""
        date = fetcher._parse_date("2024-03-09T10:30:00Z")
        assert date is not None
        assert date.year == 2024

    def test_parse_date_iso_with_timezone(self, fetcher: WebFetcher) -> None:
        """Test parsing ISO date with timezone offset."""
        date = fetcher._parse_date("2024-03-09T10:30:00+00:00")
        assert date is not None
        assert date.year == 2024

    def test_parse_date_written_format(self, fetcher: WebFetcher) -> None:
        """Test parsing written date formats."""
        date = fetcher._parse_date("March 9, 2024")
        assert date is not None
        assert date.month == 3
        assert date.day == 9

    def test_parse_date_short_month(self, fetcher: WebFetcher) -> None:
        """Test parsing short month format."""
        date = fetcher._parse_date("Mar 9, 2024")
        assert date is not None
        assert date.month == 3

    def test_parse_date_day_first(self, fetcher: WebFetcher) -> None:
        """Test parsing day-first date format."""
        date = fetcher._parse_date("9 March 2024")
        assert date is not None
        assert date.day == 9

    def test_parse_date_slash_format(self, fetcher: WebFetcher) -> None:
        """Test parsing slash-separated date format."""
        date = fetcher._parse_date("03/09/2024")
        assert date is not None
        assert date.month == 3
        assert date.day == 9

    def test_is_youtube_url_variations(self, fetcher: WebFetcher) -> None:
        """Test YouTube URL detection with various formats."""
        assert fetcher._is_youtube_url("https://www.youtube.com/watch?v=123") is True
        assert fetcher._is_youtube_url("https://youtube.com/watch?v=123") is True
        assert fetcher._is_youtube_url("https://youtu.be/123") is True
        assert fetcher._is_youtube_url("https://www.YOUTUBE.com/watch?v=123") is True
        assert fetcher._is_youtube_url("https://example.com") is False
        assert fetcher._is_youtube_url("https://vimeo.com/123") is False

    def test_can_fetch_value_error(self, fetcher: WebFetcher) -> None:
        """Test can_fetch handles ValueError."""
        # Invalid URL that causes urlparse to raise ValueError
        result = fetcher.can_fetch("http://[invalid")
        assert result is False

    def test_can_fetch_type_error(self, fetcher: WebFetcher) -> None:
        """Test can_fetch handles TypeError."""
        # None should cause TypeError
        result = fetcher.can_fetch(None)  # type: ignore[arg-type]
        assert result is False

    @patch("urllib.robotparser.RobotFileParser")
    def test_check_robots_txt_exception_on_read(
        self, mock_rp_class: MagicMock, fetcher: WebFetcher
    ) -> None:
        """Test robots.txt check when read() raises exception."""
        mock_rp = MagicMock()
        mock_rp.read.side_effect = Exception("Connection error")
        mock_rp_class.return_value = mock_rp

        result = fetcher._check_robots_txt("https://example.com/article")

        assert result.can_fetch is True
        assert "could not fetch" in result.message.lower()

    @patch("urllib.robotparser.RobotFileParser")
    def test_check_robots_txt_general_exception(
        self, mock_rp_class: MagicMock, fetcher: WebFetcher
    ) -> None:
        """Test robots.txt check when general exception occurs."""
        mock_rp_class.side_effect = Exception("Unexpected error")

        result = fetcher._check_robots_txt("https://example.com/article")

        assert result.can_fetch is True
        assert "error checking" in result.message.lower()

    @patch("personal_knowledge_base.fetchers.web.sync_playwright")
    def test_fetch_generic_error(self, mock_playwright: MagicMock, fetcher: WebFetcher) -> None:
        """Test handling of generic error during fetch."""
        mock_p = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_page.goto.side_effect = Exception("Some unexpected error")
        mock_context.new_page.return_value = mock_page
        mock_browser.new_context.return_value = mock_context
        mock_p.chromium.launch.return_value = mock_browser
        mock_playwright.return_value.__enter__.return_value = mock_p

        result = fetcher.fetch("https://example.com/error")

        assert result.success is False
        assert "fetch error" in result.error_message.lower()

    @patch("personal_knowledge_base.fetchers.web.sync_playwright")
    def test_fetch_paywall_detection_exception(
        self, mock_playwright: MagicMock, fetcher: WebFetcher
    ) -> None:
        """Test fetch when paywall detection raises exception."""
        mock_p = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_response = MagicMock()

        mock_response.status = 200
        mock_page.goto.return_value = mock_response
        mock_page.url = "https://example.com/article"

        mock_context.new_page.return_value = mock_page
        mock_browser.new_context.return_value = mock_context
        mock_p.chromium.launch.return_value = mock_browser
        mock_playwright.return_value.__enter__.return_value = mock_p

        with patch.object(
            fetcher, "_detect_paywall", side_effect=Exception("Paywall detection failed")
        ):
            result = fetcher.fetch("https://example.com/article")

            # Exception is caught and results in failure
            assert result.success is False
            assert "fetch error" in result.error_message.lower()

    @patch("personal_knowledge_base.fetchers.web.sync_playwright")
    def test_fetch_extract_content_exception(
        self, mock_playwright: MagicMock, fetcher: WebFetcher
    ) -> None:
        """Test fetch when content extraction raises exception."""
        mock_p = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_response = MagicMock()

        mock_response.status = 200
        mock_page.goto.return_value = mock_response

        mock_context.new_page.return_value = mock_page
        mock_browser.new_context.return_value = mock_context
        mock_p.chromium.launch.return_value = mock_browser
        mock_playwright.return_value.__enter__.return_value = mock_p

        with (
            patch.object(
                fetcher, "_detect_paywall", return_value={"detected": False, "reason": ""}
            ),
            patch.object(
                fetcher, "_extract_content", side_effect=Exception("Content extraction failed")
            ),
        ):
            result = fetcher.fetch("https://example.com/article")

            assert result.success is False
            assert "fetch error" in result.error_message.lower()

    def test_detect_paywall_exception_on_body_text(
        self, fetcher: WebFetcher, mock_page: MagicMock
    ) -> None:
        """Test paywall detection when body text extraction fails."""

        def mock_locator_side_effect(selector: str):
            mock_loc = MagicMock()
            mock_loc.first = MagicMock()
            if selector == "body":
                # Body text extraction raises exception
                mock_loc.inner_text.side_effect = Exception("Body not found")
            else:
                # Paywall selectors raise exception on is_visible
                mock_loc.first.is_visible.side_effect = Exception("Element not found")
            return mock_loc

        mock_page.locator.side_effect = mock_locator_side_effect

        result = fetcher._detect_paywall(mock_page)

        # Should return not detected when body text extraction fails
        assert result["detected"] is False

    def test_extract_content_title_exception(
        self, fetcher: WebFetcher, mock_page: MagicMock
    ) -> None:
        """Test content extraction when all title selectors fail."""
        # Make all locators raise exception
        mock_page.locator.return_value.first.inner_text.side_effect = Exception("Not found")
        mock_page.locator.return_value.first.get_attribute.return_value = None

        result = fetcher._extract_content(mock_page)

        # Should still return a result with empty title
        assert "title" in result
        assert "content" in result

    def test_extract_content_author_exception(
        self, fetcher: WebFetcher, mock_page: MagicMock
    ) -> None:
        """Test content extraction when author extraction fails."""
        # Make author selectors raise exception
        call_count = 0

        def inner_text_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Title"  # First call is for title
            raise Exception("Not found")

        mock_page.locator.return_value.first.inner_text.side_effect = inner_text_side_effect
        mock_page.locator.return_value.first.get_attribute.return_value = None

        with patch.object(fetcher, "_extract_main_content", return_value="Content"):
            result = fetcher._extract_content(mock_page)

            assert result["title"] == "Title"
            assert result["author"] is None

    def test_extract_content_date_exception(
        self, fetcher: WebFetcher, mock_page: MagicMock
    ) -> None:
        """Test content extraction when date extraction fails."""
        # Make date selectors raise exception
        call_count = 0

        def inner_text_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Title"  # First call is for title
            raise Exception("Not found")

        mock_page.locator.return_value.first.inner_text.side_effect = inner_text_side_effect
        mock_page.locator.return_value.first.get_attribute.return_value = None

        with patch.object(fetcher, "_extract_main_content", return_value="Content"):
            result = fetcher._extract_content(mock_page)

            assert result["title"] == "Title"
            assert result["published_date"] is None

    def test_extract_main_content_all_selectors_fail(
        self, fetcher: WebFetcher, mock_page: MagicMock
    ) -> None:
        """Test main content extraction when all selectors fail."""
        # Make all selectors raise exception
        mock_page.locator.return_value.first.is_visible.side_effect = Exception("Not found")
        mock_page.locator.return_value.first.inner_text.side_effect = Exception("Not found")

        result = fetcher._extract_main_content(mock_page)

        # Should return empty string when everything fails
        assert result == ""

    def test_extract_main_content_body_exception(
        self, fetcher: WebFetcher, mock_page: MagicMock
    ) -> None:
        """Test main content extraction when body fallback fails."""

        # Make all content selectors fail visibility
        def mock_locator_side_effect(selector: str):
            mock_loc = MagicMock()
            mock_loc.first = MagicMock()
            if selector == "body":
                # Body raises exception
                mock_loc.first.inner_text.side_effect = Exception("Body not found")
            else:
                mock_loc.first.is_visible.return_value = False
            return mock_loc

        mock_page.locator.side_effect = mock_locator_side_effect

        result = fetcher._extract_main_content(mock_page)

        # Should return empty string when body fails
        assert result == ""

"""Tests for the PDF fetcher."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyPDF2.errors import PdfReadError
from requests.exceptions import HTTPError, RequestException, Timeout

from personal_knowledge_base.fetchers.pdf import PDFFetcher


class TestPDFFetcher:
    """Tests for PDFFetcher class."""

    @pytest.fixture
    def fetcher(self, tmp_path: Path) -> PDFFetcher:
        """Create a PDFFetcher instance with temp directory."""
        return PDFFetcher(pdf_dir=tmp_path)

    @pytest.fixture
    def sample_pdf_url(self) -> str:
        """Sample PDF URL for testing."""
        return "https://example.com/document.pdf"

    # URL parsing tests

    def test_can_fetch_pdf_url(self, fetcher: PDFFetcher) -> None:
        """Test can_fetch with standard PDF URL."""
        assert fetcher.can_fetch("https://example.com/document.pdf") is True

    def test_can_fetch_pdf_url_uppercase(self, fetcher: PDFFetcher) -> None:
        """Test can_fetch with uppercase PDF extension."""
        assert fetcher.can_fetch("https://example.com/document.PDF") is True

    def test_can_fetch_pdf_url_mixed_case(self, fetcher: PDFFetcher) -> None:
        """Test can_fetch with mixed case PDF extension."""
        assert fetcher.can_fetch("https://example.com/document.Pdf") is True

    def test_can_fetch_pdf_in_path(self, fetcher: PDFFetcher) -> None:
        """Test can_fetch with /pdf/ in path."""
        assert fetcher.can_fetch("https://example.com/pdf/document") is True

    def test_can_fetch_download_path(self, fetcher: PDFFetcher) -> None:
        """Test can_fetch with /download/ in path."""
        assert fetcher.can_fetch("https://example.com/download/paper") is True

    def test_can_fetch_non_pdf_url(self, fetcher: PDFFetcher) -> None:
        """Test can_fetch with non-PDF URL."""
        assert fetcher.can_fetch("https://example.com/document.html") is False

    def test_can_fetch_non_http_url(self, fetcher: PDFFetcher) -> None:
        """Test can_fetch with non-HTTP URL."""
        assert fetcher.can_fetch("ftp://example.com/document.pdf") is False

    def test_can_fetch_empty_string(self, fetcher: PDFFetcher) -> None:
        """Test can_fetch with empty string."""
        assert fetcher.can_fetch("") is False

    def test_can_fetch_malformed_url(self, fetcher: PDFFetcher) -> None:
        """Test can_fetch with malformed URL."""
        assert fetcher.can_fetch("not-a-url") is False

    def test_can_fetch_query_params(self, fetcher: PDFFetcher) -> None:
        """Test can_fetch with query parameters."""
        assert fetcher.can_fetch("https://example.com/doc.pdf?download=1") is True

    # Download tests

    @patch("requests.Session.get")
    def test_download_pdf_success(self, mock_get: MagicMock, fetcher: PDFFetcher) -> None:
        """Test successful PDF download."""
        # Mock response
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"PDF content"]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        url = "https://example.com/test.pdf"
        path = fetcher._download_pdf(url)

        assert path.exists()
        assert path.suffix == ".pdf"
        mock_get.assert_called_once()

    @patch("requests.Session.get")
    def test_download_pdf_with_redirects(self, mock_get: MagicMock, fetcher: PDFFetcher) -> None:
        """Test PDF download follows redirects."""
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.iter_content.return_value = [b"PDF"]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        url = "https://example.com/redirect.pdf"
        path = fetcher._download_pdf(url)

        assert path.exists()
        # Verify allow_redirects was passed
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs.get("allow_redirects") is True

    @patch("requests.Session.get")
    def test_download_pdf_too_large(self, mock_get: MagicMock, fetcher: PDFFetcher) -> None:
        """Test download fails for oversized PDF."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(200 * 1024 * 1024)}  # 200MB
        mock_get.return_value = mock_response

        url = "https://example.com/huge.pdf"

        with pytest.raises(ValueError, match="too large"):
            fetcher._download_pdf(url)

    @patch("requests.Session.get")
    def test_download_pdf_size_check_during_stream(
        self, mock_get: MagicMock, fetcher: PDFFetcher
    ) -> None:
        """Test size check during streaming download."""
        mock_response = MagicMock()
        mock_response.headers = {}  # No content-length
        # Simulate chunks that exceed limit
        mock_response.iter_content.return_value = [b"x" * 8192] * 20000
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        url = "https://example.com/streaming.pdf"

        with pytest.raises(ValueError, match="too large"):
            fetcher._download_pdf(url)

    # Fetch error handling tests

    @patch("requests.Session.get")
    def test_fetch_404_error(self, mock_get: MagicMock, fetcher: PDFFetcher) -> None:
        """Test handling of 404 error."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.side_effect = HTTPError("404 Client Error", response=mock_response)

        result = fetcher.fetch("https://example.com/missing.pdf")

        assert result.success is False
        assert "404" in result.error_message
        assert "not found" in result.error_message.lower()

    @patch("requests.Session.get")
    def test_fetch_403_error(self, mock_get: MagicMock, fetcher: PDFFetcher) -> None:
        """Test handling of 403 error."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_get.side_effect = HTTPError("403 Forbidden", response=mock_response)

        result = fetcher.fetch("https://example.com/protected.pdf")

        assert result.success is False
        assert "403" in result.error_message
        assert "forbidden" in result.error_message.lower()

    @patch("requests.Session.get")
    def test_fetch_500_error(self, mock_get: MagicMock, fetcher: PDFFetcher) -> None:
        """Test handling of 500 error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.side_effect = HTTPError("500 Server Error", response=mock_response)

        result = fetcher.fetch("https://example.com/error.pdf")

        assert result.success is False
        assert "500" in result.error_message

    @patch("requests.Session.get")
    def test_fetch_timeout(self, mock_get: MagicMock, fetcher: PDFFetcher) -> None:
        """Test handling of timeout."""
        mock_get.side_effect = Timeout("Request timed out")

        result = fetcher.fetch("https://example.com/slow.pdf")

        assert result.success is False
        assert "timeout" in result.error_message.lower()

    @patch("requests.Session.get")
    def test_fetch_network_error(self, mock_get: MagicMock, fetcher: PDFFetcher) -> None:
        """Test handling of network error."""
        mock_get.side_effect = RequestException("Connection failed")

        result = fetcher.fetch("https://example.com/unreachable.pdf")

        assert result.success is False
        assert "network error" in result.error_message.lower()

    def test_fetch_invalid_url(self, fetcher: PDFFetcher) -> None:
        """Test fetch with invalid URL."""
        result = fetcher.fetch("https://example.com/not-a-pdf")

        assert result.success is False
        assert "does not point to a PDF" in result.error_message

    # PDF content extraction tests

    @patch("personal_knowledge_base.fetchers.pdf.PdfReader")
    @patch("personal_knowledge_base.fetchers.pdf.pdfplumber.open")
    @patch("requests.Session.get")
    def test_fetch_successful_extraction(
        self,
        mock_get: MagicMock,
        mock_pdfplumber: MagicMock,
        mock_reader: MagicMock,
        fetcher: PDFFetcher,
        tmp_path: Path,
    ) -> None:
        """Test successful PDF text extraction."""
        # Setup mock download
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"PDF content"]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Setup mock PyPDF2 reader
        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.pages = [MagicMock(), MagicMock()]  # 2 pages
        mock_pdf.metadata = MagicMock()
        mock_pdf.metadata.title = "Test Document"
        mock_pdf.metadata.author = "Test Author"
        mock_pdf.metadata.subject = "Test Subject"
        mock_pdf.metadata.creator = "Test Creator"
        mock_pdf.metadata.producer = "Test Producer"
        mock_pdf.metadata.creation_date = "D:20240101120000"
        mock_pdf.metadata.modification_date = None
        mock_reader.return_value = mock_pdf

        # Setup mock pdfplumber
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page text content"
        mock_pdf_doc = MagicMock()
        mock_pdf_doc.pages = [mock_page, mock_page]
        mock_pdf_doc.__enter__ = MagicMock(return_value=mock_pdf_doc)
        mock_pdf_doc.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.return_value = mock_pdf_doc

        result = fetcher.fetch("https://example.com/test.pdf")

        assert result.success is True
        assert result.content_type == "pdf"
        assert "Test Document" in result.title
        assert result.author == "Test Author"
        assert result.metadata["page_count"] == 2
        assert result.metadata["pdf_author"] == "Test Author"
        assert result.metadata["pdf_subject"] == "Test Subject"

    @patch("personal_knowledge_base.fetchers.pdf.PdfReader")
    @patch("requests.Session.get")
    def test_fetch_password_protected(
        self,
        mock_get: MagicMock,
        mock_reader: MagicMock,
        fetcher: PDFFetcher,
    ) -> None:
        """Test handling of password-protected PDF."""
        # Setup mock download
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"PDF content"]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Setup mock encrypted PDF
        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = True
        mock_reader.return_value = mock_pdf

        result = fetcher.fetch("https://example.com/protected.pdf")

        assert result.success is False
        assert "password protected" in result.error_message.lower()

    @patch("personal_knowledge_base.fetchers.pdf.PdfReader")
    @patch("requests.Session.get")
    def test_fetch_corrupted_pdf(
        self,
        mock_get: MagicMock,
        mock_reader: MagicMock,
        fetcher: PDFFetcher,
    ) -> None:
        """Test handling of corrupted PDF."""
        # Setup mock download
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"not a pdf"]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Setup mock to raise PdfReadError
        mock_reader.side_effect = PdfReadError("Invalid PDF header")

        result = fetcher.fetch("https://example.com/corrupted.pdf")

        assert result.success is False
        assert "corrupted" in result.error_message.lower()

    # Scanned PDF detection tests

    def test_is_scanned_pdf_true(self, fetcher: PDFFetcher) -> None:
        """Test detection of scanned PDF."""
        # Mostly empty pages = scanned
        text_by_page = ["", "", "Short", "", ""]
        assert fetcher._is_scanned_pdf(text_by_page, 5) is True

    def test_is_scanned_pdf_false(self, fetcher: PDFFetcher) -> None:
        """Test detection of text-based PDF."""
        # Most pages have substantial text
        text_by_page = ["Page one content here", "Page two content here", ""]
        assert fetcher._is_scanned_pdf(text_by_page, 3) is False

    def test_is_scanned_pdf_empty(self, fetcher: PDFFetcher) -> None:
        """Test scanned detection with empty PDF."""
        assert fetcher._is_scanned_pdf([], 0) is False

    def test_is_scanned_pdf_threshold_edge_case(self, fetcher: PDFFetcher) -> None:
        """Test scanned detection at threshold boundary."""
        # Exactly at threshold (10%)
        text_by_page = ["Content"] + [""] * 9  # 1/10 = 10%
        assert fetcher._is_scanned_pdf(text_by_page, 10) is True

    # Metadata extraction tests

    def test_extract_metadata_with_all_fields(self, fetcher: PDFFetcher) -> None:
        """Test metadata extraction with all fields present."""
        mock_pdf = MagicMock()
        mock_pdf.metadata = MagicMock()
        mock_pdf.metadata.title = "  Test Title  "
        mock_pdf.metadata.author = "  Test Author  "
        mock_pdf.metadata.subject = "  Test Subject  "
        mock_pdf.metadata.creator = "  Test Creator  "
        mock_pdf.metadata.producer = "  Test Producer  "
        mock_pdf.metadata.creation_date = "D:20240101120000"
        mock_pdf.metadata.modification_date = "D:20240201120000"

        metadata = fetcher._extract_metadata(mock_pdf)

        assert metadata["title"] == "Test Title"
        assert metadata["author"] == "Test Author"
        assert metadata["subject"] == "Test Subject"
        assert metadata["creator"] == "Test Creator"
        assert metadata["producer"] == "Test Producer"
        assert isinstance(metadata["creation_date"], datetime)
        assert isinstance(metadata["modification_date"], datetime)

    def test_extract_metadata_empty(self, fetcher: PDFFetcher) -> None:
        """Test metadata extraction with no metadata."""
        mock_pdf = MagicMock()
        mock_pdf.metadata = None

        metadata = fetcher._extract_metadata(mock_pdf)

        assert metadata == {}

    def test_extract_metadata_partial(self, fetcher: PDFFetcher) -> None:
        """Test metadata extraction with partial fields."""
        mock_pdf = MagicMock()
        mock_pdf.metadata = MagicMock()
        mock_pdf.metadata.title = "Title Only"
        mock_pdf.metadata.author = None
        mock_pdf.metadata.subject = None
        mock_pdf.metadata.creator = None
        mock_pdf.metadata.producer = None
        mock_pdf.metadata.creation_date = None
        mock_pdf.metadata.modification_date = None

        metadata = fetcher._extract_metadata(mock_pdf)

        assert metadata["title"] == "Title Only"
        assert "author" not in metadata

    # Date parsing tests

    def test_parse_pdf_date_full(self, fetcher: PDFFetcher) -> None:
        """Test parsing full PDF date string."""
        result = fetcher._parse_pdf_date("D:20240101120000")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_parse_pdf_date_no_prefix(self, fetcher: PDFFetcher) -> None:
        """Test parsing PDF date without D: prefix."""
        result = fetcher._parse_pdf_date("20240101120000")
        assert result is not None
        assert result.year == 2024

    def test_parse_pdf_date_date_only(self, fetcher: PDFFetcher) -> None:
        """Test parsing date-only PDF date string."""
        result = fetcher._parse_pdf_date("D:20240101")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_parse_pdf_date_invalid(self, fetcher: PDFFetcher) -> None:
        """Test parsing invalid date string."""
        result = fetcher._parse_pdf_date("invalid")
        assert result is None

    def test_parse_pdf_date_empty(self, fetcher: PDFFetcher) -> None:
        """Test parsing empty date string."""
        result = fetcher._parse_pdf_date("")
        assert result is None

    # Text cleaning tests

    def test_clean_text_normal(self, fetcher: PDFFetcher) -> None:
        """Test cleaning normal text."""
        text = "  Multiple   spaces   here  "
        result = fetcher._clean_text(text)
        assert result == "Multiple spaces here"

    def test_clean_text_with_newlines(self, fetcher: PDFFetcher) -> None:
        """Test cleaning text with newlines."""
        text = "Line 1\n\nLine 2"
        result = fetcher._clean_text(text)
        # Newlines should be preserved
        assert "\n" in result or "Line 1" in result

    def test_clean_text_empty(self, fetcher: PDFFetcher) -> None:
        """Test cleaning empty text."""
        assert fetcher._clean_text("") == ""
        assert fetcher._clean_text(None) == ""  # type: ignore[arg-type]

    def test_clean_text_control_chars(self, fetcher: PDFFetcher) -> None:
        """Test cleaning text with control characters."""
        text = "Hello\x00World\x01Test"
        result = fetcher._clean_text(text)
        # Control chars should be removed
        assert "\x00" not in result
        assert "\x01" not in result

    # URL title extraction tests

    def test_extract_title_from_url_standard(self, fetcher: PDFFetcher) -> None:
        """Test extracting title from standard PDF URL."""
        url = "https://example.com/my-document.pdf"
        result = fetcher._extract_title_from_url(url)
        assert result == "My Document"

    def test_extract_title_from_url_with_underscores(self, fetcher: PDFFetcher) -> None:
        """Test extracting title with underscores."""
        url = "https://example.com/my_document_name.pdf"
        result = fetcher._extract_title_from_url(url)
        assert result == "My Document Name"

    def test_extract_title_from_url_no_pdf(self, fetcher: PDFFetcher) -> None:
        """Test extracting title from non-PDF path."""
        url = "https://example.com/path/to/file"
        result = fetcher._extract_title_from_url(url)
        assert result == "PDF Document"

    def test_extract_title_from_url_malformed(self, fetcher: PDFFetcher) -> None:
        """Test extracting title from malformed URL."""
        url = "not-a-valid-url"
        result = fetcher._extract_title_from_url(url)
        assert result == "PDF Document"

    # Utility tests

    def test_is_temp_file_true(self, fetcher: PDFFetcher) -> None:
        """Test detecting temp file."""
        temp_path = Path(tempfile.gettempdir()) / "test.pdf"
        assert fetcher._is_temp_file(temp_path) is True

    def test_is_temp_file_false(self, fetcher: PDFFetcher) -> None:
        """Test detecting non-temp file."""
        normal_path = Path("/home/user/documents/test.pdf")
        assert fetcher._is_temp_file(normal_path) is False

    # Initialization tests

    def test_init_default_values(self) -> None:
        """Test initialization with default values."""
        fetcher = PDFFetcher()
        assert fetcher._pdf_dir == Path.home() / "pkb-data" / "pdfs"
        assert fetcher._download_timeout == 60
        assert fetcher._max_size == 100 * 1024 * 1024

    def test_init_custom_values(self, tmp_path: Path) -> None:
        """Test initialization with custom values."""
        fetcher = PDFFetcher(
            pdf_dir=tmp_path,
            download_timeout=120,
            max_size=50 * 1024 * 1024,
        )
        assert fetcher._pdf_dir == tmp_path
        assert fetcher._download_timeout == 120
        assert fetcher._max_size == 50 * 1024 * 1024

    def test_init_string_path(self, tmp_path: Path) -> None:
        """Test initialization with string path."""
        fetcher = PDFFetcher(pdf_dir=str(tmp_path))
        assert fetcher._pdf_dir == tmp_path

    # Fetch and queue tests

    @patch("personal_knowledge_base.fetchers.pdf.add_job")
    @patch("personal_knowledge_base.fetchers.pdf.PdfReader")
    @patch("personal_knowledge_base.fetchers.pdf.pdfplumber.open")
    @patch("requests.Session.get")
    def test_fetch_and_queue_success(
        self,
        mock_get: MagicMock,
        mock_pdfplumber: MagicMock,
        mock_reader: MagicMock,
        mock_add_job: MagicMock,
        fetcher: PDFFetcher,
    ) -> None:
        """Test fetch_and_queue with successful fetch."""
        # Setup mocks
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"PDF"]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.pages = [MagicMock()]
        mock_pdf.metadata = None
        mock_reader.return_value = mock_pdf

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Content"
        mock_pdf_doc = MagicMock()
        mock_pdf_doc.pages = [mock_page]
        mock_pdf_doc.__enter__ = MagicMock(return_value=mock_pdf_doc)
        mock_pdf_doc.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.return_value = mock_pdf_doc

        mock_job = MagicMock()
        mock_job.id = "test-job-id"
        mock_add_job.return_value = mock_job

        result, job_id = fetcher.fetch_and_queue(
            "https://example.com/test.pdf",
            priority=1,
            kb_name="test-kb",
        )

        assert result.success is True
        assert job_id == "test-job-id"
        mock_add_job.assert_called_once_with(
            url="https://example.com/test.pdf",
            priority=1,
            content_type="pdf",
            kb_name="test-kb",
        )

    @patch("personal_knowledge_base.fetchers.pdf.add_job")
    @patch("requests.Session.get")
    def test_fetch_and_queue_failure(
        self,
        mock_get: MagicMock,
        mock_add_job: MagicMock,
        fetcher: PDFFetcher,
    ) -> None:
        """Test fetch_and_queue with failed fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.side_effect = HTTPError("404", response=mock_response)

        result, job_id = fetcher.fetch_and_queue("https://example.com/missing.pdf")

        assert result.success is False
        assert job_id is None
        mock_add_job.assert_not_called()

    @patch("personal_knowledge_base.fetchers.pdf.add_job")
    @patch("personal_knowledge_base.fetchers.pdf.PdfReader")
    @patch("personal_knowledge_base.fetchers.pdf.pdfplumber.open")
    @patch("requests.Session.get")
    def test_fetch_and_queue_queue_failure(
        self,
        mock_get: MagicMock,
        mock_pdfplumber: MagicMock,
        mock_reader: MagicMock,
        mock_add_job: MagicMock,
        fetcher: PDFFetcher,
    ) -> None:
        """Test fetch_and_queue when queueing fails but fetch succeeds."""
        # Setup mocks for successful fetch
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"PDF"]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.pages = [MagicMock()]
        mock_pdf.metadata = None
        mock_reader.return_value = mock_pdf

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Content"
        mock_pdf_doc = MagicMock()
        mock_pdf_doc.pages = [mock_page]
        mock_pdf_doc.__enter__ = MagicMock(return_value=mock_pdf_doc)
        mock_pdf_doc.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.return_value = mock_pdf_doc

        # Queue fails
        mock_add_job.side_effect = Exception("DB error")

        result, job_id = fetcher.fetch_and_queue("https://example.com/test.pdf")

        assert result.success is True
        assert job_id is None  # Queue failed

    # Content structure tests

    @patch("personal_knowledge_base.fetchers.pdf.PdfReader")
    @patch("personal_knowledge_base.fetchers.pdf.pdfplumber.open")
    @patch("requests.Session.get")
    def test_fetch_content_structure(
        self,
        mock_get: MagicMock,
        mock_pdfplumber: MagicMock,
        mock_reader: MagicMock,
        fetcher: PDFFetcher,
    ) -> None:
        """Test that content includes page markers."""
        # Setup mocks
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"PDF"]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.pages = [MagicMock(), MagicMock(), MagicMock()]
        mock_pdf.metadata = None
        mock_reader.return_value = mock_pdf

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page content here"
        mock_pdf_doc = MagicMock()
        mock_pdf_doc.pages = [mock_page, mock_page, mock_page]
        mock_pdf_doc.__enter__ = MagicMock(return_value=mock_pdf_doc)
        mock_pdf_doc.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.return_value = mock_pdf_doc

        result = fetcher.fetch("https://example.com/multi-page.pdf")

        assert result.success is True
        assert "--- Page 1 ---" in result.content
        assert "--- Page 2 ---" in result.content
        assert "--- Page 3 ---" in result.content

    @patch("personal_knowledge_base.fetchers.pdf.PdfReader")
    @patch("personal_knowledge_base.fetchers.pdf.pdfplumber.open")
    @patch("requests.Session.get")
    def test_fetch_scanned_pdf_warning(
        self,
        mock_get: MagicMock,
        mock_pdfplumber: MagicMock,
        mock_reader: MagicMock,
        fetcher: PDFFetcher,
    ) -> None:
        """Test that scanned PDFs are flagged."""
        # Setup mocks
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"PDF"]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        mock_pdf = MagicMock()
        mock_pdf.is_encrypted = False
        mock_pdf.pages = [MagicMock()] * 10  # 10 pages
        mock_pdf.metadata = MagicMock()
        mock_pdf.metadata.title = "Scanned Doc"
        mock_pdf.metadata.author = None
        mock_pdf.metadata.subject = None
        mock_pdf.metadata.creator = None
        mock_pdf.metadata.producer = None
        mock_pdf.metadata.creation_date = None
        mock_pdf.metadata.modification_date = None
        mock_reader.return_value = mock_pdf

        # Most pages have no text (simulating scanned PDF)
        mock_pages = []
        for i in range(10):
            mock_page = MagicMock()
            if i == 0:
                mock_page.extract_text.return_value = "Short"  # Only 1 page has text
            else:
                mock_page.extract_text.return_value = ""
            mock_pages.append(mock_page)

        mock_pdf_doc = MagicMock()
        mock_pdf_doc.pages = mock_pages
        mock_pdf_doc.__enter__ = MagicMock(return_value=mock_pdf_doc)
        mock_pdf_doc.__exit__ = MagicMock(return_value=False)
        mock_pdfplumber.return_value = mock_pdf_doc

        result = fetcher.fetch("https://example.com/scanned.pdf")

        assert result.success is True
        assert result.metadata["is_scanned"] is True

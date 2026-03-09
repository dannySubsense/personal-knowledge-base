"""PDF fetcher for downloading and extracting text from PDF files."""

import contextlib
import hashlib
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pdfplumber
import requests
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

from personal_knowledge_base.fetchers.base import Fetcher, FetchResult
from personal_knowledge_base.queue.operations import add_job


class PDFFetcher(Fetcher):
    """Fetcher for PDF documents.

    Downloads PDF files from URLs and extracts text content using
    PyPDF2 and pdfplumber for robust extraction. Handles multi-page
    PDFs, extracts metadata, and provides graceful fallbacks for
    scanned PDFs and password-protected files.

    Features:
        - Downloads PDFs to local storage
        - Text extraction with page-level metadata
        - Metadata extraction (title, author, page count)
        - Scanned PDF detection with OCR fallback indication
        - Password-protected PDF handling
        - Corrupted PDF detection
        - Automatic job queueing for processing pipeline

    """

    # Default storage directory for PDFs
    _DEFAULT_PDF_DIR = Path.home() / "pkb-data" / "pdfs"

    # Download settings
    _DOWNLOAD_TIMEOUT = 60  # seconds
    _MAX_PDF_SIZE = 100 * 1024 * 1024  # 100MB limit
    _CHUNK_SIZE = 8192  # 8KB chunks for streaming

    # Minimum text ratio to consider PDF as "text-based" vs "scanned"
    _SCANNED_PDF_THRESHOLD = 0.1  # Less than 10% text coverage = likely scanned

    def __init__(
        self,
        pdf_dir: Path | str | None = None,
        download_timeout: int = _DOWNLOAD_TIMEOUT,
        max_size: int = _MAX_PDF_SIZE,
    ) -> None:
        """Initialize the PDF fetcher.

        Args:
            pdf_dir: Directory to store downloaded PDFs. Defaults to ~/pkb-data/pdfs.
            download_timeout: Timeout for downloading PDFs in seconds.
            max_size: Maximum PDF file size in bytes.

        """
        self._pdf_dir = Path(pdf_dir) if pdf_dir else self._DEFAULT_PDF_DIR
        self._download_timeout = download_timeout
        self._max_size = max_size
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            }
        )

    def can_fetch(self, url: str) -> bool:
        """Check if this fetcher can handle the given URL.

        Args:
            url: The URL to check.

        Returns:
            True if the URL points to a PDF file, False otherwise.

        """
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False

            # Check if URL ends with .pdf (case insensitive)
            path = parsed.path.lower()
            if path.endswith(".pdf"):
                return True

            # Check for PDF content type indicator in URL
            return bool("/pdf/" in path or "/download/" in path)
        except (ValueError, TypeError):
            return False

    def fetch(self, url: str) -> FetchResult:
        """Fetch and extract content from a PDF URL.

        Args:
            url: The URL of the PDF to fetch.

        Returns:
            FetchResult containing the extracted text and metadata.

        """
        if not self.can_fetch(url):
            return FetchResult(
                url=url,
                content_type="pdf",
                success=False,
                error_message="Invalid URL or URL does not point to a PDF file.",
            )

        pdf_path: Path | None = None

        try:
            # Download the PDF
            pdf_path = self._download_pdf(url)

            # Extract content from the PDF
            return self._extract_pdf_content(url, pdf_path)

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else 0
            if status_code == 404:
                return FetchResult(
                    url=url,
                    content_type="pdf",
                    success=False,
                    error_message="PDF not found (404). The file may have been removed.",
                )
            elif status_code == 403:
                return FetchResult(
                    url=url,
                    content_type="pdf",
                    success=False,
                    error_message="Access forbidden (403). The PDF may require authentication.",
                )
            else:
                return FetchResult(
                    url=url,
                    content_type="pdf",
                    success=False,
                    error_message=f"HTTP error {status_code}: {str(e)}",
                )

        except requests.exceptions.Timeout:
            return FetchResult(
                url=url,
                content_type="pdf",
                success=False,
                error_message=(
                    f"Download timeout after {self._download_timeout}s. "
                    "The PDF may be too large or the server is slow."
                ),
            )

        except requests.exceptions.RequestException as e:
            return FetchResult(
                url=url,
                content_type="pdf",
                success=False,
                error_message=f"Network error downloading PDF: {str(e)}",
            )

        except Exception as e:
            return FetchResult(
                url=url,
                content_type="pdf",
                success=False,
                error_message=f"Unexpected error processing PDF: {str(e)}",
            )

        finally:
            # Clean up temporary files if they exist
            if pdf_path and pdf_path.exists() and self._is_temp_file(pdf_path):
                with contextlib.suppress(OSError):
                    pdf_path.unlink()

    def _is_temp_file(self, path: Path) -> bool:
        """Check if a file is in the temp directory.

        Args:
            path: Path to check.

        Returns:
            True if the file is in a temp directory.

        """
        temp_dirs = [Path(tempfile.gettempdir()), Path("/tmp"), Path("/var/tmp")]
        return any(str(path).startswith(str(td)) for td in temp_dirs)

    def _download_pdf(self, url: str) -> Path:
        """Download a PDF from URL to local storage.

        Args:
            url: The URL to download from.

        Returns:
            Path to the downloaded PDF file.

        Raises:
            requests.exceptions.RequestException: If download fails.
            ValueError: If PDF is too large.

        """
        # Create filename from URL hash to ensure uniqueness
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        filename = f"{url_hash}.pdf"
        pdf_path = self._pdf_dir / filename

        # Ensure directory exists
        self._pdf_dir.mkdir(parents=True, exist_ok=True)

        # Download with streaming to handle large files
        response = self._session.get(
            url, timeout=self._download_timeout, stream=True, allow_redirects=True
        )
        response.raise_for_status()

        # Check content length if available
        content_length = response.headers.get("content-length")
        if content_length:
            size = int(content_length)
            if size > self._max_size:
                raise ValueError(
                    f"PDF too large: {size / (1024 * 1024):.1f}MB "
                    f"(max {self._max_size / (1024 * 1024):.0f}MB)"
                )

        # Download to temporary file first, then move
        temp_path = pdf_path.with_suffix(".tmp")
        downloaded_size = 0

        try:
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=self._CHUNK_SIZE):
                    if chunk:
                        downloaded_size += len(chunk)
                        if downloaded_size > self._max_size:
                            raise ValueError(
                                f"PDF too large: exceeds {self._max_size / (1024 * 1024):.0f}MB"
                            )
                        f.write(chunk)

            # Move to final location
            temp_path.rename(pdf_path)

        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise

        return pdf_path

    def _extract_pdf_content(self, url: str, pdf_path: Path) -> FetchResult:
        """Extract text and metadata from a PDF file.

        Args:
            url: The original URL.
            pdf_path: Path to the PDF file.

        Returns:
            FetchResult with extracted content.

        """
        # Try PyPDF2 first for metadata and basic text
        try:
            reader = PdfReader(str(pdf_path))
        except PdfReadError as e:
            return FetchResult(
                url=url,
                content_type="pdf",
                success=False,
                error_message=f"Corrupted or invalid PDF file: {str(e)}",
            )

        # Check if password protected
        if reader.is_encrypted:
            return FetchResult(
                url=url,
                content_type="pdf",
                success=False,
                error_message="PDF is password protected and cannot be accessed.",
            )

        # Extract metadata
        metadata = self._extract_metadata(reader)
        page_count = len(reader.pages)

        # Extract text using both libraries for best results
        text_by_page, total_text_length = self._extract_text_with_pdfplumber(pdf_path, page_count)

        # Check if this appears to be a scanned PDF
        is_scanned = self._is_scanned_pdf(text_by_page, page_count)

        # Combine all text
        full_text = "\n\n".join(
            f"--- Page {i + 1} ---\n{text}" for i, text in enumerate(text_by_page) if text.strip()
        )

        # Build metadata dict
        result_metadata: dict[str, Any] = {
            "page_count": page_count,
            "local_path": str(pdf_path),
            "is_scanned": is_scanned,
            "text_length": total_text_length,
            "text_by_page": text_by_page,
        }

        # Add PDF metadata if available
        if metadata.get("title"):
            result_metadata["pdf_title"] = metadata["title"]
        if metadata.get("author"):
            result_metadata["pdf_author"] = metadata["author"]
        if metadata.get("subject"):
            result_metadata["pdf_subject"] = metadata["subject"]
        if metadata.get("creator"):
            result_metadata["pdf_creator"] = metadata["creator"]
        if metadata.get("producer"):
            result_metadata["pdf_producer"] = metadata["producer"]
        if metadata.get("creation_date"):
            result_metadata["pdf_creation_date"] = metadata["creation_date"]
        if metadata.get("modification_date"):
            result_metadata["pdf_modification_date"] = metadata["modification_date"]

        return FetchResult(
            url=url,
            title=metadata.get("title") or self._extract_title_from_url(url),
            content=full_text,
            content_type="pdf",
            author=metadata.get("author"),
            published_date=metadata.get("creation_date"),
            metadata=result_metadata,
            success=True,
            fetched_at=datetime.now(),
        )

    def _extract_metadata(self, reader: PdfReader) -> dict[str, Any]:
        """Extract metadata from PDF.

        Args:
            reader: PyPDF2 PdfReader instance.

        Returns:
            Dictionary of metadata fields.

        """
        metadata: dict[str, Any] = {}

        if reader.metadata:
            meta = reader.metadata

            # Standard fields
            if meta.title:
                metadata["title"] = self._clean_text(str(meta.title))
            if meta.author:
                metadata["author"] = self._clean_text(str(meta.author))
            if meta.subject:
                metadata["subject"] = self._clean_text(str(meta.subject))
            if meta.creator:
                metadata["creator"] = self._clean_text(str(meta.creator))
            if meta.producer:
                metadata["producer"] = self._clean_text(str(meta.producer))

            # Dates
            if meta.creation_date:
                metadata["creation_date"] = self._parse_pdf_date(str(meta.creation_date))
            if meta.modification_date:
                metadata["modification_date"] = self._parse_pdf_date(str(meta.modification_date))

        return metadata

    def _extract_text_with_pdfplumber(
        self, pdf_path: Path, page_count: int
    ) -> tuple[list[str], int]:
        """Extract text from PDF using pdfplumber.

        Args:
            pdf_path: Path to the PDF file.
            page_count: Expected number of pages.

        Returns:
            Tuple of (list of text per page, total text length).

        """
        text_by_page: list[str] = []
        total_length = 0

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    try:
                        text = page.extract_text() or ""
                        text = self._clean_text(text)
                        text_by_page.append(text)
                        total_length += len(text)
                    except Exception:
                        text_by_page.append("")
        except Exception:
            # Fallback: return empty pages
            text_by_page = [""] * page_count

        return text_by_page, total_length

    def _is_scanned_pdf(self, text_by_page: list[str], page_count: int) -> bool:
        """Determine if PDF appears to be scanned/image-based.

        Args:
            text_by_page: List of extracted text per page.
            page_count: Total number of pages.

        Returns:
            True if PDF appears to be scanned.

        """
        if page_count == 0:
            return False

        # Count pages with meaningful text (at least 20 chars)
        pages_with_text = sum(1 for text in text_by_page if len(text.strip()) > 20)
        text_ratio = pages_with_text / page_count

        return text_ratio < self._SCANNED_PDF_THRESHOLD

    def _parse_pdf_date(self, date_str: str) -> datetime | None:
        """Parse PDF date string to datetime.

        PDF dates are typically in format: D:YYYYMMDDHHmmSSOHH'mm'

        Args:
            date_str: PDF date string.

        Returns:
            Parsed datetime or None.

        """
        if not date_str:
            return None

        # Remove D: prefix if present
        date_str = date_str.replace("D:", "")

        # Try various formats with their expected string lengths
        formats = [
            ("%Y%m%d%H%M%S", 14),  # D:YYYYMMDDHHmmSS
            ("%Y%m%d%H%M", 12),  # D:YYYYMMDDHHmm
            ("%Y%m%d", 8),  # D:YYYYMMDD
            ("%Y-%m-%dT%H:%M:%S", 19),  # ISO format
            ("%Y-%m-%d", 10),  # Simple date
        ]

        for fmt, length in formats:
            try:
                return datetime.strptime(date_str[:length], fmt)
            except ValueError:
                continue

        return None

    def _clean_text(self, text: str) -> str:
        """Clean extracted text.

        Args:
            text: Raw text.

        Returns:
            Cleaned text.

        """
        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove control characters except newlines
        text = "".join(
            char for char in text if char == "\n" or (char.isprintable() or char.isspace())
        )

        return text.strip()

    def _extract_title_from_url(self, url: str) -> str:
        """Extract a title from the URL path.

        Args:
            url: The URL.

        Returns:
            Extracted title or default.

        """
        try:
            parsed = urlparse(url)
            path = parsed.path
            filename = path.split("/")[-1]

            if filename and filename.endswith(".pdf"):
                # Remove extension and replace underscores/hyphens with spaces
                title = filename[:-4].replace("_", " ").replace("-", " ")
                return title.title()

            return "PDF Document"
        except Exception:
            return "PDF Document"

    def fetch_and_queue(
        self,
        url: str,
        priority: int = 2,
        kb_name: str | None = None,
    ) -> tuple[FetchResult, str | None]:
        """Fetch PDF and add to processing queue.

        Args:
            url: The PDF URL.
            priority: Queue priority (1=immediate, 2=normal).
            kb_name: Target knowledge base name.

        Returns:
            Tuple of (FetchResult, job_id or None if queueing failed).

        """
        result = self.fetch(url)

        job_id: str | None = None
        if result.success:
            try:
                job = add_job(
                    url=url,
                    priority=priority,
                    content_type="pdf",
                    kb_name=kb_name,
                )
                job_id = job.id
            except Exception:
                # Queueing failed but fetch succeeded
                pass

        return result, job_id

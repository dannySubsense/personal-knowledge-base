"""Image fetcher for downloading and storing images with metadata."""

import contextlib
import hashlib
import imghdr
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from personal_knowledge_base.fetchers.base import Fetcher, FetchResult
from personal_knowledge_base.queue.operations import add_job


@dataclass
class ImageMetadata:
    """Metadata for a downloaded image.

    Attributes:
        source_url: The original URL where the image was fetched from.
        source_page: Optional URL of the page containing the image.
        content_type: MIME type of the image (e.g., image/png, image/jpeg).
        size_bytes: Size of the image file in bytes.
        width: Image width in pixels (if available).
        height: Image height in pixels (if available).
        filename: The generated unique filename for the stored image.
        local_path: Full path to the stored image file.
        downloaded_at: Timestamp when the image was downloaded.

    """

    source_url: str
    content_type: str
    size_bytes: int
    filename: str
    local_path: str
    downloaded_at: datetime = field(default_factory=datetime.now)
    source_page: str | None = None
    width: int | None = None
    height: int | None = None


class ImageFetcher(Fetcher):
    """Fetcher for downloading and storing images.

    Downloads images from URLs, extracts metadata, and stores them
    in a local directory with unique filenames based on content hash.

    Features:
        - Downloads images from HTTP/HTTPS URLs
        - Generates unique filenames using content hash
        - Extracts image metadata (size, dimensions, content-type)
        - Handles common errors (404, timeout, invalid images)
        - Adds jobs to the ingestion queue
        - Returns structured content with metadata

    """

    # Default storage directory
    _DEFAULT_STORAGE_DIR = Path.home() / "pkb-data" / "images"

    # Request timeout in seconds
    _DEFAULT_TIMEOUT = 30

    # Maximum image size (100 MB)
    _MAX_IMAGE_SIZE = 100 * 1024 * 1024

    # Supported image MIME types
    _SUPPORTED_TYPES = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
    }

    def __init__(
        self,
        storage_dir: Path | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        add_to_queue: bool = True,
    ) -> None:
        """Initialize the image fetcher.

        Args:
            storage_dir: Directory to store downloaded images.
                        Defaults to ~/pkb-data/images.
            timeout: Request timeout in seconds.
            add_to_queue: Whether to add successful downloads to the queue.

        """
        self._storage_dir = storage_dir or self._DEFAULT_STORAGE_DIR
        self._timeout = timeout
        self._add_to_queue = add_to_queue

        # Ensure storage directory exists
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def can_fetch(self, url: str) -> bool:
        """Check if this fetcher can handle the given URL.

        This fetcher handles HTTP/HTTPS URLs that point to image files.

        Args:
            url: The URL to check.

        Returns:
            True if this fetcher can handle the URL, False otherwise.

        """
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False

            # Check if URL has image extension or is likely an image
            path_lower = parsed.path.lower()
            image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".tiff")
            return any(path_lower.endswith(ext) for ext in image_extensions)
        except (ValueError, TypeError):
            return False

    def fetch(
        self,
        url: str,
        source_page: str | None = None,
        kb_name: str | None = None,
    ) -> FetchResult:
        """Fetch and store an image from the given URL.

        Args:
            url: The URL of the image to fetch.
            source_page: Optional URL of the page containing the image.
            kb_name: Optional target knowledge base name.

        Returns:
            FetchResult containing image metadata or error information.

        """
        # Validate URL
        if not self.can_fetch(url):
            return FetchResult(
                url=url,
                content_type="image",
                success=False,
                error_message="Invalid URL or URL does not point to an image.",
            )

        try:
            return self._download_image(url, source_page, kb_name)
        except Exception as e:
            return FetchResult(
                url=url,
                content_type="image",
                success=False,
                error_message=f"Unexpected error: {str(e)}",
            )

    def _download_image(
        self,
        url: str,
        source_page: str | None = None,
        kb_name: str | None = None,
    ) -> FetchResult:
        """Download the image and extract metadata.

        Args:
            url: The URL of the image.
            source_page: Optional source page URL.
            kb_name: Optional target knowledge base.

        Returns:
            FetchResult with image metadata.

        """
        # Download the image
        try:
            response = requests.get(url, timeout=self._timeout, stream=True)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return FetchResult(
                url=url,
                content_type="image",
                success=False,
                error_message="Request timed out. The server may be slow or unresponsive.",
            )
        except requests.exceptions.ConnectionError:
            return FetchResult(
                url=url,
                content_type="image",
                success=False,
                error_message="Connection error. Could not connect to the server.",
            )
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            if status_code == 404:
                return FetchResult(
                    url=url,
                    content_type="image",
                    success=False,
                    error_message="Image not found (404).",
                )
            elif status_code == 403:
                return FetchResult(
                    url=url,
                    content_type="image",
                    success=False,
                    error_message="Access forbidden (403).",
                )
            else:
                return FetchResult(
                    url=url,
                    content_type="image",
                    success=False,
                    error_message=f"HTTP error {status_code}.",
                )
        except requests.exceptions.RequestException as e:
            return FetchResult(
                url=url,
                content_type="image",
                success=False,
                error_message=f"Request failed: {str(e)}",
            )

        # Check content length if available
        content_length = response.headers.get("Content-Length")
        if content_length:
            size = int(content_length)
            if size > self._MAX_IMAGE_SIZE:
                return FetchResult(
                    url=url,
                    content_type="image",
                    success=False,
                    error_message=f"Image too large ({size / 1024 / 1024:.1f} MB). "
                    f"Maximum size is {self._MAX_IMAGE_SIZE / 1024 / 1024:.0f} MB.",
                )

        # Download the content
        try:
            image_data = response.content
        except Exception as e:
            return FetchResult(
                url=url,
                content_type="image",
                success=False,
                error_message=f"Failed to download image data: {str(e)}",
            )

        # Validate image data
        if len(image_data) == 0:
            return FetchResult(
                url=url,
                content_type="image",
                success=False,
                error_message="Downloaded image is empty.",
            )

        # Detect image type
        content_type = response.headers.get("Content-Type", "").lower()
        file_extension = self._get_file_extension(image_data, content_type, url)

        if not file_extension:
            return FetchResult(
                url=url,
                content_type="image",
                success=False,
                error_message="Invalid or unsupported image format.",
            )

        # Generate unique filename based on content hash
        content_hash = hashlib.sha256(image_data).hexdigest()[:16]
        filename = f"{content_hash}{file_extension}"

        # Organize by year/month for better file management
        now = datetime.now()
        subdir = self._storage_dir / f"{now.year}" / f"{now.month:02d}"
        subdir.mkdir(parents=True, exist_ok=True)

        local_path = subdir / filename

        # Save the image
        try:
            with open(local_path, "wb") as f:
                f.write(image_data)
        except OSError as e:
            return FetchResult(
                url=url,
                content_type="image",
                success=False,
                error_message=f"Failed to save image: {str(e)}",
            )

        # Get image dimensions if possible
        width, height = self._get_image_dimensions(image_data, file_extension)

        # Create metadata
        image_metadata = ImageMetadata(
            source_url=url,
            source_page=source_page,
            content_type=content_type or f"image/{file_extension.lstrip('.')}",
            size_bytes=len(image_data),
            width=width,
            height=height,
            filename=filename,
            local_path=str(local_path),
        )

        # Add to queue if enabled
        if self._add_to_queue:
            with contextlib.suppress(Exception):
                add_job(
                    url=str(local_path),
                    priority=2,
                    content_type="image",
                    kb_name=kb_name,
                )

        # Build metadata dict for FetchResult
        metadata: dict[str, Any] = {
            "image_metadata": {
                "source_url": image_metadata.source_url,
                "source_page": image_metadata.source_page,
                "content_type": image_metadata.content_type,
                "size_bytes": image_metadata.size_bytes,
                "width": image_metadata.width,
                "height": image_metadata.height,
                "filename": image_metadata.filename,
                "local_path": image_metadata.local_path,
                "downloaded_at": image_metadata.downloaded_at.isoformat(),
            }
        }

        # Create a text representation for the content field
        content_text = self._create_content_text(image_metadata)

        return FetchResult(
            url=url,
            title=f"Image: {filename}",
            content=content_text,
            content_type="image",
            metadata=metadata,
            success=True,
            fetched_at=datetime.now(),
        )

    def _get_file_extension(self, image_data: bytes, content_type: str, url: str) -> str | None:
        """Determine the file extension based on image data and headers.

        Args:
            image_data: Raw image bytes.
            content_type: Content-Type header value.
            url: Original URL (for fallback).

        Returns:
            File extension including the dot, or None if unknown.

        """
        # First try content type from headers
        if content_type:
            # Remove any charset or other parameters
            content_type = content_type.split(";")[0].strip()
            if content_type in self._SUPPORTED_TYPES:
                return self._SUPPORTED_TYPES[content_type]

        # Try to detect from image data using imghdr
        image_type = imghdr.what(None, h=image_data)
        if image_type:
            extension_map = {
                "jpeg": ".jpg",
                "png": ".png",
                "gif": ".gif",
                "webp": ".webp",
                "svg": ".svg",
                "bmp": ".bmp",
                "tiff": ".tiff",
            }
            if image_type in extension_map:
                return extension_map[image_type]

        # Fallback to URL extension
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".tiff"):
            if path_lower.endswith(ext):
                if ext == ".jpeg":
                    return ".jpg"
                return ext

        return None

    def _get_image_dimensions(
        self, image_data: bytes, file_extension: str
    ) -> tuple[int | None, int | None]:
        """Try to extract image dimensions from the image data.

        Args:
            image_data: Raw image bytes.
            file_extension: File extension to determine image type.

        Returns:
            Tuple of (width, height) or (None, None) if dimensions cannot be determined.

        """
        try:
            ext = file_extension.lower()

            if ext in (".jpg", ".jpeg"):
                return self._get_jpeg_dimensions(image_data)
            elif ext == ".png":
                return self._get_png_dimensions(image_data)
            elif ext == ".gif":
                return self._get_gif_dimensions(image_data)
            elif ext == ".webp":
                return self._get_webp_dimensions(image_data)
            elif ext == ".bmp":
                return self._get_bmp_dimensions(image_data)
        except Exception:
            pass

        return None, None

    def _get_jpeg_dimensions(self, data: bytes) -> tuple[int | None, int | None]:
        """Extract dimensions from JPEG data."""
        # JPEG markers
        sof_markers = (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF)

        i = 0
        while i < len(data) - 1:
            if data[i] == 0xFF:
                marker = data[i + 1]

                # Skip padding bytes
                if marker == 0xFF:
                    i += 1
                    continue

                # Skip markers without length
                if marker in (
                    0x00,
                    0x01,
                    0xD0,
                    0xD1,
                    0xD2,
                    0xD3,
                    0xD4,
                    0xD5,
                    0xD6,
                    0xD7,
                    0xD8,
                    0xD9,
                ):
                    i += 2
                    continue

                # SOF markers contain dimensions
                if marker in sof_markers:
                    if i + 9 < len(data):
                        height = (data[i + 5] << 8) | data[i + 6]
                        width = (data[i + 7] << 8) | data[i + 8]
                        return width, height
                    return None, None

                # Get segment length and skip
                if i + 3 < len(data):
                    length = (data[i + 2] << 8) | data[i + 3]
                    i += 2 + length
                else:
                    break
            else:
                i += 1

        return None, None

    def _get_png_dimensions(self, data: bytes) -> tuple[int | None, int | None]:
        """Extract dimensions from PNG data."""
        # PNG IHDR chunk is at a fixed position
        if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
            width = (data[16] << 24) | (data[17] << 16) | (data[18] << 8) | data[19]
            height = (data[20] << 24) | (data[21] << 16) | (data[22] << 8) | data[23]
            return width, height
        return None, None

    def _get_gif_dimensions(self, data: bytes) -> tuple[int | None, int | None]:
        """Extract dimensions from GIF data."""
        if len(data) >= 10 and data[:6] in (b"GIF87a", b"GIF89a"):
            width = data[6] | (data[7] << 8)
            height = data[8] | (data[9] << 8)
            return width, height
        return None, None

    def _get_webp_dimensions(self, data: bytes) -> tuple[int | None, int | None]:
        """Extract dimensions from WebP data."""
        if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
            return None, None

        chunk_type = data[12:16]

        if chunk_type == b"VP8 ":
            # Simple lossy format
            if len(data) >= 30 and data[23] == 0x9D and data[24] == 0x01 and data[25] == 0x2A:
                width = (data[26] | (data[27] << 8)) & 0x3FFF
                height = (data[28] | (data[29] << 8)) & 0x3FFF
                return width, height
        elif chunk_type == b"VP8L" and len(data) >= 25:
            # Lossless format
            bits = data[21] | (data[22] << 8) | (data[23] << 16) | (data[24] << 24)
            width = (bits & 0x3FFF) + 1
            height = ((bits >> 14) & 0x3FFF) + 1
            return width, height
        elif chunk_type == b"VP8X" and len(data) >= 30:
            # Extended format
            width = 1 + (data[24] | (data[25] << 8) | (data[26] << 16))
            height = 1 + (data[27] | (data[28] << 8) | (data[29] << 16))
            return width, height

        return None, None

    def _get_bmp_dimensions(self, data: bytes) -> tuple[int | None, int | None]:
        """Extract dimensions from BMP data."""
        if len(data) >= 26 and data[:2] == b"BM":
            width = data[18] | (data[19] << 8) | (data[20] << 16) | (data[21] << 24)
            # Height is a signed 32-bit integer (can be negative for top-down BMPs)
            height_raw = data[22] | (data[23] << 8) | (data[24] << 16) | (data[25] << 24)
            # Convert to signed 32-bit
            if height_raw >= 2**31:
                height_raw -= 2**32
            # Height can be negative for top-down BMPs
            if height_raw < 0:
                height_raw = -height_raw
            return width, height_raw
        return None, None

    def _create_content_text(self, metadata: ImageMetadata) -> str:
        """Create a text representation of the image for indexing.

        Args:
            metadata: Image metadata.

        Returns:
            Text representation of the image.

        """
        lines = [
            f"Image: {metadata.filename}",
            f"Source URL: {metadata.source_url}",
        ]

        if metadata.source_page:
            lines.append(f"Source Page: {metadata.source_page}")

        lines.extend(
            [
                f"Content Type: {metadata.content_type}",
                f"Size: {metadata.size_bytes / 1024:.1f} KB",
            ]
        )

        if metadata.width and metadata.height:
            lines.append(f"Dimensions: {metadata.width}x{metadata.height}")

        lines.append(f"Local Path: {metadata.local_path}")

        return "\n".join(lines)

    @property
    def storage_dir(self) -> Path:
        """Get the storage directory path.

        Returns:
            Path to the image storage directory.

        """
        return self._storage_dir

"""Tests for the image fetcher."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from personal_knowledge_base.fetchers.image import ImageFetcher, ImageMetadata


class TestImageFetcher:
    """Tests for ImageFetcher class."""

    @pytest.fixture
    def fetcher(self, tmp_path: Path) -> ImageFetcher:
        """Create an ImageFetcher instance for testing."""
        return ImageFetcher(storage_dir=tmp_path, add_to_queue=False)

    @pytest.fixture
    def sample_png_data(self) -> bytes:
        """Create a minimal valid PNG file for testing."""
        import zlib

        signature = b"\x89PNG\r\n\x1a\n"
        ihdr_length = b"\x00\x00\x00\x0d"
        ihdr_type = b"IHDR"
        ihdr_data = bytes(
            [
                0x00,
                0x00,
                0x00,
                0x02,
                0x00,
                0x00,
                0x00,
                0x03,
                0x08,
                0x02,
                0x00,
                0x00,
                0x00,
            ]
        )
        ihdr_crc = zlib.crc32(ihdr_type + ihdr_data).to_bytes(4, "big")
        idat_type = b"IDAT"
        idat_data = zlib.compress(b"\x00\x00\x00\x00\x00\x00\x00\x00")
        idat_length = len(idat_data).to_bytes(4, "big")
        idat_crc = zlib.crc32(idat_type + idat_data).to_bytes(4, "big")
        iend_length = b"\x00\x00\x00\x00"
        iend_type = b"IEND"
        iend_crc = b"\xae\x42\x60\x82"
        return (
            signature
            + ihdr_length
            + ihdr_type
            + ihdr_data
            + ihdr_crc
            + idat_length
            + idat_type
            + idat_data
            + idat_crc
            + iend_length
            + iend_type
            + iend_crc
        )

    @pytest.fixture
    def sample_gif_data(self) -> bytes:
        """Create a minimal valid GIF file for testing."""
        # GIF89a 4x5 image - properly structured
        return bytes(
            [
                0x47,
                0x49,
                0x46,
                0x38,
                0x39,
                0x61,  # "GIF89a"
                0x04,
                0x00,  # width: 4 (little endian)
                0x05,
                0x00,  # height: 5 (little endian)
                0x80,
                0x00,
                0x00,  # global color table flag, color resolution, sorted, size
                0x00,
                0x00,
                0x00,  # background color, pixel aspect ratio
                0x00,
                0x00,
                0x00,
                0xFF,
                0xFF,
                0xFF,  # color table (black and white)
                0x2C,  # image separator
                0x00,
                0x00,
                0x00,
                0x00,  # position
                0x04,
                0x00,
                0x05,
                0x00,  # size
                0x00,  # LCT flag
                0x02,
                0x02,
                0x44,
                0x01,
                0x00,  # image data
                0x3B,  # trailer
            ]
        )

    def test_can_fetch_jpeg_url(self, fetcher: ImageFetcher) -> None:
        """Test can_fetch with JPEG URL."""
        assert fetcher.can_fetch("https://example.com/image.jpg") is True

    def test_can_fetch_png_url(self, fetcher: ImageFetcher) -> None:
        """Test can_fetch with PNG URL."""
        assert fetcher.can_fetch("https://example.com/image.png") is True

    def test_can_fetch_gif_url(self, fetcher: ImageFetcher) -> None:
        """Test can_fetch with GIF URL."""
        assert fetcher.can_fetch("https://example.com/image.gif") is True

    def test_can_fetch_webp_url(self, fetcher: ImageFetcher) -> None:
        """Test can_fetch with WebP URL."""
        assert fetcher.can_fetch("https://example.com/image.webp") is True

    def test_can_fetch_uppercase_extension(self, fetcher: ImageFetcher) -> None:
        """Test can_fetch with uppercase extension."""
        assert fetcher.can_fetch("https://example.com/image.JPG") is True

    def test_can_fetch_url_with_query_params(self, fetcher: ImageFetcher) -> None:
        """Test can_fetch with URL containing query parameters."""
        assert fetcher.can_fetch("https://example.com/image.jpg?size=large") is True

    def test_can_fetch_http_url(self, fetcher: ImageFetcher) -> None:
        """Test can_fetch with HTTP URL."""
        assert fetcher.can_fetch("http://example.com/image.jpg") is True

    def test_can_fetch_non_image_url(self, fetcher: ImageFetcher) -> None:
        """Test can_fetch returns False for non-image URL."""
        assert fetcher.can_fetch("https://example.com/page.html") is False

    def test_can_fetch_ftp_url(self, fetcher: ImageFetcher) -> None:
        """Test can_fetch returns False for FTP URL."""
        assert fetcher.can_fetch("ftp://example.com/image.jpg") is False

    def test_can_fetch_invalid_url(self, fetcher: ImageFetcher) -> None:
        """Test can_fetch with invalid URL."""
        assert fetcher.can_fetch("not-a-url") is False

    def test_can_fetch_empty_string(self, fetcher: ImageFetcher) -> None:
        """Test can_fetch with empty string."""
        assert fetcher.can_fetch("") is False

    def test_can_fetch_none(self, fetcher: ImageFetcher) -> None:
        """Test can_fetch with None."""
        result = fetcher.can_fetch(None)  # type: ignore[arg-type]
        assert result is False

    def test_fetcher_initialization_defaults(self) -> None:
        """Test fetcher initialization with default values."""
        fetcher = ImageFetcher(add_to_queue=False)
        assert fetcher._storage_dir == Path.home() / "pkb-data" / "images"
        assert fetcher._timeout == 30
        assert fetcher._add_to_queue is False

    def test_fetcher_initialization_custom(self, tmp_path: Path) -> None:
        """Test fetcher initialization with custom values."""
        fetcher = ImageFetcher(
            storage_dir=tmp_path / "custom",
            timeout=60,
            add_to_queue=True,
        )
        assert fetcher._storage_dir == tmp_path / "custom"
        assert fetcher._timeout == 60
        assert fetcher._add_to_queue is True

    def test_fetcher_creates_storage_directory(self, tmp_path: Path) -> None:
        """Test that fetcher creates storage directory on init."""
        storage_dir = tmp_path / "new_images"
        assert not storage_dir.exists()
        _ = ImageFetcher(storage_dir=storage_dir, add_to_queue=False)
        assert storage_dir.exists()

    def test_storage_dir_property(self, fetcher: ImageFetcher) -> None:
        """Test the storage_dir property."""
        assert isinstance(fetcher.storage_dir, Path)

    def test_get_file_extension_from_content_type_jpeg(self, fetcher: ImageFetcher) -> None:
        """Test getting extension from JPEG content type."""
        ext = fetcher._get_file_extension(b"data", "image/jpeg", "http://example.com/img")
        assert ext == ".jpg"

    def test_get_file_extension_from_content_type_png(self, fetcher: ImageFetcher) -> None:
        """Test getting extension from PNG content type."""
        ext = fetcher._get_file_extension(b"data", "image/png", "http://example.com/img")
        assert ext == ".png"

    def test_get_file_extension_from_content_type_with_charset(self, fetcher: ImageFetcher) -> None:
        """Test getting extension from content type with charset."""
        ext = fetcher._get_file_extension(
            b"data", "image/jpeg; charset=utf-8", "http://example.com/img"
        )
        assert ext == ".jpg"

    def test_get_file_extension_from_image_data_png(
        self, fetcher: ImageFetcher, sample_png_data: bytes
    ) -> None:
        """Test getting extension from PNG image data."""
        ext = fetcher._get_file_extension(sample_png_data, "", "http://example.com/img")
        assert ext == ".png"

    def test_get_file_extension_from_image_data_gif(
        self, fetcher: ImageFetcher, sample_gif_data: bytes
    ) -> None:
        """Test getting extension from GIF image data."""
        ext = fetcher._get_file_extension(sample_gif_data, "", "http://example.com/img")
        assert ext == ".gif"

    def test_get_file_extension_from_url(self, fetcher: ImageFetcher) -> None:
        """Test getting extension from URL."""
        ext = fetcher._get_file_extension(
            b"data", "application/octet-stream", "http://example.com/image.jpg"
        )
        assert ext == ".jpg"

    def test_get_file_extension_from_url_jpeg(self, fetcher: ImageFetcher) -> None:
        """Test getting extension from URL with .jpeg extension."""
        ext = fetcher._get_file_extension(b"data", "", "http://example.com/image.jpeg")
        assert ext == ".jpg"

    def test_get_file_extension_unknown(self, fetcher: ImageFetcher) -> None:
        """Test getting extension for unknown type."""
        ext = fetcher._get_file_extension(
            b"unknown", "application/unknown", "http://example.com/file"
        )
        assert ext is None

    def test_get_png_dimensions(self, fetcher: ImageFetcher, sample_png_data: bytes) -> None:
        """Test extracting dimensions from PNG data."""
        width, height = fetcher._get_png_dimensions(sample_png_data)
        assert width == 2
        assert height == 3

    def test_get_png_dimensions_invalid(self, fetcher: ImageFetcher) -> None:
        """Test extracting dimensions from invalid PNG data."""
        width, height = fetcher._get_png_dimensions(b"not a png")
        assert width is None
        assert height is None

    def test_get_gif_dimensions(self, fetcher: ImageFetcher) -> None:
        """Test extracting dimensions from GIF data."""
        # Valid GIF89a header with 4x5 dimensions
        # GIF header: "GIF" + version (6 bytes), width (2 bytes LE), height (2 bytes LE)
        gif_data = b"GIF89a" + b"\x04\x00" + b"\x05\x00"
        width, height = fetcher._get_gif_dimensions(gif_data)
        assert width == 4
        assert height == 5

    def test_get_gif_dimensions_invalid(self, fetcher: ImageFetcher) -> None:
        """Test extracting dimensions from invalid GIF data."""
        width, height = fetcher._get_gif_dimensions(b"not a gif")
        assert width is None
        assert height is None

    def test_get_image_dimensions_png(self, fetcher: ImageFetcher, sample_png_data: bytes) -> None:
        """Test getting dimensions for PNG."""
        width, height = fetcher._get_image_dimensions(sample_png_data, ".png")
        assert width == 2
        assert height == 3

    def test_get_image_dimensions_gif(self, fetcher: ImageFetcher) -> None:
        """Test getting dimensions for GIF."""
        # Valid GIF89a header with 4x5 dimensions
        gif_data = b"GIF89a" + b"\x04\x00" + b"\x05\x00"
        width, height = fetcher._get_image_dimensions(gif_data, ".gif")
        assert width == 4
        assert height == 5

    def test_get_image_dimensions_unsupported(self, fetcher: ImageFetcher) -> None:
        """Test getting dimensions for unsupported format."""
        width, height = fetcher._get_image_dimensions(b"data", ".unknown")
        assert width is None
        assert height is None

    def test_get_jpeg_dimensions(self, fetcher: ImageFetcher) -> None:
        """Test extracting dimensions from JPEG data."""
        # Minimal JPEG with SOF0 marker containing 100x200 dimensions
        # SOI + APP0 + DQT + SOF0 + DHT + SOS + EOI
        jpeg_data = bytes(
            [
                0xFF,
                0xD8,  # SOI
                0xFF,
                0xE0,
                0x00,
                0x10,  # APP0 marker and length
                0x4A,
                0x46,
                0x49,
                0x46,
                0x00,  # JFIF identifier
                0x01,
                0x01,
                0x00,
                0x00,
                0x01,
                0x00,
                0x01,
                0x00,
                0x00,  # version and density
                0xFF,
                0xDB,
                0x00,
                0x43,
                0x00,  # DQT marker
            ]
            + [0x10] * 64
            + [  # quantization table
                0xFF,
                0xC0,
                0x00,
                0x0B,  # SOF0 marker and length
                0x08,  # precision
                0x00,
                0xC8,  # height: 200
                0x00,
                0x64,  # width: 100
                0x01,
                0x01,
                0x11,
                0x00,  # components
                0xFF,
                0xDA,
                0x00,
                0x08,  # SOS marker
                0x01,
                0x01,
                0x00,
                0x00,
                0x3F,
                0x00,
                0x7F,  # scan data
                0xFF,
                0xD9,  # EOI
            ]
        )
        width, height = fetcher._get_image_dimensions(jpeg_data, ".jpg")
        assert width == 100
        assert height == 200

    def test_get_jpeg_dimensions_no_sof(self, fetcher: ImageFetcher) -> None:
        """Test extracting dimensions from JPEG without SOF marker."""
        # JPEG without SOF marker
        jpeg_data = bytes(
            [
                0xFF,
                0xD8,  # SOI
                0xFF,
                0xD9,  # EOI (no SOF)
            ]
        )
        width, height = fetcher._get_image_dimensions(jpeg_data, ".jpg")
        assert width is None
        assert height is None

    def test_create_content_text_basic(self, fetcher: ImageFetcher) -> None:
        """Test creating content text with basic metadata."""
        metadata = ImageMetadata(
            source_url="http://example.com/img.jpg",
            content_type="image/jpeg",
            size_bytes=1024,
            filename="abc123.jpg",
            local_path="/data/images/2024/01/abc123.jpg",
        )
        text = fetcher._create_content_text(metadata)
        assert "Image: abc123.jpg" in text
        assert "Source URL: http://example.com/img.jpg" in text
        assert "Content Type: image/jpeg" in text
        assert "Size: 1.0 KB" in text
        assert "Local Path: /data/images/2024/01/abc123.jpg" in text

    def test_create_content_text_with_source_page(self, fetcher: ImageFetcher) -> None:
        """Test creating content text with source page."""
        metadata = ImageMetadata(
            source_url="http://example.com/img.jpg",
            source_page="http://example.com/article",
            content_type="image/jpeg",
            size_bytes=1024,
            filename="abc123.jpg",
            local_path="/data/images/abc123.jpg",
        )
        text = fetcher._create_content_text(metadata)
        assert "Source Page: http://example.com/article" in text

    def test_create_content_text_with_dimensions(self, fetcher: ImageFetcher) -> None:
        """Test creating content text with dimensions."""
        metadata = ImageMetadata(
            source_url="http://example.com/img.jpg",
            content_type="image/jpeg",
            size_bytes=1024,
            filename="abc123.jpg",
            local_path="/data/images/abc123.jpg",
            width=1920,
            height=1080,
        )
        text = fetcher._create_content_text(metadata)
        assert "Dimensions: 1920x1080" in text

    def test_fetch_invalid_url(self, fetcher: ImageFetcher) -> None:
        """Test fetch with invalid URL."""
        result = fetcher.fetch("not-a-url")
        assert result.success is False
        assert "invalid" in result.error_message.lower()

    def test_fetch_non_image_url(self, fetcher: ImageFetcher) -> None:
        """Test fetch with non-image URL."""
        result = fetcher.fetch("https://example.com/page.html")
        assert result.success is False
        assert "image" in result.error_message.lower()

    @patch("requests.get")
    def test_fetch_unexpected_error(self, mock_get: MagicMock, fetcher: ImageFetcher) -> None:
        """Test handling of unexpected error during fetch."""
        mock_get.side_effect = RuntimeError("Unexpected error")
        result = fetcher.fetch("https://example.com/image.jpg")
        assert result.success is False
        assert "unexpected error" in result.error_message.lower()

    @patch("requests.get")
    def test_fetch_timeout(self, mock_get: MagicMock, fetcher: ImageFetcher) -> None:
        """Test handling of timeout error."""
        mock_get.side_effect = requests.exceptions.Timeout()
        result = fetcher.fetch("https://example.com/image.jpg")
        assert result.success is False
        assert "timed out" in result.error_message.lower()

    @patch("requests.get")
    def test_fetch_connection_error(self, mock_get: MagicMock, fetcher: ImageFetcher) -> None:
        """Test handling of connection error."""
        mock_get.side_effect = requests.exceptions.ConnectionError()
        result = fetcher.fetch("https://example.com/image.jpg")
        assert result.success is False
        assert "connection" in result.error_message.lower()

    @patch("requests.get")
    def test_fetch_http_404(self, mock_get: MagicMock, fetcher: ImageFetcher) -> None:
        """Test handling of HTTP 404 error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=MagicMock(status_code=404)
        )
        mock_get.return_value = mock_response
        result = fetcher.fetch("https://example.com/image.jpg")
        assert result.success is False
        assert "404" in result.error_message

    @patch("requests.get")
    def test_fetch_http_403(self, mock_get: MagicMock, fetcher: ImageFetcher) -> None:
        """Test handling of HTTP 403 error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=MagicMock(status_code=403)
        )
        mock_get.return_value = mock_response
        result = fetcher.fetch("https://example.com/image.jpg")
        assert result.success is False
        assert "403" in result.error_message

    @patch("requests.get")
    def test_fetch_http_500(self, mock_get: MagicMock, fetcher: ImageFetcher) -> None:
        """Test handling of HTTP 500 error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=MagicMock(status_code=500)
        )
        mock_get.return_value = mock_response
        result = fetcher.fetch("https://example.com/image.jpg")
        assert result.success is False
        assert "500" in result.error_message

    @patch("requests.get")
    def test_fetch_request_exception(self, mock_get: MagicMock, fetcher: ImageFetcher) -> None:
        """Test handling of generic request exception."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        result = fetcher.fetch("https://example.com/image.jpg")
        assert result.success is False
        assert "request failed" in result.error_message.lower()

    @patch("requests.get")
    def test_fetch_empty_response(self, mock_get: MagicMock, fetcher: ImageFetcher) -> None:
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.content = b""
        mock_response.headers = {"Content-Type": "image/jpeg"}
        mock_get.return_value = mock_response
        result = fetcher.fetch("https://example.com/image.jpg")
        assert result.success is False
        assert "empty" in result.error_message.lower()

    @patch("requests.get")
    def test_fetch_image_too_large(self, mock_get: MagicMock, fetcher: ImageFetcher) -> None:
        """Test handling of image that exceeds size limit."""
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": str(200 * 1024 * 1024)}
        mock_get.return_value = mock_response
        result = fetcher.fetch("https://example.com/image.jpg")
        assert result.success is False
        assert "too large" in result.error_message.lower()

    @patch("requests.get")
    def test_fetch_invalid_image_format(self, mock_get: MagicMock, fetcher: ImageFetcher) -> None:
        """Test handling of invalid image format."""
        mock_response = MagicMock()
        mock_response.content = b"not an image"
        mock_response.headers = {"Content-Type": "application/octet-stream"}
        mock_get.return_value = mock_response
        result = fetcher.fetch("https://example.com/file.txt")
        assert result.success is False
        assert (
            "unsupported" in result.error_message.lower()
            or "invalid" in result.error_message.lower()
        )

    @patch("requests.get")
    def test_fetch_success_png(
        self, mock_get: MagicMock, fetcher: ImageFetcher, sample_png_data: bytes
    ) -> None:
        """Test successful PNG download."""
        mock_response = MagicMock()
        mock_response.content = sample_png_data
        mock_response.headers = {"Content-Type": "image/png"}
        mock_get.return_value = mock_response
        result = fetcher.fetch("https://example.com/image.png")
        assert result.success is True
        assert result.content_type == "image"
        assert result.title.startswith("Image: ")
        assert ".png" in result.title
        assert "Source URL: https://example.com/image.png" in result.content
        assert "image_metadata" in result.metadata
        assert result.metadata["image_metadata"]["width"] == 2
        assert result.metadata["image_metadata"]["height"] == 3

    @patch("requests.get")
    def test_fetch_success_gif(self, mock_get: MagicMock, fetcher: ImageFetcher) -> None:
        """Test successful GIF download."""
        # Valid GIF89a header with 4x5 dimensions
        gif_data = b"GIF89a" + b"\x04\x00" + b"\x05\x00"
        mock_response = MagicMock()
        mock_response.content = gif_data
        mock_response.headers = {"Content-Type": "image/gif"}
        mock_get.return_value = mock_response
        result = fetcher.fetch("https://example.com/image.gif")
        assert result.success is True
        assert result.content_type == "image"
        assert result.metadata["image_metadata"]["width"] == 4
        assert result.metadata["image_metadata"]["height"] == 5

    @patch("requests.get")
    def test_fetch_success_with_source_page(
        self, mock_get: MagicMock, fetcher: ImageFetcher, sample_png_data: bytes
    ) -> None:
        """Test successful download with source page."""
        mock_response = MagicMock()
        mock_response.content = sample_png_data
        mock_response.headers = {"Content-Type": "image/png"}
        mock_get.return_value = mock_response
        result = fetcher.fetch(
            "https://example.com/image.png",
            source_page="https://example.com/article",
        )
        assert result.success is True
        assert "Source Page: https://example.com/article" in result.content
        assert result.metadata["image_metadata"]["source_page"] == "https://example.com/article"

    @patch("requests.get")
    def test_fetch_saves_file(
        self, mock_get: MagicMock, fetcher: ImageFetcher, sample_png_data: bytes
    ) -> None:
        """Test that fetch saves the file to storage."""
        mock_response = MagicMock()
        mock_response.content = sample_png_data
        mock_response.headers = {"Content-Type": "image/png"}
        mock_get.return_value = mock_response
        result = fetcher.fetch("https://example.com/image.png")
        assert result.success is True
        local_path = result.metadata["image_metadata"]["local_path"]
        assert Path(local_path).exists()
        assert Path(local_path).read_bytes() == sample_png_data

    @patch("requests.get")
    def test_fetch_generates_unique_filename(
        self, mock_get: MagicMock, fetcher: ImageFetcher, sample_png_data: bytes
    ) -> None:
        """Test that fetch generates unique filename based on content hash."""
        mock_response = MagicMock()
        mock_response.content = sample_png_data
        mock_response.headers = {"Content-Type": "image/png"}
        mock_get.return_value = mock_response
        result = fetcher.fetch("https://example.com/image.png")
        assert result.success is True
        filename = result.metadata["image_metadata"]["filename"]
        assert filename.endswith(".png")
        assert len(filename) == 20

    @patch("requests.get")
    def test_fetch_organizes_by_date(
        self, mock_get: MagicMock, tmp_path: Path, sample_png_data: bytes
    ) -> None:
        """Test that fetch organizes files by year/month."""
        fetcher = ImageFetcher(storage_dir=tmp_path, add_to_queue=False)
        mock_response = MagicMock()
        mock_response.content = sample_png_data
        mock_response.headers = {"Content-Type": "image/png"}
        mock_get.return_value = mock_response
        result = fetcher.fetch("https://example.com/image.png")
        assert result.success is True
        local_path = Path(result.metadata["image_metadata"]["local_path"])
        now = datetime.now()
        assert local_path.parent.parent.name == str(now.year)
        assert local_path.parent.name == f"{now.month:02d}"

    @patch("requests.get")
    @patch("personal_knowledge_base.fetchers.image.add_job")
    def test_fetch_adds_to_queue(
        self, mock_add_job: MagicMock, mock_get: MagicMock, tmp_path: Path, sample_png_data: bytes
    ) -> None:
        """Test that successful fetch adds job to queue."""
        fetcher = ImageFetcher(storage_dir=tmp_path, add_to_queue=True)
        mock_response = MagicMock()
        mock_response.content = sample_png_data
        mock_response.headers = {"Content-Type": "image/png"}
        mock_get.return_value = mock_response
        result = fetcher.fetch(
            "https://example.com/image.png",
            kb_name="test-kb",
        )
        assert result.success is True
        mock_add_job.assert_called_once()
        call_kwargs = mock_add_job.call_args.kwargs
        assert call_kwargs["content_type"] == "image"
        assert call_kwargs["kb_name"] == "test-kb"
        assert call_kwargs["priority"] == 2

    @patch("requests.get")
    @patch("personal_knowledge_base.fetchers.image.add_job")
    def test_fetch_queue_failure_does_not_fail_fetch(
        self,
        mock_add_job: MagicMock,
        mock_get: MagicMock,
        fetcher: ImageFetcher,
        sample_png_data: bytes,
    ) -> None:
        """Test that queue failure doesn't cause fetch to fail."""
        mock_add_job.side_effect = Exception("Queue error")
        mock_response = MagicMock()
        mock_response.content = sample_png_data
        mock_response.headers = {"Content-Type": "image/png"}
        mock_get.return_value = mock_response
        result = fetcher.fetch("https://example.com/image.png")
        assert result.success is True

    @patch("requests.get")
    def test_fetch_does_not_add_to_queue_when_disabled(
        self, mock_get: MagicMock, tmp_path: Path, sample_png_data: bytes
    ) -> None:
        """Test that fetch doesn't add to queue when disabled."""
        fetcher = ImageFetcher(storage_dir=tmp_path, add_to_queue=False)
        mock_response = MagicMock()
        mock_response.content = sample_png_data
        mock_response.headers = {"Content-Type": "image/png"}
        mock_get.return_value = mock_response
        with patch("personal_knowledge_base.fetchers.image.add_job") as mock_add_job:
            result = fetcher.fetch("https://example.com/image.png")
            assert result.success is True
            mock_add_job.assert_not_called()

    def test_image_metadata_defaults(self) -> None:
        """Test ImageMetadata with default values."""
        metadata = ImageMetadata(
            source_url="http://example.com/img.jpg",
            content_type="image/jpeg",
            size_bytes=1024,
            filename="abc123.jpg",
            local_path="/data/images/abc123.jpg",
        )
        assert metadata.source_url == "http://example.com/img.jpg"
        assert metadata.content_type == "image/jpeg"
        assert metadata.size_bytes == 1024
        assert metadata.filename == "abc123.jpg"
        assert metadata.local_path == "/data/images/abc123.jpg"
        assert metadata.source_page is None
        assert metadata.width is None
        assert metadata.height is None
        assert isinstance(metadata.downloaded_at, datetime)

    def test_image_metadata_full(self) -> None:
        """Test ImageMetadata with all values."""
        now = datetime.now()
        metadata = ImageMetadata(
            source_url="http://example.com/img.jpg",
            source_page="http://example.com/article",
            content_type="image/jpeg",
            size_bytes=2048,
            width=1920,
            height=1080,
            filename="abc123.jpg",
            local_path="/data/images/abc123.jpg",
            downloaded_at=now,
        )
        assert metadata.source_page == "http://example.com/article"
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.downloaded_at == now

    def test_get_webp_dimensions_simple_lossy(self, fetcher: ImageFetcher) -> None:
        """Test extracting dimensions from simple lossy WebP."""
        # VP8 WebP with proper structure
        data = bytes(
            [
                0x52,
                0x49,
                0x46,
                0x46,  # "RIFF"
                0x20,
                0x00,
                0x00,
                0x00,  # file size
                0x57,
                0x45,
                0x42,
                0x50,  # "WEBP"
                0x56,
                0x50,
                0x38,
                0x20,  # "VP8 "
                0x12,
                0x00,
                0x00,
                0x00,  # chunk size
                0x00,
                0x00,
                0x00,  # 3 bytes (bytes 20-22)
                0x9D,
                0x01,
                0x2A,  # sync code (bytes 23-25)
                0x01,
                0x00,  # width (14 bits): 1 (bytes 26-27)
                0x02,
                0x00,  # height (14 bits): 2 (bytes 28-29)
            ]
        )
        width, height = fetcher._get_webp_dimensions(data)
        assert width == 1
        assert height == 2

    def test_get_webp_dimensions_invalid(self, fetcher: ImageFetcher) -> None:
        """Test extracting dimensions from invalid WebP data."""
        width, height = fetcher._get_webp_dimensions(b"not a webp")
        assert width is None
        assert height is None

    def test_get_bmp_dimensions(self, fetcher: ImageFetcher) -> None:
        """Test extracting dimensions from BMP data."""
        data = bytes(
            [
                0x42,
                0x4D,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x28,
                0x00,
                0x00,
                0x00,
                0x08,
                0x00,
                0x00,
                0x00,
                0x06,
                0x00,
                0x00,
                0x00,
            ]
        )
        width, height = fetcher._get_bmp_dimensions(data)
        assert width == 8
        assert height == 6

    def test_get_bmp_dimensions_negative_height(self, fetcher: ImageFetcher) -> None:
        """Test extracting dimensions from top-down BMP."""
        # BMP with negative height (top-down) - height is stored as signed 32-bit int
        # -6 in two's complement = 0xFFFFFFFA
        data = bytes(
            [
                0x42,
                0x4D,  # "BM"
                0x00,
                0x00,
                0x00,
                0x00,  # file size
                0x00,
                0x00,
                0x00,
                0x00,  # reserved
                0x00,
                0x00,
                0x00,
                0x00,  # offset
                0x28,
                0x00,
                0x00,
                0x00,  # DIB header size (40)
                0x08,
                0x00,
                0x00,
                0x00,  # width: 8
                0xFA,
                0xFF,
                0xFF,
                0xFF,  # height: -6 (top-down BMP)
            ]
        )
        width, height = fetcher._get_bmp_dimensions(data)
        assert width == 8
        # Negative height should be converted to positive
        assert height == 6

    def test_get_bmp_dimensions_invalid(self, fetcher: ImageFetcher) -> None:
        """Test extracting dimensions from invalid BMP data."""
        width, height = fetcher._get_bmp_dimensions(b"not a bmp")
        assert width is None
        assert height is None

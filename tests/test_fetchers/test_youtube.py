"""Tests for the YouTube fetcher."""

from unittest.mock import MagicMock, patch

import pytest
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from youtube_transcript_api._transcripts import FetchedTranscript, FetchedTranscriptSnippet

from personal_knowledge_base.fetchers.youtube import YouTubeFetcher


def create_mock_transcript(snippets_data: list[dict]) -> FetchedTranscript:
    """Create a mock FetchedTranscript from snippet data.

    Args:
        snippets_data: List of dicts with 'text', 'start', 'duration' keys.

    Returns:
        FetchedTranscript object.

    """
    snippets = [
        FetchedTranscriptSnippet(
            text=data["text"],
            start=data["start"],
            duration=data["duration"],
        )
        for data in snippets_data
    ]
    return FetchedTranscript(
        snippets=snippets,
        video_id="dQw4w9WgXcQ",
        language="English",
        language_code="en",
        is_generated=False,
    )


class TestYouTubeFetcher:
    """Tests for YouTubeFetcher class."""

    @pytest.fixture
    def fetcher(self) -> YouTubeFetcher:
        """Create a YouTubeFetcher instance for testing."""
        return YouTubeFetcher()

    def test_get_api_caching(self, fetcher: YouTubeFetcher) -> None:
        """Test that _get_api returns cached instance on second call."""
        # First call creates the instance
        api1 = fetcher._get_api()
        assert api1 is not None
        assert fetcher._transcript_api is not None

        # Second call should return the same cached instance
        api2 = fetcher._get_api()
        assert api2 is api1  # Same object reference
        assert fetcher._transcript_api is api1

    @pytest.fixture
    def mock_transcript_data(self) -> list[dict]:
        """Sample transcript data for mocking."""
        return [
            {"text": "Hello everyone", "start": 0.0, "duration": 2.0},
            {"text": "Welcome to the video", "start": 2.0, "duration": 3.0},
            {"text": "Today we will learn", "start": 5.0, "duration": 2.5},
        ]

    # URL parsing tests

    def test_can_fetch_standard_url(self, fetcher: YouTubeFetcher) -> None:
        """Test can_fetch with standard YouTube URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert fetcher.can_fetch(url) is True

    def test_can_fetch_short_url(self, fetcher: YouTubeFetcher) -> None:
        """Test can_fetch with short youtu.be URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert fetcher.can_fetch(url) is True

    def test_can_fetch_embed_url(self, fetcher: YouTubeFetcher) -> None:
        """Test can_fetch with embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert fetcher.can_fetch(url) is True

    def test_can_fetch_shorts_url(self, fetcher: YouTubeFetcher) -> None:
        """Test can_fetch with shorts URL."""
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        assert fetcher.can_fetch(url) is True

    def test_can_fetch_live_url(self, fetcher: YouTubeFetcher) -> None:
        """Test can_fetch with live URL."""
        url = "https://www.youtube.com/live/dQw4w9WgXcQ"
        assert fetcher.can_fetch(url) is True

    def test_can_fetch_with_additional_params(self, fetcher: YouTubeFetcher) -> None:
        """Test can_fetch with additional URL parameters."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s&feature=share"
        assert fetcher.can_fetch(url) is True

    def test_can_fetch_invalid_url(self, fetcher: YouTubeFetcher) -> None:
        """Test can_fetch with non-YouTube URL."""
        url = "https://www.google.com/search?q=test"
        assert fetcher.can_fetch(url) is False

    def test_can_fetch_malformed_url(self, fetcher: YouTubeFetcher) -> None:
        """Test can_fetch with malformed URL."""
        url = "not-a-url"
        assert fetcher.can_fetch(url) is False

    def test_can_fetch_empty_string(self, fetcher: YouTubeFetcher) -> None:
        """Test can_fetch with empty string."""
        assert fetcher.can_fetch("") is False

    def test_extract_video_id_standard(self, fetcher: YouTubeFetcher) -> None:
        """Test extracting video ID from standard URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert fetcher._extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_short(self, fetcher: YouTubeFetcher) -> None:
        """Test extracting video ID from short URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert fetcher._extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_invalid(self, fetcher: YouTubeFetcher) -> None:
        """Test extracting video ID from invalid URL."""
        url = "https://www.google.com"
        assert fetcher._extract_video_id(url) is None

    # Fetch tests with mocking

    @patch.object(YouTubeFetcher, "_fetch_transcript")
    def test_fetch_success(
        self,
        mock_fetch: MagicMock,
        fetcher: YouTubeFetcher,
        mock_transcript_data: list[dict],
    ) -> None:
        """Test successful transcript fetch."""
        mock_fetch.return_value = create_mock_transcript(mock_transcript_data)

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = fetcher.fetch(url)

        assert result.success is True
        assert result.url == url
        assert result.content_type == "youtube"
        assert "Hello everyone" in result.content
        assert "Welcome to the video" in result.content
        assert "Today we will learn" in result.content
        assert result.metadata["video_id"] == "dQw4w9WgXcQ"
        assert result.metadata["segment_count"] == 3

    @patch.object(YouTubeFetcher, "_fetch_transcript")
    def test_fetch_success_with_timestamps(
        self,
        mock_fetch: MagicMock,
        fetcher: YouTubeFetcher,
        mock_transcript_data: list[dict],
    ) -> None:
        """Test that transcript includes timestamps in metadata."""
        mock_fetch.return_value = create_mock_transcript(mock_transcript_data)

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = fetcher.fetch(url)

        assert result.success is True
        timestamped = result.metadata["transcript_with_timestamps"]
        assert "[0:00] Hello everyone" in timestamped
        assert "[0:02] Welcome to the video" in timestamped
        assert "[0:05] Today we will learn" in timestamped

    @patch.object(YouTubeFetcher, "_fetch_transcript")
    def test_fetch_success_duration(
        self,
        mock_fetch: MagicMock,
        fetcher: YouTubeFetcher,
        mock_transcript_data: list[dict],
    ) -> None:
        """Test that duration is calculated correctly."""
        mock_fetch.return_value = create_mock_transcript(mock_transcript_data)

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = fetcher.fetch(url)

        assert result.success is True
        assert result.metadata["duration_seconds"] == 7.5  # 5.0 + 2.5
        assert result.metadata["duration_formatted"] == "0:07"

    def test_fetch_invalid_url(self, fetcher: YouTubeFetcher) -> None:
        """Test fetch with invalid URL."""
        url = "https://www.google.com"
        result = fetcher.fetch(url)

        assert result.success is False
        assert "Invalid YouTube URL" in result.error_message

    @patch.object(YouTubeFetcher, "_fetch_transcript")
    def test_fetch_video_unavailable(self, mock_fetch: MagicMock, fetcher: YouTubeFetcher) -> None:
        """Test handling of unavailable video."""
        mock_fetch.side_effect = VideoUnavailable("")

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = fetcher.fetch(url)

        assert result.success is False
        assert "unavailable" in result.error_message.lower()

    @patch.object(YouTubeFetcher, "_fetch_transcript")
    def test_fetch_no_transcript_found(
        self, mock_fetch: MagicMock, fetcher: YouTubeFetcher
    ) -> None:
        """Test handling when no transcript exists."""
        mock_fetch.side_effect = NoTranscriptFound("", "", "")

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = fetcher.fetch(url)

        assert result.success is False
        assert "No transcript available" in result.error_message

    @patch.object(YouTubeFetcher, "_fetch_transcript")
    def test_fetch_transcripts_disabled(
        self, mock_fetch: MagicMock, fetcher: YouTubeFetcher
    ) -> None:
        """Test handling when transcripts are disabled."""
        mock_fetch.side_effect = TranscriptsDisabled("")

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = fetcher.fetch(url)

        assert result.success is False
        assert "disabled" in result.error_message.lower()

    @patch.object(YouTubeFetcher, "_fetch_transcript")
    def test_fetch_generic_error(self, mock_fetch: MagicMock, fetcher: YouTubeFetcher) -> None:
        """Test handling of generic errors."""
        mock_fetch.side_effect = Exception("Network error")

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = fetcher.fetch(url)

        assert result.success is False
        assert "Network error" in result.error_message

    # Timestamp formatting tests

    def test_format_timestamp_seconds_only(self, fetcher: YouTubeFetcher) -> None:
        """Test formatting timestamp with seconds only."""
        assert fetcher._format_timestamp(45.0) == "0:45"

    def test_format_timestamp_minutes_seconds(self, fetcher: YouTubeFetcher) -> None:
        """Test formatting timestamp with minutes and seconds."""
        assert fetcher._format_timestamp(125.0) == "2:05"

    def test_format_timestamp_hours(self, fetcher: YouTubeFetcher) -> None:
        """Test formatting timestamp with hours."""
        assert fetcher._format_timestamp(3665.0) == "1:01:05"

    def test_format_timestamp_zero(self, fetcher: YouTubeFetcher) -> None:
        """Test formatting zero timestamp."""
        assert fetcher._format_timestamp(0.0) == "0:00"

    # fetch_with_title tests

    @patch.object(YouTubeFetcher, "_fetch_transcript")
    def test_fetch_with_title_success(
        self,
        mock_fetch: MagicMock,
        fetcher: YouTubeFetcher,
        mock_transcript_data: list[dict],
    ) -> None:
        """Test fetch_with_title with provided title."""
        mock_fetch.return_value = create_mock_transcript(mock_transcript_data)

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = fetcher.fetch_with_title(url, title="My Custom Title")

        assert result.success is True
        assert result.title == "My Custom Title"
        # Verify all other fields are preserved
        assert result.url == url
        assert result.content_type == "youtube"
        assert "Hello everyone" in result.content
        assert result.metadata["video_id"] == "dQw4w9WgXcQ"

    @patch.object(YouTubeFetcher, "_fetch_transcript")
    def test_fetch_with_title_no_title(
        self,
        mock_fetch: MagicMock,
        fetcher: YouTubeFetcher,
        mock_transcript_data: list[dict],
    ) -> None:
        """Test fetch_with_title without providing title."""
        mock_fetch.return_value = create_mock_transcript(mock_transcript_data)

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = fetcher.fetch_with_title(url)

        assert result.success is True
        assert "YouTube Video" in result.title

    @patch.object(YouTubeFetcher, "_fetch_transcript")
    def test_fetch_with_title_failure(self, mock_fetch: MagicMock, fetcher: YouTubeFetcher) -> None:
        """Test fetch_with_title when fetch fails."""
        mock_fetch.side_effect = VideoUnavailable("")

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = fetcher.fetch_with_title(url, title="My Title")

        assert result.success is False
        # Title should not matter on failure

    # Edge cases

    @patch.object(YouTubeFetcher, "_fetch_transcript")
    def test_fetch_empty_transcript(self, mock_fetch: MagicMock, fetcher: YouTubeFetcher) -> None:
        """Test handling of empty transcript."""
        mock_fetch.return_value = create_mock_transcript([])

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = fetcher.fetch(url)

        assert result.success is True
        assert result.content == ""
        assert result.metadata["segment_count"] == 0
        assert result.metadata["duration_seconds"] == 0.0

    @patch.object(YouTubeFetcher, "_fetch_transcript")
    def test_fetch_single_segment(self, mock_fetch: MagicMock, fetcher: YouTubeFetcher) -> None:
        """Test handling of single segment transcript."""
        mock_fetch.return_value = create_mock_transcript(
            [{"text": "Hello", "start": 0.0, "duration": 1.0}]
        )

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = fetcher.fetch(url)

        assert result.success is True
        assert result.content == "Hello"
        assert result.metadata["segment_count"] == 1

    @patch.object(YouTubeFetcher, "_fetch_transcript")
    def test_fetch_long_duration(self, mock_fetch: MagicMock, fetcher: YouTubeFetcher) -> None:
        """Test handling of long video duration."""
        mock_fetch.return_value = create_mock_transcript(
            [{"text": "End", "start": 7200.0, "duration": 5.0}]  # 2 hours
        )

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = fetcher.fetch(url)

        assert result.success is True
        assert result.metadata["duration_formatted"] == "2:00:05"

    def test_extract_video_id_with_list_param(self, fetcher: YouTubeFetcher) -> None:
        """Test extracting video ID from URL with list parameter."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLsomeplaylist"
        assert fetcher._extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_mobile_url(self, fetcher: YouTubeFetcher) -> None:
        """Test extracting video ID from mobile URL."""
        url = "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        assert fetcher._extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_mobile_url_no_www(self, fetcher: YouTubeFetcher) -> None:
        """Test extracting video ID from mobile URL without www."""
        url = "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        assert fetcher._extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_query_param_fallback(self, fetcher: YouTubeFetcher) -> None:
        """Test extracting video ID using query param fallback parsing."""
        # URL that doesn't match regex patterns but has v= in query
        url = "https://youtube.com/watch?feature=share&v=dQw4w9WgXcQ"
        assert fetcher._extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_youtube_com_no_www(self, fetcher: YouTubeFetcher) -> None:
        """Test extracting video ID from youtube.com without www."""
        url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        assert fetcher._extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_fallback_parsing(self, fetcher: YouTubeFetcher) -> None:
        """Test extracting video ID using urlparse fallback (no regex match)."""
        # URL that doesn't match regex patterns (no /watch?v= pattern)
        # but has youtube.com domain with v= in query params
        url = "https://youtube.com/other/path/here?v=dQw4w9WgXcQ"
        assert fetcher._extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_fallback_parsing_www(self, fetcher: YouTubeFetcher) -> None:
        """Test fallback parsing with www.youtube.com domain."""
        url = "https://www.youtube.com/other/path?v=dQw4w9WgXcQ"
        assert fetcher._extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_fallback_parsing_mobile(self, fetcher: YouTubeFetcher) -> None:
        """Test fallback parsing with m.youtube.com mobile domain."""
        url = "https://m.youtube.com/other/path?v=dQw4w9WgXcQ"
        assert fetcher._extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_fallback_invalid_id_length(self, fetcher: YouTubeFetcher) -> None:
        """Test fallback parsing returns None for invalid video ID length."""
        url = "https://youtube.com/other/path?v=short"
        assert fetcher._extract_video_id(url) is None

    def test_extract_video_id_fallback_no_v_param(self, fetcher: YouTubeFetcher) -> None:
        """Test fallback parsing returns None when no v param present."""
        url = "https://youtube.com/other/path?feature=share"
        assert fetcher._extract_video_id(url) is None

    def test_extract_video_id_fallback_value_error(self, fetcher: YouTubeFetcher) -> None:
        """Test fallback parsing handles ValueError from urlparse."""
        # An invalid URL that triggers ValueError in urlparse
        url = "://invalid-url"
        result = fetcher._extract_video_id(url)
        assert result is None

    @patch.object(YouTubeFetcher, "_fetch_transcript")
    def test_fetch_preserves_all_metadata(
        self,
        mock_fetch: MagicMock,
        fetcher: YouTubeFetcher,
        mock_transcript_data: list[dict],
    ) -> None:
        """Test that all expected metadata is present."""
        mock_fetch.return_value = create_mock_transcript(mock_transcript_data)

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = fetcher.fetch(url)

        assert result.success is True
        assert "video_id" in result.metadata
        assert "segment_count" in result.metadata
        assert "duration_seconds" in result.metadata
        assert "duration_formatted" in result.metadata
        assert "transcript_with_timestamps" in result.metadata
        assert "language" in result.metadata
        assert "language_code" in result.metadata
        assert "is_generated" in result.metadata

    @patch.object(YouTubeFetcher, "_fetch_transcript")
    def test_fetch_language_metadata(self, mock_fetch: MagicMock, fetcher: YouTubeFetcher) -> None:
        """Test that language metadata is correctly extracted."""
        mock_fetch.return_value = create_mock_transcript(
            [{"text": "Hello", "start": 0.0, "duration": 1.0}]
        )

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = fetcher.fetch(url)

        assert result.success is True
        assert result.metadata["language"] == "English"
        assert result.metadata["language_code"] == "en"
        assert result.metadata["is_generated"] is False

    def test_fetch_transcript_calls_get_api(self, fetcher: YouTubeFetcher) -> None:
        """Test that _fetch_transcript calls _get_api and returns result."""
        # Mock the API instance returned by _get_api
        with patch.object(fetcher, "_get_api") as mock_get_api:
            mock_api = MagicMock()
            mock_transcript = create_mock_transcript(
                [{"text": "Hello", "start": 0.0, "duration": 1.0}]
            )
            mock_api.fetch.return_value = mock_transcript
            mock_get_api.return_value = mock_api

            result = fetcher._fetch_transcript("dQw4w9WgXcQ")

            # Verify _get_api was called
            mock_get_api.assert_called_once()
            # Verify api.fetch was called with video_id
            mock_api.fetch.assert_called_once_with("dQw4w9WgXcQ")
            # Verify the result is returned
            assert result is mock_transcript

    @patch.object(YouTubeFetcher, "_fetch_transcript")
    def test_fetch_with_title_returns_original_on_failure(
        self, mock_fetch: MagicMock, fetcher: YouTubeFetcher
    ) -> None:
        """Test fetch_with_title returns original result when fetch fails."""
        from youtube_transcript_api._errors import VideoUnavailable

        mock_fetch.side_effect = VideoUnavailable("")

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = fetcher.fetch_with_title(url, title="My Title")

        # Should return the failed result (not the titled one)
        assert result.success is False
        assert "unavailable" in result.error_message.lower()

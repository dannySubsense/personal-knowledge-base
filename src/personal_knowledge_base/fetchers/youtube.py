"""YouTube transcript fetcher."""

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from youtube_transcript_api._transcripts import FetchedTranscript

from personal_knowledge_base.fetchers.base import Fetcher, FetchResult


class YouTubeFetcher(Fetcher):
    """Fetcher for YouTube video transcripts.

    Extracts video ID from various YouTube URL formats and retrieves
    the transcript using youtube-transcript-api.

    Supported URL formats:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - https://www.youtube.com/v/VIDEO_ID
        - https://www.youtube.com/shorts/VIDEO_ID

    """

    # Regex patterns for extracting video ID from various YouTube URL formats
    _URL_PATTERNS = [
        # Standard watch URLs: youtube.com/watch?v=VIDEO_ID
        re.compile(r"(?:youtube\.com/watch\?v=|youtube\.com/watch\?.*&v=)([a-zA-Z0-9_-]{11})"),
        # Short URLs: youtu.be/VIDEO_ID
        re.compile(r"youtu\.be/([a-zA-Z0-9_-]{11})"),
        # Embed URLs: youtube.com/embed/VIDEO_ID
        re.compile(r"youtube\.com/embed/([a-zA-Z0-9_-]{11})"),
        # Old object URLs: youtube.com/v/VIDEO_ID
        re.compile(r"youtube\.com/v/([a-zA-Z0-9_-]{11})"),
        # Shorts URLs: youtube.com/shorts/VIDEO_ID
        re.compile(r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})"),
        # Live URLs: youtube.com/live/VIDEO_ID
        re.compile(r"youtube\.com/live/([a-zA-Z0-9_-]{11})"),
    ]

    def __init__(self) -> None:
        """Initialize the YouTube fetcher."""
        self._transcript_api: YouTubeTranscriptApi | None = None

    def _get_api(self) -> YouTubeTranscriptApi:
        """Get or create the transcript API instance.

        Returns:
            YouTubeTranscriptApi instance.

        """
        if self._transcript_api is None:
            self._transcript_api = YouTubeTranscriptApi()
        return self._transcript_api

    def can_fetch(self, url: str) -> bool:
        """Check if this fetcher can handle the given URL.

        Args:
            url: The URL to check.

        Returns:
            True if the URL is a valid YouTube URL, False otherwise.

        """
        return self._extract_video_id(url) is not None

    def _extract_video_id(self, url: str) -> str | None:
        """Extract the video ID from a YouTube URL.

        Args:
            url: The YouTube URL.

        Returns:
            The 11-character video ID, or None if not found.

        """
        for pattern in self._URL_PATTERNS:
            match = pattern.search(url)
            if match:
                return match.group(1)

        # Try parsing as a standard URL
        try:
            parsed = urlparse(url)
            if parsed.netloc in ("youtube.com", "www.youtube.com", "m.youtube.com"):
                query_params = parse_qs(parsed.query)
                if "v" in query_params:
                    video_id = query_params["v"][0]
                    if len(video_id) == 11:
                        return video_id
        except (ValueError, IndexError):
            pass

        return None

    def fetch(self, url: str) -> FetchResult:
        """Fetch the transcript for a YouTube video.

        Args:
            url: The YouTube video URL.

        Returns:
            FetchResult containing the transcript or error information.

        """
        video_id = self._extract_video_id(url)

        if video_id is None:
            return FetchResult(
                url=url,
                content_type="youtube",
                success=False,
                error_message="Invalid YouTube URL: Could not extract video ID.",
            )

        try:
            transcript = self._fetch_transcript(video_id)
            return self._process_transcript(url, video_id, transcript)
        except VideoUnavailable:
            return FetchResult(
                url=url,
                content_type="youtube",
                success=False,
                error_message="Video is unavailable (may be private or deleted).",
            )
        except NoTranscriptFound:
            return FetchResult(
                url=url,
                content_type="youtube",
                success=False,
                error_message="No transcript available for this video.",
            )
        except TranscriptsDisabled:
            return FetchResult(
                url=url,
                content_type="youtube",
                success=False,
                error_message="Transcripts are disabled for this video.",
            )
        except Exception as e:
            return FetchResult(
                url=url,
                content_type="youtube",
                success=False,
                error_message=f"Failed to fetch transcript: {str(e)}",
            )

    def _fetch_transcript(self, video_id: str) -> FetchedTranscript:
        """Fetch the raw transcript data from YouTube.

        Args:
            video_id: The YouTube video ID.

        Returns:
            FetchedTranscript object containing transcript snippets.

        Raises:
            VideoUnavailable: If the video is unavailable.
            NoTranscriptFound: If no transcript exists.
            TranscriptsDisabled: If transcripts are disabled.

        """
        api = self._get_api()
        return api.fetch(video_id)

    def _process_transcript(
        self, url: str, video_id: str, transcript: FetchedTranscript
    ) -> FetchResult:
        """Process transcript into a FetchResult.

        Args:
            url: The original URL.
            video_id: The YouTube video ID.
            transcript: FetchedTranscript object from the API.

        Returns:
            FetchResult with the processed transcript.

        """
        # Combine all text segments
        full_text = " ".join(snippet.text for snippet in transcript.snippets)

        # Build timestamped version for metadata
        timestamped_text = "\n".join(
            f"[{self._format_timestamp(snippet.start)}] {snippet.text}"
            for snippet in transcript.snippets
        )

        # Calculate duration
        if transcript.snippets:
            last_snippet = transcript.snippets[-1]
            duration = last_snippet.start + last_snippet.duration
        else:
            duration = 0.0

        # Build metadata
        metadata: dict[str, Any] = {
            "video_id": video_id,
            "segment_count": len(transcript.snippets),
            "duration_seconds": duration,
            "duration_formatted": self._format_timestamp(duration),
            "transcript_with_timestamps": timestamped_text,
            "language": transcript.language,
            "language_code": transcript.language_code,
            "is_generated": transcript.is_generated,
        }

        return FetchResult(
            url=url,
            title=f"YouTube Video ({video_id})",  # Title would need additional API call
            content=full_text,
            content_type="youtube",
            success=True,
            metadata=metadata,
        )

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS or MM:SS.

        Args:
            seconds: Time in seconds.

        Returns:
            Formatted timestamp string.

        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def fetch_with_title(self, url: str, title: str | None = None) -> FetchResult:
        """Fetch transcript with an optional pre-known title.

        This is useful when the title is obtained from another source
        (e.g., WhatsApp message context, oEmbed, etc.).

        Args:
            url: The YouTube video URL.
            title: Optional title to use instead of the generic one.

        Returns:
            FetchResult containing the transcript.

        """
        result = self.fetch(url)

        if result.success and title:
            # Create a new result with the provided title
            return FetchResult(
                url=result.url,
                title=title,
                content=result.content,
                content_type=result.content_type,
                author=result.author,
                published_date=result.published_date,
                metadata=result.metadata,
                success=result.success,
                error_message=result.error_message,
                fetched_at=result.fetched_at,
            )

        return result

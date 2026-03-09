"""Fetcher module exports."""

from personal_knowledge_base.fetchers.base import Fetcher, FetchResult
from personal_knowledge_base.fetchers.image import ImageFetcher, ImageMetadata
from personal_knowledge_base.fetchers.pdf import PDFFetcher
from personal_knowledge_base.fetchers.web import WebFetcher
from personal_knowledge_base.fetchers.youtube import YouTubeFetcher

__all__ = [
    "Fetcher",
    "FetchResult",
    "ImageFetcher",
    "ImageMetadata",
    "PDFFetcher",
    "YouTubeFetcher",
    "WebFetcher",
]

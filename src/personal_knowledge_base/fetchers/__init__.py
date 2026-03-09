"""Fetcher module exports."""

from personal_knowledge_base.fetchers.base import Fetcher, FetchResult
from personal_knowledge_base.fetchers.youtube import YouTubeFetcher

__all__ = ["Fetcher", "FetchResult", "YouTubeFetcher"]

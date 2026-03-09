"""Text chunking module for splitting content into semantic chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ContentType(Enum):
    """Types of content that require different chunking strategies."""

    PROSE = "prose"  # Regular text, articles, blog posts
    CODE = "code"  # Source code
    MARKDOWN = "markdown"  # Markdown documents
    MIXED = "mixed"  # Mixed content types


@dataclass
class ChunkingConfig:
    """Configuration for text chunking.

    Attributes:
        chunk_size: Target size of each chunk in tokens (approximated as words).
        chunk_overlap: Number of tokens to overlap between chunks for context.
        min_chunk_size: Minimum size for a chunk to be valid.
        respect_boundaries: Whether to respect semantic boundaries like paragraphs.
        content_type: Type of content being chunked.
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 50
    respect_boundaries: bool = True
    content_type: ContentType = ContentType.PROSE


@dataclass
class Chunk:
    """A single chunk of text with metadata.

    Attributes:
        text: The chunk content.
        index: Zero-based index of this chunk.
        total: Total number of chunks.
        source: Identifier for the source document.
        metadata: Additional metadata about the chunk.
    """

    text: str
    index: int
    total: int
    source: str
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Ensure metadata is never None."""
        if self.metadata is None:
            self.metadata = {}


class Chunker:
    """Split text into semantic chunks for embedding.

    This class provides intelligent text chunking that:
    - Respects semantic boundaries (paragraphs, sections)
    - Preserves context through overlapping chunks
    - Handles different content types (prose, code, markdown)
    - Returns chunks with rich metadata
    """

    # Approximate tokens per word ratio for estimation
    TOKENS_PER_WORD = 1.3

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """Initialize the chunker with configuration.

        Args:
            config: Chunking configuration. Uses defaults if not provided.
        """
        self.config = config or ChunkingConfig()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text.

        Uses a simple word-based approximation since we don't have
        access to the actual tokenizer.

        Args:
            text: The text to estimate.

        Returns:
            Estimated token count.
        """
        words = len(text.split())
        return int(words * self.TOKENS_PER_WORD)

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs.

        Args:
            text: The text to split.

        Returns:
            List of paragraphs.
        """
        # Normalize line endings and split on double newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: The text to split.

        Returns:
            List of sentences.
        """
        # Simple sentence splitting - handles common cases
        # Matches sentence-ending punctuation followed by space and capital letter
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_code_into_blocks(self, text: str) -> list[str]:
        """Split code into logical blocks.

        Attempts to split on:
        - Function/class definitions
        - Blank lines
        - Logical blocks

        Args:
            text: The code text to split.

        Returns:
            List of code blocks.
        """
        lines = text.split("\n")
        blocks = []
        current_block = []

        for line in lines:
            stripped = line.strip()

            # Check for function/class definitions (common in many languages)
            is_definition = bool(
                re.match(r"^(def\s+|class\s+|function\s+|const\s+|let\s+|var\s+)", stripped)
            )

            # Start a new block if we hit a definition and have content
            if is_definition and current_block:
                blocks.append("\n".join(current_block))
                current_block = []

            current_block.append(line)

            # End block on blank lines if block is getting large
            if not stripped and current_block:
                block_text = "\n".join(current_block).strip()
                if block_text and self._estimate_tokens(block_text) > self.config.chunk_size // 2:
                    blocks.append(block_text)
                    current_block = []

        # Add remaining lines
        if current_block:
            block_text = "\n".join(current_block).strip()
            if block_text:
                blocks.append(block_text)

        return blocks if blocks else [text]

    def _split_markdown_into_sections(self, text: str) -> list[tuple[str, str | None]]:
        """Split markdown into sections based on headers.

        Args:
            text: The markdown text to split.

        Returns:
            List of tuples (section_text, header).
        """
        # Match markdown headers (# Header, ## Header, etc.)
        header_pattern = r"^(#{1,6}\s+.+)$"
        lines = text.split("\n")

        sections = []
        current_section = []
        current_header = None

        for line in lines:
            header_match = re.match(header_pattern, line.strip())

            if header_match:
                # Save previous section
                if current_section:
                    section_text = "\n".join(current_section).strip()
                    if section_text:
                        sections.append((section_text, current_header))

                current_header = line.strip()
                current_section = [line]
            else:
                current_section.append(line)

        # Add final section
        if current_section:
            section_text = "\n".join(current_section).strip()
            if section_text:
                sections.append((section_text, current_header))

        return sections if sections else [(text, None)]

    def _create_chunks_from_units(
        self, units: list[str], source: str, overlap_units: int = 1
    ) -> list[Chunk]:
        """Create chunks from text units (paragraphs, sentences, etc.).

        Args:
            units: List of text units to chunk.
            source: Source identifier for the document.
            overlap_units: Number of units to overlap between chunks.

        Returns:
            List of chunks.
        """
        if not units:
            return []

        chunks = []
        current_chunk_units = []
        current_token_count = 0

        for _i, unit in enumerate(units):
            unit_tokens = self._estimate_tokens(unit)

            # Check if adding this unit would exceed chunk size
            if current_chunk_units and current_token_count + unit_tokens > self.config.chunk_size:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk_units)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                if self.config.chunk_overlap > 0 and len(current_chunk_units) >= overlap_units:
                    # Take last N units for overlap
                    overlap_text = " ".join(current_chunk_units[-overlap_units:])
                    overlap_tokens = self._estimate_tokens(overlap_text)
                    current_chunk_units = current_chunk_units[-overlap_units:] + [unit]
                    current_token_count = overlap_tokens + unit_tokens
                else:
                    current_chunk_units = [unit]
                    current_token_count = unit_tokens
            else:
                current_chunk_units.append(unit)
                current_token_count += unit_tokens

        # Add final chunk
        if current_chunk_units:
            chunk_text = "\n\n".join(current_chunk_units)
            chunks.append(chunk_text)

        # Create Chunk objects with metadata
        return [
            Chunk(
                text=chunk_text,
                index=i,
                total=len(chunks),
                source=source,
                metadata={
                    "token_estimate": self._estimate_tokens(chunk_text),
                    "char_count": len(chunk_text),
                },
            )
            for i, chunk_text in enumerate(chunks)
        ]

    def _chunk_prose(self, text: str, source: str) -> list[Chunk]:
        """Chunk prose content (articles, blog posts, etc.).

        Strategy:
        1. Split into paragraphs
        2. Merge paragraphs into chunks respecting chunk_size
        3. Add overlap between chunks

        Args:
            text: The prose text to chunk.
            source: Source identifier.

        Returns:
            List of chunks.
        """
        paragraphs = self._split_into_paragraphs(text)

        if not paragraphs:
            return []

        # If text is very short, return as single chunk
        if self._estimate_tokens(text) <= self.config.chunk_size:
            return [
                Chunk(
                    text=text.strip(),
                    index=0,
                    total=1,
                    source=source,
                    metadata={
                        "token_estimate": self._estimate_tokens(text),
                        "char_count": len(text),
                    },
                )
            ]

        return self._create_chunks_from_units(paragraphs, source, overlap_units=1)

    def _chunk_code(self, text: str, source: str) -> list[Chunk]:
        """Chunk code content.

        Strategy:
        1. Split into logical blocks (functions, classes)
        2. Merge blocks into chunks
        3. Try to keep related code together

        Args:
            text: The code text to chunk.
            source: Source identifier.

        Returns:
            List of chunks.
        """
        blocks = self._split_code_into_blocks(text)

        if not blocks:
            return []

        # If text is very short, return as single chunk
        if self._estimate_tokens(text) <= self.config.chunk_size:
            return [
                Chunk(
                    text=text.strip(),
                    index=0,
                    total=1,
                    source=source,
                    metadata={
                        "token_estimate": self._estimate_tokens(text),
                        "char_count": len(text),
                        "content_type": "code",
                    },
                )
            ]

        chunks = self._create_chunks_from_units(blocks, source, overlap_units=1)

        # Add content_type metadata
        for chunk in chunks:
            if chunk.metadata:
                chunk.metadata["content_type"] = "code"

        return chunks

    def _chunk_markdown(self, text: str, source: str) -> list[Chunk]:
        """Chunk markdown content.

        Strategy:
        1. Split into sections based on headers
        2. Further split large sections
        3. Preserve header context in metadata

        Args:
            text: The markdown text to chunk.
            source: Source identifier.

        Returns:
            List of chunks.
        """
        sections = self._split_markdown_into_sections(text)

        if not sections:
            return []

        # If text is very short, return as single chunk
        if self._estimate_tokens(text) <= self.config.chunk_size:
            return [
                Chunk(
                    text=text.strip(),
                    index=0,
                    total=1,
                    source=source,
                    metadata={
                        "token_estimate": self._estimate_tokens(text),
                        "char_count": len(text),
                        "content_type": "markdown",
                    },
                )
            ]

        chunks = []
        section_idx = 0

        for section_text, header in sections:
            section_tokens = self._estimate_tokens(section_text)

            if section_tokens <= self.config.chunk_size:
                # Section fits in one chunk
                chunks.append(
                    Chunk(
                        text=section_text,
                        index=section_idx,
                        total=-1,  # Will update later
                        source=source,
                        metadata={
                            "token_estimate": section_tokens,
                            "char_count": len(section_text),
                            "content_type": "markdown",
                            "section_header": header,
                        },
                    )
                )
                section_idx += 1
            else:
                # Section is too large, split it further
                paragraphs = self._split_into_paragraphs(section_text)
                section_chunks = self._create_chunks_from_units(paragraphs, source, overlap_units=1)

                for chunk in section_chunks:
                    chunk.index = section_idx
                    if chunk.metadata:
                        chunk.metadata["content_type"] = "markdown"
                        chunk.metadata["section_header"] = header
                    section_idx += 1

                chunks.extend(section_chunks)

        # Update total count
        for i, chunk in enumerate(chunks):
            chunk.index = i
            chunk.total = len(chunks)

        return chunks

    def chunk(self, text: str, source: str = "unknown") -> list[Chunk]:
        """Split text into semantic chunks.

        This is the main entry point for chunking text. It dispatches to
        the appropriate chunking strategy based on content type.

        Args:
            text: The text to chunk.
            source: Identifier for the source document.

        Returns:
            List of chunks with metadata.

        Raises:
            ValueError: If text is not a string.
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected string, got {type(text).__name__}")

        # Handle edge cases
        if not text.strip():
            return []

        # Detect content type if set to MIXED or auto-detect
        content_type = self.config.content_type

        # Simple heuristics for content type detection
        if content_type == ContentType.MIXED:
            code_indicators = ["def ", "class ", "function ", "import ", "const ", "let "]
            markdown_indicators = ["# ", "## ", "### ", "```", "| ", "- ", "* "]

            code_score = sum(1 for ind in code_indicators if ind in text[:1000])
            md_score = sum(1 for ind in markdown_indicators if ind in text[:1000])

            if code_score >= 2:
                content_type = ContentType.CODE
            elif md_score >= 2:
                content_type = ContentType.MARKDOWN
            else:
                content_type = ContentType.PROSE

        # Dispatch to appropriate chunking strategy
        if content_type == ContentType.CODE:
            result = self._chunk_code(text, source)
        elif content_type == ContentType.MARKDOWN:
            result = self._chunk_markdown(text, source)
        else:
            result = self._chunk_prose(text, source)

        # Filter out chunks that are too small (unless it's the only chunk)
        if len(result) > 1:
            result = [
                chunk
                for chunk in result
                if self._estimate_tokens(chunk.text) >= self.config.min_chunk_size
            ]

        # Re-index after filtering
        for i, chunk in enumerate(result):
            chunk.index = i
            chunk.total = len(result)

        return result

    def chunk_with_context(
        self, text: str, source: str = "unknown", context: dict[str, Any] | None = None
    ) -> list[Chunk]:
        """Chunk text with additional context metadata.

        Args:
            text: The text to chunk.
            source: Identifier for the source document.
            context: Additional context to include in chunk metadata.

        Returns:
            List of chunks with enriched metadata.
        """
        chunks = self.chunk(text, source)

        if context:
            for chunk in chunks:
                if chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata.update(context)

        return chunks

# Personal Knowledge Base

A unified system for capturing, organizing, and retrieving digital content across multiple formats and sources.

## Overview

The Personal Knowledge Base (PKB) addresses the pain point of scattered bookmarks and saved content that gets forgotten or lost across mobile apps and platforms. It enables unified content capture via WhatsApp, semantic retrieval, and scalable organization for research and project work.

**Key Features:**
- Multi-format ingestion: YouTube videos, papers, articles, PDFs, images, code repos
- Semantic search with source attribution
- Automatic retrieval during work sessions
- Stale content detection
- Local hosting (homelab)

## Quick Start

```bash
# Clone repository
git clone https://github.com/dannySubsense/personal-knowledge-base.git
cd personal-knowledge-base

# Install dependencies
pip install -r requirements.txt

# Setup Ollama (local embeddings)
ollama pull nomic-embed-text

# Initialize database
python scripts/init_db.py

# Start batch processor
python scripts/batch_processor.py --daemon
```

## Architecture

```
WhatsApp → Queue → Fetch → Extract → Chunk → Embed → Chroma → Query
```

**Components:**
- **Ingestion:** WhatsApp handler, priority queue (SQLite)
- **Processing:** Playwright fetcher, content-specific extractors, Ollama embeddings
- **Storage:** Chroma vector DB, SQLite metadata, filesystem assets
- **Query:** LangChain RAG, hybrid search, staleness detection

## Project Structure

```
personal-knowledge-base/
├── src/
│   ├── ingestion/          # Queue, fetchers, handlers
│   ├── processing/         # Extractors, chunkers, embedders
│   ├── storage/            # Chroma, SQLite interfaces
│   ├── query/              # RAG, retrieval, ranking
│   └── interface/          # WhatsApp, API, CLI
├── data/
│   ├── chroma/             # Vector database
│   ├── sqlite/             # Metadata and queue
│   ├── images/             # Stored images
│   ├── pdfs/               # Original PDFs
│   └── cache/              # Temporary files
├── scripts/                # Utilities and tools
├── tests/                  # Test suite
├── docs/                   # Documentation
└── config/                 # Configuration files
```

## Usage

**Add content via WhatsApp:**
```
Danny: https://www.youtube.com/watch?v=...
Major Tom: Queued for processing ✓
```

**Query during conversation:**
```
Danny: What do we have on factor models?
Major Tom: Found 3 papers and 2 videos on factor models...
```

## Documentation

- [Requirements](docs/requirements.md)
- [Product Requirements Document (PRD)](docs/prd.md)
- [Technical Specification](docs/technical-spec.md)
- [Implementation Plan](docs/implementation-plan.md)
- [Review Notes](docs/review.md)

## Development

**Phases:**
1. Foundation (Week 1-2): Core infrastructure
2. Content Pipeline (Week 3-4): Multi-format ingestion
3. Integration (Week 5-6): WhatsApp + Major Tom interface
4. Polish (Week 7-8): Stale detection, tutorials, suggestions

See [Implementation Plan](docs/implementation-plan.md) for details.

## Configuration

Copy `config/example.config.yml` to `config/config.yml` and customize:

```yaml
ollama:
  model: nomic-embed-text
  host: http://localhost:11434

queue:
  batch_interval_minutes: 15
  
storage:
  chroma_path: ./data/chroma
  sqlite_path: ./data/sqlite/pkb.db
```

## Requirements

- Python 3.11+
- Ollama (local)
- 8GB RAM minimum
- 50GB disk space (for local models and data)

## License

MIT License — See [LICENSE](LICENSE) for details.

## Contributing

This is a personal project. For now, contributions are limited to Danny and Major Tom.

---

**Status:** In development (Phase 1)  
**Last Updated:** 2026-03-09

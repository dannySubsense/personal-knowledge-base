# ADR 002: Chunking Configuration for nomic-embed-text Context Limit

**Date:** 2026-03-10
**Status:** Accepted
**Deciders:** Danny, Major Tom

## Context

During initial queue testing (2026-03-10), the batch processor failed to embed YouTube transcripts with the error:

> "the input length exceeds the context length"

Root cause: the chunker's default `chunk_size` is 1000 tokens, but `nomic-embed-text` has a maximum context window of ~512 tokens (~2,000 characters). A YouTube transcript was fetched as a single blob (~10,995 chars), the chunker produced 1 oversized chunk, and Ollama rejected the embedding request with HTTP 400.

## Decision

Set `chunk_size: 256` tokens as the default in `ChunkingConfig` and in `config/example.config.yml`.

## Rationale

- **nomic-embed-text context limit:** ~512 tokens. Chunks must stay under this.
- **256 tokens chosen** (not 512) to leave headroom for overlap and to produce more granular, higher-quality embeddings. Smaller chunks = more precise retrieval.
- **chunk_overlap stays at 200 chars** (not tokens) — this is already character-based in the current implementation, so no change needed there.
- **Model not changed:** nomic-embed-text is appropriate for CPU-only local hosting. It's competitive on MTEB retrieval benchmarks and has zero API cost. No reason to swap it out until GPU is available.

## Consequences

- **Positive:** Embedding requests no longer exceed context window; ingestion pipeline works end-to-end
- **Positive:** Smaller chunks produce more precise semantic search results
- **Negative:** More chunks per document = more Qdrant vectors = slightly higher storage and query latency (acceptable at current scale)
- **Breaking change:** Any existing Qdrant collections ingested with chunk_size=1000 will have inconsistent granularity. Since the DB is empty at time of this decision, no migration needed.

## Configuration

```yaml
chunking:
  chunk_size: 256       # Must stay under nomic-embed-text's 512-token context limit
  chunk_overlap: 200    # Character-based overlap for context continuity
  min_chunk_size: 50
```

## Future Consideration

If a GPU becomes available and a higher-capacity embedding model is adopted (e.g., `mxbai-embed-large` at 512-token limit, or OpenAI `text-embedding-3-small` at 8191 tokens), `chunk_size` can be increased accordingly. Any such change requires re-ingesting all existing content.

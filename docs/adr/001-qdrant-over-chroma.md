# ADR 001: Use Qdrant Instead of Chroma as Vector Database

**Date:** 2026-03-10
**Status:** Accepted
**Deciders:** Danny, Major Tom

## Context

The original spec selected Chroma as the vector database based on familiarity from a prior project (youtube-agent). During Slice 9 implementation, the developer agent selected Qdrant instead.

## Decision

Accept Qdrant as the vector database for this project.

## Rationale

- Qdrant is production-grade with filtering, payload indexing, gRPC, and REST interfaces
- Qdrant has a stable, typed Python client with better long-term API stability
- Chroma's in-process mode is convenient for development but introduces operational risk at scale
- The collection-per-KB design maps cleanly to Qdrant collections

## Consequences

- **Positive:** Better production performance, richer filtering, typed client
- **Negative:** Requires a running Qdrant server (added to docker-compose and CI)
- **Migration:** All spec documents updated 2026-03-10; all code references to Chroma removed

## Infrastructure

- Local dev: `docker-compose up qdrant`
- CI: Qdrant service in GitHub Actions
- Default URL: `http://localhost:6333`

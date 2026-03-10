#!/usr/bin/env python3
"""CLI entry point for batch processor."""

import sys

from personal_knowledge_base.batch.processor import BatchProcessor


def main() -> None:
    """Run the batch processor and print a summary."""
    processor = BatchProcessor()
    result = processor.run()
    print(
        f"Batch complete: {result.succeeded} succeeded, "
        f"{result.failed} failed, {result.skipped} skipped"
    )
    sys.exit(0 if result.failed == 0 else 1)


if __name__ == "__main__":
    main()

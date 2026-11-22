"""Stable error type for embedding-contract failures."""

from __future__ import annotations


class EmbeddingError(Exception):
    """A typed, safe-to-log embedding failure.

    Messages never contain API keys, credentials, or raw provider payloads.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

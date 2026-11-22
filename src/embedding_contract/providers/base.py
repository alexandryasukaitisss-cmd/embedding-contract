"""Provider types shared by adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..identity import EmbeddingIdentity


@dataclass(frozen=True)
class ProviderHealth:
    ok: bool
    identity: EmbeddingIdentity
    details: dict[str, object]


@dataclass(frozen=True)
class EmbeddingBatch:
    identity: EmbeddingIdentity
    vectors: list[list[float]]
    usage: dict[str, object]


class EmbeddingProvider:
    """Minimal interface every adapter implements."""

    identity: EmbeddingIdentity
    batch_size: int

    def health(self) -> ProviderHealth:  # pragma: no cover - interface
        raise NotImplementedError

    def embed(self, texts: Sequence[str], *, purpose: str = "document") -> EmbeddingBatch:  # pragma: no cover
        raise NotImplementedError

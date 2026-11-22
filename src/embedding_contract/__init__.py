"""Provider-neutral embedding identity, backfill, and explicit adapters."""

from .backfill import BackfillSummary, backfill
from .errors import EmbeddingError
from .identity import (
    IDENTITY_VERSION,
    EmbeddingIdentity,
    canonical_json,
    identity_from_metadata,
)
from .providers.base import EmbeddingBatch, EmbeddingProvider, ProviderHealth
from .providers.gemini import GeminiProvider
from .providers.ollama import OllamaProvider
from .vectors import assert_dimensions, is_valid_vector, l2_normalize

__all__ = [
    "BackfillSummary",
    "EmbeddingBatch",
    "EmbeddingError",
    "EmbeddingIdentity",
    "EmbeddingProvider",
    "GeminiProvider",
    "IDENTITY_VERSION",
    "OllamaProvider",
    "ProviderHealth",
    "assert_dimensions",
    "backfill",
    "canonical_json",
    "identity_from_metadata",
    "is_valid_vector",
    "l2_normalize",
]

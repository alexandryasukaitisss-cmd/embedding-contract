"""Embedding adapters."""

from .base import EmbeddingBatch, EmbeddingProvider, ProviderHealth
from .gemini import GeminiProvider
from .ollama import OllamaProvider

__all__ = ["EmbeddingBatch", "EmbeddingProvider", "GeminiProvider", "OllamaProvider", "ProviderHealth"]

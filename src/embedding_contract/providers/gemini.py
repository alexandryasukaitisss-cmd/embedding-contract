"""Explicit Gemini API embedding adapter.

Cloud use is opt-in. This adapter never falls back to another provider, never
logs the API key, and requires HTTPS outside loopback tests.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..errors import EmbeddingError
from ..http import DIRECT_NO_REDIRECT_OPENER
from ..identity import EmbeddingIdentity
from ..vectors import assert_dimensions, l2_normalize
from .base import EmbeddingBatch, EmbeddingProvider, ProviderHealth

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_2 = "gemini-embedding-2"
GEMINI_1 = "gemini-embedding-001"
DEFAULT_CONTRACT = {
    "query_prefix": "task: search result | query: ",
    "document_prefix": "title: none | text: ",
}


class GeminiProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        model: str = GEMINI_2,
        dimensions: int = 768,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        prompt_contract: Mapping[str, str] | None = None,
        task_type: str | None = None,
        batch_size: int = 16,
        timeout: float = 30.0,
    ) -> None:
        self.model = model.strip()
        if self.model not in {GEMINI_1, GEMINI_2}:
            raise EmbeddingError("model_not_supported", "Only gemini-embedding-001 and gemini-embedding-2 are supported")
        self.dimensions = int(dimensions)
        self.base_url = self._checked_base_url(base_url)
        if prompt_contract is not None and not isinstance(prompt_contract, Mapping):
            raise EmbeddingError("invalid_config", "prompt_contract must be a mapping")
        self._api_key = (api_key if api_key is not None else os.environ.get("GEMINI_API_KEY") or "").strip()
        if any(character.isspace() or ord(character) < 0x20 for character in self._api_key):
            raise EmbeddingError("auth_invalid", "GEMINI_API_KEY contains whitespace or control characters")
        self.batch_size = int(batch_size)
        self.timeout = float(timeout)
        if not self.model:
            raise EmbeddingError("invalid_config", "model must not be empty")
        if self.dimensions < 1:
            raise EmbeddingError("invalid_config", "dimensions must be positive")
        if not 1 <= self.batch_size <= 64:
            raise EmbeddingError("invalid_config", "batch_size must be between 1 and 64")
        if self.timeout <= 0:
            raise EmbeddingError("invalid_config", "timeout must be positive")
        self._document_task = task_type or "RETRIEVAL_DOCUMENT"
        self._query_task = "RETRIEVAL_QUERY" if self._document_task == "RETRIEVAL_DOCUMENT" else self._document_task
        if self.model == GEMINI_1:
            # Query and document deliberately use different task types; both
            # belong to one index, so the pair is part of the identity.
            self.prompt_contract = MappingProxyType({
                "query_task": self._query_task,
                "document_task": self._document_task,
            })
            self.task_type = None
        elif self.model == GEMINI_2:
            self.prompt_contract = MappingProxyType(dict(prompt_contract or DEFAULT_CONTRACT))
            self.task_type = None
        else:
            self.prompt_contract = MappingProxyType(dict(prompt_contract or {}))
            self.task_type = task_type
        for prefix in self.prompt_contract.values():
            if not isinstance(prefix, str):
                raise EmbeddingError("invalid_config", "prompt contract values must be strings")

    def __repr__(self) -> str:
        return f"GeminiProvider(model={self.model!r}, dimensions={self.dimensions})"

    @staticmethod
    def _checked_base_url(value: str) -> str:
        parsed = urllib.parse.urlparse(value)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise EmbeddingError("invalid_base_url", "base URL must not contain credentials, query, or fragment")
        if parsed.scheme == "http" and parsed.hostname == "127.0.0.1":
            pass
        elif parsed.scheme != "https" or parsed.hostname != "generativelanguage.googleapis.com":
            raise EmbeddingError(
                "invalid_base_url",
                "Gemini base URL must be the official generativelanguage.googleapis.com HTTPS origin (or loopback HTTP for tests)",
            )
        path = parsed.path.rstrip("/")
        if path not in ("", "/v1beta", "/v1"):
            raise EmbeddingError("invalid_base_url", "Gemini base URL path must be /v1beta or /v1")
        return f"{parsed.scheme}://{parsed.hostname}" + (f":{parsed.port}" if parsed.port else "") + (path or "/v1beta")

    @property
    def identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity(
            provider="gemini",
            model=self.model,
            dimensions=self.dimensions,
            prompt_contract=self.prompt_contract,
            task_type=self.task_type,
        )

    def _url(self, method: str) -> str:
        return f"{self.base_url}/models/{urllib.parse.quote(self.model, safe='')}:{method}"

    def _request_json(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._api_key:
            raise EmbeddingError("auth_missing", "GEMINI_API_KEY is required")
        request = urllib.request.Request(
            self._url(method),
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json", "accept": "application/json", "x-goog-api-key": self._api_key},
            method="POST",
        )
        try:
            with DIRECT_NO_REDIRECT_OPENER.open(request, timeout=self.timeout) as response:
                body = response.read(4 * 1024 * 1024)
        except urllib.error.HTTPError as error:
            # Do not echo provider bodies: they can contain submitted text or
            # credentials in an error message.
            error.close()
            raise EmbeddingError("provider_http_error", f"Gemini returned HTTP {error.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise EmbeddingError("provider_unreachable", "Gemini is unreachable") from None
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise EmbeddingError("invalid_provider_response", "Gemini returned invalid JSON") from None
        if not isinstance(parsed, dict):
            raise EmbeddingError("invalid_provider_response", "Gemini returned a non-object response")
        return parsed

    def _payload(self, text: str, purpose: str) -> dict[str, Any]:
        if purpose not in {"query", "document"}:
            raise EmbeddingError("invalid_purpose", "purpose must be query or document")
        prefix = self.prompt_contract.get("query_prefix" if purpose == "query" else "document_prefix", "")
        config: dict[str, Any] = {"outputDimensionality": self.dimensions}
        if self.model == GEMINI_1:
            config["taskType"] = self.prompt_contract.get(
                "query_task" if purpose == "query" else "document_task"
            )
        return {
            "content": {"parts": [{"text": prefix + text}]},
            "embedContentConfig": config,
        }

    def health(self) -> ProviderHealth:
        batch = self.embed(["health probe"], purpose="document")
        return ProviderHealth(
            ok=True,
            identity=batch.identity,
            details={"provider": "gemini", "model": self.model, "dimensions": self.dimensions},
        )

    def embed(self, texts: Sequence[str], *, purpose: str = "document") -> EmbeddingBatch:
        inputs = list(texts)
        if not inputs or len(inputs) > self.batch_size:
            raise EmbeddingError("invalid_batch", f"batch must contain 1 through {self.batch_size} texts")
        if any(not isinstance(text, str) for text in inputs):
            raise EmbeddingError("invalid_input", "every input must be a string")
        requests = [self._payload(text, purpose) for text in inputs]
        if len(requests) == 1:
            payload = requests[0]
            response = self._request_json("embedContent", payload)
            embeddings = response.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != 1:
                single = response.get("embedding")
                embeddings = [single] if isinstance(single, dict) else None
        else:
            payload = {
                "requests": [
                    {"model": f"models/{self.model}", **request}
                    for request in requests
                ]
            }
            response = self._request_json("batchEmbedContents", payload)
            embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise EmbeddingError("invalid_provider_response", "Gemini returned the wrong number of embeddings")
        vectors: list[list[float]] = []
        for item in embeddings:
            values = item.get("values") if isinstance(item, dict) else None
            if not isinstance(values, list):
                raise EmbeddingError("invalid_provider_response", "Gemini embedding has no values")
            vectors.append(l2_normalize(assert_dimensions(values, self.dimensions), self.dimensions))
        return EmbeddingBatch(identity=self.identity, vectors=vectors, usage={})

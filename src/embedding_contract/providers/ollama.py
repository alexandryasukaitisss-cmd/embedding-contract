"""Local Ollama embedding adapter.

The adapter is loopback-only by default and has no cloud fallback. It pins the
model, dimensions, prompt contract, and (optionally) the model artifact digest.
"""

from __future__ import annotations

import json
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


class OllamaProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        base_url: str = "http://127.0.0.1:11434",
        prompt_contract: Mapping[str, str] | None = None,
        expected_digest: str | None = None,
        batch_size: int = 16,
        timeout: float = 30.0,
        keep_alive: str | None = None,
        allow_remote: bool = False,
    ) -> None:
        self.model = model.strip()
        self.dimensions = int(dimensions)
        self.base_url = self._checked_base_url(base_url, allow_remote)
        if prompt_contract is not None and not isinstance(prompt_contract, Mapping):
            raise EmbeddingError("invalid_config", "prompt_contract must be a mapping")
        self.prompt_contract = MappingProxyType(dict(prompt_contract or {}))
        self.expected_digest = expected_digest.lower() if expected_digest else None
        self.batch_size = int(batch_size)
        self.timeout = float(timeout)
        self.keep_alive = keep_alive
        self._artifact_digest: str | None = None
        self.verify_artifact = True
        if not self.model:
            raise EmbeddingError("invalid_config", "model must not be empty")
        if self.model.lower().endswith((":cloud", "-cloud")):
            raise EmbeddingError("model_not_supported", "cloud-tagged Ollama models are not allowed")
        if self.dimensions < 1:
            raise EmbeddingError("invalid_config", "dimensions must be positive")
        if not 1 <= self.batch_size <= 64:
            raise EmbeddingError("invalid_config", "batch_size must be between 1 and 64")
        if self.timeout <= 0:
            raise EmbeddingError("invalid_config", "timeout must be positive")
        if self.expected_digest and not _is_sha256(self.expected_digest):
            raise EmbeddingError("invalid_config", "expected_digest must be a SHA-256 hex digest")

    @staticmethod
    def _checked_base_url(value: str, allow_remote: bool) -> str:
        parsed = urllib.parse.urlparse(value)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise EmbeddingError("invalid_base_url", "base URL must not contain credentials, query, or fragment")
        if parsed.path not in ("", "/"):
            raise EmbeddingError("invalid_base_url", "base URL must not contain a path")
        if allow_remote:
            if parsed.scheme != "https" or not parsed.hostname:
                raise EmbeddingError("invalid_base_url", "remote Ollama requires an https origin")
        else:
            if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port:
                raise EmbeddingError(
                    "remote_endpoint_forbidden",
                    "local Ollama must use an uncredentialed http://127.0.0.1:<port> origin",
                )
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return f"{parsed.scheme}://{parsed.hostname}:{port}"

    @property
    def identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity(
            provider="ollama",
            model=self.model,
            dimensions=self.dimensions,
            prompt_contract=self.prompt_contract,
            artifact_digest=self._artifact_digest,
        )

    def _url(self, path: str) -> str:
        return self.base_url + path

    def _request_json(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._url(path),
            data=data,
            headers={"content-type": "application/json", "accept": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with DIRECT_NO_REDIRECT_OPENER.open(request, timeout=self.timeout) as response:
                body = response.read(4 * 1024 * 1024)
        except urllib.error.HTTPError as error:
            error.close()
            raise EmbeddingError("provider_http_error", f"Ollama returned HTTP {error.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise EmbeddingError("provider_unreachable", "Ollama is unreachable") from None
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise EmbeddingError("invalid_provider_response", "Ollama returned invalid JSON") from None
        if not isinstance(parsed, dict):
            raise EmbeddingError("invalid_provider_response", "Ollama returned a non-object response")
        return parsed

    def _inspect(self) -> str:
        payload = self._request_json("/api/tags")
        models = payload.get("models")
        if not isinstance(models, list):
            raise EmbeddingError("invalid_provider_response", "Ollama inventory has no model list")
        match = next((item for item in models if isinstance(item, dict) and item.get("name") == self.model), None)
        if match is None:
            raise EmbeddingError("model_not_found", f"Ollama does not have {self.model}")
        digest = str(match.get("digest") or "").lower()
        if digest.startswith("sha256:"):
            digest = digest[7:]
        if not _is_sha256(digest):
            raise EmbeddingError("invalid_provider_response", "Ollama model digest is not a SHA-256 value")
        if self.expected_digest and digest != self.expected_digest:
            raise EmbeddingError("model_digest_mismatch", "Ollama model digest does not match the pinned artifact")
        self._artifact_digest = digest
        return digest

    def health(self) -> ProviderHealth:
        digest = self._inspect()
        batch = self._embed_raw(["health probe"], purpose="document", identity=self.identity)
        return ProviderHealth(
            ok=True,
            identity=batch.identity,
            details={
                "provider": "ollama",
                "model": self.model,
                "artifact_digest": digest,
                "dimensions": self.dimensions,
            },
        )

    def embed(self, texts: Sequence[str], *, purpose: str = "document") -> EmbeddingBatch:
        # Re-inspect on every call when verification is enabled so a replaced
        # local blob cannot reuse the previous identity.
        if self._artifact_digest is None or self.verify_artifact:
            self._inspect()
        return self._embed_raw(texts, purpose=purpose, identity=self.identity)

    def _embed_raw(self, texts: Sequence[str], *, purpose: str, identity: EmbeddingIdentity) -> EmbeddingBatch:
        if purpose not in {"query", "document"}:
            raise EmbeddingError("invalid_purpose", "purpose must be query or document")
        inputs = list(texts)
        if not inputs or len(inputs) > self.batch_size:
            raise EmbeddingError("invalid_batch", f"batch must contain 1 through {self.batch_size} texts")
        if any(not isinstance(text, str) for text in inputs):
            raise EmbeddingError("invalid_input", "every input must be a string")
        prefix = self.prompt_contract.get("query_prefix" if purpose == "query" else "document_prefix", "")
        payload: dict[str, Any] = {
            "model": self.model,
            "input": [prefix + text for text in inputs],
            "truncate": False,
            "dimensions": self.dimensions,
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        response = self._request_json("/api/embed", payload)
        if response.get("model") != self.model:
            raise EmbeddingError("model_identity_mismatch", "Ollama response model does not match")
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise EmbeddingError("invalid_provider_response", "Ollama returned the wrong number of embeddings")
        if self.verify_artifact:
            current = self._inspect()
            if current != identity.artifact_digest:
                raise EmbeddingError("model_digest_changed", "Ollama model blob changed during the request")
        vectors = [
            l2_normalize(assert_dimensions(vector, self.dimensions), self.dimensions)
            for vector in embeddings
        ]
        return EmbeddingBatch(identity=identity, vectors=vectors, usage={})


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)

"""Provider-neutral embedding identity.

An identity binds every input that can change the vector space: provider,
model, dimensions, prompt/task contract, and an optional provider artifact
digest. Vectors with different identities must never share an index.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .errors import EmbeddingError

IDENTITY_VERSION = 1


def canonical_json(value: Any) -> str:
    """Serialize a JSON value with sorted keys and no insignificant whitespace."""

    if isinstance(value, Mapping):
        return "{" + ",".join(
            f"{json.dumps(str(key), ensure_ascii=False)}:{canonical_json(value[key])}"
            for key in sorted(value, key=str)
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _clean_text(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise EmbeddingError("invalid_identity", f"{field_name} must be a string")
    text = value.strip()
    if not text and not allow_empty:
        raise EmbeddingError("invalid_identity", f"{field_name} must not be empty")
    return text


@dataclass(frozen=True)
class EmbeddingIdentity:
    """The exact contract behind a set of vectors."""

    provider: str
    model: str
    dimensions: int
    prompt_contract: Mapping[str, str] = field(default_factory=dict)
    task_type: str | None = None
    artifact_digest: str | None = None
    version: int = IDENTITY_VERSION

    def __post_init__(self) -> None:
        provider = _clean_text(self.provider, "provider")
        model = _clean_text(self.model, "model")
        if isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int) or self.dimensions < 1:
            raise EmbeddingError("invalid_identity", "dimensions must be a positive integer")
        if not isinstance(self.version, int) or self.version != IDENTITY_VERSION:
            raise EmbeddingError("invalid_identity", f"identity version must be {IDENTITY_VERSION}")
        if not isinstance(self.prompt_contract, MappingABC):
            raise EmbeddingError("invalid_identity", "prompt_contract must be a mapping")
        contract = dict(self.prompt_contract)
        for key, value in contract.items():
            if not isinstance(key, str) or not key:
                raise EmbeddingError("invalid_identity", "prompt_contract keys must be non-empty strings")
            if not isinstance(value, str):
                raise EmbeddingError("invalid_identity", f"prompt_contract[{key!r}] must be a string")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "prompt_contract", MappingProxyType(contract))
        if self.task_type is not None:
            _clean_text(self.task_type, "task_type")
        if self.artifact_digest is not None:
            digest = _clean_text(self.artifact_digest, "artifact_digest").lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise EmbeddingError("invalid_identity", "artifact_digest must be one SHA-256 hex digest")

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "provider": self.provider.strip(),
            "model": self.model.strip(),
            "dimensions": self.dimensions,
            "prompt_contract": dict(sorted(self.prompt_contract.items())),
            "task_type": self.task_type,
            "artifact_digest": self.artifact_digest.lower() if self.artifact_digest else None,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.payload).encode("utf-8")).hexdigest()

    def __hash__(self) -> int:
        return hash(self.digest)

    def metadata(self) -> dict[str, Any]:
        """The metadata a consumer should persist with each vector."""

        return {
            "identity_version": self.version,
            "identity_digest": self.digest,
            "provider": self.provider.strip(),
            "model": self.model.strip(),
            "dimensions": self.dimensions,
            "prompt_contract": dict(sorted(self.prompt_contract.items())),
            "task_type": self.task_type,
            "artifact_digest": self.artifact_digest.lower() if self.artifact_digest else None,
        }

    def matches_metadata(self, metadata: Mapping[str, Any] | None) -> bool:
        """True only when all identity-changing fields match exactly."""

        try:
            return identity_from_metadata(metadata) == self
        except (EmbeddingError, TypeError):
            return False


def identity_from_metadata(metadata: Mapping[str, Any]) -> EmbeddingIdentity:
    """Reconstruct and strictly verify an identity from persisted metadata."""

    if not isinstance(metadata, MappingABC):
        raise EmbeddingError("invalid_identity", "identity metadata must be a mapping")
    required = (
        "identity_version",
        "identity_digest",
        "provider",
        "model",
        "dimensions",
        "prompt_contract",
        "task_type",
        "artifact_digest",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise EmbeddingError("invalid_identity", "identity metadata is missing required fields")
    version = metadata["identity_version"]
    dimensions = metadata["dimensions"]
    if type(version) is not int or version != IDENTITY_VERSION:
        raise EmbeddingError("invalid_identity", "identity_version must be an integer 1")
    if type(dimensions) is not int or dimensions < 1:
        raise EmbeddingError("invalid_identity", "dimensions must be a positive integer")
    if not isinstance(metadata["provider"], str) or not isinstance(metadata["model"], str):
        raise EmbeddingError("invalid_identity", "provider and model must be strings")
    if not isinstance(metadata["prompt_contract"], MappingABC):
        raise EmbeddingError("invalid_identity", "prompt_contract must be a mapping")
    task_type = metadata["task_type"]
    if task_type is not None and not isinstance(task_type, str):
        raise EmbeddingError("invalid_identity", "task_type must be a string or null")
    artifact_digest = metadata["artifact_digest"]
    if artifact_digest is not None and not isinstance(artifact_digest, str):
        raise EmbeddingError("invalid_identity", "artifact_digest must be a string or null")
    identity = EmbeddingIdentity(
        provider=metadata["provider"],
        model=metadata["model"],
        dimensions=dimensions,
        prompt_contract=metadata["prompt_contract"],
        task_type=task_type,
        artifact_digest=artifact_digest,
        version=version,
    )
    stored_digest = metadata["identity_digest"]
    if not isinstance(stored_digest, str) or stored_digest != identity.digest:
        raise EmbeddingError("identity_digest_mismatch", "stored identity digest does not match its fields")
    return identity

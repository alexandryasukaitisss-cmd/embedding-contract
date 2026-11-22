"""Storage-neutral backfill for records that own their vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .errors import EmbeddingError
from .identity import EmbeddingIdentity
from .vectors import assert_dimensions, is_valid_vector, l2_normalize


@dataclass
class BackfillSummary:
    total: int
    embedded: int
    skipped_existing: int
    skipped_empty: int
    replaced_invalid: int
    replaced_stale: int
    model: str
    dimensions: int
    identity_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "embedded": self.embedded,
            "skipped_existing": self.skipped_existing,
            "skipped_empty": self.skipped_empty,
            "replaced_invalid": self.replaced_invalid,
            "replaced_stale": self.replaced_stale,
            "model": self.model,
            "dimensions": self.dimensions,
            "identity_digest": self.identity_digest,
        }


class BackfillProvider(Protocol):
    @property
    def batch_size(self) -> int: ...

    def health(self) -> Any: ...

    def embed(self, texts: Sequence[str], *, purpose: str = "document") -> Any: ...


def _get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _set(record: Any, key: str, value: Any) -> None:
    if isinstance(record, dict):
        record[key] = value
    else:
        setattr(record, key, value)


def backfill(
    records: Iterable[Any],
    provider: BackfillProvider,
    *,
    get_text: Callable[[Any, int], str | None] | None = None,
    get_embedding: Callable[[Any, int], Any] | None = None,
    get_metadata: Callable[[Any, int], Mapping[str, Any] | None] | None = None,
    set_embedding: Callable[[Any, list[float], int], None] | None = None,
    set_metadata: Callable[[Any, Mapping[str, Any], int], None] | None = None,
    set_record: Callable[[Any, list[float], Mapping[str, Any], int], None] | None = None,
    batch_size: int | None = None,
    force: bool = False,
) -> BackfillSummary:
    """Embed missing, invalid, or stale vectors in bounded batches.

    A valid existing vector is preserved only when its metadata matches the
    current provider identity. The function never mutates a record before its
    replacement vector was produced.
    """

    items = list(records)
    health = provider.health()
    identity: EmbeddingIdentity = health.identity
    text_of = get_text or (lambda record, index: _get(record, "text"))
    vector_of = get_embedding or (lambda record, index: _get(record, "embedding"))
    metadata_of = get_metadata or (lambda record, index: _get(record, "embedding_metadata"))
    set_vector = set_embedding or (lambda record, vector, index: _set(record, "embedding", vector))
    set_meta = set_metadata or (lambda record, metadata, index: _set(record, "embedding_metadata", metadata))

    limit = provider.batch_size if batch_size is None else int(batch_size)
    if not 1 <= limit <= provider.batch_size:
        raise EmbeddingError("invalid_batch_size", f"batch_size must be between 1 and {provider.batch_size}")

    summary = BackfillSummary(
        total=len(items),
        embedded=0,
        skipped_existing=0,
        skipped_empty=0,
        replaced_invalid=0,
        replaced_stale=0,
        model=identity.model,
        dimensions=identity.dimensions,
        identity_digest=identity.digest,
    )

    pending: list[tuple[int, Any, str, bool, bool]] = []
    for index, record in enumerate(items):
        existing = vector_of(record, index)
        existing_metadata = metadata_of(record, index)
        valid = is_valid_vector(existing, identity.dimensions)
        matches = identity.matches_metadata(existing_metadata)
        if not force and valid and matches:
            summary.skipped_existing += 1
            continue
        text = text_of(record, index)
        if not isinstance(text, str) or not text.strip():
            # An empty record with no vector is harmless. An empty record whose
            # existing vector must be replaced is not: keep the old vector would
            # silently publish a stale or foreign space.
            if existing is not None and (force or not valid or not matches):
                raise EmbeddingError(
                    "empty_replacement_text",
                    f"record {index} needs a replacement vector but has no text",
                )
            summary.skipped_empty += 1
            continue
        pending.append((index, record, text, existing is not None and not valid, valid and not matches))

    for offset in range(0, len(pending), limit):
        batch = pending[offset : offset + limit]
        result = provider.embed([entry[2] for entry in batch], purpose="document")
        if result.identity != identity:
            raise EmbeddingError("identity_changed", "provider identity changed during backfill")
        if len(result.vectors) != len(batch):
            raise EmbeddingError("invalid_provider_response", "provider returned the wrong vector count")
        # Validate every vector before mutating any record in this batch.
        vectors = [
            l2_normalize(assert_dimensions(vector, identity.dimensions), identity.dimensions)
            for vector in result.vectors
        ]
        for (index, record, _text, invalid, stale), vector in zip(batch, vectors):
            metadata = identity.metadata()
            if set_record is not None:
                set_record(record, vector, metadata, index)
            else:
                set_vector(record, vector, index)
                set_meta(record, metadata, index)
            summary.embedded += 1
            if invalid:
                summary.replaced_invalid += 1
            if stale:
                summary.replaced_stale += 1

    return summary

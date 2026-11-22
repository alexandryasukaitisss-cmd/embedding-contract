"""Vector validation and normalization."""

from __future__ import annotations

import math
from typing import Iterable

from .errors import EmbeddingError


def assert_dimensions(values: Iterable[float], dimensions: int) -> list[float]:
    """Return a finite float vector of exactly `dimensions` values."""

    vector = list(values)
    if not isinstance(dimensions, int) or dimensions < 1:
        raise EmbeddingError("invalid_dimensions", "dimensions must be a positive integer")
    if len(vector) != dimensions:
        raise EmbeddingError(
            "dimension_mismatch",
            f"expected {dimensions} values, received {len(vector)}",
        )
    result: list[float] = []
    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingError("invalid_vector", f"value {index} is not a number")
        number = float(value)
        if not math.isfinite(number):
            raise EmbeddingError("invalid_vector", f"value {index} is not finite")
        result.append(number)
    if not any(value != 0.0 for value in result):
        raise EmbeddingError("invalid_vector", "vector has zero magnitude")
    return result


def l2_normalize(values: Iterable[float], dimensions: int | None = None) -> list[float]:
    """Validate and L2-normalize one vector.

    `dimensions` is optional for callers that already validated the width.
    """

    vector = list(values)
    if dimensions is not None:
        vector = assert_dimensions(vector, dimensions)
    elif not vector:
        raise EmbeddingError("invalid_vector", "vector is empty")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise EmbeddingError("invalid_vector", "vector has zero or non-finite magnitude")
    return [value / norm for value in vector]


def is_valid_vector(values: object, dimensions: int) -> bool:
    try:
        assert_dimensions(values, dimensions)  # type: ignore[arg-type]
        return True
    except (EmbeddingError, TypeError):
        return False

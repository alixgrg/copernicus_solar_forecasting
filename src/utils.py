"""Generic utilities shared across the project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_exists(path: str | Path, message: str | None = None) -> Path:
    """Return a Path if it exists, otherwise raise a clear FileNotFoundError."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(message or f"Path not found: {path}")
    return path


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_indices(
    indices: Iterable[int] | int | None,
    n_available: int,
    limit: int | None = None,
) -> np.ndarray:
    """Normalize index input into a 1D integer array."""
    if indices is None:
        size = n_available if limit is None else min(limit, n_available)
        return np.arange(size, dtype=int)

    if isinstance(indices, int):
        values = np.array([indices], dtype=int)
    else:
        values = np.asarray(list(indices), dtype=int)

    if values.ndim != 1:
        raise ValueError("indices must be a 1D iterable of integers.")
    if len(values) == 0:
        raise ValueError("indices cannot be empty.")
    if values.min() < 0 or values.max() >= n_available:
        raise IndexError(f"indices must be in [0, {n_available - 1}].")
    if limit is not None:
        values = values[:limit]
    return values


def describe_numeric_array(name: str, array: np.ndarray) -> dict[str, object]:
    """Return compact descriptive statistics for one numeric array."""
    values = np.asarray(array)
    return {
        "array": name,
        "shape": values.shape,
        "dtype": str(values.dtype),
        "min": float(np.nanmin(values)),
        "p25": float(np.nanpercentile(values, 25)),
        "median": float(np.nanpercentile(values, 50)),
        "p75": float(np.nanpercentile(values, 75)),
        "max": float(np.nanmax(values)),
        "nan_count": int(np.isnan(values).sum()) if np.issubdtype(values.dtype, np.number) else 0,
    }


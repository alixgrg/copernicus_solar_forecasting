"""Preprocessing helpers: quality checks, splits, and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from config import INPUT_VARIABLES, NORMALIZATION_EPS, VALIDATION_FRACTION


@dataclass(frozen=True)
class StandardizationStats:
    """Mean and standard deviation used to standardize arrays."""

    mean: float
    std: float


def quality_report(
    arrays: dict[str, np.ndarray],
    y: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute basic quality checks for X variables and optionally y."""
    rows = []
    for name, values in arrays.items():
        rows.append(_array_quality_row(name, values))
    if y is not None:
        rows.append(_array_quality_row("y", y))
    return pd.DataFrame(rows)


def _array_quality_row(name: str, values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values)
    finite_mask = np.isfinite(values) if np.issubdtype(values.dtype, np.number) else np.zeros(values.shape, dtype=bool)
    finite_values = values[finite_mask] if finite_mask.any() else np.array([])
    return {
        "array": name,
        "shape": values.shape,
        "dtype": str(values.dtype),
        "nan_count": int(np.isnan(values).sum()) if np.issubdtype(values.dtype, np.number) else 0,
        "inf_count": int(np.isinf(values).sum()) if np.issubdtype(values.dtype, np.number) else 0,
        "min": float(np.min(finite_values)) if len(finite_values) else np.nan,
        "max": float(np.max(finite_values)) if len(finite_values) else np.nan,
    }


def temporal_train_validation_split(
    n_samples: int,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a chronological train/validation split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")
    if n_samples < 2:
        raise ValueError("At least two samples are required to create a split.")

    split_at = int(np.floor(n_samples * (1 - validation_fraction)))
    split_at = min(max(split_at, 1), n_samples - 1)
    train_indices = np.arange(split_at, dtype=int)
    validation_indices = np.arange(split_at, n_samples, dtype=int)
    return train_indices, validation_indices


def fit_standardizer(
    arrays: dict[str, np.ndarray],
    variables: Iterable[str] = INPUT_VARIABLES,
    eps: float = NORMALIZATION_EPS,
) -> dict[str, StandardizationStats]:
    """Fit one global mean/std pair per variable."""
    stats = {}
    for variable in tuple(variables):
        values = np.asarray(arrays[variable], dtype="float64")
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        stats[variable] = StandardizationStats(mean=mean, std=max(std, eps))
    return stats


def transform_with_standardizer(
    arrays: dict[str, np.ndarray],
    stats: dict[str, StandardizationStats],
) -> dict[str, np.ndarray]:
    """Apply previously fitted standardization statistics."""
    transformed = {}
    for variable, values in arrays.items():
        if variable not in stats:
            transformed[variable] = values
            continue
        variable_stats = stats[variable]
        transformed[variable] = (np.asarray(values) - variable_stats.mean) / variable_stats.std
    return transformed


def standardizer_to_frame(stats: dict[str, StandardizationStats]) -> pd.DataFrame:
    """Convert standardization stats to a display-friendly DataFrame."""
    return pd.DataFrame(
        [
            {"variable": variable, "mean": value.mean, "std": value.std}
            for variable, value in stats.items()
        ]
    )


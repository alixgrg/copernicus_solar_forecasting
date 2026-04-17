"""Outils d'analyse exploratoire pour les tableaux Copernicus prétraités."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FORECAST_HORIZONS_MINUTES


def descriptive_stats(arrays: dict[str, np.ndarray], y: np.ndarray | None = None) -> pd.DataFrame:
    """Calcule des statistiques descriptives pour chaque variable."""
    rows = []
    for name, values in arrays.items():
        rows.append(_stats_row(name, values))
    if y is not None:
        rows.append(_stats_row("target", y))
    return pd.DataFrame(rows)


def _stats_row(name: str, values: np.ndarray) -> dict[str, object]:
    """Construit une ligne de statistiques descriptives pour un tableau."""
    values = np.asarray(values)
    return {
        "variable": name,
        "shape": values.shape,
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
        "median": float(np.nanmedian(values)),
        "max": float(np.nanmax(values)),
    }


def target_horizon_stats(y: np.ndarray) -> pd.DataFrame:
    """Calcule les statistiques de la cible pour chaque horizon de prévision."""
    y = np.asarray(y)
    rows = []
    for horizon_index, horizon in enumerate(FORECAST_HORIZONS_MINUTES):
        values = y[:, horizon_index]
        rows.append(
            {
                "horizon_min": horizon,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "min": float(np.nanmin(values)),
                "median": float(np.nanmedian(values)),
                "max": float(np.nanmax(values)),
            }
        )
    return pd.DataFrame(rows)


def temporal_ghi_summary(ghi: np.ndarray, label: str = "GHI") -> pd.DataFrame:
    """Résume les niveaux de GHI sur les pas temporels."""
    ghi = np.asarray(ghi)
    rows = []
    for time_index in range(ghi.shape[1]):
        values = ghi[:, time_index]
        rows.append(
            {
                "variable": label,
                "time_index": time_index,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
        )
    return pd.DataFrame(rows)


def sample_level_means(
    arrays: dict[str, np.ndarray],
    y: np.ndarray | None = None,
) -> pd.DataFrame:
    """Agrège chaque échantillon en une moyenne par variable."""
    data = {}
    for name, values in arrays.items():
        values = np.asarray(values)
        data[f"{name}_mean"] = values.mean(axis=tuple(range(1, values.ndim)))
    if y is not None:
        y = np.asarray(y)
        data["target_mean"] = y.mean(axis=tuple(range(1, y.ndim)))
    return pd.DataFrame(data)

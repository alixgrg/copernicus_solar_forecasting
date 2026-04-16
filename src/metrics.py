"""Metrics for Copernicus solar forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FORECAST_HORIZONS_MINUTES


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Global RMSE on a full prediction tensor."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Global MAE on a full prediction tensor."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_forecasts(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return global metrics on a forecast tensor."""
    return pd.DataFrame(
        {
            "metric": ["RMSE", "MAE"],
            "value": [rmse(y_true, y_pred), mae(y_true, y_pred)],
        }
    )


def evaluate_by_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: list[int] | tuple[int, ...] = FORECAST_HORIZONS_MINUTES,
) -> pd.DataFrame:
    """Return RMSE and MAE for each forecast horizon."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    rows = []
    for h_idx, horizon in enumerate(horizons):
        rows.append(
            {
                "horizon_min": horizon,
                "RMSE": rmse(y_true[:, h_idx], y_pred[:, h_idx]),
                "MAE": mae(y_true[:, h_idx], y_pred[:, h_idx]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_spatial_means(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: list[int] | tuple[int, ...] = FORECAST_HORIZONS_MINUTES,
) -> pd.DataFrame:
    """
    Evaluate forecasts on spatial means only.

    Useful to distinguish:
    - error on the mean irradiance level
    - error on the fine spatial structure
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    y_true_mean = y_true.mean(axis=(2, 3))
    y_pred_mean = y_pred.mean(axis=(2, 3))

    rows = []
    for h_idx, horizon in enumerate(horizons):
        rows.append(
            {
                "horizon_min": horizon,
                "RMSE_spatial_mean": rmse(y_true_mean[:, h_idx], y_pred_mean[:, h_idx]),
                "MAE_spatial_mean": mae(y_true_mean[:, h_idx], y_pred_mean[:, h_idx]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_model_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    horizons: list[int] | tuple[int, ...] = FORECAST_HORIZONS_MINUTES,
) -> dict[str, pd.DataFrame]:
    """Convenience wrapper returning all evaluation tables for one model."""
    return {
        "global": evaluate_forecasts(y_true, y_pred).assign(model=model_name),
        "by_horizon": evaluate_by_horizon(y_true, y_pred, horizons=horizons).assign(model=model_name),
        "spatial_means": evaluate_spatial_means(y_true, y_pred, horizons=horizons).assign(model=model_name),
    }


def compare_global_scores(results: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Aggregate global scores from several model result bundles.

    Expected format:
        {
            "model_a": {"global": ..., "by_horizon": ..., ...},
            "model_b": {"global": ..., "by_horizon": ..., ...},
        }
    """
    rows = []
    for model_name, bundle in results.items():
        global_df = bundle["global"].copy()
        row = {"model": model_name}
        for _, record in global_df.iterrows():
            row[str(record["metric"])] = float(record["value"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
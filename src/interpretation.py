"""Model interpretation helpers for tabular solar forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


MetricFn = Callable[[np.ndarray, np.ndarray], float]


def rmse_flat(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE for 2D target arrays."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@dataclass(frozen=True)
class PermutationImportanceResult:
    """Container for permutation importance results."""

    importance: pd.DataFrame
    baseline_score: float


def permutation_importance_multioutput(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    metric_fn: MetricFn = rmse_flat,
    n_repeats: int = 5,
    random_state: int = 42,
    max_features: int | None = None,
) -> PermutationImportanceResult:
    """
    Compute permutation importance for any model exposing predict(X).

    Importance is measured as the increase in the supplied error metric after
    shuffling one feature. Higher values mean the feature is more useful.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape {X.shape}.")
    if len(feature_names) != X.shape[1]:
        raise ValueError("feature_names length must match X.shape[1].")

    rng = np.random.default_rng(random_state)
    baseline_pred = model.predict(X)
    baseline_score = metric_fn(y, baseline_pred)

    feature_indices = np.arange(X.shape[1])
    if max_features is not None and max_features < len(feature_indices):
        variances = np.nanvar(X, axis=0)
        feature_indices = np.argsort(variances)[-max_features:]

    rows = []
    for feature_idx in feature_indices:
        deltas = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, feature_idx] = rng.permutation(X_perm[:, feature_idx])
            perm_score = metric_fn(y, model.predict(X_perm))
            deltas.append(perm_score - baseline_score)
        rows.append(
            {
                "feature": feature_names[feature_idx],
                "importance_mean": float(np.mean(deltas)),
                "importance_std": float(np.std(deltas)),
                "baseline_score": baseline_score,
            }
        )

    importance = (
        pd.DataFrame(rows)
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    return PermutationImportanceResult(importance=importance, baseline_score=baseline_score)


def model_feature_importances(model, feature_names: list[str]) -> pd.DataFrame:
    """
    Extract native feature importances or coefficients when available.

    MultiOutputRegressor estimators are averaged across outputs.
    """
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        estimators = [model]

    values = []
    for estimator in estimators:
        if hasattr(estimator, "feature_importances_"):
            values.append(np.asarray(estimator.feature_importances_, dtype=np.float64))
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_, dtype=np.float64)
            values.append(np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef))

    if not values:
        return pd.DataFrame(columns=["feature", "importance"])

    importance = np.mean(np.vstack(values), axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def compute_tree_shap_values(
    model,
    X: np.ndarray,
    feature_names: list[str],
    output_index: int = 0,
    max_samples: int = 100,
):
    """
    Compute SHAP values for one output of a tree model if shap is installed.

    Returns a tuple (shap_values, X_sample, feature_names). The import is local
    so the rest of the project works without the optional dependency.
    """
    try:
        import shap  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The optional dependency 'shap' is not installed. "
            "Use permutation_importance_multioutput as a fallback."
        ) from exc

    estimators = getattr(model, "estimators_", None)
    estimator = estimators[output_index] if estimators is not None else model
    X_sample = np.asarray(X)[:max_samples]
    explainer = shap.Explainer(estimator, X_sample, feature_names=feature_names)
    shap_values = explainer(X_sample)
    return shap_values, X_sample, feature_names


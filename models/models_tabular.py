from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor


def flatten_target(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 4:
        raise ValueError(f"Expected y with 4 dimensions, got shape {y.shape}.")
    return y.reshape(y.shape[0], -1)


def unflatten_target(y_flat: np.ndarray, horizon_count: int = 4, roi_size: int = 51) -> np.ndarray:
    y_flat = np.asarray(y_flat)
    expected = horizon_count * roi_size * roi_size
    if y_flat.ndim != 2 or y_flat.shape[1] != expected:
        raise ValueError(f"Expected flat target with second dimension {expected}, got {y_flat.shape}.")
    return y_flat.reshape(y_flat.shape[0], horizon_count, roi_size, roi_size)


def prepare_tabular_inputs(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    fillna_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    common_cols = [col for col in X_train.columns if col in X_val.columns]
    X_train_np = X_train[common_cols].fillna(fillna_value).to_numpy(dtype=np.float32)
    X_val_np = X_val[common_cols].fillna(fillna_value).to_numpy(dtype=np.float32)
    return X_train_np, X_val_np, common_cols


def fit_ridge_multioutput(
    X_train: np.ndarray,
    y_train_flat: np.ndarray,
    alpha: float = 1.0,
    random_state: int = 42,
):
    model = Ridge(alpha=alpha, random_state=random_state)
    model.fit(X_train, y_train_flat)
    return model


def fit_elasticnet_multioutput(
    X_train: np.ndarray,
    y_train_flat: np.ndarray,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
    random_state: int = 42,
    max_iter: int = 3000,
):
    base = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state,
        max_iter=max_iter,
    )
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train_flat)
    return model


def fit_random_forest_multioutput(
    X_train: np.ndarray,
    y_train_flat: np.ndarray,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    n_jobs: int = -1,
    random_state: int = 42,
):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(X_train, y_train_flat)
    return model


def fit_extra_trees_multioutput(
    X_train: np.ndarray,
    y_train_flat: np.ndarray,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
    n_jobs: int = -1,
    random_state: int = 42,
):
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(X_train, y_train_flat)
    return model


def fit_hist_gb_multioutput(
    X_train: np.ndarray,
    y_train_flat: np.ndarray,
    learning_rate: float = 0.05,
    max_iter: int = 200,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.0,
    early_stopping: bool = True,
    validation_fraction: float = 0.2,
    random_state: int = 42,
):
    base = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train_flat)
    return model


def predict_tensor(model, X: np.ndarray, horizon_count: int = 4, roi_size: int = 51) -> np.ndarray:
    y_pred_flat = model.predict(X)
    return unflatten_target(y_pred_flat, horizon_count=horizon_count, roi_size=roi_size)
"""Baselines for Copernicus solar forecasting."""

from __future__ import annotations

import numpy as np

from src.data_loading import extract_roi


def _ensure_roi_spatial(array: np.ndarray) -> np.ndarray:
    """
    Ensure an array is spatially cropped to the 51x51 RoI.

    Supported shapes:
        (n, H, W)
        (n, t, H, W)

    If already in RoI shape, return unchanged.
    """
    array = np.asarray(array)

    if array.ndim == 3:
        # (n, H, W)
        if array.shape[1:] == (51, 51):
            return array
        return extract_roi(array)

    if array.ndim == 4:
        # (n, t, H, W)
        if array.shape[2:] == (51, 51):
            return array
        return extract_roi(array)

    raise ValueError(f"Unsupported shape for ROI handling: {array.shape}")


def persistence_last_ghi_baseline(arrays: dict[str, np.ndarray], n_horizons: int = 4) -> np.ndarray:
    """
    Persistence baseline on raw GHI.

    Predict each future horizon as the last observed GHI frame.

    Expected:
        arrays["GHI"] shape = (n, 4, H, W)

    Returns:
        y_pred shape = (n, n_horizons, 51, 51)
    """
    ghi = np.asarray(arrays["GHI"], dtype=np.float32)
    last_ghi = ghi[:, -1]  # (n, H, W)
    last_ghi_roi = _ensure_roi_spatial(last_ghi)  # (n, 51, 51)
    y_pred = np.repeat(last_ghi_roi[:, np.newaxis, :, :], repeats=n_horizons, axis=1)
    return y_pred.astype(np.float32)


def persistence_csi_baseline(
    arrays_raw: dict[str, np.ndarray],
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Persistence baseline on the clear-sky index.

    Principle:
        CSI_hat(t+h) = CSI(t)
        GHI_hat(t+h) = CSI_hat(t+h) * CLS(t+h)

    Expected:
        arrays_raw["GHI"] shape = (n, 4, H, W)
        arrays_raw["CLS"] shape = (n, 8, H, W)

    Returns:
        y_pred shape = (n, 4, 51, 51)
    """
    ghi = np.asarray(arrays_raw["GHI"], dtype=np.float32)   # (n, 4, H, W)
    cls = np.asarray(arrays_raw["CLS"], dtype=np.float32)   # (n, 8, H, W)

    cls_past = cls[:, : ghi.shape[1]]      # (n, 4, H, W)
    cls_future = cls[:, ghi.shape[1] :]    # (n, 4, H, W)

    csi = ghi / np.maximum(cls_past, eps)
    last_csi = csi[:, -1]                  # (n, H, W)
    csi_hat = np.repeat(last_csi[:, np.newaxis, :, :], repeats=cls_future.shape[1], axis=1)

    ghi_hat_future = csi_hat * cls_future  # (n, 4, H, W)
    y_pred = _ensure_roi_spatial(ghi_hat_future)  # (n, 4, 51, 51)
    return y_pred.astype(np.float32)


def mean_image_baseline(y_train: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Predict the train mean image for all validation examples.

    Useful as a very weak reference.
    """
    y_train = np.asarray(y_train, dtype=np.float32)
    mean_target = y_train.mean(axis=0)  # (4, 51, 51)
    y_pred = np.repeat(mean_target[np.newaxis, :, :, :], repeats=n_samples, axis=0)
    return y_pred.astype(np.float32)
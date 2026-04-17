"""Modèles de référence pour la prévision solaire Copernicus."""

from __future__ import annotations

import numpy as np

from src.data_loading import extract_roi


def _ensure_roi_spatial(array: np.ndarray) -> np.ndarray:
    """
    Garantit qu'un tableau est ramené à la région d'intérêt 51 par 51.

    Formes prises en charge:
        (n, H, W)
        (n, t, H, W)

    Si le tableau est déjà à la bonne taille, il est renvoyé tel quel.
    """
    array = np.asarray(array)

    if array.ndim == 3:
        # Format avec une seule carte par échantillon.
        if array.shape[1:] == (51, 51):
            return array
        return extract_roi(array)

    if array.ndim == 4:
        # Format avec plusieurs pas temporels par échantillon.
        if array.shape[2:] == (51, 51):
            return array
        return extract_roi(array)

    raise ValueError(f"Unsupported shape for ROI handling: {array.shape}")


def persistence_last_ghi_baseline(arrays: dict[str, np.ndarray], n_horizons: int = 4) -> np.ndarray:
    """
    Calcule un modèle de référence de persistance sur le GHI brut.

    Chaque horizon futur reçoit la dernière image GHI observée.

    Entrée attendue:
        arrays["GHI"] a pour forme (n, 4, H, W)

    Sortie:
        y_pred a pour forme (n, n_horizons, 51, 51)
    """
    ghi = np.asarray(arrays["GHI"], dtype=np.float32)
    last_ghi = ghi[:, -1]
    last_ghi_roi = _ensure_roi_spatial(last_ghi)
    y_pred = np.repeat(last_ghi_roi[:, np.newaxis, :, :], repeats=n_horizons, axis=1)
    return y_pred.astype(np.float32)


def persistence_csi_baseline(
    arrays_raw: dict[str, np.ndarray],
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Calcule un modèle de référence de persistance sur l'indice de ciel clair.

    Principe:
        CSI_hat(t+h) = CSI(t)
        GHI_hat(t+h) = CSI_hat(t+h) * CLS(t+h)

    Entrées attendues:
        arrays_raw["GHI"] a pour forme (n, 4, H, W)
        arrays_raw["CLS"] a pour forme (n, 8, H, W)

    Sortie:
        y_pred a pour forme (n, 4, 51, 51)
    """
    ghi = np.asarray(arrays_raw["GHI"], dtype=np.float32)
    cls = np.asarray(arrays_raw["CLS"], dtype=np.float32)

    cls_past = cls[:, : ghi.shape[1]]
    cls_future = cls[:, ghi.shape[1] :]

    csi = ghi / np.maximum(cls_past, eps)
    last_csi = csi[:, -1]
    csi_hat = np.repeat(last_csi[:, np.newaxis, :, :], repeats=cls_future.shape[1], axis=1)

    ghi_hat_future = csi_hat * cls_future
    y_pred = _ensure_roi_spatial(ghi_hat_future)
    return y_pred.astype(np.float32)


def mean_image_baseline(y_train: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Prédit l'image moyenne d'entraînement pour tous les exemples.

    Ce modèle sert de référence volontairement simple.
    """
    y_train = np.asarray(y_train, dtype=np.float32)
    mean_target = y_train.mean(axis=0)
    y_pred = np.repeat(mean_target[np.newaxis, :, :, :], repeats=n_samples, axis=0)
    return y_pred.astype(np.float32)

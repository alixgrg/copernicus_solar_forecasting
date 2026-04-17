"""Feature construction for Copernicus solar forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_physical_inputs(
    arrays: dict[str, np.ndarray],
    eps: float = 1e-6,
    keep_raw_ghi: bool = False,
    encode_angles: bool = True,
) -> dict[str, np.ndarray]:
    """
    Build physically-informed representations from processed arrays.

    Expected shapes:
        GHI: (n, 4, H, W)
        CLS: (n, 8, H, W)
        SZA: (n, 8, H, W)
        SAA: (n, 8, H, W)

    Returns a dictionary containing:
        - CSI: clear-sky index on past frames
        - CLS
        - SZA / SAA or their sin/cos encodings
        - optionally raw GHI
    """
    out: dict[str, np.ndarray] = {}

    ghi = np.asarray(arrays["GHI"], dtype=np.float32)
    cls = np.asarray(arrays["CLS"], dtype=np.float32)
    sza = np.asarray(arrays["SZA"], dtype=np.float32)
    saa = np.asarray(arrays["SAA"], dtype=np.float32)

    cls_past = cls[:, : ghi.shape[1], :, :]
    csi = ghi / np.maximum(cls_past, eps)

    out["CSI"] = csi
    out["CLS"] = cls

    if keep_raw_ghi:
        out["GHI"] = ghi

    if encode_angles:
        # Heuristic:
        # - if values look like degrees, convert to radians
        # - otherwise assume already in radians
        sza_rad = np.deg2rad(sza) if np.nanmax(np.abs(sza)) > 2 * np.pi + 1 else sza
        saa_rad = np.deg2rad(saa) if np.nanmax(np.abs(saa)) > 2 * np.pi + 1 else saa

        out["SZA_sin"] = np.sin(sza_rad).astype(np.float32)
        out["SZA_cos"] = np.cos(sza_rad).astype(np.float32)
        out["SAA_sin"] = np.sin(saa_rad).astype(np.float32)
        out["SAA_cos"] = np.cos(saa_rad).astype(np.float32)
    else:
        out["SZA"] = sza
        out["SAA"] = saa

    return out


def build_spatial_feature_tensor(
    arrays: dict[str, np.ndarray],
    include_csi: bool = True,
    include_cls: bool = True,
    include_angles: bool = True,
    include_csi_deltas: bool = True,
    include_csi_anomaly: bool = True,
    include_raw_ghi: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a channel-first tensor for CNN/spatial models.

    Input arrays are assumed to be time-first:
        (n_samples, time, height, width)

    Output:
        feature tensor of shape (n_samples, channels, height, width)
    """
    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    if include_raw_ghi and "GHI" in arrays:
        ghi = np.asarray(arrays["GHI"])
        feature_blocks.append(ghi)
        feature_names.extend([f"GHI_t{i}" for i in range(ghi.shape[1])])

    if include_csi and "CSI" in arrays:
        csi = np.asarray(arrays["CSI"])
        feature_blocks.append(csi)
        feature_names.extend([f"CSI_t{i}" for i in range(csi.shape[1])])

    if include_cls and "CLS" in arrays:
        cls = np.asarray(arrays["CLS"])
        feature_blocks.append(cls)
        feature_names.extend([f"CLS_t{i}" for i in range(cls.shape[1])])

    if include_angles:
        for name in ("SZA_sin", "SZA_cos", "SAA_sin", "SAA_cos", "SZA", "SAA"):
            if name in arrays:
                values = np.asarray(arrays[name])
                feature_blocks.append(values)
                feature_names.extend([f"{name}_t{i}" for i in range(values.shape[1])])

    if include_csi_deltas and "CSI" in arrays:
        csi = np.asarray(arrays["CSI"])
        csi_deltas = np.diff(csi, axis=1)
        feature_blocks.append(csi_deltas)
        feature_names.extend([f"CSI_delta_t{i}_to_t{i+1}" for i in range(csi_deltas.shape[1])])

    if include_csi_anomaly and "CSI" in arrays:
        csi = np.asarray(arrays["CSI"])
        csi_anomaly = csi - 1.0
        feature_blocks.append(csi_anomaly)
        feature_names.extend([f"CSI_minus_1_t{i}" for i in range(csi_anomaly.shape[1])])

    if not feature_blocks:
        raise ValueError("No spatial features were built.")

    tensor = np.concatenate(feature_blocks, axis=1)
    return tensor, feature_names


def build_tabular_features(
    arrays: dict[str, np.ndarray],
    add_quantiles: bool = True,
    add_quadrants: bool = True,
) -> pd.DataFrame:
    """
    Build tabular features for classical ML / clustering / XAI.

    Arrays are expected to be time-first:
        (n_samples, time, H, W)
    """
    features: dict[str, np.ndarray] = {}

    def _add_summary_block(prefix: str, values: np.ndarray) -> None:
        # Per-time-step mean and std
        mean_by_t = values.mean(axis=(2, 3))
        std_by_t = values.std(axis=(2, 3))
        min_by_t = values.min(axis=(2, 3))
        max_by_t = values.max(axis=(2, 3))

        for t in range(values.shape[1]):
            features[f"{prefix}_mean_t{t}"] = mean_by_t[:, t]
            features[f"{prefix}_std_t{t}"] = std_by_t[:, t]
            features[f"{prefix}_min_t{t}"] = min_by_t[:, t]
            features[f"{prefix}_max_t{t}"] = max_by_t[:, t]

        # Global summaries
        features[f"{prefix}_mean_global"] = values.mean(axis=(1, 2, 3))
        features[f"{prefix}_std_global"] = values.std(axis=(1, 2, 3))

        # Temporal trend on spatial mean
        if values.shape[1] >= 2:
            features[f"{prefix}_trend_last_minus_first"] = mean_by_t[:, -1] - mean_by_t[:, 0]

        if add_quantiles:
            flat = values.reshape(values.shape[0], values.shape[1], -1)
            q25 = np.quantile(flat, 0.25, axis=2)
            q50 = np.quantile(flat, 0.50, axis=2)
            q75 = np.quantile(flat, 0.75, axis=2)
            for t in range(values.shape[1]):
                features[f"{prefix}_q25_t{t}"] = q25[:, t]
                features[f"{prefix}_q50_t{t}"] = q50[:, t]
                features[f"{prefix}_q75_t{t}"] = q75[:, t]

        if add_quadrants:
            h, w = values.shape[2], values.shape[3]
            h2, w2 = h // 2, w // 2
            quadrants = {
                "q1": values[:, :, :h2, :w2],
                "q2": values[:, :, :h2, w2:],
                "q3": values[:, :, h2:, :w2],
                "q4": values[:, :, h2:, w2:],
            }
            for qname, qvalues in quadrants.items():
                qmean = qvalues.mean(axis=(2, 3))
                for t in range(values.shape[1]):
                    features[f"{prefix}_{qname}_mean_t{t}"] = qmean[:, t]

    for name, values in arrays.items():
        values = np.asarray(values)
        if values.ndim != 4:
            continue
        _add_summary_block(name, values)

    return pd.DataFrame(features)


def feature_summary(feature_tensor: np.ndarray, feature_names: list[str]) -> dict[str, object]:
    """Compact summary of a feature tensor."""
    return {
        "shape": feature_tensor.shape,
        "n_features": int(feature_tensor.shape[1]),
        "first_features": feature_names[:15],
    }


def build_advanced_features(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Crée des features agrégées pour booster les modèles tabulaires.
    """
    features = {}
    ghi = arrays["GHI"]  # (n, 4, 51, 51)
    cls = arrays["CLS"]  # (n, 8, 51, 51)
    
    # 1. Statistiques Globales par frame (Moyenne/Std de l'image)
    for t in range(4):
        features[f"ghi_mean_t-{45-t*15}"] = ghi[:, t].mean(axis=(1, 2))
        features[f"ghi_std_t-{45-t*15}"] = ghi[:, t].std(axis=(1, 2))

    # 2. Dynamique (Différence temporelle) : Les nuages bougent-ils ?
    # Différence entre la frame actuelle (t) et la précédente (t-1)
    ghi_diff = ghi[:, 1:] - ghi[:, :-1]
    for t in range(3):
        features[f"ghi_diff_mean_t{t}"] = ghi_diff[:, t].mean(axis=(1, 2))

    # 3. Clear Sky Index (CSI) moyen 
    # Le CSI est plus stable que le GHI pour un modèle tabulaire
    cls_past = cls[:, :4]
    csi = ghi / (cls_past + 1e-6)
    features["csi_last_frame_mean"] = csi[:, -1].mean(axis=(1, 2))
    
    # 4. Position du soleil (SZA moyen sur la zone)
    sza_future = arrays["SZA"][:, 4:] # Les 4 horizons futurs
    for t in range(4):
        features[f"sza_future_t+{15+t*15}"] = sza_future[:, t].mean(axis=(1, 2))

    return pd.DataFrame(features)
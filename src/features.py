"""Construction des variables pour la prévision solaire Copernicus."""

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
    Construit des représentations inspirées de la physique à partir des tableaux prétraités.

    Formes attendues:
        GHI: (n, 4, H, W)
        CLS: (n, 8, H, W)
        SZA: (n, 8, H, W)
        SAA: (n, 8, H, W)

    Renvoie un dictionnaire contenant:
        - CSI: indice de ciel clair sur les images passées
        - CLS
        - SZA et SAA ou leurs encodages sinus et cosinus
        - GHI brut si demandé
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
        # Les valeurs qui ressemblent à des degrés sont converties en radians.
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
    Construit un tenseur avec les canaux en premier pour les modèles spatiaux.

    Les tableaux d'entrée sont supposés avoir le temps en première dimension:
        (n_samples, time, height, width)

    Sortie:
        tenseur de variables de forme (n_samples, channels, height, width)
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
    Construit des variables tabulaires pour le machine learning classique.

    Les tableaux sont attendus avec le temps en première dimension:
        (n_samples, time, H, W)
    """
    features: dict[str, np.ndarray] = {}

    def _add_summary_block(prefix: str, values: np.ndarray) -> None:
        """Ajoute les statistiques agrégées d'une variable au dictionnaire de sortie."""
        # Statistiques par pas temporel.
        mean_by_t = values.mean(axis=(2, 3))
        std_by_t = values.std(axis=(2, 3))
        min_by_t = values.min(axis=(2, 3))
        max_by_t = values.max(axis=(2, 3))

        for t in range(values.shape[1]):
            features[f"{prefix}_mean_t{t}"] = mean_by_t[:, t]
            features[f"{prefix}_std_t{t}"] = std_by_t[:, t]
            features[f"{prefix}_min_t{t}"] = min_by_t[:, t]
            features[f"{prefix}_max_t{t}"] = max_by_t[:, t]

        # Résumés globaux.
        features[f"{prefix}_mean_global"] = values.mean(axis=(1, 2, 3))
        features[f"{prefix}_std_global"] = values.std(axis=(1, 2, 3))

        # Tendance temporelle de la moyenne spatiale.
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
    """Résume brièvement un tenseur de variables."""
    return {
        "shape": feature_tensor.shape,
        "n_features": int(feature_tensor.shape[1]),
        "first_features": feature_names[:15],
    }


def build_advanced_features(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Crée des variables agrégées pour renforcer les modèles tabulaires.
    """
    features = {}
    ghi = arrays["GHI"]
    cls = arrays["CLS"]
    
    # Statistiques globales par image.
    for t in range(4):
        features[f"ghi_mean_t-{45-t*15}"] = ghi[:, t].mean(axis=(1, 2))
        features[f"ghi_std_t-{45-t*15}"] = ghi[:, t].std(axis=(1, 2))

    # Dynamique temporelle entre les images successives.
    ghi_diff = ghi[:, 1:] - ghi[:, :-1]
    for t in range(3):
        features[f"ghi_diff_mean_t{t}"] = ghi_diff[:, t].mean(axis=(1, 2))

    # CSI moyen, plus stable que le GHI pour un modèle tabulaire.
    cls_past = cls[:, :4]
    csi = ghi / (cls_past + 1e-6)
    features["csi_last_frame_mean"] = csi[:, -1].mean(axis=(1, 2))
    
    # Position future du soleil sur les quatre horizons.
    sza_future = arrays["SZA"][:, 4:]
    for t in range(4):
        features[f"sza_future_t+{15+t*15}"] = sza_future[:, t].mean(axis=(1, 2))

    return pd.DataFrame(features)


def build_exogenous_features(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Construit des variables exogènes optionnelles, comme le vent si disponible.

    Les fichiers Copernicus publics utilisés ici exposent seulement GHI, CLS, SZA
    et SAA. Cette fonction prépare le notebook à des variantes de données plus
    riches sans échouer lorsque les champs de vent U et V sont absents.
    """
    wind_aliases = {
        "U": ("U", "u", "WIND_U", "wind_u", "U10", "u10"),
        "V": ("V", "v", "WIND_V", "wind_v", "V10", "v10"),
    }
    features: dict[str, np.ndarray] = {}

    selected = {}
    for canonical, aliases in wind_aliases.items():
        for alias in aliases:
            if alias in arrays:
                selected[canonical] = np.asarray(arrays[alias], dtype=np.float32)
                break

    for name, values in selected.items():
        if values.ndim == 4:
            mean_by_t = values.mean(axis=(2, 3))
            std_by_t = values.std(axis=(2, 3))
            for t in range(values.shape[1]):
                features[f"wind_{name.lower()}_mean_t{t}"] = mean_by_t[:, t]
                features[f"wind_{name.lower()}_std_t{t}"] = std_by_t[:, t]
            features[f"wind_{name.lower()}_mean_global"] = values.mean(axis=(1, 2, 3))
            features[f"wind_{name.lower()}_std_global"] = values.std(axis=(1, 2, 3))
        elif values.ndim == 2:
            for t in range(values.shape[1]):
                features[f"wind_{name.lower()}_t{t}"] = values[:, t]
            features[f"wind_{name.lower()}_mean_global"] = values.mean(axis=1)
        elif values.ndim == 1:
            features[f"wind_{name.lower()}"] = values

    if "U" in selected and "V" in selected:
        u = selected["U"]
        v = selected["V"]
        if u.shape == v.shape:
            speed = np.sqrt(u**2 + v**2)
            if speed.ndim == 4:
                speed_mean = speed.mean(axis=(2, 3))
                for t in range(speed.shape[1]):
                    features[f"wind_speed_mean_t{t}"] = speed_mean[:, t]
                direction = np.arctan2(v.mean(axis=(1, 2, 3)), u.mean(axis=(1, 2, 3)))
            elif speed.ndim == 2:
                for t in range(speed.shape[1]):
                    features[f"wind_speed_t{t}"] = speed[:, t]
                direction = np.arctan2(v.mean(axis=1), u.mean(axis=1))
            else:
                direction = np.arctan2(v, u)
            features["wind_direction_sin"] = np.sin(direction)
            features["wind_direction_cos"] = np.cos(direction)

    return pd.DataFrame(features)


def _quadrant_slices(h: int, w: int, n_rows: int = 2, n_cols: int = 2):
    """Construit les découpes spatiales correspondant à une grille de quadrants."""
    h_edges = np.linspace(0, h, n_rows + 1, dtype=int)
    w_edges = np.linspace(0, w, n_cols + 1, dtype=int)
    return [
        (slice(h_edges[i], h_edges[i + 1]), slice(w_edges[j], w_edges[j + 1]))
        for i in range(n_rows)
        for j in range(n_cols)
    ]


def _weighted_center_of_mass(batch_maps: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule les centres de masse normalisés d'un lot de cartes.

    batch_maps a pour forme (n, h, w). La sortie contient cy et cx normalisés
    dans l'intervalle [0, 1].
    """
    n, h, w = batch_maps.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    mass = np.maximum(batch_maps, 0.0)
    total = mass.sum(axis=(1, 2)) + eps

    cy = (mass * yy[None, :, :]).sum(axis=(1, 2)) / total
    cx = (mass * xx[None, :, :]).sum(axis=(1, 2)) / total

    return cy / max(h - 1, 1), cx / max(w - 1, 1)


def _gradient_magnitude(batch_maps: np.ndarray) -> np.ndarray:
    """
    Calcule la magnitude du gradient pour un lot de cartes de forme (n, h, w).
    """
    gy = np.diff(batch_maps, axis=1, append=batch_maps[:, -1:, :])
    gx = np.diff(batch_maps, axis=2, append=batch_maps[:, :, -1:])
    return np.sqrt(gx**2 + gy**2)


def _safe_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Calcule une corrélation robuste avec une petite stabilisation numérique."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum()) * np.sqrt((b**2).sum()) + eps
    return float((a * b).sum() / denom)


def _best_shift_single(frame_prev: np.ndarray, frame_curr: np.ndarray, max_shift: int = 4):
    """
    Estime un pseudo vecteur de mouvement (dy, dx) en maximisant la corrélation
    sur de petits décalages entiers.
    """
    h, w = frame_prev.shape
    best_score = -np.inf
    best_dy, best_dx = 0, 0

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            y0_prev = max(0, -dy)
            y1_prev = min(h, h - dy)
            x0_prev = max(0, -dx)
            x1_prev = min(w, w - dx)

            y0_curr = max(0, dy)
            y1_curr = min(h, h + dy)
            x0_curr = max(0, dx)
            x1_curr = min(w, w + dx)

            prev_crop = frame_prev[y0_prev:y1_prev, x0_prev:x1_prev]
            curr_crop = frame_curr[y0_curr:y1_curr, x0_curr:x1_curr]

            if prev_crop.size == 0 or curr_crop.size == 0:
                continue

            score = _safe_corr(prev_crop, curr_crop)
            if score > best_score:
                best_score = score
                best_dy, best_dx = dy, dx

    return best_dy, best_dx, best_score


def build_spatial_dynamics_features(
    arrays: dict[str, np.ndarray],
    eps: float = 1e-6,
    max_shift: int = 4,
    downsample: int = 2,
) -> pd.DataFrame:
    """
    Construit des variables centrées sur la dynamique nuageuse et la structure locale.
    """
    ghi = np.asarray(arrays["GHI"], dtype=np.float32)
    cls = np.asarray(arrays["CLS"], dtype=np.float32)[:, :4]
    csi = ghi / np.maximum(cls, eps)

    n, t, h, w = ghi.shape
    features: dict[str, np.ndarray] = {}

    # Différences temporelles par quadrant entre le dernier pas et le précédent.
    quad_slices = _quadrant_slices(h, w, n_rows=2, n_cols=2)
    ghi_last = ghi[:, -1]
    ghi_prev = ghi[:, -2]
    csi_last = csi[:, -1]
    csi_prev = csi[:, -2]

    for q_idx, (hs, ws) in enumerate(quad_slices):
        ghi_diff_q = ghi_last[:, hs, ws].mean(axis=(1, 2)) - ghi_prev[:, hs, ws].mean(axis=(1, 2))
        csi_diff_q = csi_last[:, hs, ws].mean(axis=(1, 2)) - csi_prev[:, hs, ws].mean(axis=(1, 2))

        features[f"quad{q_idx}_ghi_diff_last_minus_prev"] = ghi_diff_q
        features[f"quad{q_idx}_csi_diff_last_minus_prev"] = csi_diff_q

    # Variables de gradient sur la dernière carte CSI.
    grad_last = _gradient_magnitude(csi_last)
    features["grad_csi_last_mean"] = grad_last.mean(axis=(1, 2))
    features["grad_csi_last_std"] = grad_last.std(axis=(1, 2))
    features["grad_csi_last_p90"] = np.quantile(grad_last.reshape(n, -1), 0.90, axis=1)

    # Centres de masse des zones sombres et lumineuses.
    dark_mass = np.maximum(1.0 - csi_last, 0.0)
    bright_mass = np.maximum(csi_last, 0.0)

    dark_cy, dark_cx = _weighted_center_of_mass(dark_mass)
    bright_cy, bright_cx = _weighted_center_of_mass(bright_mass)

    features["dark_com_y_last"] = dark_cy
    features["dark_com_x_last"] = dark_cx
    features["bright_com_y_last"] = bright_cy
    features["bright_com_x_last"] = bright_cx

    # Pseudo mouvement entre les deux dernières cartes d'anomalie nuageuse.
    dark_prev = np.maximum(1.0 - csi_prev, 0.0)[:, ::downsample, ::downsample]
    dark_last_ds = np.maximum(1.0 - csi_last, 0.0)[:, ::downsample, ::downsample]

    dy_list, dx_list, score_list = [], [], []
    for i in range(n):
        dy, dx, score = _best_shift_single(dark_prev[i], dark_last_ds[i], max_shift=max_shift)
        dy_list.append(dy)
        dx_list.append(dx)
        score_list.append(score)

    features["pseudo_motion_dy_t-15_to_t"] = np.asarray(dy_list, dtype=np.float32)
    features["pseudo_motion_dx_t-15_to_t"] = np.asarray(dx_list, dtype=np.float32)
    features["pseudo_motion_score_t-15_to_t"] = np.asarray(score_list, dtype=np.float32)

    return pd.DataFrame(features)


def build_advanced_features(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Crée des variables agrégées, spatiales et dynamiques pour les modèles tabulaires.
    """
    features = {}
    ghi = arrays["GHI"]
    cls = arrays["CLS"]

    # Statistiques globales par image.
    for t in range(4):
        features[f"ghi_mean_t-{45-t*15}"] = ghi[:, t].mean(axis=(1, 2))
        features[f"ghi_std_t-{45-t*15}"] = ghi[:, t].std(axis=(1, 2))

    # Dynamique globale.
    ghi_diff = ghi[:, 1:] - ghi[:, :-1]
    for t in range(3):
        features[f"ghi_diff_mean_t{t}"] = ghi_diff[:, t].mean(axis=(1, 2))
        features[f"ghi_diff_std_t{t}"] = ghi_diff[:, t].std(axis=(1, 2))

    # CSI moyen.
    cls_past = cls[:, :4]
    csi = ghi / (cls_past + 1e-6)
    features["csi_last_frame_mean"] = csi[:, -1].mean(axis=(1, 2))
    features["csi_last_frame_std"] = csi[:, -1].std(axis=(1, 2))

    # Position future du soleil.
    sza_future = arrays["SZA"][:, 4:]
    for t in range(4):
        features[f"sza_future_t+{15+t*15}"] = sza_future[:, t].mean(axis=(1, 2))

    base_df = pd.DataFrame(features)
    dyn_df = build_spatial_dynamics_features(arrays)

    return pd.concat([base_df, dyn_df], axis=1)

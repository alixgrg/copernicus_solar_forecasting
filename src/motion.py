"""Variables de mouvement nuageux et modèles de référence advectifs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def has_opencv() -> bool:
    """Indique si OpenCV est disponible."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalise une image entre 0 et 1 à partir de percentiles robustes."""
    frame = np.asarray(frame, dtype=np.float32)
    finite = np.isfinite(frame)
    if not finite.any():
        return np.zeros_like(frame, dtype=np.float32)
    values = frame[finite]
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        return np.zeros_like(frame, dtype=np.float32)
    out = np.clip((frame - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32)


def farneback_flow_pair(prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
    """
    Calcule un flot optique dense de Farneback lorsque OpenCV est installé.

    Renvoie un tableau de forme (hauteur, largeur, 2), où la dernière dimension
    contient (dx, dy). Le notebook se rabat sur la corrélation de phase si OpenCV
    est absent.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for Farneback optical flow.") from exc

    prev_norm = _normalize_frame(prev_frame)
    next_norm = _normalize_frame(next_frame)
    flow = cv2.calcOpticalFlowFarneback(
        prev_norm,
        next_norm,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow.astype(np.float32)


def phase_correlation_shift(prev_frame: np.ndarray, next_frame: np.ndarray) -> tuple[float, float]:
    """
    Estime une translation globale (dx, dy) par corrélation de phase.

    Cette méthode légère sert de solution de repli quand OpenCV n'est pas disponible.
    Elle capture un déplacement dominant des nuages plutôt qu'un champ dense.
    """
    prev = _normalize_frame(prev_frame)
    nxt = _normalize_frame(next_frame)
    prev = prev - prev.mean()
    nxt = nxt - nxt.mean()

    cross_power = np.fft.fft2(prev) * np.fft.fft2(nxt).conj()
    cross_power /= np.maximum(np.abs(cross_power), 1e-8)
    corr = np.fft.ifft2(cross_power).real

    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    height, width = corr.shape
    if peak_x > width // 2:
        peak_x -= width
    if peak_y > height // 2:
        peak_y -= height

    # Convention de signe: déplacement de l'image précédente vers la suivante.
    return float(-peak_x), float(-peak_y)


def cloud_centroid_shift(prev_frame: np.ndarray, next_frame: np.ndarray) -> tuple[float, float]:
    """
    Estime le mouvement à partir du déplacement des barycentres de nébulosité.

    La nébulosité est approximée par max(0, 1 - CSI). Cette approche est moins
    précise que le flot optique, mais reste informative lorsque la translation
    dominante est trop faible ou trop diffuse pour la corrélation de phase.
    """
    prev_cloud = np.maximum(0.0, 1.0 - np.asarray(prev_frame, dtype=np.float32))
    next_cloud = np.maximum(0.0, 1.0 - np.asarray(next_frame, dtype=np.float32))

    def _centroid(weights: np.ndarray) -> tuple[float, float]:
        """Calcule le barycentre pondéré d'une carte de nébulosité."""
        total = float(np.sum(weights))
        if total <= 1e-8:
            height, width = weights.shape
            return (width - 1) / 2.0, (height - 1) / 2.0
        yy, xx = np.indices(weights.shape)
        cx = float(np.sum(xx * weights) / total)
        cy = float(np.sum(yy * weights) / total)
        return cx, cy

    prev_x, prev_y = _centroid(prev_cloud)
    next_x, next_y = _centroid(next_cloud)
    return float(next_x - prev_x), float(next_y - prev_y)


def estimate_motion_vectors(
    sequence: np.ndarray,
    use_farneback: bool | None = None,
) -> pd.DataFrame:
    """
    Estime des variables de mouvement nuageux à l'échelle de l'échantillon.

    Paramètres
    ----------
    sequence:
        Tableau de forme (n_samples, n_times, height, width), typiquement CSI ou GHI.
    use_farneback:
        Si True, impose Farneback via OpenCV. Si False, utilise la corrélation de
        phase. Si None, Farneback est utilisé seulement quand OpenCV est disponible.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 4 or sequence.shape[1] < 2:
        raise ValueError(f"Expected sequence with shape (n, t, h, w), got {sequence.shape}.")

    if use_farneback is None:
        use_farneback = has_opencv()

    rows = []
    for sample_idx in range(sequence.shape[0]):
        dx_values = []
        dy_values = []
        mag_values = []

        for time_idx in range(sequence.shape[1] - 1):
            prev_frame = sequence[sample_idx, time_idx]
            next_frame = sequence[sample_idx, time_idx + 1]
            centroid_dx, centroid_dy = cloud_centroid_shift(prev_frame, next_frame)

            if use_farneback:
                flow = farneback_flow_pair(prev_frame, next_frame)
                dx = float(np.nanmedian(flow[..., 0]))
                dy = float(np.nanmedian(flow[..., 1]))
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                mag = float(np.nanmedian(magnitude))
                mag_p90 = float(np.nanquantile(magnitude, 0.90))
            else:
                dx, dy = phase_correlation_shift(prev_frame, next_frame)
                if np.sqrt(dx**2 + dy**2) < 1e-8:
                    dx, dy = centroid_dx, centroid_dy
                mag = float(np.sqrt(dx**2 + dy**2))
                mag_p90 = mag

            dx_values.append(dx)
            dy_values.append(dy)
            mag_values.append(mag)
            rows.append(
                {
                    "sample": sample_idx,
                    "pair": time_idx,
                    "motion_dx": dx,
                    "motion_dy": dy,
                    "motion_magnitude": mag,
                    "motion_magnitude_p90": mag_p90,
                    "cloud_centroid_dx": centroid_dx,
                    "cloud_centroid_dy": centroid_dy,
                }
            )

    pair_frame = pd.DataFrame(rows)
    summary = pair_frame.groupby("sample").agg(
        motion_dx_mean=("motion_dx", "mean"),
        motion_dy_mean=("motion_dy", "mean"),
        motion_dx_last=("motion_dx", "last"),
        motion_dy_last=("motion_dy", "last"),
        motion_speed_mean=("motion_magnitude", "mean"),
        motion_speed_last=("motion_magnitude", "last"),
        motion_speed_p90=("motion_magnitude_p90", "max"),
        cloud_centroid_dx_mean=("cloud_centroid_dx", "mean"),
        cloud_centroid_dy_mean=("cloud_centroid_dy", "mean"),
        cloud_centroid_dx_last=("cloud_centroid_dx", "last"),
        cloud_centroid_dy_last=("cloud_centroid_dy", "last"),
    )
    summary["motion_direction_rad"] = np.arctan2(summary["motion_dy_last"], summary["motion_dx_last"])
    summary["motion_direction_sin"] = np.sin(summary["motion_direction_rad"])
    summary["motion_direction_cos"] = np.cos(summary["motion_direction_rad"])
    return summary.reset_index(drop=True).astype(np.float32)


def shift_image_nearest(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Décale une image avec un déplacement entier au plus proche voisin.

    Les bordures vides sont remplies par les valeurs de bord les plus proches.
    Cela garde le modèle de référence simple, déterministe et léger en dépendances.
    """
    image = np.asarray(image, dtype=np.float32)
    shift_x = int(np.rint(dx))
    shift_y = int(np.rint(dy))
    shifted = np.roll(image, shift=(shift_y, shift_x), axis=(0, 1))

    if shift_y > 0:
        shifted[:shift_y, :] = shifted[shift_y : shift_y + 1, :]
    elif shift_y < 0:
        shifted[shift_y:, :] = shifted[shift_y - 1 : shift_y, :]

    if shift_x > 0:
        shifted[:, :shift_x] = shifted[:, shift_x : shift_x + 1]
    elif shift_x < 0:
        shifted[:, shift_x:] = shifted[:, shift_x - 1 : shift_x]

    return shifted.astype(np.float32)


def advective_csi_baseline(
    arrays_raw: dict[str, np.ndarray],
    motion_features: pd.DataFrame | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Calcule un modèle de référence de persistance advective sur l'indice de ciel clair.

    La dernière carte CSI observée est décalée selon le vecteur de mouvement
    nuageux estimé, puis multipliée par le GHI de ciel clair futur. L'horizon h
    utilise h fois le dernier déplacement estimé.
    """
    ghi = np.asarray(arrays_raw["GHI"], dtype=np.float32)
    cls = np.asarray(arrays_raw["CLS"], dtype=np.float32)

    cls_past = cls[:, : ghi.shape[1]]
    cls_future = cls[:, ghi.shape[1] :]
    csi = ghi / np.maximum(cls_past, eps)
    last_csi = csi[:, -1]

    if motion_features is None:
        motion_features = estimate_motion_vectors(csi, use_farneback=None)

    dx_values = motion_features["motion_dx_last"].to_numpy(dtype=np.float32)
    dy_values = motion_features["motion_dy_last"].to_numpy(dtype=np.float32)

    y_pred = np.empty((ghi.shape[0], cls_future.shape[1], ghi.shape[2], ghi.shape[3]), dtype=np.float32)
    for sample_idx in range(ghi.shape[0]):
        for horizon_idx in range(cls_future.shape[1]):
            step = horizon_idx + 1
            advected_csi = shift_image_nearest(
                last_csi[sample_idx],
                dx=step * dx_values[sample_idx],
                dy=step * dy_values[sample_idx],
            )
            y_pred[sample_idx, horizon_idx] = advected_csi * cls_future[sample_idx, horizon_idx]

    return np.maximum(y_pred, 0.0).astype(np.float32)

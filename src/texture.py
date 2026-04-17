"""Spatial texture features for cloud-regime characterization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantize_image(image: np.ndarray, levels: int = 16) -> np.ndarray:
    """Quantize one image to integer gray levels."""
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.uint8)
    values = image[finite]
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return np.floor(scaled * (levels - 1)).astype(np.uint8)


def glcm_matrix(
    image: np.ndarray,
    levels: int = 16,
    offsets: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (-1, 1)),
) -> np.ndarray:
    """Build a normalized gray-level co-occurrence matrix."""
    quantized = quantize_image(image, levels=levels)
    matrix = np.zeros((levels, levels), dtype=np.float64)
    height, width = quantized.shape

    for dy, dx in offsets:
        y0 = max(0, -dy)
        y1 = min(height, height - dy)
        x0 = max(0, -dx)
        x1 = min(width, width - dx)
        ref = quantized[y0:y1, x0:x1].ravel()
        nbr = quantized[y0 + dy : y1 + dy, x0 + dx : x1 + dx].ravel()
        np.add.at(matrix, (ref, nbr), 1)
        np.add.at(matrix, (nbr, ref), 1)

    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def glcm_properties(image: np.ndarray, levels: int = 16) -> dict[str, float]:
    """Compute common GLCM texture descriptors."""
    p = glcm_matrix(image, levels=levels)
    indices = np.arange(levels, dtype=np.float64)
    i, j = np.meshgrid(indices, indices, indexing="ij")
    diff = i - j

    contrast = np.sum((diff**2) * p)
    dissimilarity = np.sum(np.abs(diff) * p)
    homogeneity = np.sum(p / (1.0 + diff**2))
    energy = np.sqrt(np.sum(p**2))
    entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))

    return {
        "glcm_contrast": float(contrast),
        "glcm_dissimilarity": float(dissimilarity),
        "glcm_homogeneity": float(homogeneity),
        "glcm_energy": float(energy),
        "glcm_entropy": float(entropy),
    }


def _gabor_kernel(
    frequency: float,
    theta: float,
    sigma: float = 2.0,
    radius: int = 5,
) -> np.ndarray:
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    x, y = np.meshgrid(axis, axis)
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    envelope = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))
    carrier = np.cos(2 * np.pi * frequency * x_theta)
    kernel = envelope * carrier
    kernel -= kernel.mean()
    norm = np.sqrt(np.sum(kernel**2))
    if norm > 0:
        kernel /= norm
    return kernel.astype(np.float32)


def gabor_properties(
    image: np.ndarray,
    frequencies: tuple[float, ...] = (0.10, 0.20),
    thetas: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> dict[str, float]:
    """Compute compact Gabor filter response statistics."""
    from scipy import ndimage

    img = np.asarray(image, dtype=np.float32)
    img = img - np.nanmean(img)
    std = np.nanstd(img)
    if std > 1e-8:
        img = img / std

    means = []
    p90s = []
    for frequency in frequencies:
        for theta in thetas:
            kernel = _gabor_kernel(frequency=frequency, theta=theta)
            response = ndimage.convolve(img, kernel, mode="nearest")
            abs_response = np.abs(response)
            means.append(float(np.nanmean(abs_response)))
            p90s.append(float(np.nanquantile(abs_response, 0.90)))

    return {
        "gabor_mean": float(np.mean(means)),
        "gabor_max_mean": float(np.max(means)),
        "gabor_p90": float(np.mean(p90s)),
        "gabor_max_p90": float(np.max(p90s)),
    }


def build_texture_features(
    arrays: dict[str, np.ndarray],
    variable: str = "CSI",
    time_index: int = -1,
    levels: int = 16,
    include_gabor: bool = True,
) -> pd.DataFrame:
    """
    Build texture descriptors from one image per sample.

    The default uses the last CSI frame, which is a good proxy for cloud cover
    complexity before forecasting.
    """
    if variable not in arrays:
        raise KeyError(f"Variable '{variable}' not found in arrays.")
    values = np.asarray(arrays[variable], dtype=np.float32)
    if values.ndim != 4:
        raise ValueError(f"Expected {variable} with shape (n, t, h, w), got {values.shape}.")

    rows = []
    for sample_idx in range(values.shape[0]):
        image = values[sample_idx, time_index]
        row = glcm_properties(image, levels=levels)
        if include_gabor:
            row.update(gabor_properties(image))
        rows.append(row)

    prefix = f"{variable.lower()}_t{time_index}_"
    return pd.DataFrame(rows).add_prefix(prefix).astype(np.float32)


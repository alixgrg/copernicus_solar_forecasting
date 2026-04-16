"""Visualization helpers for solar irradiance image sequences."""

from __future__ import annotations

import math
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FORECAST_HORIZONS_MINUTES, INPUT_VARIABLES
from src.utils import describe_numeric_array


def time_first(array: np.ndarray, n_steps: int | None = None) -> np.ndarray:
    """Return a 3D image sequence as (time, height, width)."""
    array = np.asarray(array)
    if array.ndim == 2:
        return array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D image or a 3D sequence, got shape {array.shape}.")

    if n_steps is not None:
        if array.shape[0] == n_steps:
            return array
        if array.shape[-1] == n_steps:
            return np.moveaxis(array, -1, 0)

    if array.shape[0] in {4, 8}:
        return array
    if array.shape[-1] in {4, 8}:
        return np.moveaxis(array, -1, 0)
    return array


def plot_sequence(
    sequence: np.ndarray,
    titles: Iterable[str] | None = None,
    n_steps: int | None = None,
    cmap: str = "inferno",
    suptitle: str | None = None,
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Plot all frames in an image sequence."""
    frames = time_first(sequence, n_steps=n_steps)
    n_frames = frames.shape[0]
    ncols = ncols or n_frames
    nrows = math.ceil(n_frames / ncols)
    figsize = figsize or (3.2 * ncols, 3.2 * nrows)

    vmin = np.nanpercentile(frames, 2)
    vmax = np.nanpercentile(frames, 98)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    titles = list(titles) if titles is not None else [f"t={i}" for i in range(n_frames)]

    image = None
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n_frames:
            continue
        image = ax.imshow(frames[i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(titles[i])

    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), fraction=0.025, pad=0.02)
    if suptitle:
        fig.suptitle(suptitle)
    return fig, axes


def plot_sample_overview(sample: dict[str, np.ndarray], target: np.ndarray | None = None):
    fig, axes = plt.subplots(5 if target is not None else 4, 1, figsize=(16, 18))

    variables = [variable for variable in INPUT_VARIABLES if variable in sample]
    for ax, var in zip(axes[:4], variables):
        seq = time_first(sample[var])
        ax.imshow(seq[0], cmap="inferno")
        ax.set_title(f"{var} - first frame")
        ax.axis("off")

    if target is not None:
        seq = time_first(target)
        axes[-1].imshow(seq[0], cmap="inferno")
        axes[-1].set_title("Target - first future frame")
        axes[-1].axis("off")

    plt.tight_layout()
    return fig, axes


def plot_value_distribution(array: np.ndarray, title: str = "", bins: int = 50):
    values = np.asarray(array).ravel()
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    return fig, ax


def describe_array(name: str, array: np.ndarray) -> dict[str, object]:
    """Return compact descriptive statistics for one array."""
    return describe_numeric_array(name, array)


def describe_sample(sample: dict[str, np.ndarray]) -> pd.DataFrame:
    """Describe every numeric array in a loaded sample."""
    rows = []
    for name, array in sample.items():
        if name == "datetime":
            continue
        rows.append(describe_array(name, array))
    return pd.DataFrame(rows)


def horizon_titles(horizons: Iterable[int] = FORECAST_HORIZONS_MINUTES) -> list[str]:
    return [f"t+{h} min" for h in horizons]

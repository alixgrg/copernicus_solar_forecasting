"""Outils de visualisation pour les séquences d'images d'irradiance solaire."""

from __future__ import annotations

import math
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FORECAST_HORIZONS_MINUTES, INPUT_VARIABLES
from src.utils import describe_numeric_array


def time_first(array: np.ndarray, n_steps: int | None = None) -> np.ndarray:
    """Renvoie une séquence d'images 3D au format (temps, hauteur, largeur)."""
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
    """Trace toutes les images d'une séquence."""
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
    """Trace un aperçu des variables d'un échantillon et de sa cible éventuelle."""
    fig, axes = plt.subplots(5 if target is not None else 4, 1, figsize=(16, 18))

    variables = [variable for variable in INPUT_VARIABLES if variable in sample]
    for ax, var in zip(axes[:4], variables):
        seq = time_first(sample[var])
        ax.imshow(seq[0], cmap="inferno")
        ax.set_title(f"{var} - première image")
        ax.axis("off")

    if target is not None:
        seq = time_first(target)
        axes[-1].imshow(seq[0], cmap="inferno")
        axes[-1].set_title("Cible - première image future")
        axes[-1].axis("off")

    plt.tight_layout()
    return fig, axes


def plot_value_distribution(array: np.ndarray, title: str = "", bins: int = 50):
    """Trace la distribution des valeurs finies d'un tableau."""
    values = np.asarray(array).ravel()
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    return fig, ax


def describe_array(name: str, array: np.ndarray) -> dict[str, object]:
    """Renvoie des statistiques descriptives compactes pour un tableau."""
    return describe_numeric_array(name, array)


def describe_sample(sample: dict[str, np.ndarray]) -> pd.DataFrame:
    """Décrit chaque tableau numérique d'un échantillon chargé."""
    rows = []
    for name, array in sample.items():
        if name == "datetime":
            continue
        rows.append(describe_array(name, array))
    return pd.DataFrame(rows)


def horizon_titles(horizons: Iterable[int] = FORECAST_HORIZONS_MINUTES) -> list[str]:
    """Construit les titres associés aux horizons de prévision."""
    return [f"t+{h} min" for h in horizons]



def plot_error_analysis(y_true, y_pred, title="Analyse des Erreurs"):
    """Trace un diagnostic visuel des erreurs de prédiction."""
    # On aplatit pour avoir tous les pixels
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    errors = y_pred_flat - y_true_flat

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Nuage de points entre valeur réelle et valeur prédite.
    ax[0].scatter(y_true_flat[::100], y_pred_flat[::100], alpha=0.1, s=1)
    ax[0].plot([0, 1000], [0, 1000], 'r--')
    ax[0].set_xlabel("GHI réel")
    ax[0].set_ylabel("GHI prédit")
    ax[0].set_title("Biais du modèle")

    # Distribution de l'erreur.
    ax[1].hist(errors, bins=50, color='skyblue', edgecolor='black')
    ax[1].axvline(0, color='red', linestyle='--')
    ax[1].set_title("Distribution de l'erreur")
    
    plt.suptitle(title)
    plt.show()


def plot_forecast_triplet(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_idx: int = 0,
    horizon_idx: int = 0,
    model_name: str = "model",
    cmap: str = "inferno",
):
    """Trace vérité terrain, prédiction et erreur pour un échantillon et un horizon."""
    truth = np.asarray(y_true)[sample_idx, horizon_idx]
    pred = np.asarray(y_pred)[sample_idx, horizon_idx]
    error = pred - truth

    vmin = float(np.nanpercentile(np.stack([truth, pred]), 2))
    vmax = float(np.nanpercentile(np.stack([truth, pred]), 98))
    err_abs = float(np.nanpercentile(np.abs(error), 98))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    images = [
        axes[0].imshow(truth, cmap=cmap, vmin=vmin, vmax=vmax),
        axes[1].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax),
        axes[2].imshow(error, cmap="coolwarm", vmin=-err_abs, vmax=err_abs),
    ]
    titles = [
        "Vérité",
        f"Prédiction - {model_name}",
        "Erreur prédiction - vérité",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(images[0], ax=axes[:2].tolist(), fraction=0.035, pad=0.02)
    fig.colorbar(images[2], ax=axes[2], fraction=0.046, pad=0.04)
    return fig, axes


def plot_motion_summary(
    motion_features: pd.DataFrame,
    sample_idx: int = 0,
    ax=None,
    title: str = "Vecteur de mouvement estime",
):
    """Trace un vecteur de mouvement nuageux pour un échantillon."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    row = motion_features.iloc[sample_idx]
    dx = float(row.get("motion_dx_last", 0.0))
    dy = float(row.get("motion_dy_last", 0.0))
    ax.quiver([0], [0], [dx], [dy], angles="xy", scale_units="xy", scale=1, width=0.012)
    lim = max(1.0, abs(dx), abs(dy)) * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="0.8", linewidth=1)
    ax.axvline(0, color="0.8", linewidth=1)
    ax.set_xlabel("dx pixels / pas de 15 min")
    ax.set_ylabel("dy pixels / pas de 15 min")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    return fig, ax


def plot_metric_by_horizon(
    diagnostics_by_horizon: pd.DataFrame,
    models: list[str] | None = None,
    metric: str = "RMSE",
    title: str | None = None,
):
    """Trace une métrique par horizon pour une sélection de modèles."""
    df = diagnostics_by_horizon.copy()
    if models is not None:
        df = df[df["model"].isin(models)]

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, part in df.groupby("model"):
        part = part.sort_values("horizon_min")
        ax.plot(part["horizon_min"], part[metric], marker="o", label=model_name)
    ax.set_xlabel("Horizon de prévision (minutes)")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} par horizon")
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_cluster_metric(
    cluster_metrics: pd.DataFrame,
    metric: str = "RMSE",
    title: str | None = None,
):
    """Trace la performance des modèles par cluster ou régime interprété."""
    pivot = cluster_metrics.pivot_table(index="regime", columns="model", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} par régime météo")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig, ax

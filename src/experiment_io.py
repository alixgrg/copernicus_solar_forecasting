"""Sauvegarde et recharge les sorties de modèles locales ou Colab."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import PROJECT_ROOT
from src.metrics import build_model_diagnostics, evaluate_forecasts


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "model_outputs"


def model_slug(model_name: str) -> str:
    """Renvoie un identifiant de modèle compatible avec le système de fichiers."""
    allowed = []
    for char in model_name.lower().strip():
        if char.isalnum():
            allowed.append(char)
        elif char in {"_", "-", ".", " "}:
            allowed.append("_")
    slug = "".join(allowed).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "model"


def save_model_output(
    model_name: str,
    y_pred: np.ndarray,
    y_true: np.ndarray | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    profile: str | None = None,
    split_name: str = "validation",
    reference_name: str = "persistence_csi",
    backend: str = "local",
    extra_metadata: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> dict[str, Path]:
    """
    Sauvegarde les prédictions d'un modèle avec leurs métadonnées.

    Les prédictions sont l'artefact de référence. Les métriques peuvent être
    sauvegardées pour consultation, mais les comparaisons finales les recalculent
    à partir de y_true afin de garder tous les modèles alignés.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = model_slug(model_name)
    pred_path = output_dir / f"{slug}_predictions.npz"
    metadata_path = output_dir / f"{slug}_metadata.json"
    metrics_path = output_dir / f"{slug}_metrics.csv"

    if not overwrite and (pred_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"Artifacts already exist for model '{model_name}' in {output_dir}.")

    y_pred = np.asarray(y_pred, dtype=np.float32)
    np.savez_compressed(pred_path, y_pred=y_pred)

    metadata = {
        "model_name": model_name,
        "slug": slug,
        "backend": backend,
        "profile": profile,
        "split_name": split_name,
        "reference_name": reference_name,
        "prediction_shape": list(y_pred.shape),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    paths = {
        "predictions": pred_path,
        "metadata": metadata_path,
    }

    if y_true is not None:
        evaluate_forecasts(np.asarray(y_true, dtype=np.float32), y_pred).assign(model=model_name).to_csv(
            metrics_path,
            index=False,
        )
        paths["metrics"] = metrics_path

    return paths


def save_prediction_bundle(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    profile: str | None = None,
    backend: str = "local",
    reference_name: str = "persistence_csi",
    extra_metadata: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> pd.DataFrame:
    """Sauvegarde plusieurs jeux de prédictions et renvoie une table manifeste."""
    rows = []
    for model_name, y_pred in predictions.items():
        paths = save_model_output(
            model_name=model_name,
            y_pred=y_pred,
            y_true=None,
            output_dir=output_dir,
            profile=profile,
            reference_name=reference_name,
            backend=backend,
            extra_metadata=extra_metadata,
            overwrite=overwrite,
        )
        rows.append(
            {
                "model": model_name,
                "backend": backend,
                "predictions": str(paths["predictions"]),
                "metadata": str(paths["metadata"]),
            }
        )

    manifest = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    manifest_path = output_dir / f"manifest_{backend}.csv"
    manifest.to_csv(manifest_path, index=False)

    if y_true is not None:
        diagnostics = build_model_diagnostics(y_true, predictions, reference_name=reference_name)
        diagnostics["global"].to_csv(output_dir / f"global_metrics_{backend}.csv", index=False)
        diagnostics["by_horizon"].to_csv(output_dir / f"by_horizon_metrics_{backend}.csv", index=False)
        diagnostics["spatial_means"].to_csv(output_dir / f"spatial_mean_metrics_{backend}.csv", index=False)
        diagnostics["spatial_structure"].to_csv(output_dir / f"spatial_structure_metrics_{backend}.csv", index=False)

    return manifest


def list_saved_outputs(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    """Liste les artefacts de modèles sauvegardés dans un dossier de sortie."""
    output_dir = Path(output_dir)
    rows = []
    for metadata_path in sorted(output_dir.glob("*_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        slug = metadata.get("slug", metadata_path.name.replace("_metadata.json", ""))
        pred_path = output_dir / f"{slug}_predictions.npz"
        rows.append(
            {
                "model": metadata.get("model_name", slug),
                "slug": slug,
                "backend": metadata.get("backend"),
                "profile": metadata.get("profile"),
                "prediction_shape": tuple(metadata.get("prediction_shape", [])),
                "created_at_utc": metadata.get("created_at_utc"),
                "predictions_path": pred_path,
                "metadata_path": metadata_path,
                "exists": pred_path.exists(),
            }
        )
    return pd.DataFrame(rows)


def load_saved_predictions(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_names: list[str] | None = None,
    strict_shape: tuple[int, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Recharge des artefacts de prédictions sauvegardés."""
    output_dir = Path(output_dir)
    listing = list_saved_outputs(output_dir)
    if listing.empty:
        return {}

    predictions = {}
    wanted = set(model_names) if model_names is not None else None
    for _, row in listing.iterrows():
        model_name = row["model"]
        if wanted is not None and model_name not in wanted:
            continue
        pred_path = Path(row["predictions_path"])
        if not pred_path.exists():
            continue
        with np.load(pred_path, allow_pickle=False) as archive:
            y_pred = np.asarray(archive["y_pred"], dtype=np.float32)
        if strict_shape is not None and tuple(y_pred.shape) != tuple(strict_shape):
            raise ValueError(
                f"Saved prediction for {model_name} has shape {y_pred.shape}, "
                f"expected {strict_shape}."
            )
        predictions[model_name] = y_pred
    return predictions


def diagnostics_from_saved_outputs(
    y_true: np.ndarray,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    reference_name: str = "persistence_csi",
    include_predictions: dict[str, np.ndarray] | None = None,
) -> dict[str, pd.DataFrame]:
    """Recharge les prédictions sauvegardées et construit les diagnostics comparatifs."""
    predictions = {}
    if include_predictions:
        predictions.update(include_predictions)
    predictions.update(load_saved_predictions(output_dir, strict_shape=tuple(np.asarray(y_true).shape)))
    if reference_name not in predictions:
        raise KeyError(
            f"Reference '{reference_name}' is missing. Save it locally or pass it via include_predictions."
        )
    return build_model_diagnostics(y_true, predictions, reference_name=reference_name)

"""Métriques d'évaluation pour la prévision solaire Copernicus."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from config import FORECAST_HORIZONS_MINUTES


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule la RMSE globale sur un tenseur complet de prévisions."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule la MAE globale sur un tenseur complet de prévisions."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule l'erreur signée moyenne. Une valeur positive indique une surestimation."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(y_pred - y_true))


def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule l'erreur absolue médiane sur un tenseur complet de prévisions."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.median(np.abs(y_true - y_pred)))


def p90_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule le percentile 90 de l'erreur absolue sur un tenseur complet."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.quantile(np.abs(y_true - y_pred), 0.90))


def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Calcule la RMSE divisée par le GHI moyen observé."""
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(rmse(y_true, y_pred) / (np.mean(y_true) + eps))


def spatial_mean_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule la corrélation entre moyennes spatiales observées et prédites."""
    y_true_mean = np.asarray(y_true, dtype=np.float64).mean(axis=(2, 3)).ravel()
    y_pred_mean = np.asarray(y_pred, dtype=np.float64).mean(axis=(2, 3)).ravel()
    if np.std(y_true_mean) < 1e-12 or np.std(y_pred_mean) < 1e-12:
        return np.nan
    return float(np.corrcoef(y_true_mean, y_pred_mean)[0, 1])


def evaluate_forecasts(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Renvoie les métriques globales d'un tenseur de prévisions."""
    return pd.DataFrame(
        {
            "metric": ["RMSE", "MAE", "MBE", "MedAE", "P90AE", "nRMSE", "R2", "MAPE", "corr_spatial_mean"],
            "value": [
                rmse(y_true, y_pred),
                mae(y_true, y_pred),
                mean_bias_error(y_true, y_pred),
                median_absolute_error(y_true, y_pred),
                p90_absolute_error(y_true, y_pred),
                normalized_rmse(y_true, y_pred),
                r2_score(y_true, y_pred),
                mape(y_true, y_pred),
                spatial_mean_correlation(y_true, y_pred),
            ],
        }
    )


def evaluate_by_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: list[int] | tuple[int, ...] = FORECAST_HORIZONS_MINUTES,
) -> pd.DataFrame:
    """Renvoie la RMSE et la MAE pour chaque horizon de prévision."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    rows = []
    for h_idx, horizon in enumerate(horizons):
        rows.append(
            {
                "horizon_min": horizon,
                "RMSE": rmse(y_true[:, h_idx], y_pred[:, h_idx]),
                "MAE": mae(y_true[:, h_idx], y_pred[:, h_idx]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_spatial_means(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: list[int] | tuple[int, ...] = FORECAST_HORIZONS_MINUTES,
) -> pd.DataFrame:
    """
    Évalue les prévisions uniquement sur les moyennes spatiales.

    Cette métrique distingue l'erreur sur le niveau moyen d'irradiance de
    l'erreur sur la structure spatiale fine.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    y_true_mean = y_true.mean(axis=(2, 3))
    y_pred_mean = y_pred.mean(axis=(2, 3))

    rows = []
    for h_idx, horizon in enumerate(horizons):
        rows.append(
            {
                "horizon_min": horizon,
                "RMSE_spatial_mean": rmse(y_true_mean[:, h_idx], y_pred_mean[:, h_idx]),
                "MAE_spatial_mean": mae(y_true_mean[:, h_idx], y_pred_mean[:, h_idx]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_model_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    horizons: list[int] | tuple[int, ...] = FORECAST_HORIZONS_MINUTES,
) -> dict[str, pd.DataFrame]:
    """Renvoie toutes les tables d'évaluation pour un modèle."""
    return {
        "global": evaluate_forecasts(y_true, y_pred).assign(model=model_name),
        "by_horizon": evaluate_by_horizon(y_true, y_pred, horizons=horizons).assign(model=model_name),
        "spatial_means": evaluate_spatial_means(y_true, y_pred, horizons=horizons).assign(model=model_name),
    }


def compare_global_scores(results: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Agrège les scores globaux de plusieurs ensembles de résultats.

    Format attendu:
        {
            "model_a": {"global": ..., "by_horizon": ..., ...},
            "model_b": {"global": ..., "by_horizon": ..., ...},
        }
    """
    rows = []
    for model_name, bundle in results.items():
        global_df = bundle["global"].copy()
        row = {"model": model_name}
        for _, record in global_df.iterrows():
            row[str(record["metric"])] = float(record["value"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("RMSE", ascending=True).reset_index(drop=True)

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule le coefficient de détermination global."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / (ss_tot + 1e-8)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule l'erreur absolue moyenne en pourcentage globale."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    # On ajoute un petit epsilon pour éviter la division par zéro la nuit
    epsilon = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (np.maximum(y_true, epsilon))))) * 100

def forecast_skill_score(mse_model: float, mse_persistence: float) -> float:
    """
    Calcule le Skill Score (SS). 
    SS > 0 : Le modèle est meilleur que la persistance.
    SS = 1 : Prédiction parfaite.
    """
    return 1 - (mse_model / (mse_persistence + 1e-8))


def skill_score_from_rmse(model_rmse: float, persistence_rmse: float) -> float:
    """Calcule le skill score à partir de RMSE via l'aide fondée sur la MSE."""
    return forecast_skill_score(model_rmse**2, persistence_rmse**2)

def add_skill_scores(results_df: pd.DataFrame, persistence_rmse: float) -> pd.DataFrame:
    """Ajoute une colonne Skill Score à votre tableau de comparaison."""
    persistence_mse = persistence_rmse ** 2
    results_df["MSE"] = results_df["value"] ** 2 # Si la métrique est RMSE
    # Calculer le SS uniquement pour les lignes RMSE
    results_df["skill_score"] = results_df.apply(
        lambda row: forecast_skill_score(row["value"]**2, persistence_mse) 
        if row["metric"] == "RMSE" else None, axis=1
    )
    return results_df


def spatial_mean_residual(y_true, baseline_pred):
    """Calcule le résidu moyen spatial entre la vérité et un modèle de référence."""
    return (np.asarray(y_true) - np.asarray(baseline_pred)).mean(axis=(2, 3))


def global_metrics_row(model_name, y_true, y_pred, reference_rmse=None):
    """Construit une ligne de métriques globales pour un modèle."""
    row = {"model": model_name}
    for _, record in evaluate_forecasts(y_true, y_pred).iterrows():
        row[str(record["metric"])] = float(record["value"])
    if reference_rmse is not None:
        row["skill_RMSE_vs_CSI"] = skill_score_from_rmse(row["RMSE"], reference_rmse)
    return row


def cluster_quality(X, labels):
    """Calcule des indicateurs de qualité pour un partitionnement."""
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "metric": ["silhouette", "calinski_harabasz", "davies_bouldin"],
            "value": [
                silhouette_score(X, labels),
                calinski_harabasz_score(X, labels),
                davies_bouldin_score(X, labels),
            ],
        }
    )


def metrics_by_cluster(y_true, y_pred, labels, model_name, cluster_name_map, reference_pred=None):
    """Évalue un modèle séparément pour chaque cluster interprété."""
    rows = []
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        ref_rmse = rmse(y_true[mask], reference_pred[mask]) if reference_pred is not None else None
        model_rmse = rmse(y_true[mask], y_pred[mask])
        row = {
            "model": model_name,
            "cluster": int(cluster),
            "regime": cluster_name_map.get(cluster, str(cluster)),
            "n": int(mask.sum()),
            "RMSE": model_rmse,
            "MAE": float(np.mean(np.abs(y_pred[mask] - y_true[mask]))),
            "bias": float(np.mean(y_pred[mask] - y_true[mask])),
            "P90AE": float(np.quantile(np.abs(y_pred[mask] - y_true[mask]), 0.90)),
            "nRMSE": float(model_rmse / (np.mean(y_true[mask]) + 1e-8)),
        }
        if ref_rmse is not None:
            row["skill_RMSE_vs_CSI"] = forecast_skill_score(model_rmse**2, ref_rmse**2)
        rows.append(row)
    return pd.DataFrame(rows)


def skill_score_metric(model_metric: float, reference_metric: float, eps: float = 1e-8) -> float:
    """Calcule un skill score directement sur une métrique comme la RMSE ou la MAE."""
    return 1.0 - (model_metric / (reference_metric + eps))


def centered_spatial_fields(y: np.ndarray) -> np.ndarray:
    """Retire la moyenne spatiale de chaque carte pour conserver la structure locale."""
    y = np.asarray(y, dtype=np.float64)
    return y - y.mean(axis=(2, 3), keepdims=True)


def evaluate_spatial_structure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: list[int] | tuple[int, ...] = FORECAST_HORIZONS_MINUTES,
) -> pd.DataFrame:
    """
    Évalue uniquement la structure spatiale après retrait du niveau moyen d'irradiance.

    Si un modèle améliore RMSE_spatial_mean mais pas RMSE_structure, il corrige
    surtout le niveau moyen et non la forme du motif nuageux.
    """
    y_true_c = centered_spatial_fields(y_true)
    y_pred_c = centered_spatial_fields(y_pred)

    rows = []
    for h_idx, horizon in enumerate(horizons):
        rows.append(
            {
                "horizon_min": horizon,
                "RMSE_structure": rmse(y_true_c[:, h_idx], y_pred_c[:, h_idx]),
                "MAE_structure": mae(y_true_c[:, h_idx], y_pred_c[:, h_idx]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_by_horizon_detailed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reference_pred: np.ndarray | None = None,
    horizons: list[int] | tuple[int, ...] = FORECAST_HORIZONS_MINUTES,
) -> pd.DataFrame:
    """
    Produit des diagnostics détaillés par horizon.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    rows = []
    for h_idx, horizon in enumerate(horizons):
        yt = y_true[:, h_idx]
        yp = y_pred[:, h_idx]

        yt_mean = yt.mean(axis=(1, 2))
        yp_mean = yp.mean(axis=(1, 2))
        if np.std(yt_mean) < 1e-12 or np.std(yp_mean) < 1e-12:
            corr_mean = np.nan
        else:
            corr_mean = float(np.corrcoef(yt_mean, yp_mean)[0, 1])

        row = {
            "horizon_min": horizon,
            "RMSE": rmse(yt, yp),
            "MAE": mae(yt, yp),
            "bias": mean_bias_error(yt, yp),
            "P90AE": p90_absolute_error(yt, yp),
            "corr_spatial_mean": corr_mean,
        }

        if reference_pred is not None:
            yr = np.asarray(reference_pred, dtype=np.float64)[:, h_idx]
            ref_rmse = rmse(yt, yr)
            ref_mae = mae(yt, yr)
            row["skill_RMSE_vs_ref"] = skill_score_metric(row["RMSE"], ref_rmse)
            row["skill_MAE_vs_ref"] = skill_score_metric(row["MAE"], ref_mae)

        rows.append(row)

    return pd.DataFrame(rows)


def build_model_diagnostics(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    reference_name: str = "persistence_csi",
) -> dict[str, pd.DataFrame]:
    """
    Construit toutes les tables de comparaison à partir d'un dictionnaire de prédictions.
    """
    if reference_name not in predictions:
        raise KeyError(f"reference_name='{reference_name}' not found in predictions.")

    reference_pred = predictions[reference_name]

    global_rows = []
    by_horizon_tables = []
    spatial_mean_tables = []
    spatial_structure_tables = []

    ref_rmse = rmse(y_true, reference_pred)

    for model_name, pred in predictions.items():
        global_rows.append(global_metrics_row(model_name, y_true, pred, reference_rmse=ref_rmse))

        by_h = evaluate_by_horizon_detailed(
            y_true,
            pred,
            reference_pred=reference_pred,
        ).assign(model=model_name)
        by_horizon_tables.append(by_h)

        sp_mean = evaluate_spatial_means(y_true, pred).assign(model=model_name)
        spatial_mean_tables.append(sp_mean)

        sp_struct = evaluate_spatial_structure(y_true, pred).assign(model=model_name)
        spatial_structure_tables.append(sp_struct)

    global_df = pd.DataFrame(global_rows).sort_values("RMSE").reset_index(drop=True)
    by_horizon_df = pd.concat(by_horizon_tables, ignore_index=True)
    spatial_mean_df = pd.concat(spatial_mean_tables, ignore_index=True)
    spatial_structure_df = pd.concat(spatial_structure_tables, ignore_index=True)

    return {
        "global": global_df,
        "by_horizon": by_horizon_df,
        "spatial_means": spatial_mean_df,
        "spatial_structure": spatial_structure_df,
    }


def cluster_balance_report(labels: np.ndarray) -> pd.DataFrame:
    """Décrit les effectifs et proportions de chaque cluster."""
    labels = np.asarray(labels)
    counts = pd.Series(labels).value_counts().sort_index()
    shares = counts / counts.sum()
    return pd.DataFrame(
        {
            "cluster": counts.index.astype(int),
            "n": counts.values.astype(int),
            "share": shares.values,
        }
    )

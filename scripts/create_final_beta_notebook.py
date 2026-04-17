"""Create the final_beta notebook following the requested A-E structure."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Copernicus_notebook_final_beta.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


cells = [
    md(
        """
        # Copernicus Solar Forecasting - final beta

        Structure du rendu final:

        - Bloc A: Baselines
        - Bloc B: ML tabulaire
        - Bloc C: Non supervise
        - Bloc D: Deep Learning
        - Bloc E: Interpretation

        La baseline de reference reste `persistence_csi`. Tous les skill scores RMSE sont calcules relativement a cette reference.
        """
    ),
    md(
        """
        ## Etat d'avancement

        Fait:

        - Persistance brute, persistance CSI et baseline advective CSI.
        - Features tabulaires: statistiques globales, CSI, angles, motion, texture, spatial dynamics.
        - Modeles ML: Ridge/ElasticNet, ExtraTrees, HistGradientBoosting, RandomForest.
        - Clustering KMeans avec interpretation des regimes et performances par cluster.
        - CNN residuel et ConvLSTM residuel disponibles si TensorFlow est installe, avec sauvegarde possible des sorties Colab.
        - XAI: feature importance native, permutation importance, SHAP optionnel sur arbre.
        - Figures: sequences GHI/CLS/CSI, vecteur de mouvement, verite/prediction/erreur, performance par horizon, performance par cluster.

        Reste a faire pour la version finale:

        - Relancer avec `PROFILE = "sample"` puis `PROFILE = "full"` si le temps machine le permet.
        - Executer le Bloc D sur Google Colab avec GPU et TensorFlow.
        - Copier les predictions DL sauvegardees dans `OUTPUT_DIR`, puis relancer la comparaison finale.
        - Remplacer les commentaires "lecture" par les valeurs finales obtenues sur `sample` ou `full`.
        """
    ),
    md("## 0. Setup"),
    code(
        """
        from pathlib import Path
        import sys
        import warnings

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import display

        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.decomposition import PCA
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.model_selection import ParameterGrid
        from sklearn.preprocessing import StandardScaler

        ROOT = Path.cwd()
        if ROOT.name == "notebooks":
            ROOT = ROOT.parent
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from config import FORECAST_HORIZONS_MINUTES, INPUT_VARIABLES
        from src.baselines import mean_image_baseline, persistence_csi_baseline, persistence_last_ghi_baseline
        from src.data_loading import open_processed_profile, prepare_processed_profile, processed_profile_exists
        from src.deep_learning import (
            build_convlstm_residual_model,
            build_small_residual_cnn,
            fit_mlp_residual_mean,
            has_tensorflow,
            prepare_cnn_training_data,
            prepare_convlstm_training_data,
            target_from_channels_last,
            add_residual_mean_to_baseline,
        )
        from src.eda import descriptive_stats, target_horizon_stats
        from src.experiment_io import list_saved_outputs, load_saved_predictions, save_prediction_bundle
        from src.features import (
            build_advanced_features,
            build_exogenous_features,
            build_physical_inputs,
            build_spatial_dynamics_features,
            build_spatial_feature_tensor,
            build_tabular_features,
        )
        from src.interpretation import (
            compute_tree_shap_values,
            model_feature_importances,
            permutation_importance_multioutput,
        )
        from src.metrics import (
            build_model_diagnostics,
            cluster_balance_report,
            cluster_quality,
            evaluate_model_bundle,
            global_metrics_row,
            metrics_by_cluster,
            rmse,
            spatial_mean_residual,
        )
        from src.motion import advective_csi_baseline, estimate_motion_vectors, has_opencv
        from src.preprocessing import temporal_train_validation_split
        from src.texture import build_texture_features
        from src.visualization import (
            horizon_titles,
            plot_cluster_metric,
            plot_forecast_triplet,
            plot_metric_by_horizon,
            plot_motion_summary,
            plot_sequence,
        )
        from models.models_tabular import (
            fit_elasticnet_multioutput,
            fit_extra_trees_multioutput,
            fit_hist_gb_multioutput,
            fit_random_forest_multioutput,
            fit_ridge_multioutput,
            patchwise_predictions_to_map,
            patchwise_target_means,
        )

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        """
    ),
    code(
        """
        RANDOM_STATE = 42
        PROFILE = "dev"  # "dev" pour smoke test, puis "sample" ou "full"
        MAIN_REFERENCE_NAME = "persistence_csi"

        RUN_FAST = PROFILE == "dev"
        RUN_LOCAL_ML = True
        RUN_DEEP_LEARNING = False  # mettre True sur Colab
        RUN_XAI = True
        RUN_SHAP = False  # mettre True si shap est installe

        SAVE_OUTPUTS = True
        LOAD_SAVED_OUTPUTS = True
        OVERWRITE_OUTPUTS = True
        OUTPUT_DIR = ROOT / "reports" / "model_outputs" / "final_beta" / PROFILE
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        np.random.seed(RANDOM_STATE)
        print("ROOT:", ROOT)
        print("OUTPUT_DIR:", OUTPUT_DIR)
        """
    ),
    md("## 1. Chargement, split et EDA rapide"),
    code(
        """
        if not processed_profile_exists(PROFILE, split="train"):
            prepare_processed_profile(PROFILE, split="train", overwrite=False)

        data = open_processed_profile(PROFILE, split="train", variables=INPUT_VARIABLES, mmap_mode=None)
        arrays = {name: np.asarray(values, dtype=np.float32) for name, values in data["X"].items()}
        y = np.asarray(data["y"], dtype=np.float32)

        train_idx, val_idx = temporal_train_validation_split(len(y), validation_fraction=0.2)
        train_arrays_raw = {name: values[train_idx] for name, values in arrays.items()}
        val_arrays_raw = {name: values[val_idx] for name, values in arrays.items()}
        train_target = y[train_idx]
        val_target = y[val_idx]

        print(f"Profil: {PROFILE}")
        print(f"Samples: total={len(y)}, train={len(train_idx)}, validation={len(val_idx)}")
        print({name: value.shape for name, value in arrays.items()})
        display(descriptive_stats(arrays, y))
        display(target_horizon_stats(y))
        """
    ),
    code(
        """
        sample_id = 0
        sample_phys = build_physical_inputs({name: values[[sample_id]] for name, values in arrays.items()}, keep_raw_ghi=True)
        sample_csi = sample_phys["CSI"][0]

        plot_sequence(arrays["GHI"][sample_id], titles=["t-45", "t-30", "t-15", "t"], suptitle="Sequence GHI passee")
        plt.show()
        plot_sequence(arrays["CLS"][sample_id], titles=["t-45", "t-30", "t-15", "t", "t+15", "t+30", "t+45", "t+60"], suptitle="Sequence CLS")
        plt.show()
        plot_sequence(sample_csi, titles=["t-45", "t-30", "t-15", "t"], suptitle="Sequence CSI passee")
        plt.show()
        """
    ),
    md("## 2. Construction des features communes"),
    code(
        """
        train_phys = build_physical_inputs(train_arrays_raw, keep_raw_ghi=True, encode_angles=True)
        val_phys = build_physical_inputs(val_arrays_raw, keep_raw_ghi=True, encode_angles=True)

        train_motion = estimate_motion_vectors(train_phys["CSI"], use_farneback=None).add_prefix("flow_")
        val_motion = estimate_motion_vectors(val_phys["CSI"], use_farneback=None).add_prefix("flow_")
        train_texture = build_texture_features(train_phys, variable="CSI", time_index=-1).add_prefix("tex_")
        val_texture = build_texture_features(val_phys, variable="CSI", time_index=-1).add_prefix("tex_")
        train_spatial_dyn = build_spatial_dynamics_features(train_arrays_raw).add_prefix("spdyn_")
        val_spatial_dyn = build_spatial_dynamics_features(val_arrays_raw).add_prefix("spdyn_")
        train_exogenous = build_exogenous_features(train_arrays_raw).add_prefix("exo_")
        val_exogenous = build_exogenous_features(val_arrays_raw).add_prefix("exo_")

        feature_blocks_train = [
            build_tabular_features(train_phys).add_prefix("phys_"),
            build_advanced_features(train_arrays_raw).add_prefix("adv_"),
            train_motion,
            train_texture,
            train_spatial_dyn,
        ]
        feature_blocks_val = [
            build_tabular_features(val_phys).add_prefix("phys_"),
            build_advanced_features(val_arrays_raw).add_prefix("adv_"),
            val_motion,
            val_texture,
            val_spatial_dyn,
        ]
        if len(train_exogenous.columns):
            feature_blocks_train.append(train_exogenous)
            feature_blocks_val.append(val_exogenous)
        else:
            print("Vent U/V absent des fichiers: pas de features exogenes ajoutees.")

        X_train_features = pd.concat(feature_blocks_train, axis=1)
        X_val_features = pd.concat(feature_blocks_val, axis=1)
        common_cols = [col for col in X_train_features.columns if col in X_val_features.columns]
        X_train_features = X_train_features[common_cols].replace([np.inf, -np.inf], np.nan)
        X_val_features = X_val_features[common_cols].replace([np.inf, -np.inf], np.nan)
        medians = X_train_features.median(numeric_only=True).fillna(0.0)
        X_train_features = X_train_features.fillna(medians)
        X_val_features = X_val_features.fillna(medians)

        scaler = StandardScaler()
        X_train_tab = scaler.fit_transform(X_train_features).astype(np.float32)
        X_val_tab = scaler.transform(X_val_features).astype(np.float32)
        feature_names = list(X_train_features.columns)

        print("OpenCV disponible:", has_opencv())
        print("Feature matrix:", X_train_tab.shape)
        display(X_train_features.describe().T.head(30))
        plot_motion_summary(train_motion, sample_idx=0)
        plt.show()
        """
    ),
    md(
        """
        ## Bloc A - Baselines

        Baselines retenues:

        - Persistance brute: repete la derniere image GHI.
        - Persistance CSI: repete le dernier clear-sky index puis remultiplie par le CLS futur.
        - Advective CSI: deplace le dernier CSI selon le mouvement nuageux estime.
        """
    ),
    code(
        """
        y_pred_persistence_raw = persistence_last_ghi_baseline(val_arrays_raw)
        y_pred_persistence_csi = persistence_csi_baseline(val_arrays_raw)
        y_pred_train_csi = persistence_csi_baseline(train_arrays_raw)
        y_pred_advective_csi = advective_csi_baseline(
            val_arrays_raw,
            motion_features=val_motion.rename(columns=lambda col: col.replace("flow_", "")),
        )
        y_pred_mean = mean_image_baseline(train_target, n_samples=len(val_target))

        predictions = {
            "persistence_raw": y_pred_persistence_raw,
            "persistence_csi": y_pred_persistence_csi,
            "advective_csi": y_pred_advective_csi,
            "mean_image": y_pred_mean,
        }
        baseline_diagnostics = build_model_diagnostics(val_target, predictions, reference_name=MAIN_REFERENCE_NAME)
        display(baseline_diagnostics["global"])
        display(baseline_diagnostics["by_horizon"].sort_values(["horizon_min", "RMSE"]))
        """
    ),
    md(
        """
        ## Bloc B - ML tabulaire

        Les modeles apprennent une correction residuelle moyenne par horizon par rapport a la persistance CSI. Cela force les modeles tabulaires a corriger une baseline physique forte plutot qu'a reconstruire toute l'image.
        """
    ),
    code(
        """
        residual_train = spatial_mean_residual(train_target, y_pred_train_csi)
        residual_val = spatial_mean_residual(val_target, y_pred_persistence_csi)
        model_registry = {}
        supervised_predictions = {}

        if RUN_LOCAL_ML:
            ridge_model = fit_ridge_multioutput(X_train_tab, residual_train, alpha=10.0, random_state=RANDOM_STATE)
            supervised_predictions["ridge_residual_csi"] = add_residual_mean_to_baseline(y_pred_persistence_csi, ridge_model.predict(X_val_tab))
            model_registry["ridge_residual_csi"] = ridge_model

            elastic_model = fit_elasticnet_multioutput(
                X_train_tab,
                residual_train,
                alpha=0.001,
                l1_ratio=0.2,
                random_state=RANDOM_STATE,
                max_iter=2000,
            )
            supervised_predictions["elasticnet_residual_csi"] = add_residual_mean_to_baseline(y_pred_persistence_csi, elastic_model.predict(X_val_tab))
            model_registry["elasticnet_residual_csi"] = elastic_model

            extra_model = fit_extra_trees_multioutput(
                X_train_tab,
                residual_train,
                n_estimators=40 if RUN_FAST else 250,
                max_depth=8 if RUN_FAST else None,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
            )
            supervised_predictions["extra_trees_residual_csi"] = add_residual_mean_to_baseline(y_pred_persistence_csi, extra_model.predict(X_val_tab))
            model_registry["extra_trees_residual_csi"] = extra_model

            rf_model = fit_random_forest_multioutput(
                X_train_tab,
                residual_train,
                n_estimators=30 if RUN_FAST else 200,
                max_depth=6 if RUN_FAST else None,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
            )
            supervised_predictions["random_forest_residual_csi"] = add_residual_mean_to_baseline(y_pred_persistence_csi, rf_model.predict(X_val_tab))
            model_registry["random_forest_residual_csi"] = rf_model
        """
    ),
    code(
        """
        hgb_grid = {
            "learning_rate": [0.05, 0.10] if RUN_FAST else [0.03, 0.05, 0.08],
            "max_iter": [40] if RUN_FAST else [150, 250],
            "max_leaf_nodes": [15, 31],
            "min_samples_leaf": [5] if RUN_FAST else [10, 20],
            "l2_regularization": [0.0] if RUN_FAST else [0.0, 0.1],
        }
        inner_train_idx, inner_val_idx = temporal_train_validation_split(len(residual_train), validation_fraction=0.25)
        hgb_rows = []
        best_hgb_score = np.inf
        best_hgb_params = None

        for params in ParameterGrid(hgb_grid):
            candidate = fit_hist_gb_multioutput(
                X_train_tab[inner_train_idx],
                residual_train[inner_train_idx],
                random_state=RANDOM_STATE,
                **params,
            )
            pred = candidate.predict(X_train_tab[inner_val_idx])
            score = float(np.sqrt(np.mean((pred - residual_train[inner_val_idx]) ** 2)))
            row = dict(params)
            row["inner_RMSE_residual_mean"] = score
            hgb_rows.append(row)
            if score < best_hgb_score:
                best_hgb_score = score
                best_hgb_params = params

        hgb_tuning = pd.DataFrame(hgb_rows).sort_values("inner_RMSE_residual_mean")
        display(hgb_tuning)
        print("Best HistGB params:", best_hgb_params)

        if RUN_LOCAL_ML:
            hgb_model = fit_hist_gb_multioutput(X_train_tab, residual_train, random_state=RANDOM_STATE, **best_hgb_params)
            supervised_predictions["hist_gb_residual_csi"] = add_residual_mean_to_baseline(y_pred_persistence_csi, hgb_model.predict(X_val_tab))
            model_registry["hist_gb_residual_csi"] = hgb_model

        predictions.update(supervised_predictions)
        ml_diagnostics = build_model_diagnostics(val_target, predictions, reference_name=MAIN_REFERENCE_NAME)
        display(ml_diagnostics["global"])
        plot_metric_by_horizon(ml_diagnostics["by_horizon"], models=list(ml_diagnostics["global"]["model"].head(6)), metric="RMSE")
        plt.show()
        """
    ),
    md("### Variante spatiale patch-wise"),
    code(
        """
        spatial_predictions = {}
        residual_train_maps = train_target - y_pred_train_csi
        residual_patch_train, patch_names = patchwise_target_means(residual_train_maps, n_rows=2, n_cols=2)
        residual_patch_val, _ = patchwise_target_means(val_target - y_pred_persistence_csi, n_rows=2, n_cols=2)

        if RUN_LOCAL_ML:
            patch_hgb_model = fit_hist_gb_multioutput(
                X_train_tab,
                residual_patch_train,
                learning_rate=0.05,
                max_iter=60 if RUN_FAST else 180,
                max_leaf_nodes=15,
                min_samples_leaf=5 if RUN_FAST else 10,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            )
            patch_pred = patch_hgb_model.predict(X_val_tab)
            spatial_predictions["hist_gb_patch2x2_residual_csi"] = patchwise_predictions_to_map(
                y_pred_persistence_csi,
                patch_pred,
                n_rows=2,
                n_cols=2,
            )
            model_registry["hist_gb_patch2x2_residual_csi"] = patch_hgb_model

        predictions.update(spatial_predictions)
        patch_diagnostics = build_model_diagnostics(val_target, predictions, reference_name=MAIN_REFERENCE_NAME)
        display(patch_diagnostics["global"])
        """
    ),
    md(
        """
        ## Bloc C - Non supervise

        Le clustering est realise sur les features meteo, texture et motion. Il sert a interpreter les regimes de ciel, puis a evaluer la robustesse des modeles par regime.
        """
    ),
    code(
        """
        cluster_feature_names = [
            col for col in feature_names
            if any(token in col for token in ["CSI_mean", "CSI_std", "CSI_trend", "GHI_mean", "GHI_std", "flow_", "tex_", "spdyn_"])
        ]
        if len(cluster_feature_names) < 3:
            cluster_feature_names = feature_names

        X_cluster_train = X_train_features[cluster_feature_names].to_numpy(dtype=np.float32)
        X_cluster_val = X_val_features[cluster_feature_names].to_numpy(dtype=np.float32)
        cluster_scaler = StandardScaler()
        X_cluster_train_scaled = cluster_scaler.fit_transform(X_cluster_train)
        X_cluster_val_scaled = cluster_scaler.transform(X_cluster_val)

        cluster_pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
        X_cluster_train_emb = cluster_pca.fit_transform(X_cluster_train_scaled)
        X_cluster_val_emb = cluster_pca.transform(X_cluster_val_scaled)

        k_values = [2, 3] if RUN_FAST else [2, 3, 4, 5]
        cluster_rows = []
        best_kmeans = None
        best_score = -np.inf
        min_cluster_size_required = max(3, int(0.05 * len(X_cluster_train_emb)))
        for k in k_values:
            km = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE)
            labels = km.fit_predict(X_cluster_train_emb)
            quality = cluster_quality(X_cluster_train_emb, labels)
            qmap = dict(zip(quality["metric"], quality["value"])) if not quality.empty else {}
            counts = pd.Series(labels).value_counts()
            row = {
                "method": "kmeans",
                "k": k,
                "silhouette": qmap.get("silhouette", np.nan),
                "calinski_harabasz": qmap.get("calinski_harabasz", np.nan),
                "davies_bouldin": qmap.get("davies_bouldin", np.nan),
                "min_cluster_size": int(counts.min()),
            }
            cluster_rows.append(row)
            if row["min_cluster_size"] >= min_cluster_size_required and row["silhouette"] > best_score:
                best_score = row["silhouette"]
                best_kmeans = km

        cluster_search = pd.DataFrame(cluster_rows).sort_values("silhouette", ascending=False)
        display(cluster_search)
        if best_kmeans is None:
            best_k = int(cluster_search.iloc[0]["k"])
            best_kmeans = KMeans(n_clusters=best_k, n_init=30, random_state=RANDOM_STATE).fit(X_cluster_train_emb)

        train_clusters = best_kmeans.predict(X_cluster_train_emb)
        val_clusters = best_kmeans.predict(X_cluster_val_emb)
        display(cluster_balance_report(train_clusters))
        display(cluster_balance_report(val_clusters))
        """
    ),
    code(
        """
        cluster_frame = X_val_features.copy()
        cluster_frame["cluster"] = val_clusters
        csi_col = "phys_CSI_mean_t3" if "phys_CSI_mean_t3" in cluster_frame.columns else cluster_feature_names[0]
        texture_col = "tex_csi_t-1_glcm_entropy" if "tex_csi_t-1_glcm_entropy" in cluster_frame.columns else csi_col
        motion_col = "flow_motion_speed_last" if "flow_motion_speed_last" in cluster_frame.columns else csi_col

        cluster_summary = (
            cluster_frame
            .groupby("cluster")
            .agg(
                n=(csi_col, "size"),
                csi_mean=(csi_col, "mean"),
                texture_entropy=(texture_col, "mean"),
                motion_speed=(motion_col, "mean"),
            )
            .sort_values(["csi_mean", "texture_entropy"])
        )

        regime_names = ["couvert", "partiellement_nuageux", "ciel_clair", "tres_variable", "mixte"]
        cluster_name_map = {cluster: regime_names[i] for i, cluster in enumerate(cluster_summary.index)}
        cluster_summary["regime"] = [cluster_name_map[idx] for idx in cluster_summary.index]
        display(cluster_summary.reset_index())

        cluster_perf_tables = []
        for model_name in [MAIN_REFERENCE_NAME, patch_diagnostics["global"].iloc[0]["model"]]:
            cluster_perf_tables.append(
                metrics_by_cluster(
                    val_target,
                    predictions[model_name],
                    val_clusters,
                    model_name,
                    cluster_name_map=cluster_name_map,
                    reference_pred=predictions[MAIN_REFERENCE_NAME],
                )
            )
        cluster_perf = pd.concat(cluster_perf_tables, ignore_index=True)
        display(cluster_perf.sort_values(["regime", "RMSE"]))
        plot_cluster_metric(cluster_perf, metric="RMSE")
        plt.show()
        """
    ),
    md(
        """
        ## Bloc D - Deep Learning

        A executer sur Colab pour TensorFlow/GPU:

        - CNN residuel: entree cartes en canaux, sortie residual maps.
        - ConvLSTM residuel: entree sequentielle `(CSI, GHI, CLS, SZA/SAA sin/cos)`, sortie residual maps.

        Les predictions DL peuvent etre sauvegardees dans `OUTPUT_DIR`, puis rechargees localement dans la comparaison finale.
        """
    ),
    code(
        """
        dl_predictions = {}
        if RUN_DEEP_LEARNING:
            spatial_train_tensor, spatial_feature_names = build_spatial_feature_tensor(
                build_physical_inputs(train_arrays_raw, keep_raw_ghi=False, encode_angles=True)
            )
            spatial_val_tensor, _ = build_spatial_feature_tensor(
                build_physical_inputs(val_arrays_raw, keep_raw_ghi=False, encode_angles=True)
            )
            X_train_cnn, y_train_cnn = prepare_cnn_training_data(spatial_train_tensor, train_target, baseline=y_pred_train_csi)
            X_val_cnn, y_val_cnn = prepare_cnn_training_data(spatial_val_tensor, val_target, baseline=y_pred_persistence_csi)

            X_train_convlstm, y_train_convlstm, convlstm_channels = prepare_convlstm_training_data(
                train_arrays_raw,
                train_target,
                baseline=y_pred_train_csi,
            )
            X_val_convlstm, y_val_convlstm, _ = prepare_convlstm_training_data(
                val_arrays_raw,
                val_target,
                baseline=y_pred_persistence_csi,
            )

            if has_tensorflow():
                cnn_model = build_small_residual_cnn(
                    input_shape=X_train_cnn.shape[1:],
                    n_horizons=len(FORECAST_HORIZONS_MINUTES),
                    learning_rate=1e-3,
                )
                cnn_model.fit(
                    X_train_cnn,
                    y_train_cnn,
                    validation_data=(X_val_cnn, y_val_cnn),
                    epochs=3 if RUN_FAST else 20,
                    batch_size=8 if RUN_FAST else 32,
                    verbose=1,
                )
                cnn_residual = target_from_channels_last(cnn_model.predict(X_val_cnn, verbose=0))
                dl_predictions["cnn_residual_csi_fullmap"] = np.maximum(y_pred_persistence_csi + cnn_residual, 0.0)

                convlstm_model = build_convlstm_residual_model(
                    input_shape=X_train_convlstm.shape[1:],
                    n_horizons=len(FORECAST_HORIZONS_MINUTES),
                    learning_rate=1e-3,
                )
                convlstm_model.fit(
                    X_train_convlstm,
                    y_train_convlstm,
                    validation_data=(X_val_convlstm, y_val_convlstm),
                    epochs=3 if RUN_FAST else 25,
                    batch_size=4 if RUN_FAST else 16,
                    verbose=1,
                )
                convlstm_residual = target_from_channels_last(convlstm_model.predict(X_val_convlstm, verbose=0))
                dl_predictions["convlstm_residual_csi_fullmap"] = np.maximum(y_pred_persistence_csi + convlstm_residual, 0.0)
            else:
                print("TensorFlow indisponible: executer cette cellule sur Colab.")
        else:
            print("RUN_DEEP_LEARNING=False: bloc DL saute dans cet environnement.")

        if len(dl_predictions):
            predictions.update(dl_predictions)
            display(build_model_diagnostics(val_target, predictions, reference_name=MAIN_REFERENCE_NAME)["global"])
        """
    ),
    md("## Sauvegarde et rechargement des outputs"),
    code(
        """
        if SAVE_OUTPUTS:
            manifest = save_prediction_bundle(
                predictions,
                y_true=val_target,
                output_dir=OUTPUT_DIR,
                profile=PROFILE,
                backend="final_beta_colab" if RUN_DEEP_LEARNING else "final_beta_local",
                reference_name=MAIN_REFERENCE_NAME,
                extra_metadata={"notebook": "Copernicus_notebook_final_beta.ipynb"},
                overwrite=OVERWRITE_OUTPUTS,
            )
            display(manifest)

        saved_listing = list_saved_outputs(OUTPUT_DIR)
        display(saved_listing)

        if LOAD_SAVED_OUTPUTS:
            saved_predictions = load_saved_predictions(OUTPUT_DIR, strict_shape=tuple(val_target.shape))
            predictions.update(saved_predictions)
        """
    ),
    md(
        """
        ## Comparaison finale propre

        Toutes les predictions disponibles sont comparees avec le meme `val_target`.
        """
    ),
    code(
        """
        final_diagnostics = build_model_diagnostics(val_target, predictions, reference_name=MAIN_REFERENCE_NAME)
        final_table = final_diagnostics["global"]
        display(final_table)
        display(final_diagnostics["by_horizon"].sort_values(["horizon_min", "RMSE"]))
        display(final_diagnostics["spatial_structure"].sort_values(["horizon_min", "RMSE_structure"]))

        top_models = list(final_table["model"].head(min(7, len(final_table))))
        plot_metric_by_horizon(final_diagnostics["by_horizon"], models=top_models, metric="RMSE", title="RMSE par horizon - meilleurs modeles")
        plt.show()

        best_model_name = final_table.iloc[0]["model"]
        print("Meilleur modele global:", best_model_name)
        plot_forecast_triplet(val_target, predictions[best_model_name], sample_idx=0, horizon_idx=0, model_name=best_model_name)
        plt.show()

        winners = final_table.query("model != @MAIN_REFERENCE_NAME and skill_RMSE_vs_CSI > 0")
        if len(winners):
            print("Modeles qui battent la persistance CSI:")
            display(winners[["model", "RMSE", "MAE", "skill_RMSE_vs_CSI"]])
        else:
            print("Aucun modele ne bat la persistance CSI sur ce run.")
        """
    ),
    md(
        """
        ## Bloc E - Interpretation / XAI

        On interprete le meilleur modele arbre disponible. La lecture physique attendue:

        - CSI et CLS expliquent le niveau d'irradiance attendu.
        - Angles solaires capturent la geometrie du soleil.
        - Motion features indiquent si le deplacement nuageux est utile.
        - Texture et spatial dynamics caracterisent la complexite des nuages.
        """
    ),
    code(
        """
        if RUN_XAI:
            candidate_names = [
                "hist_gb_patch2x2_residual_csi",
                "hist_gb_residual_csi",
                "extra_trees_residual_csi",
                "random_forest_residual_csi",
                "ridge_residual_csi",
            ]
            available = [name for name in candidate_names if name in model_registry]
            if not available:
                print("Aucun modele interpretable disponible dans model_registry.")
            else:
                xai_name = (
                    final_table[final_table["model"].isin(available)]
                    .sort_values("RMSE")
                    .iloc[0]["model"]
                )
                xai_model = model_registry[xai_name]
                if "patch2x2" in xai_name:
                    y_xai = residual_patch_val
                else:
                    y_xai = residual_val

                print("Modele interprete:", xai_name)
                native_imp = model_feature_importances(xai_model, feature_names)
                if len(native_imp):
                    display(native_imp.head(25))
                    fig, ax = plt.subplots(figsize=(9, 5))
                    top = native_imp.head(15).sort_values("importance")
                    ax.barh(top["feature"], top["importance"])
                    ax.set_title(f"Feature importance native - {xai_name}")
                    plt.tight_layout()
                    plt.show()

                perm = permutation_importance_multioutput(
                    xai_model,
                    X_val_tab,
                    y_xai,
                    feature_names,
                    n_repeats=3,
                    random_state=RANDOM_STATE,
                    max_features=60 if RUN_FAST else None,
                )
                display(perm.importance.head(25))
                fig, ax = plt.subplots(figsize=(9, 5))
                top = perm.importance.head(15).sort_values("importance_mean")
                ax.barh(top["feature"], top["importance_mean"])
                ax.set_title(f"Permutation importance - {xai_name}")
                plt.tight_layout()
                plt.show()

                if RUN_SHAP:
                    try:
                        shap_values, X_shap, shap_feature_names = compute_tree_shap_values(
                            xai_model,
                            X_val_tab,
                            feature_names,
                            output_index=0,
                            max_samples=100,
                        )
                        import shap
                        shap.plots.bar(shap_values, max_display=15, show=False)
                        plt.tight_layout()
                        plt.show()
                    except Exception as exc:
                        print("SHAP non execute:", exc)
                else:
                    print("RUN_SHAP=False: SHAP garde en option pour l'environnement avec shap installe.")
        """
    ),
    md(
        """
        ## Conclusion beta

        Cette version beta est structuree comme le rendu final. Elle peut etre executee localement pour les blocs A, B, C et E. Le bloc D est prevu pour Colab; ses predictions doivent ensuite etre sauvegardees dans `OUTPUT_DIR` et rechargees dans la comparaison finale.
        """
    ),
]


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)

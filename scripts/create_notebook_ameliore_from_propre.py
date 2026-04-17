"""Create an enhanced notebook from the clean final notebook.

This script intentionally does not edit the source notebook in place.

Input:
    notebooks/Copernicus_notebook_final_rendu_propre.ipynb

Output:
    notebooks/Copernicus_notebook_final_rendu_propre_ameliore.ipynb
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "notebooks" / "Copernicus_notebook_final_rendu_propre.ipynb"
TARGET_NOTEBOOK = ROOT / "notebooks" / "Copernicus_notebook_final_rendu_propre_ameliore.ipynb"


def clean(source: str) -> str:
    return textwrap.dedent(source).strip()


def md(source: str):
    return nbf.v4.new_markdown_cell(clean(source))


def ensure_import(source: str, old: str, new: str) -> str:
    if new in source:
        return source
    return source.replace(old, new)


def clear_outputs(nb) -> None:
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def replace_first_markdown_title(nb) -> None:
    for cell in nb.cells:
        if cell.cell_type == "markdown" and cell.source.lstrip().startswith("# "):
            cell.source = clean(
                """
                # Copernicus Solar Forecasting - notebook final ameliore

                Version generee automatiquement depuis `Copernicus_notebook_final_rendu_propre.ipynb`.

                Ajouts visibles dans cette version:

                - features de mouvement nuageux et baseline `advective_csi`;
                - features de texture spatiale GLCM et Gabor;
                - branchement conditionnel pour variables exogenes de vent `U/V`;
                - section Deep Learning orientee ConvLSTM;
                - clustering enrichi avec mouvement et texture.
                """
            )
            return


def enhance_setup_cell(cell) -> None:
    source = cell.source
    source = ensure_import(
        source,
        "    build_spatial_feature_tensor,\n    build_tabular_features,\n)",
        "    build_spatial_feature_tensor,\n    build_tabular_features,\n    build_exogenous_features,\n)",
    )
    source = ensure_import(
        source,
        "    build_small_residual_cnn,\n    fit_mlp_residual_mean,",
        "    build_convlstm_residual_model,\n    build_small_residual_cnn,\n    fit_mlp_residual_mean,",
    )
    source = ensure_import(
        source,
        "    prepare_cnn_training_data,\n    target_from_channels_last,",
        "    prepare_cnn_training_data,\n    prepare_convlstm_training_data,\n    target_from_channels_last,",
    )
    if "from src.motion import" not in source:
        source = source.replace(
            "from src.preprocessing import temporal_train_validation_split\n",
            "from src.motion import advective_csi_baseline, estimate_motion_vectors, has_opencv\n"
            "from src.texture import build_texture_features\n"
            "from src.preprocessing import temporal_train_validation_split\n",
        )
    if "RUN_OPTICAL_FLOW" not in source:
        source = source.replace(
            "RUN_DL = True   # plutot sur Colab",
            "RUN_DL = True   # plutot sur Colab\nRUN_OPTICAL_FLOW = True\nRUN_TEXTURE_FEATURES = True",
        )
        source = source.replace(
            "RUN_DL = True   # plutôt sur Colab",
            "RUN_DL = True   # plutôt sur Colab\nRUN_OPTICAL_FLOW = True\nRUN_TEXTURE_FEATURES = True",
        )
    cell.source = source


def enhance_load_cell(cell) -> None:
    cell.source = clean(
        """
        if not processed_profile_exists(PROFILE, split="train"):
            prepare_processed_profile(PROFILE, split="train", overwrite=False)

        data = open_processed_profile(PROFILE, split="train", variables=INPUT_VARIABLES, mmap_mode=None)
        arrays = {name: np.asarray(values, dtype=np.float32) for name, values in data["X"].items()}
        y = np.asarray(data["y"], dtype=np.float32)

        available_variables = sorted(arrays)
        wind_variables = [
            name for name in available_variables
            if name.lower() in {"u", "v", "u10", "v10", "wind_u", "wind_v"}
        ]
        if len(wind_variables) == 0:
            print("Variables exogenes vent: non disponibles dans ce jeu Copernicus (GHI/CLS/SZA/SAA seulement).")
        else:
            print("Variables exogenes vent detectees:", wind_variables)

        train_idx, val_idx = temporal_train_validation_split(len(y), validation_fraction=0.2)
        train_arrays_raw = {name: values[train_idx] for name, values in arrays.items()}
        val_arrays_raw = {name: values[val_idx] for name, values in arrays.items()}
        train_target = y[train_idx]
        val_target = y[val_idx]

        print(f"Profil: {PROFILE}")
        print(f"Samples: total={len(y)}, train={len(train_idx)}, validation={len(val_idx)}")
        print({name: value.shape for name, value in arrays.items()})
        print("Target:", y.shape)
        """
    )


def enhance_feature_cells(nb, offset: int = 0) -> None:
    nb.cells[8 + offset].source = clean(
        """
        ## 3. Features enrichies

        Les features conservent le socle physique (`CSI`, `CLS`, angles solaires), puis ajoutent trois ameliorations ciblees.

        - Mouvement nuageux: estimation d'un deplacement dominant entre images successives. Si OpenCV est installe, Farneback peut etre utilise; sinon le notebook utilise une phase-correlation completee par le barycentre de nebulosite.
        - Texture spatiale: descripteurs GLCM et Gabor sur le dernier CSI pour reperer les champs uniformes, stratus, ou les situations tres variables.
        - Variables exogenes: le code accepte des composantes de vent `U/V` si elles existent dans une variante enrichie du dataset. Dans les fichiers actuels du challenge, elles ne sont pas presentes.
        """
    )
    nb.cells[9 + offset].source = clean(
        """
        train_phys = build_physical_inputs(train_arrays_raw, keep_raw_ghi=True, encode_angles=True)
        val_phys = build_physical_inputs(val_arrays_raw, keep_raw_ghi=True, encode_angles=True)

        feature_blocks_train = [
            build_tabular_features(train_phys).add_prefix("phys_"),
            build_advanced_features(train_arrays_raw).add_prefix("adv_"),
        ]
        feature_blocks_val = [
            build_tabular_features(val_phys).add_prefix("phys_"),
            build_advanced_features(val_arrays_raw).add_prefix("adv_"),
        ]

        if RUN_OPTICAL_FLOW:
            train_motion_features = estimate_motion_vectors(train_phys["CSI"], use_farneback=None).add_prefix("flow_")
            val_motion_features = estimate_motion_vectors(val_phys["CSI"], use_farneback=None).add_prefix("flow_")
            feature_blocks_train.append(train_motion_features)
            feature_blocks_val.append(val_motion_features)
            print("Optical flow:", "Farneback/OpenCV" if has_opencv() else "phase-correlation + barycentre fallback")
            display(train_motion_features.describe().T)

        if RUN_TEXTURE_FEATURES:
            train_texture_features = build_texture_features(train_phys, variable="CSI", time_index=-1).add_prefix("tex_")
            val_texture_features = build_texture_features(val_phys, variable="CSI", time_index=-1).add_prefix("tex_")
            feature_blocks_train.append(train_texture_features)
            feature_blocks_val.append(val_texture_features)
            display(train_texture_features.describe().T)

        train_exogenous_features = build_exogenous_features(train_arrays_raw).add_prefix("exo_")
        val_exogenous_features = build_exogenous_features(val_arrays_raw).add_prefix("exo_")
        if len(train_exogenous_features.columns):
            feature_blocks_train.append(train_exogenous_features)
            feature_blocks_val.append(val_exogenous_features)
        else:
            print("Aucune feature exogene vent ajoutee: variables U/V absentes des fichiers.")

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

        print("Feature matrix train:", X_train_tab.shape)
        display(X_train_features.describe().T.head(30))
        """
    )
    nb.cells[11 + offset].source = clean(
        """
        ### Justification des features retenues

        Les features tabulaires resument le niveau moyen, la dispersion et la dynamique temporelle. Les features de mouvement apportent une information directionnelle: elles repondent a la limite de la persistance statique, qui suppose implicitement que les masses nuageuses restent au meme endroit. Les features de texture completent le clustering, car deux situations peuvent avoir le meme CSI moyen mais une organisation spatiale tres differente.
        """
    )


def enhance_baseline_cells(nb, offset: int = 0) -> None:
    nb.cells[13 + offset].source = clean(
        """
        y_pred_persistence_raw = persistence_last_ghi_baseline(val_arrays_raw)
        y_pred_persistence_csi = persistence_csi_baseline(val_arrays_raw)
        y_pred_mean = mean_image_baseline(train_target, n_samples=len(val_target))
        y_pred_train_csi = persistence_csi_baseline(train_arrays_raw)

        # Persistance advective: on deplace le dernier CSI selon le mouvement estime.
        if RUN_OPTICAL_FLOW:
            y_pred_advective_csi = advective_csi_baseline(
                val_arrays_raw,
                motion_features=val_motion_features.rename(columns=lambda col: col.replace("flow_", "")),
            )
            y_pred_train_advective_csi = advective_csi_baseline(
                train_arrays_raw,
                motion_features=train_motion_features.rename(columns=lambda col: col.replace("flow_", "")),
            )
        else:
            y_pred_advective_csi = y_pred_persistence_csi
            y_pred_train_advective_csi = y_pred_train_csi

        predictions = {
            "persistence_raw": y_pred_persistence_raw,
            "persistence_csi": y_pred_persistence_csi,
            "advective_csi": y_pred_advective_csi,
            "mean_image": y_pred_mean,
        }

        csi_reference_rmse = rmse(val_target, y_pred_persistence_csi)
        baseline_table = pd.DataFrame(
            [
                global_metrics_row(name, val_target, pred, csi_reference_rmse)
                for name, pred in predictions.items()
            ]
        ).sort_values("RMSE")
        display(baseline_table)
        """
    )
    nb.cells[14 + offset].source = clean(
        """
        ### Analyse des baselines

        La baseline `persistence_csi` reste la reference du skill score. `advective_csi` teste une hypothese plus ambitieuse: conserver la structure de CSI mais la deplacer dans la direction estimee du mouvement nuageux. Sur petit echantillon, elle peut etre instable; sur `sample` ou `full`, elle indique si le mouvement apporte reellement de l'information.
        """
    )


def enhance_clustering_cell(cell) -> None:
    source = cell.source
    source = source.replace(
        '["CSI_mean", "CSI_std", "CSI_trend", "GHI_mean", "GHI_std", "adv_csi"]',
        '["CSI_mean", "CSI_std", "CSI_trend", "GHI_mean", "GHI_std", "adv_csi", "flow_", "tex_"]',
    )
    if "best_kmeans is None" not in source:
        source = source.replace(
            clean(
                """
                cluster_search = pd.DataFrame(cluster_rows).sort_values("silhouette", ascending=False)
                display(cluster_search)
                print("Best k:", best_k)

                train_clusters = best_kmeans.predict(X_cluster_train_emb)
                val_clusters = best_kmeans.predict(X_cluster_val_emb)
                """
            ),
            clean(
                """
                cluster_search = pd.DataFrame(cluster_rows).sort_values("silhouette", ascending=False)
                display(cluster_search)

                if best_kmeans is None:
                    best_k = int(cluster_search.iloc[0]["k"])
                    best_kmeans = KMeans(n_clusters=best_k, n_init=30, random_state=RANDOM_STATE).fit(X_cluster_train_emb)
                    print("Aucun k ne respecte la contrainte de taille sur ce petit profil; fallback au meilleur silhouette.")

                print("Best k:", best_k)

                train_clusters = best_kmeans.predict(X_cluster_train_emb)
                val_clusters = best_kmeans.predict(X_cluster_val_emb)
                """
            ),
        )
    cell.source = source


def enhance_deep_learning_cells(nb, offset: int = 0) -> None:
    nb.cells[42 + offset].source = clean(
        """
        dl_predictions = {}

        if RUN_DL:
            # Baseline neuronale tabulaire legere, toujours executable avec scikit-learn.
            mlp_model = fit_mlp_residual_mean(
                X_train_tab,
                residual_train,
                hidden_layer_sizes=(64, 32),
                random_state=RANDOM_STATE,
                max_iter=80 if RUN_FAST else 300,
            )
            mlp_residual = mlp_model.predict(X_val_tab)
            dl_predictions["mlp_residual_csi"] = add_residual_mean_to_baseline(
                y_pred_persistence_csi,
                mlp_residual,
            )

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
            print("ConvLSTM input:", X_train_convlstm.shape, "| channels:", convlstm_channels)

            if has_tensorflow():
                convlstm_model = build_convlstm_residual_model(
                    input_shape=X_train_convlstm.shape[1:],
                    n_horizons=len(FORECAST_HORIZONS_MINUTES),
                    learning_rate=1e-3,
                )
                history = convlstm_model.fit(
                    X_train_convlstm,
                    y_train_convlstm,
                    validation_data=(X_val_convlstm, y_val_convlstm),
                    epochs=3 if RUN_FAST else 25,
                    batch_size=4 if RUN_FAST else 16,
                    verbose=0,
                )
                residual_pred_val = target_from_channels_last(convlstm_model.predict(X_val_convlstm, verbose=0))
                y_pred_convlstm = np.maximum(y_pred_persistence_csi + residual_pred_val, 0.0)
                dl_predictions["convlstm_residual_csi_fullmap"] = y_pred_convlstm
                print("ConvLSTM executee.")
            else:
                print("TensorFlow indisponible : ConvLSTM non executee.")
        else:
            print("RUN_DL = False : section deep learning sautee.")
        """
    )
    nb.cells[43 + offset].source = clean(
        """
        ### Pourquoi garder cette section

        La CNN simple traite les images passees comme des canaux independants. Une ConvLSTM est mieux alignee avec la problematique: elle lit une sequence d'images et apprend directement la dynamique spatio-temporelle. Le notebook tente donc une ConvLSTM si TensorFlow est disponible; sinon il garde un MLP residuel comme fallback leger pour que le rendu reste executable.
        """
    )


def enhance_conclusion(nb) -> None:
    nb.cells[-1].source = clean(
        """
        ## Conclusion operationnelle

        Les ameliorations ajoutees renforcent la logique physique du notebook:

        - la persistance CSI reste la reference principale du skill score;
        - l'advective CSI teste explicitement l'apport du deplacement nuageux;
        - les modeles supervises exploitent desormais des features de mouvement et de texture;
        - le clustering peut expliquer les regimes non seulement par le niveau de CSI, mais aussi par la complexite spatiale;
        - la section deep learning est prete pour une ConvLSTM, plus adaptee qu'une CNN simple au caractere sequentiel du probleme.

        Les variables de vent ne sont pas presentes dans les fichiers Copernicus fournis. Le pipeline accepte toutefois des champs `U/V` si une source meteo externe ou une variante enrichie du challenge est ajoutee plus tard.
        """
    )


def main() -> None:
    if not SOURCE_NOTEBOOK.exists():
        raise FileNotFoundError(SOURCE_NOTEBOOK)

    source_nb = nbf.read(SOURCE_NOTEBOOK, as_version=4)
    nb = deepcopy(source_nb)

    replace_first_markdown_title(nb)
    nb.cells.insert(
        1,
        md(
            """
            ## Note de generation

            Ce fichier est une nouvelle sortie separee. Le notebook propre d'origine n'est pas modifie par ce script.

            Fichier source: `Copernicus_notebook_final_rendu_propre.ipynb`  
            Fichier genere: `Copernicus_notebook_final_rendu_propre_ameliore.ipynb`
            """
        ),
    )

    # The inserted generation note shifts original indices by +1.
    enhance_setup_cell(nb.cells[3])
    enhance_load_cell(nb.cells[4])
    inserted_offset = 1
    enhance_feature_cells(nb, offset=inserted_offset)
    enhance_baseline_cells(nb, offset=inserted_offset)
    enhance_clustering_cell(nb.cells[16 + inserted_offset])
    enhance_deep_learning_cells(nb, offset=inserted_offset)
    enhance_conclusion(nb)
    clear_outputs(nb)

    nb.metadata.setdefault("codex", {})
    nb.metadata["codex"]["generated_from"] = SOURCE_NOTEBOOK.name
    nb.metadata["codex"]["enhancements"] = [
        "cloud_motion_features",
        "advective_csi_baseline",
        "texture_features_glcm_gabor",
        "conditional_wind_features",
        "convlstm_section",
    ]

    nbf.write(nb, TARGET_NOTEBOOK)
    print(TARGET_NOTEBOOK)


if __name__ == "__main__":
    main()

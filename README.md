# Copernicus Solar Forecasting

Projet d'apprentissage automatique consacré à la prévision solaire à court terme à partir de séquences d'images satellitaires Copernicus.

Le notebook de rendu est `notebooks/Copernicus_notebook_final_beta.ipynb`.

## Objectif

Le but est de prédire les quatre prochaines cartes de GHI, aux horizons 15, 30, 45 et 60 minutes, à partir des dernières observations satellitaires et de variables physiques associées.

Le problème est traité comme une régression spatio-temporelle sur images. Les entrées couvrent une zone 81 par 81 pixels, tandis que la cible correspond à la région centrale 51 par 51 pixels.

## Données

Les fichiers de données ne sont pas inclus dans le dépôt, car ils sont trop volumineux.

Les fichiers bruts attendus doivent être placés dans `data/raw/`:

- `X_train_copernicus.npz`
- `X_test_copernicus.npz`
- `y_train_zRvpCeO_nQsYtKN.csv`
- `y_sub.csv`

Les profils prétraités sont générés dans `data/processed/` par les fonctions de `src/data_loading.py`. Ce dossier est ignoré par Git.

## Méthode

Le notebook final suit la chaîne de traitement suivante:

- chargement et prétraitement des tableaux Copernicus
- analyse exploratoire des variables et des cibles
- construction de variables physiques, tabulaires, spatiales et de texture
- baselines de persistance sur GHI et CSI
- modèles tabulaires supervisés, dont Ridge, ElasticNet, Random Forest, Extra Trees et HistGradientBoosting
- analyse par horizon, par structure spatiale et par régimes nuageux
- interprétation par importance de variables et SHAP
- comparaison avec des prédictions d'apprentissage profond sauvegardées lorsque disponibles

Les modèles d'apprentissage profond sont optionnels. Le notebook peut recharger des prédictions sauvegardées dans `reports/model_outputs/final_beta/` sans relancer TensorFlow localement.

## Installation

Créer l'environnement conda:

```bash
conda env create -f environment.yml
conda activate copernicus-solar
```

Lancer Jupyter:

```bash
jupyter lab
```

Ouvrir ensuite:

```text
notebooks/Copernicus_notebook_final_beta.ipynb
```

## Structure du projet

```text
config.py
environment.yml
README.md
data/raw/
data/processed/
models/models_tabular.py
notebooks/Copernicus_notebook_final_beta.ipynb
reports/model_outputs/
src/baselines.py
src/data_loading.py
src/deep_learning.py
src/eda.py
src/experiment_io.py
src/features.py
src/interpretation.py
src/metrics.py
src/motion.py
src/preprocessing.py
src/texture.py
src/utils.py
src/visualization.py
```

## Reproductibilité

Le notebook utilise `PROFILE = "full"` pour l'exécution finale. Les profils `dev` et `sample` restent disponibles pour des tests rapides.

Les résultats volumineux, les caches Python, les données brutes et les données prétraitées sont ignorés par Git. Pour partager une exécution déjà calculée, fournir séparément les artefacts nécessaires de `reports/model_outputs/final_beta/`.

## Auteur

Alix GREGGIO

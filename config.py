"""Configuration globale du projet de prévision solaire Copernicus."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

X_TRAIN_PATH = RAW_DATA_DIR / "X_train_copernicus.npz"
X_TEST_PATH = RAW_DATA_DIR / "X_test_copernicus.npz"
Y_TRAIN_PATH = RAW_DATA_DIR / "y_train_zRvpCeO_nQsYtKN.csv"
Y_SUBMISSION_PATH = RAW_DATA_DIR / "y_sub.csv"

INPUT_VARIABLES = ("GHI", "CLS", "SZA", "SAA")
TARGET_VARIABLE = "GHI"
FORECAST_HORIZONS_MINUTES = (15, 30, 45, 60)

RAW_IMAGE_SHAPE = (81, 81)
TARGET_IMAGE_SHAPE = (51, 51)
TARGET_ARRAY_SHAPE = (4, *TARGET_IMAGE_SHAPE)
ROI_SLICE = slice(15, 66)

PROCESSED_DTYPE = "float32"
PROCESSING_CHUNK_SIZE = 64
VALIDATION_FRACTION = 0.2
NORMALIZATION_EPS = 1e-6
FEATURE_EPS = 1e-6

PROCESSED_PROFILES = {
    "dev": {
        "n_samples": 32,
        "description": "Petit sous-ensemble déterministe pour le développement et les vérifications du notebook.",
    },
    "sample": {
        "n_samples": 256,
        "description": "Sous-ensemble réduit pour les tests rapides et les expériences courtes.",
    },
    "full": {
        "n_samples": None,
        "description": "Jeu de données complet prétraité.",
    },
}

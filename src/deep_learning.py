"""Outils optionnels d'apprentissage profond pour la prévision solaire Copernicus."""

from __future__ import annotations

import numpy as np


def has_tensorflow() -> bool:
    """Indique si TensorFlow et Keras peuvent être importés."""
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return True


def channels_first_to_last(feature_tensor: np.ndarray) -> np.ndarray:
    """
    Convertit un tenseur spatial du format (n, canaux, hauteur, largeur) vers NHWC.
    """
    feature_tensor = np.asarray(feature_tensor, dtype=np.float32)
    if feature_tensor.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape {feature_tensor.shape}.")
    return np.moveaxis(feature_tensor, 1, -1)


def target_to_channels_last(y: np.ndarray) -> np.ndarray:
    """
    Convertit les tenseurs cibles du format (n, horizon, hauteur, largeur) vers NHWC.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 4:
        raise ValueError(f"Expected a 4D target tensor, got shape {y.shape}.")
    return np.moveaxis(y, 1, -1)


def target_from_channels_last(y: np.ndarray) -> np.ndarray:
    """
    Reconvertit les tenseurs cibles NHWC vers (n, horizon, hauteur, largeur).
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 4:
        raise ValueError(f"Expected a 4D target tensor, got shape {y.shape}.")
    return np.moveaxis(y, -1, 1)


def prepare_cnn_training_data(
    feature_tensor: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prépare les tenseurs X et y pour un petit CNN.

    Si une baseline est fournie, le réseau apprend le champ résiduel y - baseline.
    Cette formulation est souvent plus stable que l'apprentissage direct du GHI.
    """
    X_nhwc = channels_first_to_last(feature_tensor)
    target = np.asarray(y, dtype=np.float32)
    if baseline is not None:
        target = target - np.asarray(baseline, dtype=np.float32)
    y_nhwc = target_to_channels_last(target)
    return X_nhwc, y_nhwc


def prepare_convlstm_training_data(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    baseline: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prépare un tenseur séquentiel pour les modèles ConvLSTM.

    Chaque pas passé reçoit les canaux suivants:
    - CSI au pas temporel correspondant
    - GHI au pas temporel correspondant
    - CLS au pas temporel correspondant
    - SZA et SAA au pas temporel correspondant si disponibles

    Formes de sortie:
        X: (n, 4, height, width, channels)
        y: (n, height, width, horizons)
    """
    ghi = np.asarray(arrays["GHI"], dtype=np.float32)
    cls = np.asarray(arrays["CLS"], dtype=np.float32)
    if ghi.ndim != 4 or cls.ndim != 4:
        raise ValueError("Expected GHI and CLS arrays with shape (n, t, h, w).")

    n_past = ghi.shape[1]
    cls_past = cls[:, :n_past]
    csi = ghi / np.maximum(cls_past, eps)

    channel_blocks = [csi, ghi, cls_past]
    channel_names = ["CSI", "GHI", "CLS"]

    for name in ("SZA", "SAA"):
        if name in arrays:
            values = np.asarray(arrays[name], dtype=np.float32)[:, :n_past]
            if np.nanmax(np.abs(values)) > 2 * np.pi + 1:
                values = np.deg2rad(values)
            channel_blocks.extend([np.sin(values).astype(np.float32), np.cos(values).astype(np.float32)])
            channel_names.extend([f"{name}_sin", f"{name}_cos"])

    X = np.stack(channel_blocks, axis=-1)
    target = np.asarray(y, dtype=np.float32)
    if baseline is not None:
        target = target - np.asarray(baseline, dtype=np.float32)
    y_nhwc = target_to_channels_last(target)
    return X.astype(np.float32), y_nhwc.astype(np.float32), channel_names


def build_small_residual_cnn(
    input_shape: tuple[int, int, int],
    n_horizons: int = 4,
    learning_rate: float = 1e-3,
):
    """
    Construit un CNN résiduel compact et entièrement convolutionnel avec TensorFlow.

    Le modèle transforme les cartes de variables en cartes de résidus pour chaque horizon.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for build_small_residual_cnn. "
            "Install tensorflow or use fit_mlp_residual_mean as a light fallback."
        ) from exc

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    outputs = tf.keras.layers.Conv2D(n_horizons, 1, padding="same", activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mae"],
    )
    return model


def build_convlstm_residual_model(
    input_shape: tuple[int, int, int, int],
    n_horizons: int = 4,
    learning_rate: float = 1e-3,
):
    """
    Construit un modèle ConvLSTM compact pour prévoir les résidus spatio-temporels.

    L'entrée suit le format (temps, hauteur, largeur, canaux). Le modèle prédit des
    cartes de résidus GHI au format (hauteur, largeur, n_horizons).
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for build_convlstm_residual_model. "
            "The notebook keeps an MLP fallback for environments without it."
        ) from exc

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.ConvLSTM2D(
        filters=24,
        kernel_size=3,
        padding="same",
        return_sequences=True,
        activation="tanh",
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ConvLSTM2D(
        filters=24,
        kernel_size=3,
        padding="same",
        return_sequences=False,
        activation="tanh",
    )(x)
    skip = tf.keras.layers.Conv2D(16, 1, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    outputs = tf.keras.layers.Conv2D(n_horizons, 1, padding="same", activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mae"],
    )
    return model


def fit_mlp_residual_mean(
    X_train: np.ndarray,
    residual_mean_train: np.ndarray,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    random_state: int = 42,
    max_iter: int = 300,
):
    """
    Fournit une alternative légère de réseau de neurones avec MLPRegressor.

    Ce modèle n'est pas un CNN, mais il permet au notebook d'exécuter une baseline
    neuronale lorsque TensorFlow ou PyTorch ne sont pas disponibles.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            random_state=random_state,
        ),
    )
    model.fit(np.asarray(X_train, dtype=np.float32), np.asarray(residual_mean_train, dtype=np.float32))
    return model


def add_residual_mean_to_baseline(
    baseline: np.ndarray,
    residual_mean_pred: np.ndarray,
    clip_min: float = 0.0,
) -> np.ndarray:
    """
    Convertit des résidus moyens par horizon en prévisions complètes d'images.
    """
    baseline = np.asarray(baseline, dtype=np.float32)
    residual_mean_pred = np.asarray(residual_mean_pred, dtype=np.float32)
    if residual_mean_pred.ndim != 2 or residual_mean_pred.shape[1] != baseline.shape[1]:
        raise ValueError(
            "Expected residual_mean_pred with shape "
            f"(n_samples, {baseline.shape[1]}), got {residual_mean_pred.shape}."
        )
    y_pred = baseline + residual_mean_pred[:, :, np.newaxis, np.newaxis]
    return np.maximum(y_pred, clip_min).astype(np.float32)

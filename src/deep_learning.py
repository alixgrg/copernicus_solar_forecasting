"""Optional deep-learning helpers for Copernicus solar forecasting."""

from __future__ import annotations

import numpy as np


def has_tensorflow() -> bool:
    """Return True when TensorFlow/Keras can be imported."""
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return True


def channels_first_to_last(feature_tensor: np.ndarray) -> np.ndarray:
    """
    Convert a spatial tensor from (n, channels, height, width) to NHWC.
    """
    feature_tensor = np.asarray(feature_tensor, dtype=np.float32)
    if feature_tensor.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape {feature_tensor.shape}.")
    return np.moveaxis(feature_tensor, 1, -1)


def target_to_channels_last(y: np.ndarray) -> np.ndarray:
    """
    Convert target tensors from (n, horizon, height, width) to NHWC.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 4:
        raise ValueError(f"Expected a 4D target tensor, got shape {y.shape}.")
    return np.moveaxis(y, 1, -1)


def target_from_channels_last(y: np.ndarray) -> np.ndarray:
    """
    Convert target tensors from NHWC back to (n, horizon, height, width).
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
    Prepare X/y tensors for a small CNN.

    If a baseline is supplied, the network learns the residual field
    y - baseline. This is usually easier than learning absolute GHI directly.
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
    Prepare a sequence tensor for ConvLSTM models.

    Each past step receives channels:
    - CSI at that past time
    - GHI at that past time
    - CLS at that past time
    - SZA/SAA at that past time when available

    Output shape:
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
    Build a compact fully-convolutional residual CNN with TensorFlow/Keras.

    The model maps 51x51 feature maps to 51x51 residual maps for each horizon.
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
    Build a compact ConvLSTM model for spatio-temporal residual forecasting.

    The input shape is (time, height, width, channels). The model predicts
    residual GHI maps with shape (height, width, n_horizons).
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
    Lightweight neural-network fallback based on scikit-learn MLPRegressor.

    This is not a CNN, but it lets the notebook run a neural baseline when
    TensorFlow/PyTorch are unavailable.
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
    Turn horizon-level residual predictions into full image forecasts.
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

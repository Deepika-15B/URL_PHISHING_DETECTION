"""Standalone inference pipeline for the trained IEEE phishing FNN.

This module loads only persisted artifacts and transforms a dataframe with the
full preprocessed feature schema into a standardized phishing prediction.  It
does not retrain models, extract URL features, or depend on Flask.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"
_MODEL_PATH = _MODELS_DIR / "fnn_model.keras"
_SCALER_PATH = _MODELS_DIR / "scaler.pkl"
_TOP14_PATH = _MODELS_DIR / "top14_features.pkl"
_FEATURE_NAMES_PATH = _MODELS_DIR / "preprocessed_feature_names.pkl"


def load_model(model_path: Path = _MODEL_PATH) -> tf.keras.Model:
    """Load the trained Keras FNN and raise a clear error if it is unavailable."""
    if not model_path.exists():
        raise FileNotFoundError(f"FNN model not found: {model_path}")
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as error:
        raise RuntimeError(f"Failed to load FNN model from {model_path}: {error}") from error


def load_artifacts(
    scaler_path: Path = _SCALER_PATH,
    top14_path: Path = _TOP14_PATH,
    feature_names_path: Path = _FEATURE_NAMES_PATH,
) -> dict[str, Any]:
    """Load and validate the scaler plus both ordered feature-name artifacts."""
    for path, label in [
        (scaler_path, "Scaler"),
        (top14_path, "Top 14 feature list"),
        (feature_names_path, "Preprocessed feature-name list"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    try:
        with scaler_path.open("rb") as file:
            scaler = pickle.load(file)
        with top14_path.open("rb") as file:
            top14_features = pickle.load(file)
        with feature_names_path.open("rb") as file:
            feature_names = pickle.load(file)
    except Exception as error:
        raise RuntimeError(f"Failed to load preprocessing artifacts: {error}") from error

    if not isinstance(scaler, MinMaxScaler) or not hasattr(scaler, "n_features_in_"):
        raise TypeError("Loaded scaler is not a fitted sklearn MinMaxScaler.")
    if not isinstance(feature_names, list) or not feature_names:
        raise TypeError("preprocessed_feature_names.pkl must contain a non-empty list.")
    if not isinstance(top14_features, list) or not top14_features:
        raise TypeError("top14_features.pkl must contain a non-empty list.")
    if scaler.n_features_in_ != len(feature_names):
        raise ValueError(
            "Scaler dimension does not match stored feature names: "
            f"{scaler.n_features_in_} != {len(feature_names)}."
        )

    unknown_top14 = [name for name in top14_features if name not in feature_names]
    if unknown_top14:
        raise ValueError(f"Top 14 features absent from preprocessed schema: {unknown_top14}")

    return {
        "scaler": scaler,
        "top14_features": top14_features,
        "feature_names": feature_names,
    }


def validate_input(input_data: pd.DataFrame, required_features: list[str]) -> None:
    """Ensure the caller supplies a dataframe with every required feature.

    Extra columns (including a target column) are allowed, but missing features
    are never silently filled because that would invalidate trained-model input.
    """
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame containing extracted feature values.")
    if input_data.empty:
        raise ValueError("Input DataFrame must contain at least one prediction row.")

    missing_features = [name for name in required_features if name not in input_data.columns]
    if missing_features:
        raise ValueError(
            "Input is missing required preprocessed feature columns: "
            + ", ".join(missing_features)
        )


def align_feature_order(input_data: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Reorder input columns exactly as the 101-feature scaler was trained."""
    validate_input(input_data, feature_names)
    aligned = input_data.loc[:, feature_names].copy()
    try:
        # Numeric coercion catches malformed extracted values before prediction.
        return aligned.astype(np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Input contains non-numeric required feature values: {error}") from error


def select_top14_features(
    scaled_features: pd.DataFrame,
    top14_features: list[str],
) -> pd.DataFrame:
    """Select the ordered persisted FNN input schema after all-feature scaling."""
    validate_input(scaled_features, top14_features)
    selected = scaled_features.loc[:, top14_features]
    if selected.shape[1] != len(top14_features):
        raise ValueError("Incorrect Top 14 feature dimensions after selection.")
    return selected


def apply_scaling(
    aligned_features: pd.DataFrame,
    scaler: MinMaxScaler,
    feature_names: list[str],
) -> pd.DataFrame:
    """Apply the saved 101-feature scaler without fitting it again.

    The training scaler was fitted on all preprocessed features.  Scaling must
    therefore occur before selecting the Top 14 model inputs.
    """
    if aligned_features.shape[1] != scaler.n_features_in_:
        raise ValueError(
            "Incorrect feature dimensions for saved scaler: "
            f"expected {scaler.n_features_in_}, received {aligned_features.shape[1]}."
        )
    try:
        scaled_values = scaler.transform(aligned_features.to_numpy())
    except Exception as error:
        raise RuntimeError(f"Saved scaler failed to transform input: {error}") from error
    return pd.DataFrame(scaled_values, columns=feature_names, index=aligned_features.index)


def predict_probability(model: tf.keras.Model, model_features: pd.DataFrame) -> np.ndarray:
    """Return one sigmoid phishing probability per input row from the trained FNN."""
    expected_dimensions = int(model.input_shape[-1])
    if model_features.shape[1] != expected_dimensions:
        raise ValueError(
            f"Incorrect FNN input dimensions: expected {expected_dimensions}, "
            f"received {model_features.shape[1]}."
        )
    try:
        probabilities = model.predict(model_features, verbose=0).reshape(-1)
    except Exception as error:
        raise RuntimeError(f"FNN prediction failed: {error}") from error
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("FNN prediction produced non-finite probabilities.")
    return probabilities


def predict_label(probability: float) -> str:
    """Convert the sigmoid output to the required Safe/Phishing label."""
    return "Phishing" if probability >= 0.5 else "Safe"


def predict(input_data: pd.DataFrame, artifacts: dict[str, Any] | None = None,
            model: tf.keras.Model | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    """Run the complete validation, alignment, scaling, and FNN prediction flow.

    ``artifacts`` and ``model`` can be passed by a service that loads them once
    at startup; omitting them retains a simple standalone one-call interface.
    A single input row returns one dictionary; multiple rows return a list.
    """
    model = model or load_model()
    artifacts = artifacts or load_artifacts()

    validate_input(input_data, artifacts["feature_names"])
    aligned = align_feature_order(input_data, artifacts["feature_names"])
    scaled = apply_scaling(aligned, artifacts["scaler"], artifacts["feature_names"])
    top14 = select_top14_features(scaled, artifacts["top14_features"])
    probabilities = predict_probability(model, top14)

    results = [
        {
            "prediction": predict_label(float(probability)),
            "probability": round(float(probability), 4),
            "confidence": round(max(float(probability), 1 - float(probability)) * 100, 2),
        }
        for probability in probabilities
    ]
    return results[0] if len(results) == 1 else results


if __name__ == "__main__":
    # Self-test uses one synthetic row in the exact scaler input schema.  It is
    # only a plumbing check and does not alter data, artifacts, or model weights.
    loaded_model = load_model()
    loaded_artifacts = load_artifacts()
    rng = np.random.default_rng(42)
    dummy_sample = pd.DataFrame(
        rng.uniform(0.0, 1.0, size=(1, len(loaded_artifacts["feature_names"]))),
        columns=loaded_artifacts["feature_names"],
    )
    result = predict(dummy_sample, artifacts=loaded_artifacts, model=loaded_model)

    print("Inference Pipeline successfully implemented.")
    print("Artifacts Loaded:")
    print("[OK] FNN Model\n[OK] Scaler\n[OK] Top14 Features\n[OK] Feature Names")
    print("Pipeline:\n[OK] Feature Validation\n[OK] Feature Ordering\n[OK] Scaling\n[OK] Prediction")
    print("Output Format:")
    print(result)


"""Experimental FNN training using the persisted Top 20 selected features."""
from __future__ import annotations
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import pickle
import random
import re
import time
from io import StringIO
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_curve)
from sklearn.model_selection import train_test_split
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dataset_small_preprocessed.csv"
FEATURES_PATH = ROOT / "models" / "top20_features.pkl"
MODEL_PATH = ROOT / "models" / "fnn_model_top20.keras"
HISTORY_PATH = ROOT / "models" / "training_history_top20.pkl"
REPORTS = ROOT / "reports"
TARGET = "phishing"
SEED = 42


def load_data():
    dataframe = pd.read_csv(DATA_PATH)
    with FEATURES_PATH.open("rb") as handle:
        selected = pickle.load(handle)
    if not isinstance(selected, list) or len(selected) != 20:
        raise ValueError("top20_features.pkl must contain exactly 20 features.")
    missing = [name for name in selected if name not in dataframe.columns]
    if missing:
        raise ValueError(f"Top 20 features missing from dataset: {missing}")
    x = dataframe[selected].astype(np.float32)
    y = dataframe[TARGET].astype(np.int32)
    print("✓ Dataset loaded")
    print("✓ Top20 features loaded")
    return x, y, selected


def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name="selected_features_top20"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    print("✓ Model built")
    return model


def evaluate(model, x_test, y_test) -> dict[str, Any]:
    probabilities = model.predict(x_test, verbose=0).ravel()
    truth = y_test.to_numpy()
    predicted = (probabilities >= 0.5).astype(np.int32)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": float(auc(*roc_curve(truth, probabilities)[:2])),
        "matrix": confusion_matrix(truth, predicted),
        "classification": classification_report(truth, predicted, target_names=["legitimate", "phishing"], zero_division=0),
        "truth": truth, "probabilities": probabilities,
    }


def save_artifacts(model, history, metrics, selected, dataset_size, train_size, test_size, elapsed, early_stopped, lr_activated):
    REPORTS.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    with HISTORY_PATH.open("wb") as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    epochs = range(1, len(history.history["loss"]) + 1)
    for key, valkey, title, ylabel, filename in [
        ("accuracy", "val_accuracy", "Top 20 Training and Validation Accuracy", "Accuracy", "training_accuracy_top20.png"),
        ("loss", "val_loss", "Top 20 Training and Validation Loss", "Binary crossentropy loss", "training_loss_top20.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5)); ax.plot(epochs, history.history[key], label="Training"); ax.plot(epochs, history.history[valkey], label="Validation")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(REPORTS / filename, dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5)); image = ax.imshow(metrics["matrix"], cmap="Blues"); fig.colorbar(image, ax=ax)
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["Legitimate","Phishing"], yticklabels=["Legitimate","Phishing"], xlabel="Predicted label", ylabel="True label", title="Top 20 FNN Confusion Matrix")
    for row in range(2):
        for col in range(2): ax.text(col, row, str(metrics["matrix"][row,col]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(REPORTS / "confusion_matrix_top20.png", dpi=200); plt.close(fig)
    fpr, tpr, _ = roc_curve(metrics["truth"], metrics["probabilities"])
    fig, ax = plt.subplots(figsize=(7,5)); ax.plot(fpr, tpr, label=f"Top 20 FNN (AUC = {metrics['roc_auc']:.4f})"); ax.plot([0,1],[0,1],"--",color="grey"); ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Top 20 FNN ROC Curve"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(REPORTS / "roc_curve_top20.png", dpi=200); plt.close(fig)
    (REPORTS / "classification_report_top20.txt").write_text(metrics["classification"], encoding="utf-8")
    (REPORTS / "training_report_top20.txt").write_text(f"""IEEE PHISHING DETECTION - TOP 20 FNN TRAINING REPORT

Dataset size: {dataset_size:,}
Training samples: {train_size:,}
Validation samples: {test_size:,}
Input features (20): {', '.join(selected)}
Split: train_test_split(test_size=0.20, stratify=y, random_state=42)
Architecture: Dense(64) -> Dropout(0.20) -> Dense(32) -> Dropout(0.20) -> Dense(1)
Optimizer: Adam (learning_rate=0.0005)
Loss: Binary Crossentropy
Metrics: Accuracy, Precision, Recall, AUC
Training configuration: epochs=100, batch_size=32
EarlyStopping: monitor=val_loss, patience=8, restore_best_weights=True
ReduceLROnPlateau: monitor=val_loss, factor=0.5, patience=3, min_lr=1e-6, verbose=1
Epochs actually trained: {len(history.history['loss'])}
ReduceLROnPlateau activated: {lr_activated}
EarlyStopping triggered: {early_stopped}
Training time: {elapsed:.2f} seconds

Accuracy: {metrics['accuracy']:.6f}
Precision: {metrics['precision']:.6f}
Recall: {metrics['recall']:.6f}
F1 Score: {metrics['f1']:.6f}
ROC-AUC: {metrics['roc_auc']:.6f}
Confusion matrix:\n{metrics['matrix']}
""", encoding="utf-8")
    print("✓ Reports generated")
    print("✓ Model saved")


def baseline_metrics() -> dict[str, float]:
    text = (REPORTS / "hyperparameter_tuning_report.txt").read_text(encoding="utf-8")
    labels = {"accuracy":"New Accuracy", "precision":"New Precision", "recall":"New Recall", "f1":"New F1", "roc_auc":"New ROC-AUC"}
    return {key: float(re.search(rf"^{re.escape(label)}: ([0-9.]+)$", text, re.MULTILINE).group(1)) for key, label in labels.items()}


def write_comparison(top14, top20):
    lines = ["IEEE PHISHING DETECTION - TOP 14 VS TOP 20 COMPARISON", ""]
    for key, label in [("accuracy","Accuracy"),("precision","Precision"),("recall","Recall"),("f1","F1 Score"),("roc_auc","ROC-AUC")]:
        diff = top20[key] - top14[key]
        lines += [f"Top14 {label}: {top14[key]:.6f}", f"Top20 {label}: {top20[key]:.6f}", f"Difference ({label}, Top20 - Top14): {diff:+.6f}", ""]
    recommendation = "Keep Top14 Model" if top20["accuracy"] <= top14["accuracy"] else "Replace with Top20 Model"
    outcome = "Top20 improved performance." if recommendation == "Replace with Top20 Model" else "Top20 reduced or did not improve performance."
    lines += [f"Performance conclusion: {outcome}", f"Recommendation: {recommendation}"]
    (REPORTS / "top14_vs_top20_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_top20():
    random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
    x, y, selected = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=SEED)
    model = build_model(len(selected))
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    start = time.perf_counter()
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=32, callbacks=[early, reduce], verbose=2)
    elapsed = time.perf_counter() - start
    early_stopped = len(history.history["loss"]) < 100
    rates = history.history.get("learning_rate", history.history.get("lr", []))
    lr_activated = bool(rates and min(rates) < 0.0005)
    print("✓ Training completed")
    metrics = evaluate(model, x_test, y_test); print("✓ Model evaluated")
    save_artifacts(model, history, metrics, selected, len(x), len(x_train), len(x_test), elapsed, early_stopped, lr_activated)
    top14 = baseline_metrics(); write_comparison(top14, metrics)
    print("Top20 Model Training Completed")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]: print(f"{key.replace('_',' ').title()}: {metrics[key]:.6f}")
    print("Recommendation:", "Replace with Top20 Model" if metrics["accuracy"] > top14["accuracy"] else "Keep Top14 Model")
    return metrics


if __name__ == "__main__":
    train_top20()



"""Standalone Phase 2 TabNet training and comparison workflow.

This module deliberately does not alter the Phase 1/2 preprocessing or any
existing FNN, DNN, Wide & Deep, backend, frontend, or inference code.
"""
from __future__ import annotations

import pickle
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed_v2"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures_phase2"

TABNET_MODEL_PATH = MODEL_DIR / "tabnet_phase2.zip"
TABNET_SCALER_PATH = MODEL_DIR / "scaler_tabnet.pkl"


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load duplicate-free splits and select the persisted Top-20 features."""
    with (MODEL_DIR / "top20_features.pkl").open("rb") as handle:
        features = list(pickle.load(handle))

    frames = {name: pd.read_csv(DATA_DIR / f"{name}.csv") for name in ("train", "validation", "test")}
    missing = set(features).difference(frames["train"].columns)
    if missing:
        raise ValueError(f"Selected features missing from data: {sorted(missing)}")
    if any("label" not in frame for frame in frames.values()):
        raise ValueError("Each input split must contain a 'label' column.")

    def split(name: str) -> tuple[np.ndarray, np.ndarray]:
        return (
            frames[name].loc[:, features].to_numpy(dtype=np.float32),
            frames[name]["label"].to_numpy(dtype=np.int64),
        )

    x_train, y_train = split("train")
    x_val, y_val = split("validation")
    x_test, y_test = split("test")
    return x_train, y_train, x_val, y_val, x_test, y_test


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predicted = (probabilities >= 0.5).astype(np.int64)
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "predictions": predicted,
        "probabilities": probabilities,
        "confusion_matrix": confusion_matrix(y_true, predicted),
    }


def keras_baseline(model_file: str, scaler_file: str, x_test: np.ndarray, y_test: np.ndarray) -> tuple[dict[str, object], int, float]:
    """Evaluate an already-trained baseline without modifying it."""
    from tensorflow.keras.models import load_model

    with (MODEL_DIR / scaler_file).open("rb") as handle:
        scaler = pickle.load(handle)
    model = load_model(MODEL_DIR / model_file, compile=False)
    scaled = scaler.transform(x_test)
    started = time.perf_counter()
    if model_file.startswith("wide_deep"):
        probability = model.predict([scaled, scaled], batch_size=1024, verbose=0).ravel()
    else:
        probability = model.predict(scaled, batch_size=1024, verbose=0).ravel()
    elapsed = time.perf_counter() - started
    return metrics(y_test, probability), int(model.count_params()), elapsed / len(x_test) * 1000


def report_value(report_file: str, field: str, default: str = "N/A") -> str:
    text = (REPORT_DIR / report_file).read_text(encoding="utf-8")
    match = re.search(field, text)
    return match.group(1) if match else default


def mcnemar_pvalue(reference: np.ndarray, challenger: np.ndarray, y_true: np.ndarray) -> float:
    """Two-sided exact McNemar test for paired correctness outcomes."""
    ref_correct = reference == y_true
    challenger_correct = challenger == y_true
    b = int(np.sum(ref_correct & ~challenger_correct))
    c = int(np.sum(~ref_correct & challenger_correct))
    return binomtest(min(b, c), n=b + c, p=0.5).pvalue if b + c else 1.0


def save_figures(y_test: np.ndarray, result: dict[str, object]) -> None:
    fpr, tpr, _ = roc_curve(y_test, result["probabilities"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"TabNet (AUC = {result['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="TabNet ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curve_tabnet.png", dpi=220)
    plt.close(fig)

    cm = result["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Phishing", "Legitimate"],
           yticklabels=["Phishing", "Legitimate"], xlabel="Predicted", ylabel="Actual",
           title="TabNet Confusion Matrix")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix_tabnet.png", dpi=220)
    plt.close(fig)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    with TABNET_SCALER_PATH.open("wb") as handle:
        pickle.dump(scaler, handle)

    model = TabNetClassifier(n_d=16, n_a=16, n_steps=5, gamma=1.5, lambda_sparse=1e-4,
                             optimizer_fn=torch.optim.Adam, optimizer_params={"lr": 0.001},
                             mask_type="sparsemax", seed=42, verbose=0)
    started = time.perf_counter()
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_name=["validation"],
              eval_metric=["auc"], max_epochs=100, patience=10, batch_size=1024,
              virtual_batch_size=128, num_workers=0, drop_last=False, compute_importance=False)
    training_seconds = time.perf_counter() - started
    probabilities = model.predict_proba(x_test)[:, 1]
    tabnet = metrics(y_test, probabilities)
    model.save_model(str(TABNET_MODEL_PATH.with_suffix("")))
    model_size = TABNET_MODEL_PATH.stat().st_size
    epochs = len(model.history["loss"])
    tabnet_parameters = sum(parameter.numel() for parameter in model.network.parameters())
    save_figures(y_test, tabnet)

    class_text = classification_report(y_test, tabnet["predictions"], target_names=["Phishing", "Legitimate"], digits=4)
    (REPORT_DIR / "classification_report_tabnet.txt").write_text(class_text, encoding="utf-8")
    report = ["PHIUSIIL TABNET MODEL REPORT - PHASE 2", "=" * 38, "", "Data and preprocessing", "-" * 22,
              "- Corrected duplicate-free data: processed_v2 train/validation/test splits", "- Features: persisted Top-20 feature selection", "- StandardScaler fit on train only; validation and test transformed with that scaler", "",
              "Configuration", "-" * 13, "- n_d=16, n_a=16, n_steps=5, gamma=1.5, lambda_sparse=1e-4", "- Optimizer: Adam; learning rate: 0.001", "- Batch size: 1024; virtual batch size: 128", "- Maximum epochs: 100; validation early stopping patience: 10", "",
              "Training summary", "-" * 16, f"- Training time (seconds): {training_seconds:.2f}", f"- Epochs trained: {epochs}", f"- Number of parameters: {tabnet_parameters}", f"- Model size (bytes): {model_size}", "",
              "Corrected test-set evaluation", "-" * 29]
    report += [f"- {key.replace('_', ' ').upper()}: {tabnet[key]:.6f}" for key in ("accuracy", "precision", "recall", "f1", "roc_auc")]
    report += [f"- Confusion matrix: {tabnet['confusion_matrix'].tolist()}", "", "Classification report", "-" * 21, class_text]
    (REPORT_DIR / "tabnet_phase2_report.txt").write_text("\n".join(report), encoding="utf-8")

    # Existing baselines are read-only; their own fitted scalers are retained.
    raw_x_test = scaler.inverse_transform(x_test).astype(np.float32)
    baselines = {
        "FNN": keras_baseline("fnn_phase2_v2.keras", "scaler_phase2_v2.pkl", raw_x_test, y_test),
        "DNN": keras_baseline("dnn_phase2.keras", "scaler_dnn_phase2.pkl", raw_x_test, y_test),
        "Wide & Deep": keras_baseline("wide_deep_phase2.keras", "scaler_wide_deep.pkl", raw_x_test, y_test),
    }
    rows = [("FNN", *baselines["FNN"]), ("DNN", *baselines["DNN"]), ("Wide & Deep", *baselines["Wide & Deep"]),
            ("TabNet", tabnet, tabnet_parameters, (time.perf_counter() - time.perf_counter()))]
    # Measure TabNet prediction latency after training; exclude preprocessing for all models.
    started = time.perf_counter(); model.predict_proba(x_test); tabnet_ms = (time.perf_counter() - started) / len(x_test) * 1000
    rows[-1] = ("TabNet", tabnet, tabnet_parameters, tabnet_ms)
    metadata = {
        "FNN": (report_value("fnn_phase2_report_v2.txt", r"Training time \(seconds\):\s*([0-9.]+)"), report_value("fnn_phase2_report_v2.txt", r"Epochs completed:\s*(\d+)"), (MODEL_DIR / "fnn_phase2_v2.keras").stat().st_size),
        "DNN": (report_value("dnn_phase2_report.txt", r"Training time \(seconds\):\s*([0-9.]+)"), report_value("dnn_phase2_report.txt", r"Epochs completed:\s*(\d+)"), (MODEL_DIR / "dnn_phase2.keras").stat().st_size),
        "Wide & Deep": (report_value("wide_deep_phase2_report.txt", r"Training time \(seconds\):\s*([0-9.]+)"), report_value("wide_deep_phase2_report.txt", r"Epochs completed:\s*(\d+)"), (MODEL_DIR / "wide_deep_phase2.keras").stat().st_size),
        "TabNet": (f"{training_seconds:.2f}", str(epochs), model_size),
    }
    comparison = ["FINAL PHIUSIIL MODEL COMPARISON", "=" * 31, "", "All metrics use the corrected duplicate-free test split and the Top-20 features.", "",
                  "Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Train s | Parameters | Size bytes | Epochs | Inference ms/sample",
                  "-" * 126]
    for name, outcome, parameters, inference_ms in rows:
        train_s, completed_epochs, size = metadata[name]
        comparison.append(f"{name} | {outcome['accuracy']:.6f} | {outcome['precision']:.6f} | {outcome['recall']:.6f} | {outcome['f1']:.6f} | {outcome['roc_auc']:.6f} | {train_s} | {parameters} | {size} | {completed_epochs} | {inference_ms:.6f}")
    pvalues = {name: mcnemar_pvalue(baselines[name][0]["predictions"], tabnet["predictions"], y_test) for name in baselines}
    best_name = max(rows, key=lambda item: (item[1]["f1"], item[1]["roc_auc"], item[1]["accuracy"]))[0]
    comparison += ["", "Deployment assessment", "-" * 21,
                   "- Computational complexity: FNN is the simplest dense network; DNN and Wide & Deep use more dense operations; TabNet additionally performs five sequential attentive decision steps.",
                   "- Memory usage: FNN has the smallest parameter footprint. TabNet requires additional attention-mask activations, so its runtime memory demand is higher than the compact FNN.",
                   "- Inference speed: the measured test-batch latency is reported above; sequential TabNet attention generally trades latency for feature-selection interpretability.",
                   "- Real-time suitability: prefer FNN for the Flask system when it is statistically tied on detection quality, because it has the smallest architecture and lowest expected serving overhead.",
                   "", "Conclusions", "-" * 11,
                   f"- Best overall test performance by the stated F1/ROC-AUC/accuracy tie-break: {best_name}.",
                   "- Statistical meaning: exact paired McNemar p-values for TabNet versus FNN/DNN/Wide & Deep are " + ", ".join(f"{name}={value:.4g}" for name, value in pvalues.items()) + ". At alpha=0.05, differences with p>=0.05 are not statistically significant; this single holdout test does not establish significance for any smaller p-value without a pre-specified multiple-comparison procedure.",
                   "- Dataset saturation: near-identical near-perfect metrics would be consistent with saturation from highly discriminative engineered URL and HTML features; this should be confirmed on a temporally or source-disjoint external test set.",
                   "- Deployment recommendation: integrate FNN when its holdout performance remains statistically tied, retaining TabNet as an offline interpretability/benchmark model."]
    (REPORT_DIR / "final_model_comparison.txt").write_text("\n".join(comparison), encoding="utf-8")
    print(f"Completed TabNet training: accuracy={tabnet['accuracy']:.6f}, ROC-AUC={tabnet['roc_auc']:.6f}")


if __name__ == "__main__":
    main()

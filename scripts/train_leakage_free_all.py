"""Retrain and evaluate the 4 ML models on the 18 leakage-free features.

Models: FNN, DNN, TabNet, Wide & Deep
Data: data/processed_leakage_free/{train, validation, test}.csv
Features: models/leakage_free/features_18.pkl
Artifacts saved to: models/leakage_free/
Reports saved to: reports/leakage_free_model_comparison.{csv, txt}
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed_leakage_free"
TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODELS_DIR = ROOT / "models" / "leakage_free"
FEATURES_PATH = MODELS_DIR / "features_18.pkl"
REPORTS_DIR = ROOT / "reports"

EXPECTED_18_FEATURES = [
    "NoOfExternalRef",
    "LineOfCode",
    "NoOfSelfRef",
    "NoOfImage",
    "NoOfJS",
    "HasSocialNet",
    "NoOfCSS",
    "HasCopyrightInfo",
    "NoOfOtherSpecialCharsInURL",
    "LargestLineLength",
    "HasDescription",
    "NoOfDegitsInURL",
    "URLLength",
    "IsResponsive",
    "DegitRatioInURL",
    "DomainTitleMatchScore",
    "SpacialCharRatioInURL",
    "HasSubmitButton",
]


def verify_feature_invariants(feature_cols: list[str]) -> None:
    """Strictly verify the 18-feature set and absence of leakage columns."""
    print("=" * 60)
    print("POST-TRAINING/PRE-TRAINING FEATURE CHECK")
    print("=" * 60)
    print(f"FEATURE COUNT = {len(feature_cols)}")
    print("FEATURES = [")
    for f in feature_cols:
        print(f"  {f},")
    print("]")

    if len(feature_cols) != 18:
        raise ValueError(f"Expected exactly 18 features, got {len(feature_cols)}")

    if feature_cols != EXPECTED_18_FEATURES:
        raise ValueError(f"Feature ordering mismatch: {feature_cols} != {EXPECTED_18_FEATURES}")

    for banned in ["URLSimilarityIndex", "IsHTTPS"]:
        if banned in feature_cols:
            raise ValueError(f"CRITICAL ERROR: Leakage feature {banned} detected in feature list!")
    print("URLSimilarityIndex NOT PRESENT")
    print("IsHTTPS NOT PRESENT")
    print("=" * 60)


def build_fnn(input_dim: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_dim,), name="input"),
        layers.Dense(64, activation="relu", name="dense_1"),
        layers.Dropout(0.20, name="dropout_1"),
        layers.Dense(32, activation="relu", name="dense_2"),
        layers.Dropout(0.20, name="dropout_2"),
        layers.Dense(1, activation="sigmoid", name="output"),
    ], name="fnn_leakage_free")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_dnn(input_dim: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_dim,), name="input"),
        layers.Dense(128, activation="relu", name="dense_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.Dropout(0.30, name="dropout_1"),
        layers.Dense(64, activation="relu", name="dense_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.Dropout(0.30, name="dropout_2"),
        layers.Dense(32, activation="relu", name="dense_3"),
        layers.BatchNormalization(name="bn_3"),
        layers.Dropout(0.20, name="dropout_3"),
        layers.Dense(1, activation="sigmoid", name="output"),
    ], name="dnn_leakage_free")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_wide_deep(input_dim: int) -> tf.keras.Model:
    wide_input = layers.Input(shape=(input_dim,), name="wide_input")
    deep_input = layers.Input(shape=(input_dim,), name="deep_input")

    wide_branch = layers.Dense(input_dim, activation="linear", name="wide_branch")(wide_input)

    deep_branch = layers.Dense(128, name="deep_hidden_1")(deep_input)
    deep_branch = layers.BatchNormalization(name="deep_bn_1")(deep_branch)
    deep_branch = layers.ReLU(name="deep_relu_1")(deep_branch)
    deep_branch = layers.Dropout(0.30, name="deep_dropout_1")(deep_branch)

    deep_branch = layers.Dense(64, name="deep_hidden_2")(deep_branch)
    deep_branch = layers.BatchNormalization(name="deep_bn_2")(deep_branch)
    deep_branch = layers.ReLU(name="deep_relu_2")(deep_branch)
    deep_branch = layers.Dropout(0.30, name="deep_dropout_2")(deep_branch)

    deep_branch = layers.Dense(32, name="deep_hidden_3")(deep_branch)
    deep_branch = layers.BatchNormalization(name="deep_bn_3")(deep_branch)
    deep_branch = layers.ReLU(name="deep_relu_3")(deep_branch)
    deep_branch = layers.Dropout(0.20, name="deep_dropout_3")(deep_branch)

    merged = layers.Concatenate(name="merge")([wide_branch, deep_branch])
    output = layers.Dense(1, activation="sigmoid", name="output")(merged)

    model = models.Model(inputs=[wide_input, deep_input], outputs=output, name="wide_deep_leakage_free")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> dict[str, Any]:
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "Model": model_name,
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "ROC_AUC": float(roc_auc),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "ConfusionMatrix": cm.tolist(),
        "ClassificationReport": classification_report(
            y_true, y_pred, target_names=["Phishing (0)", "Legitimate (1)"], digits=4
        ),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Feature Schema
    features_18 = joblib.load(FEATURES_PATH)
    verify_feature_invariants(features_18)

    # 2. Load Datasets
    print("\nLoading datasets from data/processed_leakage_free/ ...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Train samples: {len(train_df)} (Phishing: {(train_df['label'] == 0).sum()}, Legitimate: {(train_df['label'] == 1).sum()})")
    print(f"Val samples:   {len(val_df)} (Phishing: {(val_df['label'] == 0).sum()}, Legitimate: {(val_df['label'] == 1).sum()})")
    print(f"Test samples:  {len(test_df)} (Phishing: {(test_df['label'] == 0).sum()}, Legitimate: {(test_df['label'] == 1).sum()})")

    X_train = train_df[features_18]
    y_train = train_df["label"].to_numpy(dtype=np.int32)

    X_val = val_df[features_18]
    y_val = val_df["label"].to_numpy(dtype=np.int32)

    X_test = test_df[features_18]
    y_test = test_df["label"].to_numpy(dtype=np.int32)

    # 3. Preprocessing: Fit imputer & scaler on train ONLY
    print("\nFitting Imputer and Scaler on train set ONLY...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # Save preprocessing artifacts
    joblib.dump(imputer, MODELS_DIR / "imputer.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("Saved imputer.pkl and scaler.pkl to models/leakage_free/")

    input_dim = X_train_scaled.shape[1]
    results = []
    training_times = {}

    # Callbacks for Keras models
    def get_callbacks():
        return [
            callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]

    # =========================================================================
    # MODEL 1: FNN
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training Model 1: FNN (Feedforward Neural Network)...")
    print("=" * 60)
    fnn = build_fnn(input_dim)
    t0 = time.time()
    fnn.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=32,
        callbacks=get_callbacks(),
        verbose=2,
    )
    fnn_time = time.time() - t0
    training_times["FNN"] = fnn_time

    # Evaluate FNN on untouched test set
    t_inf_start = time.time()
    fnn_prob = fnn.predict(X_test_scaled, batch_size=2048, verbose=0).ravel()
    fnn_inf_time = (time.time() - t_inf_start) / len(X_test)
    fnn_metrics = evaluate_predictions(y_test, fnn_prob, "FNN")
    fnn_metrics["TrainingTime_s"] = round(fnn_time, 2)
    fnn_metrics["InferenceTime_ms_per_sample"] = round(fnn_inf_time * 1000, 4)
    results.append(fnn_metrics)

    # Save FNN
    fnn.save(MODELS_DIR / "fnn_leakage_free.keras")
    fnn.save(MODELS_DIR / "fnn_model.keras")
    print(f"FNN complete: Test Accuracy = {fnn_metrics['Accuracy']:.4f}, F1 = {fnn_metrics['F1']:.4f}, ROC-AUC = {fnn_metrics['ROC_AUC']:.4f}")

    # =========================================================================
    # MODEL 2: DNN
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training Model 2: DNN (Deep Neural Network)...")
    print("=" * 60)
    dnn = build_dnn(input_dim)
    t0 = time.time()
    dnn.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=32,
        callbacks=get_callbacks(),
        verbose=2,
    )
    dnn_time = time.time() - t0
    training_times["DNN"] = dnn_time

    # Evaluate DNN on untouched test set
    t_inf_start = time.time()
    dnn_prob = dnn.predict(X_test_scaled, batch_size=2048, verbose=0).ravel()
    dnn_inf_time = (time.time() - t_inf_start) / len(X_test)
    dnn_metrics = evaluate_predictions(y_test, dnn_prob, "DNN")
    dnn_metrics["TrainingTime_s"] = round(dnn_time, 2)
    dnn_metrics["InferenceTime_ms_per_sample"] = round(dnn_inf_time * 1000, 4)
    results.append(dnn_metrics)

    # Save DNN
    dnn.save(MODELS_DIR / "dnn_leakage_free.keras")
    dnn.save(MODELS_DIR / "dnn_model.keras")
    print(f"DNN complete: Test Accuracy = {dnn_metrics['Accuracy']:.4f}, F1 = {dnn_metrics['F1']:.4f}, ROC-AUC = {dnn_metrics['ROC_AUC']:.4f}")

    # =========================================================================
    # MODEL 3: Wide & Deep
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training Model 3: Wide & Deep...")
    print("=" * 60)
    wide_deep = build_wide_deep(input_dim)
    t0 = time.time()
    wide_deep.fit(
        [X_train_scaled, X_train_scaled],
        y_train,
        validation_data=([X_val_scaled, X_val_scaled], y_val),
        epochs=100,
        batch_size=32,
        callbacks=get_callbacks(),
        verbose=2,
    )
    wd_time = time.time() - t0
    training_times["Wide & Deep"] = wd_time

    # Evaluate Wide & Deep on untouched test set
    t_inf_start = time.time()
    wd_prob = wide_deep.predict([X_test_scaled, X_test_scaled], batch_size=2048, verbose=0).ravel()
    wd_inf_time = (time.time() - t_inf_start) / len(X_test)
    wd_metrics = evaluate_predictions(y_test, wd_prob, "Wide & Deep")
    wd_metrics["TrainingTime_s"] = round(wd_time, 2)
    wd_metrics["InferenceTime_ms_per_sample"] = round(wd_inf_time * 1000, 4)
    results.append(wd_metrics)

    # Save Wide & Deep
    wide_deep.save(MODELS_DIR / "wide_deep_leakage_free.keras")
    wide_deep.save(MODELS_DIR / "wide_deep_model.keras")
    print(f"Wide & Deep complete: Test Accuracy = {wd_metrics['Accuracy']:.4f}, F1 = {wd_metrics['F1']:.4f}, ROC-AUC = {wd_metrics['ROC_AUC']:.4f}")

    # =========================================================================
    # MODEL 4: TabNet
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training Model 4: TabNet...")
    print("=" * 60)
    np.random.seed(42)
    torch.manual_seed(42)

    tabnet = TabNetClassifier(
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=0.001),
        scheduler_params=dict(mode="min", patience=3, factor=0.5),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        mask_type="sparsemax",
        verbose=1,
    )
    t0 = time.time()
    tabnet.fit(
        X_train_scaled.astype(np.float32),
        y_train.astype(np.int64),
        eval_set=[(X_val_scaled.astype(np.float32), y_val.astype(np.int64))],
        eval_name=["val"],
        max_epochs=100,
        batch_size=1024,
        virtual_batch_size=128,
        patience=10,
        loss_fn=torch.nn.functional.cross_entropy,
        compute_importance=False,
    )
    tabnet_time = time.time() - t0
    training_times["TabNet"] = tabnet_time

    # Evaluate TabNet on untouched test set
    t_inf_start = time.time()
    tabnet_probs = tabnet.predict_proba(X_test_scaled.astype(np.float32))
    if tabnet_probs.ndim > 1 and tabnet_probs.shape[1] > 1:
        tabnet_prob = tabnet_probs[:, 1]
    else:
        tabnet_prob = tabnet_probs.ravel()
    tabnet_inf_time = (time.time() - t_inf_start) / len(X_test)
    tabnet_metrics = evaluate_predictions(y_test, tabnet_prob, "TabNet")
    tabnet_metrics["TrainingTime_s"] = round(tabnet_time, 2)
    tabnet_metrics["InferenceTime_ms_per_sample"] = round(tabnet_inf_time * 1000, 4)
    results.append(tabnet_metrics)

    # Save TabNet
    tabnet.save_model(str(MODELS_DIR / "tabnet_leakage_free.zip"))
    tabnet.save_model(str(MODELS_DIR / "tabnet_model.zip"))
    print(f"TabNet complete: Test Accuracy = {tabnet_metrics['Accuracy']:.4f}, F1 = {tabnet_metrics['F1']:.4f}, ROC-AUC = {tabnet_metrics['ROC_AUC']:.4f}")

    # =========================================================================
    # CREATE COMPARISON REPORTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("Generating Comparison Reports...")
    print("=" * 60)

    # CSV Report
    csv_rows = []
    for r in results:
        csv_rows.append({
            "Model": r["Model"],
            "Accuracy": f"{r['Accuracy']:.6f}",
            "Precision": f"{r['Precision']:.6f}",
            "Recall": f"{r['Recall']:.6f}",
            "F1": f"{r['F1']:.6f}",
            "ROC_AUC": f"{r['ROC_AUC']:.6f}",
            "TP": r["TP"],
            "TN": r["TN"],
            "FP": r["FP"],
            "FN": r["FN"],
        })
    csv_df = pd.DataFrame(csv_rows)
    csv_path = REPORTS_DIR / "leakage_free_model_comparison.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # TXT Report
    txt_lines = [
        "=" * 72,
        "LEAKAGE-FREE ML MODEL COMPARISON REPORT (18 FEATURES)",
        "=" * 72,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "1. DATASET SPECIFICATION",
        "-" * 72,
        f"Train set:      data/processed_leakage_free/train.csv ({len(train_df)} samples: {(train_df['label'] == 0).sum()} Phishing, {(train_df['label'] == 1).sum()} Legitimate)",
        f"Validation set: data/processed_leakage_free/validation.csv ({len(val_df)} samples: {(val_df['label'] == 0).sum()} Phishing, {(val_df['label'] == 1).sum()} Legitimate)",
        f"Test set:       data/processed_leakage_free/test.csv ({len(test_df)} samples: {(test_df['label'] == 0).sum()} Phishing, {(test_df['label'] == 1).sum()} Legitimate)",
        "",
        "Target Label Convention:",
        "  0 = Phishing",
        "  1 = Legitimate",
        "",
        "2. FEATURE SPECIFICATION (18 FEATURES)",
        "-" * 72,
        "Included 18 Features:",
    ]
    for idx, f in enumerate(features_18, 1):
        txt_lines.append(f"  {idx:2d}. {f}")
    txt_lines.extend([
        "",
        "Strictly Excluded Leakage Features:",
        "  - URLSimilarityIndex (REMOVED)",
        "  - IsHTTPS (REMOVED)",
        "",
        "3. PREPROCESSING METHOD",
        "-" * 72,
        "- Imputation: SimpleImputer(strategy='median') fitted on Train set ONLY",
        "- Scaling:    StandardScaler() fitted on Train set ONLY",
        "- Validation and Test sets transformed using Train-fitted artifacts (no data leakage)",
        "",
        "4. MODEL ARCHITECTURES & CONFIGURATIONS",
        "-" * 72,
        "FNN:         Input(18) -> Dense(64, ReLU) -> Dropout(0.20) -> Dense(32, ReLU) -> Dropout(0.20) -> Dense(1, Sigmoid)",
        "DNN:         Input(18) -> Dense(128, ReLU, BN) -> Dropout(0.30) -> Dense(64, ReLU, BN) -> Dropout(0.30) -> Dense(32, ReLU, BN) -> Dropout(0.20) -> Dense(1, Sigmoid)",
        "Wide & Deep: Wide Branch: Dense(18, Linear); Deep Branch: Dense(128, BN, ReLU, Dropout(0.30)) -> Dense(64, BN, ReLU, Dropout(0.30)) -> Dense(32, BN, ReLU, Dropout(0.20)) -> Concat -> Dense(1, Sigmoid)",
        "TabNet:      TabNetClassifier(n_d=16, n_a=16, n_steps=5, gamma=1.5, lambda_sparse=1e-4, batch_size=1024, virtual_batch_size=128, lr=0.001)",
        "",
        "5. TEST SET EVALUATION METRICS (Untouched Test Set)",
        "-" * 72,
        f"{'Model':<14} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'Train(s)':<8} | {'Inf(ms)':<8}",
        "-" * 94,
    ])
    for r in results:
        txt_lines.append(
            f"{r['Model']:<14} | {r['Accuracy']:<10.6f} | {r['Precision']:<10.6f} | {r['Recall']:<10.6f} | {r['F1']:<10.6f} | {r['ROC_AUC']:<10.6f} | {r['TrainingTime_s']:<8.2f} | {r['InferenceTime_ms_per_sample']:<8.4f}"
        )

    txt_lines.extend([
        "",
        "6. CONFUSION MATRICES & COUNTS",
        "-" * 72,
        "Positive class = 1 (Legitimate), Negative class = 0 (Phishing)",
        "TP = Actual Legitimate predicted Legitimate",
        "TN = Actual Phishing predicted Phishing",
        "FP = Actual Phishing predicted Legitimate",
        "FN = Actual Legitimate predicted Phishing",
        "",
    ])
    for r in results:
        txt_lines.extend([
            f"--- {r['Model']} ---",
            f"Confusion Matrix: [[TN={r['TN']}, FP={r['FP']}], [FN={r['FN']}, TP={r['TP']}]]",
            f"TP: {r['TP']} | TN: {r['TN']} | FP: {r['FP']} | FN: {r['FN']}",
            "Classification Report:",
            r["ClassificationReport"],
            "",
        ])

    # Find best model
    best_model = max(results, key=lambda x: (x["F1"], x["Accuracy"], x["ROC_AUC"]))
    txt_lines.extend([
        "7. BEST MODEL SELECTION",
        "-" * 72,
        f"Selected Best Model: {best_model['Model']}",
        f"Rationale: Achieved highest F1 score ({best_model['F1']:.6f}) and Accuracy ({best_model['Accuracy']:.6f}) with ROC-AUC of {best_model['ROC_AUC']:.6f} on the untouched 35,185-sample test set.",
        "",
        "8. SAVED ARTIFACTS",
        "-" * 72,
        f"- models/leakage_free/fnn_leakage_free.keras",
        f"- models/leakage_free/dnn_leakage_free.keras",
        f"- models/leakage_free/wide_deep_leakage_free.keras",
        f"- models/leakage_free/tabnet_leakage_free.zip",
        f"- models/leakage_free/imputer.pkl",
        f"- models/leakage_free/scaler.pkl",
        f"- models/leakage_free/features_18.pkl",
        "=" * 72,
    ])

    txt_path = REPORTS_DIR / "leakage_free_model_comparison.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    print(f"Saved {txt_path}")
    print("\nAll models retrained and evaluated successfully!")


if __name__ == "__main__":
    main()

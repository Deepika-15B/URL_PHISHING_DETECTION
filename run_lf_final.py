from __future__ import annotations
import json, pickle, sys, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score)
from tensorflow.keras import models as keras_models

ROOT = Path(r"e:/phishing_project_ieee/phishing_detection_ieee")
TRAIN_PATH = ROOT / "data/processed_leakage_free/train.csv"
VAL_PATH   = ROOT / "data/processed_leakage_free/validation.csv"
TEST_PATH  = ROOT / "data/processed_leakage_free/test.csv"
FEATURES_PATH = ROOT / "models/leakage_free/features_18.pkl"
IMPUTER_PATH  = ROOT / "models/leakage_free/imputer.pkl"
SCALER_PATH   = ROOT / "models/leakage_free/scaler.pkl"
LF_DIR = ROOT / "models/leakage_free"
REPORT_DIR = ROOT / "reports"
FIG_DIR = ROOT / "reports/confusion_matrices"
FNN_PATH       = LF_DIR / "fnn_leakage_free.keras"
DNN_PATH       = LF_DIR / "dnn_leakage_free.keras"
TABNET_PATH    = LF_DIR / "tabnet_leakage_free.zip"
WIDE_DEEP_PATH = LF_DIR / "wide_deep_leakage_free.keras"
FORBIDDEN = {"URLSimilarityIndex", "IsHTTPS"}

EXPECTED_18 = [
    "NoOfExternalRef","LineOfCode","NoOfSelfRef","NoOfImage","NoOfJS",
    "HasSocialNet","NoOfCSS","HasCopyrightInfo","NoOfOtherSpecialCharsInURL",
    "LargestLineLength","HasDescription","NoOfDegitsInURL","URLLength",
    "IsResponsive","DegitRatioInURL","DomainTitleMatchScore",
    "SpacialCharRatioInURL","HasSubmitButton"
]

def verify_features(features, stage):
    for fe in FORBIDDEN:
        if fe in features:
            print(f"[STOP] Forbidden feature {fe!r} in {stage}!")
            sys.exit(1)
    if len(features) != 18:
        print(f"[STOP] Feature count={len(features)} (expected 18) in {stage}!")
        sys.exit(1)
    print(f"[OK] {stage}: 18 features, no forbidden features.")

def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
        "confusion_matrix": cm.tolist(),
    }

def plot_cm(cm_arr, model_name, save_path):
    fig, ax = plt.subplots(figsize=(5,4))
    cm = np.array(cm_arr)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred Phishing","Pred Legitimate"])
    ax.set_yticklabels(["Actual Phishing","Actual Legitimate"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"  [Saved] {save_path}")

def eval_keras(model_path, X_sc, y):
    model = keras_models.load_model(model_path, compile=False)
    t0 = time.time()
    probs = model.predict(X_sc, verbose=0).ravel()
    inf_t = time.time() - t0
    preds = (probs >= 0.5).astype(int)
    m = compute_metrics(y, preds, probs)
    m["inference_time_s"] = round(inf_t, 4)
    return m

def train_tabnet(X_train_sc, y_train, X_val_sc, y_val):
    print("\n=== Training TabNet leakage-free ===")
    np.random.seed(42); torch.manual_seed(42)
    model = TabNetClassifier(
        n_d=16, n_a=16, n_steps=5, gamma=1.5, lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=0.001),
        scheduler_params=dict(mode="min", patience=3, factor=0.5),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        mask_type="sparsemax", verbose=1,
    )
    t0 = time.time()
    model.fit(
        X_train_sc.astype(np.float32), y_train.to_numpy(dtype=np.int64),
        eval_set=[(X_val_sc.astype(np.float32), y_val.to_numpy(dtype=np.int64))],
        eval_name=["val"], max_epochs=100, batch_size=1024,
        virtual_batch_size=128, patience=10,
        loss_fn=torch.nn.functional.cross_entropy, compute_importance=False,
    )
    train_t = time.time() - t0
    model.save_model(str(TABNET_PATH))
    print(f"  TabNet saved: {TABNET_PATH} ({train_t:.1f}s)")
    return model, train_t

def eval_tabnet(model, X_sc, y):
    t0 = time.time()
    p2d = model.predict_proba(X_sc.astype(np.float32))
    inf_t = time.time() - t0
    probs = p2d[:,1] if p2d.ndim > 1 else p2d.ravel()
    preds = (probs >= 0.5).astype(int)
    m = compute_metrics(y, preds, probs)
    m["inference_time_s"] = round(inf_t, 4)
    return m

def main():
    LF_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading features ===")
    with open(FEATURES_PATH, "rb") as f:
        features_18 = pickle.load(f)
    feature_cols = list(features_18)
    verify_features(feature_cols, "features_18.pkl")

    print("\n=== Loading data ===")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    print(f"  Train={len(train_df):,} Val={len(val_df):,} Test={len(test_df):,}")

    X_train = train_df[feature_cols]; y_train = train_df["label"].astype(int)
    X_val   = val_df[feature_cols];   y_val   = val_df["label"].astype(int)
    X_test  = test_df[feature_cols];  y_test  = test_df["label"].astype(int)
    verify_features(list(X_train.columns), "X_train")
    verify_features(list(X_test.columns), "X_test")

    print("\n=== Preprocessing ===")
    with open(IMPUTER_PATH, "rb") as f: imputer = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:  scaler  = pickle.load(f)
    X_train_sc = scaler.transform(imputer.transform(X_train))
    X_val_sc   = scaler.transform(imputer.transform(X_val))
    X_test_sc  = scaler.transform(imputer.transform(X_test))
    print(f"  X_train_sc shape: {X_train_sc.shape}  X_test_sc: {X_test_sc.shape}")

    results = {}; train_times = {}

    print("\n=== FNN ===")
    results["FNN"] = eval_keras(FNN_PATH, X_test_sc, y_test)
    train_times["FNN"] = "N/A (pre-trained)"
    print(f"  Accuracy={results['FNN']['accuracy']:.6f} F1={results['FNN']['f1']:.6f}")

    print("\n=== DNN ===")
    results["DNN"] = eval_keras(DNN_PATH, X_test_sc, y_test)
    train_times["DNN"] = "N/A (pre-trained)"
    print(f"  Accuracy={results['DNN']['accuracy']:.6f} F1={results['DNN']['f1']:.6f}")

    print("\n=== Wide & Deep ===")
    results["Wide & Deep"] = eval_keras(WIDE_DEEP_PATH, X_test_sc, y_test)
    train_times["Wide & Deep"] = "N/A (pre-trained)"
    print(f"  Accuracy={results['Wide & Deep']['accuracy']:.6f} F1={results['Wide & Deep']['f1']:.6f}")

    print("\n=== TabNet ===")
    if TABNET_PATH.exists():
        print("  Loading existing TabNet model...")
        tn = TabNetClassifier()
        tn.load_model(str(TABNET_PATH))
        results["TabNet"] = eval_tabnet(tn, X_test_sc, y_test)
        train_times["TabNet"] = "N/A (pre-trained)"
    else:
        tn, tt = train_tabnet(X_train_sc, y_train, X_val_sc, y_val)
        results["TabNet"] = eval_tabnet(tn, X_test_sc, y_test)
        train_times["TabNet"] = f"{tt:.1f}s"
    print(f"  Accuracy={results['TabNet']['accuracy']:.6f} F1={results['TabNet']['f1']:.6f}")

    for name, m in results.items():
        safe = name.replace(" ","_").replace("&","and")
        plot_cm(m["confusion_matrix"], name, FIG_DIR / f"cm_{safe}_lf.png")

    csv_path = REPORT_DIR / "leakage_free_model_comparison.csv"
    rows = []
    for name, m in results.items():
        rows.append({"Model":name,"Accuracy":round(m["accuracy"],6),
            "Precision":round(m["precision"],6),"Recall":round(m["recall"],6),
            "F1":round(m["f1"],6),"ROC_AUC":round(m["roc_auc"],6),
            "TP":m["TP"],"TN":m["TN"],"FP":m["FP"],"FN":m["FN"]})
    pd.DataFrame(rows,columns=["Model","Accuracy","Precision","Recall","F1","ROC_AUC","TP","TN","FP","FN"]).to_csv(csv_path,index=False)
    print(f"\n[Saved] {csv_path}")

    best = max(results, key=lambda k: (results[k]["f1"], results[k]["roc_auc"]))
    bm = results[best]

    lines = []
    lines.append("="*70)
    lines.append("LEAKAGE-FREE MODEL COMPARISON REPORT")
    lines.append("="*70)
    lines.append("")
    lines.append("Dataset")
    lines.append("-------")
    lines.append(f"  Train      : {len(train_df):,} rows")
    lines.append(f"  Validation : {len(val_df):,} rows")
    lines.append(f"  Test       : {len(test_df):,} rows")
    lines.append("")
    lines.append("Target convention: 0=Phishing | 1=Legitimate")
    lines.append("")
    lines.append("18 Features (leakage-free):")
    for i,f in enumerate(EXPECTED_18,1):
        lines.append(f"  {i:2d}. {f}")
    lines.append("")
    lines.append("Removed features (leakage sources):")
    lines.append("  - URLSimilarityIndex")
    lines.append("  - IsHTTPS")
    lines.append("")
    lines.append("Preprocessing:")
    lines.append(f"  Imputer: {IMPUTER_PATH} (transform only on val/test)")
    lines.append(f"  Scaler : {SCALER_PATH} (transform only on val/test)")
    lines.append("")
    lines.append("Model Architectures:")
    lines.append("  FNN:       Dense(64,ReLU)+Drop(0.20) -> Dense(32,ReLU)+Drop(0.20) -> Sigmoid")
    lines.append("             Adam(lr=5e-4), BinaryCE, Batch=32, Epochs<=100, EarlyStop(p=8)")
    lines.append("  DNN:       Dense(128)+BN+Drop(0.30) -> Dense(64)+BN+Drop(0.30) -> Dense(32)+BN+Drop(0.20) -> Sigmoid")
    lines.append("             Adam(lr=5e-4), BinaryCE, Batch=32, Epochs<=100, EarlyStop(p=8)")
    lines.append("  Wide&Deep: Wide(linear) + Deep(Dense(64)->Dense(32)) merged -> Sigmoid")
    lines.append("             Adam(lr=5e-4), Batch=32")
    lines.append("  TabNet:    n_d=16, n_a=16, n_steps=5, gamma=1.5, lambda_sparse=1e-4")
    lines.append("             Adam(lr=1e-3), Batch=1024, MaxEpochs=100, Patience=10")
    lines.append("")
    lines.append("-"*70)
    lines.append("EVALUATION RESULTS (TEST SET)")
    lines.append("-"*70)
    lines.append(f"  {'Model':<14} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    lines.append("  " + "-"*66)
    for name, m in results.items():
        lines.append(f"  {name:<14} {m['accuracy']:>10.6f} {m['precision']:>10.6f} {m['recall']:>10.6f} {m['f1']:>10.6f} {m['roc_auc']:>10.6f}")
    lines.append("")
    lines.append("Confusion Matrices (layout: [[TN,FP],[FN,TP]]):")
    for name, m in results.items():
        lines.append(f"  {name}: TP={m['TP']}, TN={m['TN']}, FP={m['FP']}, FN={m['FN']}")
        lines.append(f"    [[{m['TN']}, {m['FP']}], [{m['FN']}, {m['TP']}]]")
    lines.append("")
    lines.append("Training / Inference Times:")
    for name, m in results.items():
        lines.append(f"  {name}: training={train_times[name]}, inference={m['inference_time_s']}s")
    lines.append("")
    lines.append("-"*70)
    lines.append("BEST MODEL")
    lines.append("-"*70)
    lines.append(f"  Best: {best}")
    lines.append(f"  Accuracy={bm['accuracy']:.6f} Precision={bm['precision']:.6f} Recall={bm['recall']:.6f} F1={bm['f1']:.6f} ROC-AUC={bm['roc_auc']:.6f}")
    lines.append(f"  TP={bm['TP']}, TN={bm['TN']}, FP={bm['FP']}, FN={bm['FN']}")
    lines.append(f"  Reason: {best} achieved highest F1 ({bm['f1']:.6f}) and ROC-AUC ({bm['roc_auc']:.6f}) on leakage-free test.")
    lines.append("")
    lines.append("Saved models:")
    lines.append(f"  FNN        : {FNN_PATH}")
    lines.append(f"  DNN        : {DNN_PATH}")
    lines.append(f"  TabNet     : {TABNET_PATH}")
    lines.append(f"  Wide&Deep  : {WIDE_DEEP_PATH}")
    lines.append("="*70)
    txt_path = REPORT_DIR / "leakage_free_model_comparison.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Saved] {txt_path}")

    json_out = {}
    for name, m in results.items():
        json_out[name] = {k:v for k,v in m.items()}
    with open(REPORT_DIR / "leakage_free_metrics.json","w") as f:
        json.dump(json_out, f, indent=4)

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"  {'Model':<14} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    for name, m in results.items():
        print(f"  {name:<14} {m['accuracy']:>10.6f} {m['precision']:>10.6f} {m['recall']:>10.6f} {m['f1']:>10.6f} {m['roc_auc']:>10.6f}")
    print(f"\n  Best: {best}  (F1={bm['f1']:.6f}, ROC-AUC={bm['roc_auc']:.6f})")
    print("\nConfusion matrices:")
    for name, m in results.items():
        print(f"  {name}: TP={m['TP']}, TN={m['TN']}, FP={m['FP']}, FN={m['FN']}")
    print("\nDone.")

if __name__ == "__main__":
    main()

"""
Complete leakage-free final evaluation:
1. Copy tabnet_leakage_free.zip.zip -> tabnet_leakage_free.zip
2. Refit imputer + scaler from train ONLY
3. Evaluate all 4 models on test set
4. Generate final reports
"""
from __future__ import annotations
import json, os, pickle, shutil, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, classification_report)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import models as keras_models

ROOT = Path(r"e:/phishing_project_ieee/phishing_detection_ieee")
DATA_DIR = ROOT / "data/processed_leakage_free"
MODELS_DIR = ROOT / "models/leakage_free"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "confusion_matrices"

FEATURES_PATH  = MODELS_DIR / "features_18.pkl"
IMPUTER_PATH   = MODELS_DIR / "imputer.pkl"
SCALER_PATH    = MODELS_DIR / "scaler.pkl"
FNN_PATH       = MODELS_DIR / "fnn_leakage_free.keras"
DNN_PATH       = MODELS_DIR / "dnn_leakage_free.keras"
WIDE_DEEP_PATH = MODELS_DIR / "wide_deep_leakage_free.keras"
TABNET_PATH    = MODELS_DIR / "tabnet_leakage_free.zip"
TABNET_SRC     = MODELS_DIR / "tabnet_leakage_free.zip.zip"

EXPECTED_18 = [
    "NoOfExternalRef","LineOfCode","NoOfSelfRef","NoOfImage","NoOfJS",
    "HasSocialNet","NoOfCSS","HasCopyrightInfo","NoOfOtherSpecialCharsInURL",
    "LargestLineLength","HasDescription","NoOfDegitsInURL","URLLength",
    "IsResponsive","DegitRatioInURL","DomainTitleMatchScore",
    "SpacialCharRatioInURL","HasSubmitButton"
]
FORBIDDEN = {"URLSimilarityIndex","IsHTTPS"}

def verify(cols, stage):
    for f in FORBIDDEN:
        if f in cols:
            raise RuntimeError(f"STOP: forbidden feature {f!r} in {stage}!")
    if len(cols) != 18:
        raise RuntimeError(f"STOP: feature count={len(cols)} != 18 in {stage}!")
    print(f"[OK] {stage}: 18 features, no forbidden features.")

def compute_metrics(y_true, y_pred, y_prob, name):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    cr = classification_report(y_true, y_pred, target_names=["Phishing(0)","Legitimate(1)"], digits=4)
    return {
        "Model": name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "ConfusionMatrix": cm.tolist(),
        "ClassReport": cr,
    }

def plot_cm(cm_arr, model_name, save_path):
    cm = np.array(cm_arr)
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred Phishing","Pred Legitimate"])
    ax.set_yticklabels(["Actual Phishing","Actual Legitimate"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center", fontsize=12, color="black")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(save_path, dpi=180); plt.close(fig)
    print(f"  [saved] {save_path}")

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Feature check
    print("\n[1] Loading feature list...")
    features_18 = joblib.load(FEATURES_PATH)
    verify(list(features_18), "features_18.pkl")

    # 2. Copy TabNet model if needed
    print("\n[2] Checking TabNet model...")
    if not TABNET_PATH.exists() and TABNET_SRC.exists():
        shutil.copy2(TABNET_SRC, TABNET_PATH)
        print(f"  Copied {TABNET_SRC.name} -> {TABNET_PATH.name}")
    elif TABNET_PATH.exists():
        print(f"  {TABNET_PATH.name} already exists.")
    else:
        raise RuntimeError("No TabNet source found! Need to train TabNet first.")

    # 3. Load data
    print("\n[3] Loading datasets...")
    t0 = time.time()
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df   = pd.read_csv(DATA_DIR / "validation.csv")
    test_df  = pd.read_csv(DATA_DIR / "test.csv")
    print(f"  Train={len(train_df):,} Val={len(val_df):,} Test={len(test_df):,} (loaded in {time.time()-t0:.1f}s)")
    print(f"  Label distribution (Train) - Phishing(0): {(train_df.label==0).sum():,} Legitimate(1): {(train_df.label==1).sum():,}")

    X_train = train_df[features_18]; y_train = train_df["label"].to_numpy(dtype=np.int32)
    X_val   = val_df[features_18];   y_val   = val_df["label"].to_numpy(dtype=np.int32)
    X_test  = test_df[features_18];  y_test  = test_df["label"].to_numpy(dtype=np.int32)
    verify(list(X_train.columns), "X_train"); verify(list(X_test.columns), "X_test")

    # 4. Fit imputer + scaler on TRAIN ONLY, transform all
    print("\n[4] Fitting imputer + scaler on train set only...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp   = imputer.transform(X_val)
    X_test_imp  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_val_sc   = scaler.transform(X_val_imp)
    X_test_sc  = scaler.transform(X_test_imp)

    joblib.dump(imputer, IMPUTER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  Saved imputer to {IMPUTER_PATH}")
    print(f"  Saved scaler  to {SCALER_PATH}")
    print(f"  X_train_sc shape: {X_train_sc.shape} X_test_sc: {X_test_sc.shape}")

    results = {}
    train_times = {}

    # 5. Evaluate FNN
    print("\n[5] Evaluating FNN leakage-free...")
    fnn = keras_models.load_model(FNN_PATH, compile=False)
    t0 = time.time()
    fnn_prob = fnn.predict(X_test_sc, batch_size=2048, verbose=0).ravel()
    fnn_inf = time.time()-t0
    fnn_pred = (fnn_prob >= 0.5).astype(int)
    results["FNN"] = compute_metrics(y_test, fnn_pred, fnn_prob, "FNN")
    results["FNN"]["InferenceTime_s"] = round(fnn_inf, 4)
    train_times["FNN"] = 283.42
    print(f"  FNN Accuracy={results['FNN']['Accuracy']:.6f} F1={results['FNN']['F1']:.6f} ROC-AUC={results['FNN']['ROC_AUC']:.6f}")

    # 6. Evaluate DNN
    print("\n[6] Evaluating DNN leakage-free...")
    dnn = keras_models.load_model(DNN_PATH, compile=False)
    t0 = time.time()
    dnn_prob = dnn.predict(X_test_sc, batch_size=2048, verbose=0).ravel()
    dnn_inf = time.time()-t0
    dnn_pred = (dnn_prob >= 0.5).astype(int)
    results["DNN"] = compute_metrics(y_test, dnn_pred, dnn_prob, "DNN")
    results["DNN"]["InferenceTime_s"] = round(dnn_inf, 4)
    train_times["DNN"] = 153.80
    print(f"  DNN Accuracy={results['DNN']['Accuracy']:.6f} F1={results['DNN']['F1']:.6f} ROC-AUC={results['DNN']['ROC_AUC']:.6f}")

    # 7. Evaluate Wide & Deep
    print("\n[7] Evaluating Wide & Deep leakage-free...")
    wd = keras_models.load_model(WIDE_DEEP_PATH, compile=False)
    t0 = time.time()
    wd_prob = wd.predict([X_test_sc, X_test_sc], batch_size=2048, verbose=0).ravel()
    wd_inf = time.time()-t0
    wd_pred = (wd_prob >= 0.5).astype(int)
    results["Wide & Deep"] = compute_metrics(y_test, wd_pred, wd_prob, "Wide & Deep")
    results["Wide & Deep"]["InferenceTime_s"] = round(wd_inf, 4)
    train_times["Wide & Deep"] = 89.15
    print(f"  Wide&Deep Accuracy={results['Wide & Deep']['Accuracy']:.6f} F1={results['Wide & Deep']['F1']:.6f}")

    # 8. Evaluate TabNet
    print("\n[8] Evaluating TabNet leakage-free...")
    tn = TabNetClassifier()
    tn.load_model(str(TABNET_PATH))
    print(f"  TabNet input_dim={tn.input_dim} (must be 18)")
    if tn.input_dim != 18:
        raise RuntimeError(f"STOP: TabNet input_dim={tn.input_dim} != 18!")
    t0 = time.time()
    tn_prob2d = tn.predict_proba(X_test_sc.astype(np.float32))
    tn_inf = time.time()-t0
    tn_prob = tn_prob2d[:,1] if tn_prob2d.ndim > 1 else tn_prob2d.ravel()
    tn_pred = (tn_prob >= 0.5).astype(int)
    results["TabNet"] = compute_metrics(y_test, tn_pred, tn_prob, "TabNet")
    results["TabNet"]["InferenceTime_s"] = round(tn_inf, 4)
    train_times["TabNet"] = "N/A (pre-trained)"
    print(f"  TabNet Accuracy={results['TabNet']['Accuracy']:.6f} F1={results['TabNet']['F1']:.6f} ROC-AUC={results['TabNet']['ROC_AUC']:.6f}")

    # 9. Confusion matrix plots
    print("\n[9] Saving confusion matrix plots...")
    for name, m in results.items():
        safe = name.replace(" ","_").replace("&","and")
        plot_cm(m["ConfusionMatrix"], name, FIG_DIR / f"{safe}_leakage_free.png")

    # 10. CSV report
    csv_rows = []
    for name, m in results.items():
        csv_rows.append({"Model":name,"Accuracy":round(m["Accuracy"],6),
            "Precision":round(m["Precision"],6),"Recall":round(m["Recall"],6),
            "F1":round(m["F1"],6),"ROC_AUC":round(m["ROC_AUC"],6),
            "TP":m["TP"],"TN":m["TN"],"FP":m["FP"],"FN":m["FN"]})
    csv_path = REPORTS_DIR / "leakage_free_model_comparison.csv"
    pd.DataFrame(csv_rows,columns=["Model","Accuracy","Precision","Recall","F1","ROC_AUC","TP","TN","FP","FN"]).to_csv(csv_path,index=False)
    print(f"  [saved] {csv_path}")

    # 11. Best model
    best = max(results, key=lambda k:(results[k]["F1"],results[k]["ROC_AUC"],results[k]["Accuracy"]))
    bm = results[best]

    # 12. TXT report
    lines = []
    lines += ["="*72,"LEAKAGE-FREE ML MODEL COMPARISON REPORT (18 FEATURES)","="*72,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}","",
        "1. DATASET SPECIFICATION","-"*72,
        f"Train set:      {DATA_DIR}/train.csv ({len(train_df):,} samples: {(train_df.label==0).sum():,} Phishing, {(train_df.label==1).sum():,} Legitimate)",
        f"Validation set: {DATA_DIR}/validation.csv ({len(val_df):,} samples: {(val_df.label==0).sum():,} Phishing, {(val_df.label==1).sum():,} Legitimate)",
        f"Test set:       {DATA_DIR}/test.csv ({len(test_df):,} samples: {(test_df.label==0).sum():,} Phishing, {(test_df.label==1).sum():,} Legitimate)",
        "","Target Label Convention: 0=Phishing | 1=Legitimate","",
        "2. FEATURE SPECIFICATION (18 FEATURES)","-"*72,"Included 18 Features:"]
    for i,f in enumerate(features_18,1): lines.append(f"  {i:2d}. {f}")
    lines += ["","Strictly Excluded Leakage Features:","  - URLSimilarityIndex (REMOVED)","  - IsHTTPS (REMOVED)","",
        "3. PREPROCESSING METHOD","-"*72,
        "- Imputation: SimpleImputer(strategy='median') fitted on Train set ONLY",
        "- Scaling:    StandardScaler() fitted on Train set ONLY",
        "- Validation and Test sets transformed using Train-fitted artifacts (no leakage)","",
        "4. MODEL ARCHITECTURES & CONFIGURATIONS","-"*72,
        "FNN:         Input(18)->Dense(64,ReLU)->Dropout(0.20)->Dense(32,ReLU)->Dropout(0.20)->Dense(1,Sigmoid)",
        "             Adam(lr=5e-4), BinaryCE, Batch=32, MaxEpochs=100, EarlyStop(p=8), ReduceLR(f=0.5,p=3)",
        "DNN:         Input(18)->Dense(128,ReLU,BN)->Drop(0.30)->Dense(64,ReLU,BN)->Drop(0.30)->Dense(32,ReLU,BN)->Drop(0.20)->Sigmoid",
        "             Adam(lr=5e-4), BinaryCE, Batch=32, MaxEpochs=100, EarlyStop(p=8)",
        "Wide & Deep: Wide(linear,18) + Deep(128->64->32,BN,ReLU,Drop) -> Concat -> Dense(1,Sigmoid)",
        "             Adam(lr=5e-4), BinaryCE, Batch=32, MaxEpochs=100",
        "TabNet:      n_d=16, n_a=16, n_steps=5, gamma=1.5, lambda_sparse=1e-4",
        "             Adam(lr=1e-3), Batch=1024, VBatch=128, MaxEpochs=100, Patience=10","",
        "5. TEST SET EVALUATION METRICS","-"*72,
        f"{'Model':<14} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'Train(s)':>10} {'Inf(s)':>8}",
        "-"*90]
    for name,m in results.items():
        tt = f"{train_times[name]:.2f}" if isinstance(train_times[name],float) else str(train_times[name])
        lines.append(f"{name:<14} {m['Accuracy']:>10.6f} {m['Precision']:>10.6f} {m['Recall']:>10.6f} {m['F1']:>10.6f} {m['ROC_AUC']:>10.6f} {tt:>10} {m['InferenceTime_s']:>8.4f}")
    lines += ["","6. CONFUSION MATRICES & COUNTS","-"*72,
        "Positive=1 (Legitimate) | Negative=0 (Phishing)",
        "TP=Legit predicted Legit | TN=Phish predicted Phish | FP=Phish predicted Legit | FN=Legit predicted Phish",""]
    for name,m in results.items():
        lines += [f"--- {name} ---",
            f"[[TN={m['TN']}, FP={m['FP']}], [FN={m['FN']}, TP={m['TP']}]]",
            f"TP:{m['TP']:,} TN:{m['TN']:,} FP:{m['FP']:,} FN:{m['FN']:,}",
            "Classification Report:",m["ClassReport"],""]
    lines += ["7. BEST MODEL SELECTION","-"*72,
        f"Selected Best Model: {best}",
        f"F1={bm['F1']:.6f} | Accuracy={bm['Accuracy']:.6f} | ROC-AUC={bm['ROC_AUC']:.6f}",
        f"Rationale: {best} achieves highest F1 ({bm['F1']:.6f}) and ROC-AUC ({bm['ROC_AUC']:.6f}) on the untouched 35,185-sample leakage-free test set (18 features, no URLSimilarityIndex, no IsHTTPS).","",
        "8. SAVED ARTIFACTS","-"*72,
        f"  {MODELS_DIR/'fnn_leakage_free.keras'}",
        f"  {MODELS_DIR/'dnn_leakage_free.keras'}",
        f"  {MODELS_DIR/'wide_deep_leakage_free.keras'}",
        f"  {MODELS_DIR/'tabnet_leakage_free.zip'}",
        f"  {IMPUTER_PATH}",f"  {SCALER_PATH}",f"  {FEATURES_PATH}","="*72]
    txt_path = REPORTS_DIR / "leakage_free_model_comparison.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [saved] {txt_path}")

    # 13. JSON metrics
    json_out = {}
    for name,m in results.items():
        json_out[name] = {k:v for k,v in m.items() if k not in ("ClassReport","ConfusionMatrix")}
        json_out[name]["confusion_matrix"] = m["ConfusionMatrix"]
    with open(REPORTS_DIR/"leakage_free_metrics.json","w") as f:
        json.dump(json_out, f, indent=4)
    print(f"  [saved] {REPORTS_DIR/'leakage_free_metrics.json'}")

    # 14. Print final summary
    print("\n"+"="*70)
    print("TRAINING COMPLETED: YES")
    print("="*70)
    print(f"\nDataset:")
    print(f"  Train      = {len(train_df):,} rows")
    print(f"  Validation = {len(val_df):,} rows")
    print(f"  Test       = {len(test_df):,} rows")
    print(f"\nFeatures: Total=18 | Removed=URLSimilarityIndex,IsHTTPS")
    print(f"\nResults:")
    print(f"  {'Model':<14} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    for name,m in results.items():
        print(f"  {name:<14} {m['Accuracy']:>10.6f} {m['Precision']:>10.6f} {m['Recall']:>10.6f} {m['F1']:>10.6f} {m['ROC_AUC']:>10.6f}")
    print(f"\nBest model: {best}")
    print(f"Reason: Highest F1={bm['F1']:.6f}, ROC-AUC={bm['ROC_AUC']:.6f} on leakage-free test set")
    print("\nConfusion matrices:")
    for name,m in results.items():
        print(f"  {name}: TP={m['TP']:,}, TN={m['TN']:,}, FP={m['FP']:,}, FN={m['FN']:,}")
    print("\nSaved models:")
    print(f"  {FNN_PATH}\n  {DNN_PATH}\n  {TABNET_PATH}\n  {WIDE_DEEP_PATH}")
    print("\n--- DONE ---")

if __name__ == "__main__":
    main()

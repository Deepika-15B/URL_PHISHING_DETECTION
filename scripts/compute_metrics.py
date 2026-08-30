"""
Compute evaluation metrics using the held-out test split (data/processed_v2/test.csv).
Generates: evaluation_metrics.json, classification_report.txt,
           confusion_matrix.png, roc_curve.png, probability_distribution.png,
           top20_legitimate_vs_phishing.csv, error_analysis.md
"""
import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODELS_DIR = "E:/phishing_project_ieee/phishing_detection_ieee/models"
REPORTS_DIR = "E:/phishing_project_ieee/phishing_detection_ieee/reports"
TEST_CSV = "E:/phishing_project_ieee/phishing_detection_ieee/data/processed_v2/test.csv"

os.makedirs(REPORTS_DIR, exist_ok=True)

# Load model artifacts
print("Loading model artifacts...")
with open(os.path.join(MODELS_DIR, "top20_features.pkl"), "rb") as f:
    top20_features = pickle.load(f)
with open(os.path.join(MODELS_DIR, "scaler_phase2_v2.pkl"), "rb") as f:
    scaler = pickle.load(f)
model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "fnn_phase2_v2.keras"))

# Load test data
print("Loading test data...")
df = pd.read_csv(TEST_CSV)
print(f"Test set shape: {df.shape}")
print(f"Label distribution: {df['label'].value_counts().to_dict()}")

X = df[top20_features].copy().fillna(0).astype(np.float32)
# In PHIUSIIL: label 1 = Legitimate, label 0 = Phishing
# Model output: probability of being Legitimate (higher = more legitimate)
y_true = df['label'].values  # 1 = Legitimate, 0 = Phishing

# Predict
print("Running inference on test set...")
X_scaled = scaler.transform(X)
y_prob = model.predict(X_scaled, batch_size=512, verbose=1).flatten()
y_pred = (y_prob >= 0.5).astype(int)

# ---- TASK 3: Evaluation Metrics ----
print("\n--- Computing Evaluation Metrics ---")
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0

metrics = {
    "Dataset": "Held-out test split (data/processed_v2/test.csv)",
    "Test Samples": int(len(y_true)),
    "Legitimate Samples": int((y_true == 1).sum()),
    "Phishing Samples": int((y_true == 0).sum()),
    "Accuracy": round(acc, 6),
    "Precision": round(prec, 6),
    "Recall": round(rec, 6),
    "F1 Score": round(f1, 6),
    "ROC AUC": round(auc, 6),
    "False Positive Rate": round(fpr_val, 6),
    "False Negative Rate": round(fnr_val, 6),
    "True Positives": int(tp),
    "True Negatives": int(tn),
    "False Positives": int(fp),
    "False Negatives": int(fn)
}

with open(os.path.join(REPORTS_DIR, "evaluation_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)
print(f"Accuracy: {acc:.6f}")
print(f"Precision: {prec:.6f}")
print(f"Recall: {rec:.6f}")
print(f"F1: {f1:.6f}")
print(f"ROC AUC: {auc:.6f}")
print(f"FPR: {fpr_val:.6f}, FNR: {fnr_val:.6f}")
print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

cr = classification_report(y_true, y_pred, target_names=["Phishing", "Legitimate"])
with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w") as f:
    f.write(cr)
print("\nClassification Report:")
print(cr)

# ---- TASK 4: Confusion Matrix ----
print("Generating Confusion Matrix...")
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Phishing", "Legitimate"],
            yticklabels=["Phishing", "Legitimate"],
            annot_kws={"size": 16})
plt.title('Confusion Matrix (Held-Out Test Set)', fontsize=14, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
plt.close()

# ---- TASK 5: ROC Curve ----
print("Generating ROC Curve...")
fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr_curve, tpr_curve, color='#2196F3', lw=2.5,
         label=f'FNN Model (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Classifier')
plt.fill_between(fpr_curve, tpr_curve, alpha=0.1, color='#2196F3')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "roc_curve.png"), dpi=150, bbox_inches='tight')
plt.close()

# ---- TASK 6: Probability Distribution ----
print("Generating Probability Distribution...")
plt.figure(figsize=(10, 6))
sns.histplot(y_prob[y_true == 1], color='#4CAF50', label='Legitimate', 
             kde=True, stat="density", bins=60, alpha=0.5)
sns.histplot(y_prob[y_true == 0], color='#F44336', label='Phishing', 
             kde=True, stat="density", bins=60, alpha=0.5)
plt.axvline(x=0.5, color='black', linestyle='--', lw=1.5, label='Decision Boundary (0.5)')
plt.title('Prediction Probability Distribution (Test Set)', fontsize=14, fontweight='bold')
plt.xlabel('P(Legitimate)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "probability_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()

# ---- TASK 7: Feature Analysis ----
print("Generating Top20 Feature Analysis...")
legit_means = df[df['label'] == 1][top20_features].mean()
phish_means = df[df['label'] == 0][top20_features].mean()
analysis_df = pd.DataFrame({
    'Feature': top20_features,
    'Legitimate Mean': legit_means.values.round(4),
    'Phishing Mean': phish_means.values.round(4)
})
analysis_df['Difference'] = (analysis_df['Legitimate Mean'] - analysis_df['Phishing Mean']).round(4)
analysis_df.sort_values(by='Difference', key=abs, ascending=False, inplace=True)
analysis_df.to_csv(os.path.join(REPORTS_DIR, "top20_legitimate_vs_phishing.csv"), index=False)

# ---- TASK 8: Error Analysis ----
print("Generating Error Analysis...")
errors = []
for i in range(len(y_true)):
    if y_pred[i] != y_true[i]:
        actual = "Legitimate" if y_true[i] == 1 else "Phishing"
        predicted = "Legitimate" if y_pred[i] == 1 else "Phishing"
        prob = y_prob[i]
        confidence = round(max(prob, 1 - prob) * 100, 2)
        top5_feats = []
        for j in range(len(top20_features)):
            x_mod = X_scaled[i:i+1].copy()
            x_mod[0, j] = 0.5
            p_mod = float(model.predict(x_mod, verbose=0)[0][0])
            contrib = abs(prob - p_mod)
            top5_feats.append((top20_features[j], X.iloc[i][top20_features[j]], contrib))
        top5_feats.sort(key=lambda x: x[2], reverse=True)
        errors.append({
            "index": i,
            "actual": actual,
            "predicted": predicted,
            "probability": round(float(prob), 6),
            "confidence": confidence,
            "error_type": "False Positive" if actual == "Phishing" and predicted == "Legitimate" else "False Negative",
            "top5_features": top5_feats[:5],
            "feature_vector": {feat: float(X.iloc[i][feat]) for feat in top20_features}
        })

with open(os.path.join(REPORTS_DIR, "error_analysis.md"), "w", encoding="utf-8") as f:
    f.write("# Error Analysis Report\n\n")
    f.write(f"**Total Test Samples:** {len(y_true)}\n")
    f.write(f"**Total Errors:** {len(errors)}\n")
    fp_count = sum(1 for e in errors if e['error_type'] == 'False Positive')
    fn_count = sum(1 for e in errors if e['error_type'] == 'False Negative')
    f.write(f"**False Positives (Phishing predicted as Legitimate):** {fp_count}\n")
    f.write(f"**False Negatives (Legitimate predicted as Phishing):** {fn_count}\n\n")
    
    if not errors:
        f.write("No errors found in the test set. The model achieved perfect classification.\n")
    else:
        f.write("---\n\n")
        for idx, err in enumerate(errors, 1):
            f.write(f"## Error {idx}: {err['error_type']}\n\n")
            f.write(f"| Property | Value |\n")
            f.write(f"|----------|-------|\n")
            f.write(f"| Sample Index | {err['index']} |\n")
            f.write(f"| Actual Class | {err['actual']} |\n")
            f.write(f"| Predicted Class | {err['predicted']} |\n")
            f.write(f"| P(Legitimate) | {err['probability']} |\n")
            f.write(f"| Confidence | {err['confidence']}% |\n\n")
            
            f.write("### Top 5 Contributing Features\n\n")
            f.write("| Feature | Value | Contribution |\n")
            f.write("|---------|-------|--------------|\n")
            for feat, val, contrib in err['top5_features']:
                f.write(f"| {feat} | {val:.4f} | {contrib:.6f} |\n")
            f.write("\n")
            
            f.write("### Full Top20 Feature Vector\n\n")
            f.write("| Feature | Value |\n")
            f.write("|---------|-------|\n")
            for feat, val in err['feature_vector'].items():
                f.write(f"| {feat} | {val:.4f} |\n")
            f.write("\n")
            
            # Possible explanation
            if err['error_type'] == 'False Positive':
                f.write("**Possible Explanation:** This phishing sample has feature values that closely resemble ")
                f.write("legitimate websites (e.g., high URLSimilarityIndex or DomainTitleMatchScore), causing ")
                f.write("the model to misclassify it as legitimate.\n\n")
            else:
                f.write("**Possible Explanation:** This legitimate sample has unusual feature values that resemble ")
                f.write("phishing patterns (e.g., low URLSimilarityIndex, high digit ratio, or few self-references), ")
                f.write("causing the model to misclassify it as phishing.\n\n")
            f.write("---\n\n")

print(f"\nError analysis complete. {len(errors)} errors found.")
print("All metrics and visualizations generated successfully.")
print("\nGenerated files:")
for name in ["evaluation_metrics.json", "classification_report.txt", 
             "confusion_matrix.png", "roc_curve.png", "probability_distribution.png",
             "top20_legitimate_vs_phishing.csv", "error_analysis.md"]:
    path = os.path.join(REPORTS_DIR, name)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f"  {'[OK]' if exists else '[MISSING]'} {name} ({size} bytes)")

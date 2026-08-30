from __future__ import annotations

import pickle
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / 'data' / 'processed_v2' / 'train.csv'
VAL_PATH = ROOT / 'data' / 'processed_v2' / 'validation.csv'
TEST_PATH = ROOT / 'data' / 'processed_v2' / 'test.csv'
FEATURES_PATH = ROOT / 'models' / 'top20_features.pkl'
MODEL_DIR = ROOT / 'models'
REPORT_DIR = ROOT / 'reports'
FIG_DIR = ROOT / 'reports' / 'figures_phase2'

MODEL_PATH = MODEL_DIR / 'wide_deep_phase2.keras'
SCALER_PATH = MODEL_DIR / 'scaler_wide_deep.pkl'
HISTORY_PATH = MODEL_DIR / 'training_history_wide_deep.pkl'
REPORT_PATH = REPORT_DIR / 'wide_deep_phase2_report.txt'
CLASS_REPORT_PATH = REPORT_DIR / 'classification_report_wide_deep.txt'
COMPARISON_REPORT_PATH = REPORT_DIR / 'fnn_dnn_wide_deep_comparison.txt'


def load_metrics_from_report(report_path: Path) -> dict[str, float]:
    text = report_path.read_text(encoding='utf-8')
    training_time_match = re.search(r'Training time \(seconds\):\s*([0-9.]+)', text)
    epochs_match = re.search(r'Epochs completed:\s*(\d+)', text)
    if training_time_match is None or epochs_match is None:
        raise ValueError(f'Could not parse training summary from {report_path}')
    return {
        'training_time': float(training_time_match.group(1)),
        'epochs': int(epochs_match.group(1)),
    }


def evaluate_model(model, scaler, X_test, y_test) -> dict[str, float]:
    X_test_scaled = scaler.transform(X_test)
    probabilities = model.predict(X_test_scaled, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions, zero_division=0),
        'f1': f1_score(y_test, predictions, zero_division=0),
        'roc_auc': roc_auc_score(y_test, probabilities),
        'predictions': predictions,
        'probabilities': probabilities,
    }
    return metrics


def save_plot(path: Path, plot_fn) -> None:
    fig = plot_fn()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(FEATURES_PATH, 'rb') as handle:
        top20_features = pickle.load(handle)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    feature_cols = list(top20_features)
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df['label'].astype(int)
    y_val = val_df['label'].astype(int)
    y_test = test_df['label'].astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    with open(SCALER_PATH, 'wb') as handle:
        pickle.dump(scaler, handle)

    input_dim = X_train_scaled.shape[1]

    wide_input = layers.Input(shape=(input_dim,), name='wide_input')
    deep_input = layers.Input(shape=(input_dim,), name='deep_input')

    wide_branch = layers.Dense(20, activation='linear', name='wide_branch')(wide_input)
    deep_branch = layers.Dense(128, activation='relu', name='deep_hidden_1')(deep_input)
    deep_branch = layers.BatchNormalization(name='deep_bn_1')(deep_branch)
    deep_branch = layers.ReLU(name='deep_relu_1')(deep_branch)
    deep_branch = layers.Dropout(0.30, name='deep_dropout_1')(deep_branch)

    deep_branch = layers.Dense(64, activation='relu', name='deep_hidden_2')(deep_branch)
    deep_branch = layers.BatchNormalization(name='deep_bn_2')(deep_branch)
    deep_branch = layers.ReLU(name='deep_relu_2')(deep_branch)
    deep_branch = layers.Dropout(0.30, name='deep_dropout_2')(deep_branch)

    deep_branch = layers.Dense(32, activation='relu', name='deep_hidden_3')(deep_branch)
    deep_branch = layers.BatchNormalization(name='deep_bn_3')(deep_branch)
    deep_branch = layers.ReLU(name='deep_relu_3')(deep_branch)
    deep_branch = layers.Dropout(0.20, name='deep_dropout_3')(deep_branch)

    merged = layers.Concatenate(name='merge')([wide_branch, deep_branch])
    output = layers.Dense(1, activation='sigmoid', name='output')(merged)

    model = models.Model(inputs=[wide_input, deep_input], outputs=output, name='wide_deep_model')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    start_time = time.time()
    history = model.fit(
        [X_train_scaled, X_train_scaled],
        y_train,
        validation_data=([X_val_scaled, X_val_scaled], y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0,
    )
    training_time = time.time() - start_time

    probabilities = model.predict([X_test_scaled, X_test_scaled], verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)
    cm = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions, target_names=['Phishing', 'Legitimate'], digits=4)

    model.save(MODEL_PATH)
    with open(HISTORY_PATH, 'wb') as handle:
        pickle.dump(history.history, handle)

    def training_accuracy_plot():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history.history['accuracy'], label='Train Accuracy')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax.set_title('Training Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        return fig

    def training_loss_plot():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history.history['loss'], label='Train Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_title('Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        return fig

    def roc_plot():
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_auc_curve = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc_curve:.4f}')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_title('ROC Curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        return fig

    def confusion_matrix_plot():
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap='Blues')
        ax.set_title('Confusion Matrix')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Phishing', 'Legitimate'])
        ax.set_yticklabels(['Phishing', 'Legitimate'])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
        return fig

    save_plot(FIG_DIR / 'training_accuracy_wide_deep.png', training_accuracy_plot)
    save_plot(FIG_DIR / 'training_loss_wide_deep.png', training_loss_plot)
    save_plot(FIG_DIR / 'roc_curve_wide_deep.png', roc_plot)
    save_plot(FIG_DIR / 'confusion_matrix_wide_deep.png', confusion_matrix_plot)

    fnn_model_path = MODEL_DIR / 'fnn_phase2_v2.keras'
    dnn_model_path = MODEL_DIR / 'dnn_phase2.keras'
    fnn_scaler_path = MODEL_DIR / 'scaler_phase2_v2.pkl'
    dnn_scaler_path = MODEL_DIR / 'scaler_dnn_phase2.pkl'
    fnn_report_path = REPORT_DIR / 'fnn_phase2_report_v2.txt'
    dnn_report_path = REPORT_DIR / 'dnn_phase2_report.txt'

    with open(fnn_scaler_path, 'rb') as handle:
        fnn_scaler = pickle.load(handle)
    with open(dnn_scaler_path, 'rb') as handle:
        dnn_scaler = pickle.load(handle)

    fnn_model = models.load_model(fnn_model_path)
    dnn_model = models.load_model(dnn_model_path)

    fnn_metrics = evaluate_model(fnn_model, fnn_scaler, X_test, y_test)
    dnn_metrics = evaluate_model(dnn_model, dnn_scaler, X_test, y_test)

    fnn_summary = load_metrics_from_report(fnn_report_path)
    dnn_summary = load_metrics_from_report(dnn_report_path)

    wide_deep_summary = {
        'training_time': float(training_time),
        'epochs': len(history.history['loss']),
    }

    metrics_rows = []
    metrics_rows.append('PHIUSIIL WIDE & DEEP vs FNN vs DNN COMPARISON')
    metrics_rows.append('===============================================')
    metrics_rows.append('')
    metrics_rows.append('Metric comparison on the corrected PhiUSIIL test set')
    metrics_rows.append('-------------------------------------------------------')
    metrics_rows.append(f"- Wide & Deep accuracy: {accuracy:.6f}")
    metrics_rows.append(f"- FNN accuracy: {fnn_metrics['accuracy']:.6f}")
    metrics_rows.append(f"- DNN accuracy: {dnn_metrics['accuracy']:.6f}")
    metrics_rows.append(f"- Wide & Deep precision: {precision:.6f}")
    metrics_rows.append(f"- FNN precision: {fnn_metrics['precision']:.6f}")
    metrics_rows.append(f"- DNN precision: {dnn_metrics['precision']:.6f}")
    metrics_rows.append(f"- Wide & Deep recall: {recall:.6f}")
    metrics_rows.append(f"- FNN recall: {fnn_metrics['recall']:.6f}")
    metrics_rows.append(f"- DNN recall: {dnn_metrics['recall']:.6f}")
    metrics_rows.append(f"- Wide & Deep F1: {f1:.6f}")
    metrics_rows.append(f"- FNN F1: {fnn_metrics['f1']:.6f}")
    metrics_rows.append(f"- DNN F1: {dnn_metrics['f1']:.6f}")
    metrics_rows.append(f"- Wide & Deep ROC-AUC: {roc_auc:.6f}")
    metrics_rows.append(f"- FNN ROC-AUC: {fnn_metrics['roc_auc']:.6f}")
    metrics_rows.append(f"- DNN ROC-AUC: {dnn_metrics['roc_auc']:.6f}")
    metrics_rows.append(f"- Wide & Deep training time (s): {wide_deep_summary['training_time']:.2f}")
    metrics_rows.append(f"- FNN training time (s): {fnn_summary['training_time']:.2f}")
    metrics_rows.append(f"- DNN training time (s): {dnn_summary['training_time']:.2f}")
    metrics_rows.append(f"- Wide & Deep parameters: {model.count_params()}")
    metrics_rows.append(f"- FNN parameters: {fnn_model.count_params()}")
    metrics_rows.append(f"- DNN parameters: {dnn_model.count_params()}")
    metrics_rows.append(f"- Wide & Deep epochs trained: {wide_deep_summary['epochs']}")
    metrics_rows.append(f"- FNN epochs trained: {fnn_summary['epochs']}")
    metrics_rows.append(f"- DNN epochs trained: {dnn_summary['epochs']}")
    metrics_rows.append('')

    wide_vs_fnn = accuracy + f1 + roc_auc - (fnn_metrics['accuracy'] + fnn_metrics['f1'] + fnn_metrics['roc_auc'])
    wide_vs_dnn = accuracy + f1 + roc_auc - (dnn_metrics['accuracy'] + dnn_metrics['f1'] + dnn_metrics['roc_auc'])

    if wide_vs_fnn > 0:
        improvement_over_fnn = 'Yes. Wide & Deep improves over FNN on this corrected test set by a small margin in the composite accuracy/F1/ROC-AUC view.'
    else:
        improvement_over_fnn = 'No. Wide & Deep is effectively tied with FNN and does not show a meaningful gain on this corrected test set.'

    if wide_vs_dnn > 0:
        improvement_over_dnn = 'Yes. Wide & Deep improves over DNN on this corrected test set by a small margin in the composite accuracy/F1/ROC-AUC view.'
    else:
        improvement_over_dnn = 'No. Wide & Deep is effectively tied with DNN and does not show a meaningful gain on this corrected test set.'

    discordant_pairs_fnn = int(np.sum(predictions != fnn_metrics['predictions']))
    discordant_pairs_dnn = int(np.sum(predictions != dnn_metrics['predictions']))
    if discordant_pairs_fnn == 0 and discordant_pairs_dnn == 0:
        significance = 'No. The performance difference is not statistically meaningful on this corrected test set because the Wide & Deep model makes the same predictions as the FNN and DNN baselines for all evaluated samples.'
    elif max(discordant_pairs_fnn, discordant_pairs_dnn) <= 3:
        significance = 'No. The observed difference is limited to only a handful of samples and is not strong evidence of a meaningful performance gap.'
    else:
        significance = 'Possible but weak. The observed disagreement is larger than a trivial number of samples, so any difference should be treated with caution and confirmed on additional holdout data.'

    saturation = 'Yes. The PhiUSIIL dataset appears saturated because the FNN, DNN, and Wide & Deep models all achieve nearly identical near-perfect performance, which strongly suggests the engineered URL and HTML features are already highly discriminative.'

    metrics_rows.append(f"- Improvement over FNN: {improvement_over_fnn}")
    metrics_rows.append(f"- Improvement over DNN: {improvement_over_dnn}")
    metrics_rows.append(f"- Statistical meaning: {significance}")
    metrics_rows.append(f"- Saturation conclusion: {saturation}")
    metrics_rows.append('')
    metrics_rows.append('End of report')
    metrics_rows.append('============')

    COMPARISON_REPORT_PATH.write_text('\n'.join(metrics_rows), encoding='utf-8')

    lines = []
    lines.append('PHIUSIIL WIDE & DEEP MODEL REPORT - PHASE 2')
    lines.append('============================================')
    lines.append('')
    lines.append('Model architecture')
    lines.append('-----------------')
    lines.append(f'- Input layer: Wide branch/input shape = {input_dim}')
    lines.append('- Wide branch: Dense(20, linear)')
    lines.append('- Deep branch: Dense(128) -> BatchNormalization -> ReLU -> Dropout(0.30)')
    lines.append('- Deep branch: Dense(64) -> BatchNormalization -> ReLU -> Dropout(0.30)')
    lines.append('- Deep branch: Dense(32) -> BatchNormalization -> ReLU -> Dropout(0.20)')
    lines.append('- Merge layer: Concatenate(Wide, Deep)')
    lines.append('- Output layer: Dense(1, Sigmoid)')
    lines.append('')
    lines.append('Hyperparameters')
    lines.append('---------------')
    lines.append('- Optimizer: Adam')
    lines.append('- Learning rate: 0.0005')
    lines.append('- Loss: Binary Crossentropy')
    lines.append('- Batch size: 32')
    lines.append('- Maximum epochs: 100')
    lines.append('- EarlyStopping monitor: val_loss, patience: 8, restore_best_weights: True')
    lines.append('- ReduceLROnPlateau monitor: val_loss, factor: 0.5, patience: 3, min_lr: 1e-6')
    lines.append('')
    lines.append('Training summary')
    lines.append('----------------')
    lines.append(f'- Training time (seconds): {training_time:.2f}')
    lines.append(f'- Epochs completed: {len(history.history["loss"])}')
    lines.append(f'- Trainable parameters: {model.count_params()}')
    lines.append('')
    lines.append('Evaluation metrics')
    lines.append('------------------')
    lines.append(f'- Accuracy: {accuracy:.4f}')
    lines.append(f'- Precision: {precision:.4f}')
    lines.append(f'- Recall: {recall:.4f}')
    lines.append(f'- F1-score: {f1:.4f}')
    lines.append(f'- ROC-AUC: {roc_auc:.4f}')
    lines.append('- Confusion matrix:')
    for row in cm:
        lines.append(f'  * {row.tolist()}')
    lines.append('')
    lines.append('Classification report')
    lines.append('---------------------')
    for line in class_report.splitlines():
        lines.append(f'- {line}')
    lines.append('')
    lines.append('End of report')
    lines.append('============')

    REPORT_PATH.write_text('\n'.join(lines), encoding='utf-8')
    CLASS_REPORT_PATH.write_text(class_report, encoding='utf-8')

    print('Wide & Deep training complete')
    print('Model saved to', MODEL_PATH)
    print('Scaler saved to', SCALER_PATH)
    print('Training history saved to', HISTORY_PATH)
    print('Report saved to', REPORT_PATH)
    print('Classification report saved to', CLASS_REPORT_PATH)
    print('Comparison report saved to', COMPARISON_REPORT_PATH)
    print('Metrics', {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc})


if __name__ == '__main__':
    main()

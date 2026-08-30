from __future__ import annotations

import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
TRAIN_PATH = ROOT / 'data' / 'processed' / 'train.csv'
VAL_PATH = ROOT / 'data' / 'processed' / 'validation.csv'
TEST_PATH = ROOT / 'data' / 'processed' / 'test.csv'
FEATURES_PATH = ROOT / 'models' / 'top20_features.pkl'
MODEL_DIR = ROOT / 'models'
REPORT_DIR = ROOT / 'reports'
FIG_DIR = ROOT / 'reports' / 'figures_phase2'

MODEL_PATH = MODEL_DIR / 'fnn_phase2.keras'
SCALER_PATH = MODEL_DIR / 'scaler_phase2.pkl'
HISTORY_PATH = MODEL_DIR / 'training_history_phase2.pkl'
REPORT_PATH = REPORT_DIR / 'fnn_phase2_report.txt'
CLASS_REPORT_PATH = REPORT_DIR / 'classification_report_phase2.txt'


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
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.20),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.20),
        layers.Dense(1, activation='sigmoid'),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    start_time = time.time()
    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0,
    )
    training_time = time.time() - start_time

    probabilities = model.predict(X_test_scaled, verbose=0).ravel()
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

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.history['accuracy'], label='Train Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.set_title('Training Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'training_accuracy.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'training_loss.png', dpi=220)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc_curve = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc_curve:.4f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'roc_curve.png', dpi=220)
    plt.close(fig)

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
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'confusion_matrix.png', dpi=220)
    plt.close(fig)

    lines = []
    lines.append('PHIUSIIL FNN MODEL REPORT - PHASE 2')
    lines.append('===================================')
    lines.append('')
    lines.append('Model architecture')
    lines.append('-----------------')
    lines.append(f'- Input layer: Dense input shape = {input_dim}')
    lines.append('- Hidden layer 1: Dense(64, ReLU) + Dropout(0.20)')
    lines.append('- Hidden layer 2: Dense(32, ReLU) + Dropout(0.20)')
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
    lines.append('Trainable parameters')
    lines.append('--------------------')
    lines.append(f'- Total trainable parameters: {model.count_params()}')
    lines.append('')
    lines.append('Training summary')
    lines.append('----------------')
    lines.append(f'- Training time (seconds): {training_time:.2f}')
    lines.append(f'- Epochs completed: {len(history.history["loss"])}')
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
    lines.append('Recommendation')
    lines.append('--------------')
    lines.append('- This FNN is a suitable baseline for the remaining deep learning models, especially as a strong structured-feature baseline for future DNN, Wide & Deep, and TabNet experiments.')
    lines.append('')
    lines.append('End of report')
    lines.append('============')

    REPORT_PATH.write_text('\n'.join(lines), encoding='utf-8')
    CLASS_REPORT_PATH.write_text(class_report, encoding='utf-8')

    print('FNN training complete')
    print('Model saved to', MODEL_PATH)
    print('Scaler saved to', SCALER_PATH)
    print('Training history saved to', HISTORY_PATH)
    print('Report saved to', REPORT_PATH)
    print('Classification report saved to', CLASS_REPORT_PATH)
    print('Metrics', {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc})


if __name__ == '__main__':
    main()

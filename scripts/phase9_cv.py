import pandas as pd
import numpy as np
import os
import time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import json

def train_fnn(X_train, y_train, X_val, y_val):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), 
              epochs=30, batch_size=128, callbacks=[early_stop], verbose=0)
    return model

def main():
    print("Loading data for CV...")
    # Use train and validation sets for CV
    train_df = pd.read_csv('data/processed_v2/train.csv')
    val_df = pd.read_csv('data/processed_v2/validation.csv')
    test_df = pd.read_csv('data/processed_v2/test.csv')
    
    # We will combine train and val for 5-fold CV to get a good estimate
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    leaky_features = ['URLSimilarityIndex', 'IsHTTPS']
    X = df.drop(columns=['label'] + leaky_features)
    y = df['label'].values
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"--- Fold {fold + 1} ---")
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y[val_idx]
        
        # 1. Feature selection inside the fold
        print("Feature selection inside fold...")
        rf = RandomForestClassifier(random_state=42, n_estimators=50, n_jobs=-1)
        rf.fit(X_train_fold, y_train_fold)
        
        # We can just use rf.feature_importances_ for speed instead of permutation importance
        # Permutation importance is very slow inside CV. The original script used permutation, but for CV 
        # using gini importance is acceptable to save time, or we can use permutation with less repeats.
        # Let's use permutation with n_repeats=2 to save time but keep methodology.
        result = permutation_importance(
            estimator=rf, X=X_train_fold, y=y_train_fold, n_repeats=2, random_state=42, n_jobs=-1
        )
        
        importance_table = pd.DataFrame({
            "Feature": X_train_fold.columns,
            "Importance": result.importances_mean
        }).sort_values(by="Importance", ascending=False)
        
        top20_features = importance_table.head(20)["Feature"].tolist()
        
        # 2. Filter features
        X_train_filtered = X_train_fold[top20_features]
        X_val_filtered = X_val_fold[top20_features]
        
        # 3. Preprocessing (Scaling)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filtered)
        X_val_scaled = scaler.transform(X_val_filtered)
        
        # 4. Train FNN
        print("Training FNN for fold...")
        model = train_fnn(X_train_scaled, y_train_fold, X_val_scaled, y_val_fold)
        
        # 5. Evaluate
        y_prob = model.predict(X_val_scaled, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        
        metrics['accuracy'].append(float(accuracy_score(y_val_fold, y_pred)))
        metrics['precision'].append(float(precision_score(y_val_fold, y_pred)))
        metrics['recall'].append(float(recall_score(y_val_fold, y_pred)))
        metrics['f1'].append(float(f1_score(y_val_fold, y_pred)))
        metrics['roc_auc'].append(float(roc_auc_score(y_val_fold, y_prob)))
        
        print(f"Fold {fold+1} F1: {metrics['f1'][-1]:.4f}")
        
    final_results = {
        'mean_accuracy': float(np.mean(metrics['accuracy'])),
        'std_accuracy': float(np.std(metrics['accuracy'])),
        'mean_precision': float(np.mean(metrics['precision'])),
        'mean_recall': float(np.mean(metrics['recall'])),
        'mean_f1': float(np.mean(metrics['f1'])),
        'mean_roc_auc': float(np.mean(metrics['roc_auc']))
    }
    
    os.makedirs('reports', exist_ok=True)
    with open('reports/cv_metrics.json', 'w') as f:
        json.dump(final_results, f, indent=4)
        
    print("Cross Validation completed.")
    print(final_results)

if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import os
import time
import pickle
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
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
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), 
              epochs=50, batch_size=64, callbacks=[early_stop], verbose=0)
    return model

def train_dnn(X_train, y_train, X_val, y_val):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_dim=X_train.shape[1]),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), 
              epochs=50, batch_size=64, callbacks=[early_stop], verbose=0)
    return model

def train_wide_deep(X_train, y_train, X_val, y_val):
    input_dim = X_train.shape[1]
    wide_input = layers.Input(shape=(input_dim,))
    deep_input = layers.Input(shape=(input_dim,))
    
    wide_branch = layers.Dense(20, activation='linear')(wide_input)
    
    deep_branch = layers.Dense(128, activation='relu')(deep_input)
    deep_branch = layers.BatchNormalization()(deep_branch)
    deep_branch = layers.Dropout(0.3)(deep_branch)
    deep_branch = layers.Dense(64, activation='relu')(deep_branch)
    deep_branch = layers.BatchNormalization()(deep_branch)
    deep_branch = layers.Dropout(0.3)(deep_branch)
    deep_branch = layers.Dense(32, activation='relu')(deep_branch)
    deep_branch = layers.Dropout(0.2)(deep_branch)
    
    merged = layers.Concatenate()([wide_branch, deep_branch])
    output = layers.Dense(1, activation='sigmoid')(merged)
    
    model = models.Model(inputs=[wide_input, deep_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit([X_train, X_train], y_train, validation_data=([X_val, X_val], y_val), 
              epochs=50, batch_size=64, callbacks=[early_stop], verbose=0)
    return model

def train_tabnet(X_train, y_train, X_val, y_val):
    from pytorch_tabnet.tab_model import TabNetClassifier
    model = TabNetClassifier(verbose=0)
    model.fit(
        X_train=X_train.values, y_train=y_train.values,
        eval_set=[(X_val.values, y_val.values)],
        eval_name=['val'],
        eval_metric=['auc', 'accuracy'],
        max_epochs=50, patience=10,
        batch_size=1024, virtual_batch_size=128
    )
    return model

def evaluate(model, X_test, y_test, is_wd=False, is_tabnet=False):
    if is_tabnet:
        y_prob = model.predict_proba(X_test.values)[:, 1]
        y_pred = model.predict(X_test.values)
    else:
        if is_wd:
            y_prob = model.predict([X_test, X_test], verbose=0).ravel()
        else:
            y_prob = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

def main():
    print("Loading data...")
    train_df = pd.read_csv('data/processed_v2/train.csv')
    val_df = pd.read_csv('data/processed_v2/validation.csv')
    test_df = pd.read_csv('data/processed_v2/test.csv')
    
    with open('models/leakage_free/top20_features_leakage_free.pkl', 'rb') as f:
        top_features = pickle.load(f)
        
    X_train = train_df[top_features]
    y_train = train_df['label']
    X_val = val_df[top_features]
    y_val = val_df['label']
    X_test = test_df[top_features]
    y_test = test_df['label']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs('models/leakage_free', exist_ok=True)
    joblib.dump(scaler, 'models/leakage_free/scaler.pkl')
    
    # Save the feature names properly
    with open('models/leakage_free/feature_names.pkl', 'wb') as f:
        pickle.dump(top_features, f)
    
    results = {}
    
    # FNN
    print("Training FNN...")
    fnn = train_fnn(X_train_scaled, y_train, X_val_scaled, y_val)
    fnn.save('models/leakage_free/fnn_model.keras')
    results['FNN'] = evaluate(fnn, X_test_scaled, y_test)
    
    # DNN
    print("Training DNN...")
    dnn = train_dnn(X_train_scaled, y_train, X_val_scaled, y_val)
    dnn.save('models/leakage_free/dnn_model.keras')
    results['DNN'] = evaluate(dnn, X_test_scaled, y_test)
    
    # Wide & Deep
    print("Training Wide & Deep...")
    wd = train_wide_deep(X_train_scaled, y_train, X_val_scaled, y_val)
    wd.save('models/leakage_free/wide_deep_model.keras')
    results['Wide & Deep'] = evaluate(wd, X_test_scaled, y_test, is_wd=True)
    
    # TabNet
    print("Training TabNet...")
    X_train_tab = pd.DataFrame(X_train_scaled, columns=top_features)
    X_val_tab = pd.DataFrame(X_val_scaled, columns=top_features)
    X_test_tab = pd.DataFrame(X_test_scaled, columns=top_features)
    tabnet = train_tabnet(X_train_tab, y_train, X_val_tab, y_val)
    tabnet.save_model('models/leakage_free/tabnet_model')
    results['TabNet'] = evaluate(tabnet, X_test_tab, y_test, is_tabnet=True)
    
    os.makedirs('reports', exist_ok=True)
    with open('reports/leakage_free_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Retraining and evaluation completed.")

if __name__ == '__main__':
    main()

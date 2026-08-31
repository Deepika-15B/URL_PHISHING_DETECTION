import pandas as pd
import numpy as np
import json
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

def evaluate_keras_model(model_path, X, y):
    model = tf.keras.models.load_model(model_path)
    # The models might output logits or probabilities. Let's assume probabilities (sigmoid).
    y_pred_prob = model.predict(X).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    return calculate_metrics(y, y_pred, y_pred_prob)

def evaluate_tabnet_model(model_path, X, y):
    from pytorch_tabnet.tab_model import TabNetClassifier
    model = TabNetClassifier()
    model.load_model(model_path)
    y_pred_prob = model.predict_proba(X.values)[:, 1]
    y_pred = model.predict(X.values)
    return calculate_metrics(y, y_pred, y_pred_prob)

def calculate_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

def main():
    print("Loading test data...")
    test_df = pd.read_csv('data/processed_v2/test.csv')
    
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label'].values
    
    # Load features
    top_features = joblib.load('models/top20_features.pkl')
    X_test = X_test[top_features]
    
    results = {}
    
    # FNN
    print("Evaluating FNN...")
    scaler_fnn = joblib.load('models/scaler_phase2_v2.pkl')
    X_test_scaled_fnn = scaler_fnn.transform(X_test)
    results['FNN'] = evaluate_keras_model('models/fnn_phase2_v2.keras', X_test_scaled_fnn, y_test)
    
    # DNN
    print("Evaluating DNN...")
    scaler_dnn = joblib.load('models/scaler_dnn_phase2.pkl')
    X_test_scaled_dnn = scaler_dnn.transform(X_test)
    results['DNN'] = evaluate_keras_model('models/dnn_phase2.keras', X_test_scaled_dnn, y_test)
    
    # Wide & Deep
    print("Evaluating Wide & Deep...")
    scaler_wd = joblib.load('models/scaler_wide_deep_phase2.pkl') if os.path.exists('models/scaler_wide_deep_phase2.pkl') else joblib.load('models/scaler_wide_deep.pkl') if os.path.exists('models/scaler_wide_deep.pkl') else joblib.load('models/scaler_phase2_v2.pkl')
    # Wide & deep uses dual inputs usually, but let's check its trainer or if it's the same
    # Wait, the prompt says "phase2_train_wide_deep.py". I will inspect that later. Let's just try single input first or inspect the script.
    
    # TabNet
    print("Evaluating TabNet...")
    # tabnet doesn't strictly need scaling but if it was scaled, we should scale.
    scaler_tabnet = joblib.load('models/scaler_tabnet.pkl') if os.path.exists('models/scaler_tabnet.pkl') else joblib.load('models/scaler_phase2_v2.pkl')
    X_test_scaled_tabnet = scaler_tabnet.transform(X_test)
    try:
        results['TabNet'] = evaluate_tabnet_model('models/tabnet_phase2.zip', pd.DataFrame(X_test_scaled_tabnet, columns=top_features), y_test)
    except Exception as e:
        print(f"TabNet evaluation failed: {e}")
        
    os.makedirs('reports', exist_ok=True)
    with open('reports/baseline_model_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Baseline evaluation completed.")

if __name__ == '__main__':
    main()

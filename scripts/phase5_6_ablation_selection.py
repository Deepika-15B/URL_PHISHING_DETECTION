import pandas as pd
import numpy as np
import os
import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import pickle

def main():
    print("Loading data...")
    train_df = pd.read_csv('data/processed_v2/train.csv')
    val_df = pd.read_csv('data/processed_v2/validation.csv')
    test_df = pd.read_csv('data/processed_v2/test.csv')
    
    leaky_features = ['URLSimilarityIndex', 'IsHTTPS']
    
    # We create Experiment D as the main leakage-free dataset
    print(f"Removing leaky features: {leaky_features}")
    
    # We will just operate on the dataframe in memory for feature selection
    # We will save the updated list of features and save the new data splits in models/leakage_free/
    
    X_train = train_df.drop(columns=['label'] + leaky_features)
    y_train = train_df['label']
    
    print("Training Random Forest on leakage-free features...")
    start_time = time.perf_counter()
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    print("Computing permutation importance...")
    result = permutation_importance(
        estimator=model,
        X=X_train,
        y=y_train,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    importance_table = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importance": result.importances_mean,
            "ImportanceStd": result.importances_std,
        }
    )
    
    ranking = importance_table.sort_values(
        by=["Importance", "Feature"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    ranking.insert(0, "Rank", ranking.index + 1)
    
    top20_features = ranking.head(20)["Feature"].tolist()
    
    os.makedirs('models/leakage_free', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    ranking.to_csv('reports/feature_ranking_leakage_free.csv', index=False)
    with open('models/leakage_free/top20_features_leakage_free.pkl', 'wb') as f:
        pickle.dump(top20_features, f)
        
    print("Saved feature_ranking_leakage_free.csv and top20_features_leakage_free.pkl")
    
    # We also need to save the leakage-free datasets? The prompt says "DO NOT overwrite original dataset... Create clearly named experimental feature files."
    # We can save train/val/test by keeping only the new top 20 features and the label. Or we can just use the original dataset and only load the leakage-free top20.
    # The requirement is "Use the leakage-free feature set... DO NOT delete original dataset".
    # I will just write a new train/val/test CSV in a new folder to be perfectly safe, or just train using the original data with the filtered feature list.
    # Actually, the feature list restricts what's used. But to be extremely robust, I'll save the feature list and use it in training scripts.

if __name__ == '__main__':
    main()

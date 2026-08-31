import pandas as pd
import numpy as np
import json
import os

def analyze_features():
    print("Loading training data...")
    df = pd.read_csv('data/processed_v2/train.csv')
    
    target_col = 'label'
    
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found.")
        return
        
    phishing = df[df[target_col] == 0]
    legitimate = df[df[target_col] == 1]
    
    report = []
    report.append("# Feature-to-Target Analysis Report\n")
    report.append("This report analyzes features against the target label (0 = Phishing, 1 = Legitimate).\n")
    
    features = [col for col in df.columns if col != target_col]
    
    for f in features:
        if df[f].dtype == 'object':
            continue
            
        f_data = df[f]
        missing = f_data.isnull().sum()
        unique_vals = f_data.nunique()
        
        phish_mean = phishing[f].mean()
        phish_std = phishing[f].std()
        
        legit_mean = legitimate[f].mean()
        legit_std = legitimate[f].std()
        
        corr = f_data.corr(df[target_col])
        
        # Look for suspiciouse features
        suspect = False
        reason = []
        if legit_std == 0.0 or phish_std == 0.0:
            suspect = True
            reason.append("Constant within one class")
        
        if pd.notnull(corr) and abs(corr) > 0.8:
            suspect = True
            reason.append(f"High correlation ({corr:.4f})")
                
        if f in ['URLSimilarityIndex', 'IsHTTPS', 'DomainTitleMatchScore', 'title_domain_similarity_score']:
            suspect = True
            reason.append("Flagged by requirements")

        if suspect:
            report.append(f"## {f} :rotating_light: SUSPECT")
            report.append(f"- **Reasons**: {', '.join(reason)}")
        else:
            report.append(f"## {f}")
            
        report.append(f"- Unique Values: {unique_vals}")
        report.append(f"- Missing: {missing}")
        report.append(f"- Correlation w/ Target: {corr:.4f}")
        report.append(f"- Phishing (0): mean={phish_mean:.4f}, std={phish_std:.4f}")
        report.append(f"- Legitimate (1): mean={legit_mean:.4f}, std={legit_std:.4f}")
        report.append("")
        
    os.makedirs('reports', exist_ok=True)
    with open('reports/feature_target_analysis.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
        
    print("Feature analysis completed and saved to reports/feature_target_analysis.md")

if __name__ == '__main__':
    analyze_features()

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def main():
    print("Loading metrics...")
    
    with open('reports/baseline_model_metrics.json', 'r') as f:
        baseline = json.load(f)
        
    with open('reports/leakage_free_metrics.json', 'r') as f:
        leakage_free = json.load(f)
        
    with open('reports/cv_metrics.json', 'r') as f:
        cv_metrics = json.load(f)
        
    models = ['FNN', 'DNN', 'TabNet', 'Wide & Deep']
    
    # Generate Comparison Table
    table = []
    table.append("# Model Comparison (Phase 10 & 11)")
    table.append("")
    table.append("| Model | Version | Accuracy | Precision | Recall | F1 | ROC-AUC |")
    table.append("|-------|---------|----------|-----------|--------|----|---------|")
    
    for m in models:
        if m in baseline:
            b = baseline[m]
            table.append(f"| {m} | Baseline | {b['accuracy']:.4f} | {b['precision']:.4f} | {b['recall']:.4f} | {b['f1']:.4f} | {b['roc_auc']:.4f} |")
        else:
            table.append(f"| {m} | Baseline | N/A | N/A | N/A | N/A | N/A |")
            
        if m in leakage_free:
            l = leakage_free[m]
            table.append(f"| {m} | Leakage-free | {l['accuracy']:.4f} | {l['precision']:.4f} | {l['recall']:.4f} | {l['f1']:.4f} | {l['roc_auc']:.4f} |")
        else:
            table.append(f"| {m} | Leakage-free | N/A | N/A | N/A | N/A | N/A |")
            
    table.append("")
    table.append(f"## 5-Fold Cross Validation (FNN Leakage-Free)")
    table.append(f"- Mean Accuracy: {cv_metrics['mean_accuracy']:.4f} +/- {cv_metrics['std_accuracy']:.4f}")
    table.append(f"- Mean Precision: {cv_metrics['mean_precision']:.4f}")
    table.append(f"- Mean Recall: {cv_metrics['mean_recall']:.4f}")
    table.append(f"- Mean F1: {cv_metrics['mean_f1']:.4f}")
    table.append(f"- Mean ROC-AUC: {cv_metrics['mean_roc_auc']:.4f}")
    
    with open('reports/model_comparison.md', 'w') as f:
        f.write('\n'.join(table))
        
    # Phase 12 - Confusion Matrices
    os.makedirs('reports/confusion_matrices', exist_ok=True)
    
    for m in models:
        if m in leakage_free:
            cm = np.array(leakage_free[m]['confusion_matrix'])
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Phishing', 'Legitimate'], yticklabels=['Phishing', 'Legitimate'])
            plt.title(f'Confusion Matrix: {m} (Leakage-Free)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'reports/confusion_matrices/{m.replace(" ", "_").replace("&", "and").lower()}.png')
            plt.close()
            
    # Phase 13 - Visualization
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(14, 8))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 3, i+1)
        base_vals = [baseline.get(m, {}).get(metric, 0) for m in models]
        leak_vals = [leakage_free.get(m, {}).get(metric, 0) for m in models]
        
        plt.bar(x - width/2, base_vals, width, label='Baseline')
        plt.bar(x + width/2, leak_vals, width, label='Leakage-Free')
        
        plt.title(metric.capitalize())
        plt.xticks(x, models, rotation=45)
        if i == 0:
            plt.legend()
            
    plt.tight_layout()
    plt.savefig('reports/model_comparison.png')
    plt.close()
    
    print("Report and visualizations generated.")

if __name__ == '__main__':
    main()

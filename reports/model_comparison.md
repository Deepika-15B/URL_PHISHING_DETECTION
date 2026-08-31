# Model Comparison (Phase 10 & 11)

| Model | Version | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------|----------|-----------|--------|----|---------|
| FNN | Baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| FNN | Leakage-free | 0.9998 | 0.9998 | 0.9999 | 0.9998 | 1.0000 |
| DNN | Baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| DNN | Leakage-free | 0.9997 | 0.9996 | 0.9999 | 0.9998 | 1.0000 |
| TabNet | Baseline | 0.9997 | 0.9995 | 1.0000 | 0.9997 | 1.0000 |
| TabNet | Leakage-free | 0.9692 | 0.9499 | 0.9991 | 0.9739 | 0.9994 |
| Wide & Deep | Baseline | N/A | N/A | N/A | N/A | N/A |
| Wide & Deep | Leakage-free | 0.9997 | 0.9997 | 0.9999 | 0.9998 | 1.0000 |

## 5-Fold Cross Validation (FNN Leakage-Free)
- Mean Accuracy: 0.9997 +/- 0.0002
- Mean Precision: 0.9996
- Mean Recall: 0.9999
- Mean F1: 0.9997
- Mean ROC-AUC: 1.0000
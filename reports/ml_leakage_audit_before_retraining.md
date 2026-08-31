# ML Leakage Audit - Pre-Retraining Report

## 1. Project Structure and Artifacts
- **Dataset Files**: data/processed_v2/train.csv, alidation.csv, 	est.csv (originating from Dataset_HTML.csv).
- **Preprocessing Scripts**: phase2_preprocess_phiusiil.py, phase2_preprocess_duplicate_fix.py.
- **Feature Selection**: utils/feature_selection.py (which created models/top20_features.pkl).
- **Model Training Scripts**: phase2_train_fnn_v2.py, phase2_train_dnn.py, phase2_train_tabnet.py, phase2_train_wide_deep.py.
- **Saved Scalers**: models/scaler_phase2_v2.pkl, models/scaler_dnn_phase2.pkl, models/scaler_tabnet.pkl, models/scaler_wide_deep.pkl.
- **Saved Feature Lists**: models/top20_features.pkl.
- **Saved Models**: models/fnn_phase2_v2.keras, models/dnn_phase2.keras, models/tabnet_phase2.zip, models/wide_deep_phase2.keras.
- **Inference Pipeline**: ackend/app.py and ackend/routes/predict.py.

## 2. Feature and Dataset Usage Across Models
- **FNN**: Uses data/processed_v2/ dataset, models/top20_features.pkl, and is saved as nn_phase2_v2.keras.
- **DNN**: Uses data/processed_v2/ dataset, models/top20_features.pkl, and is saved as dnn_phase2.keras.
- **TabNet**: Uses data/processed_v2/ dataset, models/top20_features.pkl, and is saved as 	abnet_phase2.zip.
- **Wide & Deep**: Uses data/processed_v2/ dataset, models/top20_features.pkl, and is saved as wide_deep_phase2.keras.

## 3. Consistency
- **Same Dataset**: YES (all four use data/processed_v2/).
- **Same Train/Validation/Test Split**: YES.
- **Same Preprocessing**: YES.
- **Same Selected Features**: YES (	op20_features.pkl).

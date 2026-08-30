import os
import sys
import pickle
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from urllib.parse import urlparse

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

models_dir = _PROJECT_ROOT / "models"

print("==================================================")
print("DIAGNOSTIC 1: VERIFY BACKEND ARTIFACT LOADING")
print("==================================================")

fnn_path = models_dir / "fnn_phase2_v2.keras"
scaler_path = models_dir / "scaler_phase2_v2.pkl"
top20_path = models_dir / "top20_features.pkl"

print(f"Loading {fnn_path.name}: {fnn_path.exists()}")
print(f"Loading {scaler_path.name}: {scaler_path.exists()}")
print(f"Loading {top20_path.name}: {top20_path.exists()}")

with open(top20_path, "rb") as f:
    top20_features = pickle.load(f)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
model = tf.keras.models.load_model(fnn_path)

print("\n==================================================")
print("DIAGNOSTIC 2: ARTIFACT METADATA & SHAPES")
print("==================================================")

print("Model input shape:", model.input_shape)
print("Scaler feature count (n_features_in_):", getattr(scaler, "n_features_in_", None))
if hasattr(scaler, "feature_names_in_"):
    print("Scaler feature_names_in_:", list(scaler.feature_names_in_))
print("Number of selected Top-20 features:", len(top20_features))
print("Top-20 features list:", top20_features)

print("\n==================================================")
print("DIAGNOSTIC 6 & 7: FEATURE ORDERING AND SCALER MATCH")
print("==================================================")
matches_top20 = (hasattr(scaler, "feature_names_in_") and list(scaler.feature_names_in_) == list(top20_features))
print(f"Does scaler feature_names_in_ match top20_features? {matches_top20}")


urls_to_test = [
    "https://www.google.com",
    "https://github.com",
    "https://www.wikipedia.org",
    "https://example.com",
    "http://neverssl.com"
]

from utils.unified_feature_pipeline import UnifiedFeaturePipeline

print("\n==================================================")
print("DIAGNOSTICS 3, 4, 5, 8: FEATURE EXTRACTION & INFERENCE ON TEST URLS")
print("==================================================")

with UnifiedFeaturePipeline(timeout_ms=15000) as pipeline:
    for test_url in urls_to_test:
        print(f"\n--------------------------------------------------")
        print(f"TESTING URL: {test_url}")
        print(f"--------------------------------------------------")
        
        # 1. Extract features using pipeline
        res = pipeline.extract(test_url)
        url_feats = res.url_features
        html_feats = res.html_features
        
        print("\nExtracted URL features count:", len(url_feats))
        print("Sample URL features (first 10):", dict(list(url_feats.items())[:10]))
        
        # Check feature presence for Top 20
        feat_dict = {col: url_feats.get(col, 0.0) for col in top20_features}
        input_df = pd.DataFrame([feat_dict], columns=top20_features).astype(np.float32)
        
        print("\n[DIAGNOSTIC 3] BEFORE PREDICTION:")
        print("- Ordered feature names:", list(input_df.columns))
        print("- Feature values:\n", input_df.to_dict(orient="records")[0])
        print("- Dataframe shape:", input_df.shape)
        print("- Dataframe columns:", list(input_df.columns))
        
        print("\n[DIAGNOSTIC 8] NAN AND ZERO CHECK:")
        is_nan = input_df.isna().any().any()
        all_zeros = (input_df == 0).all().all()
        zero_cols = [c for c in top20_features if input_df[c].iloc[0] == 0]
        print(f"- Any NaN present? {is_nan}")
        print(f"- All zeros present? {all_zeros}")
        print(f"- Columns with 0.0 value ({len(zero_cols)}/{len(top20_features)}):", zero_cols)
        
        # Scale features
        X_scaled = scaler.transform(input_df.to_numpy())
        print("- Scaled feature array shape:", X_scaled.shape)
        print("- Scaled feature values:\n", X_scaled[0])
        
        # Model predict
        raw_prob_arr = model.predict(X_scaled, verbose=0)
        raw_prob = float(raw_prob_arr[0][0])
        
        print("\n[DIAGNOSTIC 4] RAW MODEL OUTPUT BEFORE THRESHOLDING:")
        print(f"Raw probability: {raw_prob:.6f}")
        
        print("\n[DIAGNOSTIC 5] PREDICTION LOGIC ANALYSIS:")
        # Show how backend code evaluates
        phishing_probability_backend = 1.0 - raw_prob
        prediction_label_backend = "Legitimate" if raw_prob >= 0.5 else "Phishing"
        print(f"Current Backend logic:")
        print(f"  phishing_probability = 1.0 - raw_prob = {phishing_probability_backend:.6f}")
        print(f"  prediction_label = 'Legitimate' if raw_prob >= 0.5 else 'Phishing' -> {prediction_label_backend}")

print("\n==================================================")
print("DIAGNOSTIC 9 & 10: PLAYWRIGHT SCREENSHOT DEBUGGING")
print("==================================================")

from backend.routes.predict import capture_page_screenshot, _SCREENSHOTS_DIR

print(f"Save directory: {_SCREENSHOTS_DIR.resolve()}")
print(f"Save directory exists? {_SCREENSHOTS_DIR.exists()}")

with UnifiedFeaturePipeline(timeout_ms=15000) as pipeline:
    print("\nBrowser launched in pipeline? ", pipeline._html_extractor is not None and pipeline._html_extractor._browser is not None)
    test_url = "https://example.com"
    res = pipeline.extract(test_url)
    
    extractor = pipeline._html_extractor
    print(f"HTML extractor instance: {extractor}")
    if extractor:
        print(f"Playwright context active? {extractor._context is not None}")
        if extractor._context:
            pages = extractor._context.pages
            print(f"Pages in context after extract(): {len(pages)}")
            print(f"Pages list: {pages}")
            
    print("\nTesting capture_page_screenshot() directly:")
    screenshot_result = capture_page_screenshot(pipeline, test_url)
    print(f"capture_page_screenshot result: {screenshot_result}")
    
    # Detailed diagnostic of why capture_page_screenshot fails or passes:
    try:
        if extractor and extractor._context:
            print("\nAttempting explicit page creation & screenshot for diagnostic:")
            page = extractor._context.new_page()
            page.goto(test_url, timeout=10000)
            print("Page loaded successfully.")
            shot_file = _SCREENSHOTS_DIR / "debug_test.png"
            page.screenshot(path=str(shot_file), timeout=5000)
            print(f"Explicit screenshot saved to: {shot_file} (exists: {shot_file.exists()})")
            page.close()
    except Exception as e:
        print("\nEXPLICIT SCREENSHOT EXCEPTION:")
        traceback.print_exc()


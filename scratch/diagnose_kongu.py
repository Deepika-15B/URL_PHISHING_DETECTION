import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import json

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app import create_app
from utils.unified_feature_pipeline import UnifiedFeaturePipeline
from backend.routes.predict import build_feature_key_mapping
import tensorflow as tf

def main():
    app = create_app()
    with app.app_context():
        url = "kongu.ac.in"
        
        models_dir = _PROJECT_ROOT / "models"
        with open(models_dir / "top20_features.pkl", "rb") as f:
            top20_features = pickle.load(f)
        with open(models_dir / "scaler_phase2_v2.pkl", "rb") as f:
            scaler = pickle.load(f)
        model = tf.keras.models.load_model(models_dir / "fnn_phase2_v2.keras")
        
        print("Extracting features (this may take a few seconds)...")
        with UnifiedFeaturePipeline() as pipeline:
            result = pipeline.extract(url)
        
        url_feats = result.url_features
        html_feats = result.html_features
        html_diagnostics = result.html_diagnostics
        
        combined_feats = {**url_feats, **html_feats}
        
        feat_dict = build_feature_key_mapping(
            top20_features,
            combined_feats,
            url=url,
            html_diagnostics=html_diagnostics,
        )
        
        input_df = pd.DataFrame([feat_dict], columns=top20_features).astype(np.float32)
        X_scaled = scaler.transform(input_df)
        
        raw_prob = float(model.predict(X_scaled, verbose=0)[0][0])
        prediction = "Phishing" if raw_prob >= 0.5 else "Legitimate"
        
        out_lines = []
        out_lines.append(f"Diagnostic Report for {url}")
        out_lines.append("=" * 50)
        out_lines.append(f"Raw FNN output probability: {raw_prob:.6f}")
        out_lines.append(f"Final prediction: {prediction}")
        out_lines.append("-" * 50)
        out_lines.append("Feature Analysis:")
        
        means = scaler.mean_
        stds = scaler.scale_
        
        alias_map_path = _PROJECT_ROOT / "backend" / "feature_alias_map.json"
        alias_map = {}
        if alias_map_path.exists():
            with open(alias_map_path, "r") as f:
                alias_map = json.load(f)
        
        for i, feat in enumerate(top20_features):
            raw_val = input_df.iloc[0, i]
            scaled_val = X_scaled[0, i]
            mean_val = means[i]
            std_val = stds[i]
            
            z_score = (raw_val - mean_val) / (std_val if std_val != 0 else 1e-9)
            
            outlier_flag = "*** EXTREME OUTLIER ***" if abs(z_score) > 3.0 else ""
            
            out_lines.append(f"Feature: {feat}")
            out_lines.append(f"  Raw Value   : {raw_val}")
            out_lines.append(f"  Scaled Value: {scaled_val:.4f}")
            out_lines.append(f"  Scaler Mean : {mean_val:.4f}")
            out_lines.append(f"  Scaler Std  : {std_val:.4f}")
            out_lines.append(f"  Z-Score     : {z_score:.2f} {outlier_flag}")
            if feat in alias_map:
                map_info = alias_map[feat]
                out_lines.append(f"  Alias Logic : Type='{map_info.get('type')}', Keys='{map_info.get('keys', '')}', Fallback={map_info.get('fallback')}")
            out_lines.append("")
        
        out_path = _PROJECT_ROOT / "reports" / "kongu_feature_debug.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(out_lines), encoding="utf-8")
        print(f"Report saved to {out_path}")

if __name__ == "__main__":
    main()

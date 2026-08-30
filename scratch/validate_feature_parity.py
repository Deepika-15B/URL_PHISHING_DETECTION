"""
scratch/validate_feature_parity.py
-----------------------------------
Validates the continuous URLSimilarityIndex and DomainTitleMatchScore
computation against the previous binary approximation.

Outputs: reports/feature_parity_validation.txt
"""
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pickle
import numpy as np


def main():
    url = "https://kongu.ac.in"
    print(f"[Parity Validation] URL: {url}")

    # 1. Import only after sys.path is set
    from backend.app import create_app
    app = create_app()

    with app.app_context():
        from backend.routes.predict import build_feature_key_mapping
        from utils.unified_feature_pipeline import UnifiedFeaturePipeline

        print("[Parity Validation] Launching browser pipeline...")
        with UnifiedFeaturePipeline() as pipeline:
            result = pipeline.extract(url)

        url_feats      = result.url_features
        html_feats     = result.html_features
        html_diag      = result.html_diagnostics

        combined_feats = {**url_feats, **html_feats}

        # --- Load model artefacts ---
        models_dir = _PROJECT_ROOT / "models"
        with open(models_dir / "top20_features.pkl", "rb") as f:
            top20_features = pickle.load(f)
        with open(models_dir / "scaler_phase2_v2.pkl", "rb") as f:
            scaler = pickle.load(f)

        # --- Resolve features ---
        feat_dict = build_feature_key_mapping(
            top20_features,
            combined_feats,
            url=url,
            html_diagnostics=html_diag,
        )

        import pandas as pd
        input_df  = pd.DataFrame([feat_dict], columns=top20_features).astype("float32")
        X_scaled  = scaler.transform(input_df)

        import tensorflow as tf
        model    = tf.keras.models.load_model(models_dir / "fnn_phase2_v2.keras")
        raw_prob = float(model.predict(X_scaled, verbose=0)[0][0])
        phishing_prob = 1.0 - raw_prob
        prediction    = "Legitimate" if raw_prob >= 0.5 else "Phishing"

        # --- Collect values for report ---
        brand_name   = html_diag.get("extracted_brand_name", "(not extracted)")
        page_title   = html_diag.get("page_title",           "(not extracted)")

        new_url_sim      = feat_dict.get("URLSimilarityIndex",    0.0)
        new_domain_sim   = feat_dict.get("DomainTitleMatchScore", 0.0)
        old_binary_value = html_feats.get("title_matches_domain", 0) * 100

        report_lines = [
            "FEATURE PARITY VALIDATION REPORT",
            "=" * 60,
            f"Target URL                  : {url}",
            "",
            "--- Domain Extraction ---",
            f"Registered Domain (tldextract): {brand_name}",
            f"Page Title                    : {page_title}",
            "",
            "--- URLSimilarityIndex ---",
            f"  PREVIOUS (binary * 100)     : {old_binary_value}",
            f"  NEW (continuous fuzzy 0-100): {new_url_sim}",
            f"  Change                      : {new_url_sim - old_binary_value:+.2f}",
            "",
            "--- DomainTitleMatchScore ---",
            f"  PREVIOUS (binary * 100)     : {old_binary_value}",
            f"  NEW (continuous fuzzy 0-100): {new_domain_sim}",
            f"  Change                      : {new_domain_sim - old_binary_value:+.2f}",
            "",
            "--- Scaled Feature Values (all 20) ---",
        ]
        for i, feat_name in enumerate(top20_features):
            raw_val    = feat_dict.get(feat_name, 0.0)
            scaled_val = X_scaled[0][i]
            mean       = scaler.mean_[i]
            std        = scaler.scale_[i]
            z          = (raw_val - mean) / std if std > 0 else 0.0
            outlier    = " *** OUTLIER ***" if abs(z) > 2.5 else ""
            report_lines.append(
                f"  {feat_name:<35} raw={raw_val:>10.4f}  scaled={scaled_val:>8.4f}  z={z:>+7.2f}{outlier}"
            )

        report_lines += [
            "",
            "--- Final Prediction ---",
            f"  Raw FNN sigmoid output  : {raw_prob:.8f}",
            f"  Phishing probability    : {phishing_prob:.4f}",
            f"  Prediction              : {prediction}",
        ]

        report_text = "\n".join(report_lines) + "\n"

        report_path = _PROJECT_ROOT / "reports" / "feature_parity_validation.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")

        print(report_text)
        print(f"[Parity Validation] Report saved to {report_path}")


if __name__ == "__main__":
    main()

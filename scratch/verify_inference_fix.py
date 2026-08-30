"""
Verification script for the inference pipeline fix.
Exercises both Bug 1 (feature resolution) and Bug 2 (screenshots),
then writes reports/inference_fix_report.txt.

Usage:
    python -X utf8 scratch/verify_inference_fix.py
"""
from __future__ import annotations

import io
import sys
import pickle
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 issues
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.routes.predict import (
    build_feature_key_mapping,
    capture_page_screenshot,
    _normalize_key,
    _load_alias_map,
    _SCREENSHOTS_DIR,
    _ALIAS_MAP_PATH,
)
from utils.unified_feature_pipeline import UnifiedFeaturePipeline

_MODELS_DIR = _PROJECT_ROOT / "models"
_REPORTS_DIR = _PROJECT_ROOT / "reports"
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_URLS = [
    "https://www.google.com",
    "https://github.com",
    "https://example.com",
]

report_lines: list[str] = []


def w(text: str = "") -> None:
    print(text)
    report_lines.append(text)


def run() -> None:
    w("=" * 80)
    w("IEEE PHISHING DETECTION - INFERENCE FIX VERIFICATION REPORT")
    w("=" * 80)
    w(f"Timestamp      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"Models dir     : {_MODELS_DIR}")
    w(f"Screenshots dir: {_SCREENSHOTS_DIR}")
    w(f"Alias map      : {_ALIAS_MAP_PATH}")
    w()

    # ── Artifact loading ──────────────────────────────────────────────────
    w("=" * 60)
    w("ARTIFACT LOADING")
    w("=" * 60)
    try:
        with open(_MODELS_DIR / "top20_features.pkl", "rb") as f:
            top20_features: list[str] = pickle.load(f)
        with open(_MODELS_DIR / "scaler_phase2_v2.pkl", "rb") as f:
            scaler = pickle.load(f)
        model = tf.keras.models.load_model(_MODELS_DIR / "fnn_phase2_v2.keras")
        alias_map = _load_alias_map()
        w(f"  fnn_phase2_v2.keras  : LOADED  (input_shape={model.input_shape})")
        w(f"  scaler_phase2_v2.pkl : LOADED  (type={type(scaler).__name__}, n_features={scaler.n_features_in_})")
        w(f"  top20_features.pkl   : LOADED  ({len(top20_features)} features)")
        w(f"  feature_alias_map    : LOADED  ({len(alias_map)} entries)")
        w()
        w("  Top20 feature names (from top20_features.pkl):")
        for feat in top20_features:
            w(f"    {feat}")
        w()
        w("  Alias map entries (from feature_alias_map.json):")
        for k, v in alias_map.items():
            w(f"    {k}: {v}")
    except Exception as exc:
        w(f"  ERROR loading artifacts: {exc}")
        w(traceback.format_exc())
        return

    w()

    # ── Per-URL tests ─────────────────────────────────────────────────────
    for test_url in TEST_URLS:
        w("=" * 80)
        w(f"TEST URL: {test_url}")
        w("=" * 80)

        try:
            with UnifiedFeaturePipeline(timeout_ms=15000) as pipeline:
                result = pipeline.extract(test_url)

                # ── Bug 2: Screenshot ─────────────────────────────────────
                w()
                w("-" * 60)
                w("BUG 2 FIX: SCREENSHOT CAPTURE")
                w("-" * 60)
                screenshot_path = capture_page_screenshot(pipeline, test_url)

                screenshot_saved = False
                screenshot_size_kb = 0.0
                if screenshot_path:
                    abs_path = _PROJECT_ROOT / screenshot_path.lstrip("/")
                    screenshot_saved = abs_path.exists()
                    if screenshot_saved:
                        screenshot_size_kb = abs_path.stat().st_size / 1024
                    w(f"  Screenshot path   : {screenshot_path}")
                    w(f"  Screenshot saved  : {screenshot_saved}")
                    w(f"  File size         : {screenshot_size_kb:.1f} KB")
                else:
                    w("  Screenshot path   : None")
                    w("  Screenshot saved  : False")

            url_feats = result.url_features
            html_feats = result.html_features
            html_diag  = result.html_diagnostics
            combined   = {**url_feats, **html_feats}

        except Exception as exc:
            w(f"  ERROR during extraction: {exc}")
            w(traceback.format_exc())
            continue

        # ── All extracted keys ────────────────────────────────────────────
        w()
        w("-" * 60)
        w(f"EXTRACTED URL FEATURE KEYS ({len(url_feats)})")
        w("-" * 60)
        for k in sorted(url_feats.keys()):
            w(f"  {k}: {url_feats[k]}")

        w()
        w("-" * 60)
        w(f"EXTRACTED HTML FEATURE KEYS ({len(html_feats)})")
        w("-" * 60)
        for k in sorted(html_feats.keys()):
            w(f"  {k}: {html_feats[k]}")

        # ── Normalization table ───────────────────────────────────────────
        w()
        w("-" * 60)
        w("NORMALIZATION CANONICAL FORMS (top20 features)")
        w("-" * 60)
        for feat in top20_features:
            w(f"  '{feat}' -> '{_normalize_key(feat)}'")

        # ── Bug 1: Feature mapping ────────────────────────────────────────
        w()
        w("-" * 60)
        w("BUG 1 FIX: FEATURE MAPPING (3 passes)")
        w("-" * 60)

        mapping_error: str | None = None
        feat_dict: dict = {}
        try:
            feat_dict = build_feature_key_mapping(
                top20_features,
                combined,
                url=test_url,
                html_diagnostics=html_diag,
            )
            matched_count = len(feat_dict)
            missing_count = len(top20_features) - matched_count
            w(f"  Matched Features : {matched_count}/{len(top20_features)}")
            w(f"  Missing Features : {missing_count}/{len(top20_features)}")
        except ValueError as err:
            mapping_error = str(err)
            w(f"  MAPPING FAILED: {mapping_error}")
            w()
            continue

        # ── Feature values before scaling ─────────────────────────────────
        w()
        w("-" * 60)
        w("FEATURE VALUES BEFORE SCALING")
        w("-" * 60)
        input_df = pd.DataFrame([feat_dict], columns=top20_features).astype(np.float32)
        w(f"  DataFrame shape : {input_df.shape}")
        w(f"  Any NaN         : {input_df.isna().any().any()}")
        w(f"  All zeros       : {(input_df == 0).all().all()}")
        w()
        for col in top20_features:
            val = feat_dict[col]
            w(f"  {col:<35s} = {val}")

        # ── Feature values after scaling ──────────────────────────────────
        w()
        w("-" * 60)
        w("FEATURE VALUES AFTER SCALING")
        w("-" * 60)
        X_scaled = scaler.transform(input_df.values)
        w(f"  Scaled shape : {X_scaled.shape}")
        for i, col in enumerate(top20_features):
            w(f"  {col:<35s} = {X_scaled[0][i]:.6f}")

        # ── Raw model output ──────────────────────────────────────────────
        w()
        w("-" * 60)
        w("RAW MODEL OUTPUT")
        w("-" * 60)
        raw_prob = float(model.predict(X_scaled, verbose=0)[0][0])
        phishing_prob = 1.0 - raw_prob
        prediction_label = "Legitimate" if raw_prob >= 0.5 else "Phishing"
        confidence = round(raw_prob * 100, 2) if raw_prob >= 0.5 else round(phishing_prob * 100, 2)
        threat_score = round(phishing_prob * 100, 1)
        risk_level = "High" if phishing_prob >= 0.8 else "Medium" if phishing_prob >= 0.4 else "Low"

        w(f"  Raw probability (sigmoid) : {raw_prob:.6f}")
        w(f"  Phishing probability      : {phishing_prob:.6f}")
        w(f"  Threshold applied         : raw_prob >= 0.5 -> Legitimate, else -> Phishing")

        w()
        w("-" * 60)
        w("FINAL PREDICTION")
        w("-" * 60)
        w(f"  Prediction   : {prediction_label}")
        w(f"  Confidence   : {confidence}%")
        w(f"  Threat Score : {threat_score}")
        w(f"  Risk Level   : {risk_level}")

        # ── Screenshot summary ────────────────────────────────────────────
        w()
        w("-" * 60)
        w("SCREENSHOT SUMMARY")
        w("-" * 60)
        w(f"  Screenshot path    : {screenshot_path}")
        w(f"  Screenshot saved   : {screenshot_saved}")
        if screenshot_saved:
            w(f"  File size          : {screenshot_size_kb:.1f} KB")

        w()

    # ── Write report ──────────────────────────────────────────────────────
    report_path = _REPORTS_DIR / "inference_fix_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print()
    print("=" * 60)
    print(f"Report written to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    run()

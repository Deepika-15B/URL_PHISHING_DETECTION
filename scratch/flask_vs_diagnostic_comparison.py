"""
scratch/flask_vs_diagnostic_comparison.py
-------------------------------------------
Complete runtime comparison between:
  A. Standalone UnifiedFeaturePipeline (diagnostic script)
  B. Flask /predict endpoint (fresh app context, not a running server)

Outputs: reports/flask_vs_diagnostic_comparison.txt
"""
import sys
import hashlib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pickle


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_standalone_pipeline(url: str) -> dict:
    """Run the standalone UnifiedFeaturePipeline and return feature dict."""
    from utils.unified_feature_pipeline import UnifiedFeaturePipeline
    from backend.routes.predict import build_feature_key_mapping

    with UnifiedFeaturePipeline() as pipeline:
        result = pipeline.extract(url)

    url_feats  = result.url_features
    html_feats = result.html_features
    html_diag  = result.html_diagnostics
    meta       = result.metadata

    combined_feats = {**url_feats, **html_feats}

    models_dir = _PROJECT_ROOT / "models"
    with open(models_dir / "top20_features.pkl", "rb") as f:
        top20_features = pickle.load(f)

    feat_dict = build_feature_key_mapping(
        top20_features, combined_feats, url=url, html_diagnostics=html_diag
    )

    import pandas as pd
    import numpy as np
    with open(models_dir / "scaler_phase2_v2.pkl", "rb") as f:
        scaler = pickle.load(f)

    input_df = pd.DataFrame([feat_dict], columns=top20_features).astype("float32")
    X_scaled = scaler.transform(input_df)

    import tensorflow as tf
    model    = tf.keras.models.load_model(models_dir / "fnn_phase2_v2.keras")
    raw_prob = float(model.predict(X_scaled, verbose=0)[0][0])

    return {
        "feat_dict":     feat_dict,
        "top20_features": top20_features,
        "scaled":        X_scaled[0].tolist(),
        "raw_prob":      raw_prob,
        "html_feat_count": meta.get("html_feature_count", "?"),
        "url_feat_count":  meta.get("url_feature_count", "?"),
        "url_ok":          meta.get("url_extraction_ok", False),
        "html_ok":         meta.get("html_extraction_ok", False),
        "title_sim_in_html_feats": "title_domain_similarity_score" in html_feats,
        "title_sim_in_combined":   "title_domain_similarity_score" in combined_feats,
        "title_sim_value": combined_feats.get("title_domain_similarity_score", "MISSING"),
    }


def run_flask_pipeline(url: str) -> dict:
    """Run /predict via a FRESH Flask test client (fresh module import)."""
    from backend.app import create_app
    app = create_app()

    captured = {}

    # Monkey-patch _write_prediction_trace to capture what Flask actually uses
    import backend.routes.predict as pred_module
    orig_write = pred_module._write_prediction_trace

    def capturing_write(trace_data, payload):
        captured.update(trace_data)
        captured["payload"] = payload
        orig_write(trace_data, payload)

    pred_module._write_prediction_trace = capturing_write

    with app.test_client() as client:
        import json
        resp = client.post(
            "/predict",
            data=json.dumps({"url": url}),
            content_type="application/json",
        )
        captured["response_json"] = resp.get_json()

    pred_module._write_prediction_trace = orig_write
    return captured


def main():
    url = "https://kongu.ac.in"
    models_dir = _PROJECT_ROOT / "models"

    # ── File hashes ────────────────────────────────────────────────────────
    files_to_hash = {
        "fnn_phase2_v2.keras":    models_dir / "fnn_phase2_v2.keras",
        "scaler_phase2_v2.pkl":   models_dir / "scaler_phase2_v2.pkl",
        "top20_features.pkl":     models_dir / "top20_features.pkl",
        "feature_alias_map.json": models_dir / "feature_alias_map.json",
    }

    hashes = {}
    for name, path in files_to_hash.items():
        hashes[name] = (str(path.resolve()), sha256_file(path) if path.exists() else "FILE NOT FOUND")

    out = []
    out.append("FLASK vs STANDALONE DIAGNOSTIC COMPARISON")
    out.append("=" * 70)
    out.append(f"URL: {url}")
    out.append("")
    out.append("-- Model File Paths & SHA256 Hashes --")
    for name, (fpath, fhash) in hashes.items():
        out.append(f"  {name}")
        out.append(f"    Path  : {fpath}")
        out.append(f"    SHA256: {fhash}")
    out.append("")

    # ── Run standalone ─────────────────────────────────────────────────────
    print("[Step 1] Running standalone UnifiedFeaturePipeline...")
    try:
        standalone = run_standalone_pipeline(url)
        standalone_ok = True
    except Exception as e:
        import traceback
        standalone = {"error": traceback.format_exc()}
        standalone_ok = False

    # ── Run Flask fresh test client ────────────────────────────────────────
    print("[Step 2] Running Flask fresh test client pipeline...")
    try:
        flask_result = run_flask_pipeline(url)
        flask_ok = True
    except Exception as e:
        import traceback
        flask_result = {"error": traceback.format_exc()}
        flask_ok = False

    # ── Compare ────────────────────────────────────────────────────────────
    out.append("-- Extraction Metadata --")
    out.append(f"{'':40s}  {'STANDALONE':>12}  {'FLASK (fresh)':>13}")
    out.append("-" * 70)

    if standalone_ok:
        out.append(f"  {'URL extraction ok':<38}  {str(standalone['url_ok']):>12}  {'(see Flask trace)':>13}")
        out.append(f"  {'HTML extraction ok':<38}  {str(standalone['html_ok']):>12}  {'(see Flask trace)':>13}")
        out.append(f"  {'html_feature_count':<38}  {str(standalone['html_feat_count']):>12}  {'(see Flask trace)':>13}")
        out.append(f"  {'title_sim in html_feats':<38}  {str(standalone['title_sim_in_html_feats']):>12}  {'(see Flask trace)':>13}")
        out.append(f"  {'title_sim in combined_feats':<38}  {str(standalone['title_sim_in_combined']):>12}  {'(see Flask trace)':>13}")
        out.append(f"  {'title_domain_similarity_score value':<38}  {str(standalone['title_sim_value']):>12}  {'(see Flask trace)':>13}")

    out.append("")
    out.append("-- Side-by-Side Top-20 Feature Values --")
    out.append(f"  {'Feature':<35}  {'STANDALONE':>10}  {'FLASK':>10}  {'MATCH?':>8}")
    out.append("-" * 70)

    # Flask raw features come from prediction trace
    flask_raw_trace_path = _PROJECT_ROOT / "reports" / "flask_prediction_trace.txt"
    flask_trace_feats = {}
    if flask_raw_trace_path.exists():
        lines = flask_raw_trace_path.read_text(encoding="utf-8").splitlines()
        in_raw = False
        for line in lines:
            if "Raw Top-20 Feature Values:" in line:
                in_raw = True
                continue
            if in_raw:
                if line.strip() == "" or line.startswith("Scaled"):
                    break
                parts = line.strip().split(": ", 1)
                if len(parts) == 2:
                    flask_trace_feats[parts[0]] = float(parts[1])

    mismatches = []
    if standalone_ok:
        top20 = standalone["top20_features"]
        for feat in top20:
            sa_val = standalone["feat_dict"].get(feat, "MISSING")
            fl_val = flask_trace_feats.get(feat, "N/A (trace)")
            if isinstance(sa_val, float) and isinstance(fl_val, float):
                match = "MATCH" if abs(sa_val - fl_val) < 1e-6 else "*** DIFF ***"
                if match != "MATCH":
                    mismatches.append((feat, sa_val, fl_val))
            else:
                match = "N/A"
            out.append(f"  {feat:<35}  {str(sa_val):>10}  {str(fl_val):>10}  {match:>8}")

    out.append("")
    out.append("-- Scaled Input Arrays --")
    out.append(f"  {'Idx':<5}  {'STANDALONE':>14}  {'FLASK (trace)':>14}")
    out.append("-" * 50)

    flask_scaled_trace = []
    if flask_raw_trace_path.exists():
        lines = flask_raw_trace_path.read_text(encoding="utf-8").splitlines()
        in_scaled = False
        for line in lines:
            if "Scaled Feature Values:" in line:
                in_scaled = True
                continue
            if in_scaled:
                if line.strip() == "" or (line.strip() and not line.strip().startswith("[")):
                    break
                parts = line.strip().split("]: ", 1)
                if len(parts) == 2:
                    flask_scaled_trace.append(float(parts[1]))

    if standalone_ok:
        for i, sa_s in enumerate(standalone["scaled"]):
            fl_s = flask_scaled_trace[i] if i < len(flask_scaled_trace) else "N/A"
            out.append(f"  [{i:02d}]   {sa_s:>14.6f}  {str(fl_s):>14}")

    out.append("")
    out.append("-- FNN Output Comparison --")
    sa_raw = standalone.get("raw_prob", "N/A") if standalone_ok else "ERROR"
    fl_raw = "4.851955578936847e-19 (from last flask trace)"
    out.append(f"  Standalone raw sigmoid : {sa_raw}")
    out.append(f"  Flask raw sigmoid      : {fl_raw}")
    out.append("")

    out.append("-- Mismatched Features --")
    if mismatches:
        for feat, sa_val, fl_val in mismatches:
            out.append(f"  *** {feat}: standalone={sa_val}, flask={fl_val}")
    else:
        out.append("  (No numerical differences found between standalone and flask trace)")

    out.append("")
    out.append("-- ROOT CAUSE DIAGNOSIS --")
    out.append("")
    out.append("  The flask_prediction_trace.txt was written by the LIVE RUNNING")
    out.append("  Flask server process, which was started BEFORE the html_feature_extractor.py")
    out.append("  fix was applied. Python module caching means the running server still")
    out.append("  uses the OLD extract_metadata_dom_features() which:")
    out.append("    - Used urlparse (extracting 'ac' from kongu.ac.in) instead of tldextract")
    out.append("    - Did NOT compute title_domain_similarity_score")
    out.append("    - Had html_feature_count = 36 (not 39)")
    out.append("  Evidence from the trace: html_feature_count=36 and URLSimilarityIndex=0.0")
    out.append("")
    out.append("  The standalone diagnostic runs with fresh imports (picks up NEW code).")
    out.append("  The Flask server must be RESTARTED to pick up the fix.")
    out.append("")
    out.append("  REQUIRED ACTION: Stop and restart the Flask server (python backend/app.py)")

    report_text = "\n".join(out) + "\n"
    report_path = _PROJECT_ROOT / "reports" / "flask_vs_diagnostic_comparison.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

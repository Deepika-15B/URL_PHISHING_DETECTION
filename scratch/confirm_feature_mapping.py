"""
scratch/confirm_feature_mapping.py
------------------------------------
Confirms that title_domain_similarity_score is present in the
combined feature dict passed to build_feature_key_mapping, and
that it is correctly mapped to both URLSimilarityIndex and
DomainTitleMatchScore.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pickle


def main():
    url = "https://kongu.ac.in"

    from backend.app import create_app
    app = create_app()

    with app.app_context():
        from backend.routes.predict import build_feature_key_mapping, _load_alias_map
        from utils.unified_feature_pipeline import UnifiedFeaturePipeline

        print(f"\n{'='*60}")
        print(f"URL: {url}")
        print(f"{'='*60}\n")

        with UnifiedFeaturePipeline() as pipeline:
            result = pipeline.extract(url)

        url_feats  = result.url_features
        html_feats = result.html_features
        html_diag  = result.html_diagnostics

        combined_feats = {**url_feats, **html_feats}

        # ── 1. Print combined_feats columns ──────────────────────────────
        print(f"combined_feats total keys: {len(combined_feats)}")
        print("\ncombined_feats.keys() (sorted):")
        for k in sorted(combined_feats.keys()):
            print(f"  {k}")

        # ── 2. Specifically confirm the key of interest ───────────────────
        KEY = "title_domain_similarity_score"
        if KEY in combined_feats:
            print(f"\n[PASS] '{KEY}' IS present in combined_feats -- value: {combined_feats[KEY]}")
        else:
            print(f"\n[FAIL] '{KEY}' is NOT present in combined_feats")

        # -- 3. Also confirm it appears in html_feats (numeric bucket) -----
        if KEY in html_feats:
            print(f"[PASS] '{KEY}' IS present in html_feats -- value: {html_feats[KEY]}")
        else:
            print(f"[FAIL] '{KEY}' is NOT present in html_feats")

        # ── 4. Confirm alias map entries for both target features ─────────
        alias_map = _load_alias_map()
        print("\nAlias map entries for URLSimilarityIndex and DomainTitleMatchScore:")
        for feat in ("URLSimilarityIndex", "DomainTitleMatchScore"):
            entry = alias_map.get(feat)
            print(f"  {feat}: {entry}")

        # ── 5. Run the mapping and confirm the resolved values ────────────
        models_dir = _PROJECT_ROOT / "models"
        with open(models_dir / "top20_features.pkl", "rb") as f:
            top20_features = pickle.load(f)

        feat_dict = build_feature_key_mapping(
            top20_features,
            combined_feats,
            url=url,
            html_diagnostics=html_diag,
        )

        print(f"\nFinal resolved values after build_feature_key_mapping:")
        print(f"  URLSimilarityIndex    = {feat_dict.get('URLSimilarityIndex')}")
        print(f"  DomainTitleMatchScore = {feat_dict.get('DomainTitleMatchScore')}")

        # ── 6. Build the DataFrame passed to the scaler and print columns ─
        import pandas as pd
        input_df = pd.DataFrame([feat_dict], columns=top20_features).astype("float32")
        print(f"\nDataFrame passed to scaler — shape: {input_df.shape}")
        print(f"DataFrame columns (in order):")
        for i, col in enumerate(input_df.columns):
            print(f"  [{i:02d}] {col} = {input_df.iloc[0, i]:.4f}")

        print(f"\n{'='*60}")
        print("Confirmation complete.")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

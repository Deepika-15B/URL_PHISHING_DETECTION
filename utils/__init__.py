"""
utils/__init__.py
=================
Public interface for the phishing-detection utility package.

Sub-modules
-----------
  url_feature_extractor  — URL lexical/network feature extraction (101-col IEEE schema)
  html_feature_extractor — Rendered-HTML content feature extraction (35 numeric signals)
  unified_feature_pipeline — Submodule 3.5: Unified HTML & Hybrid Feature Extraction Pipeline
  preprocessing          — Data cleaning & sklearn-compatible transformation pipeline
  feature_selection      — Random-Forest permutation importance feature ranking
  inference              — FNN prediction pipeline (uses original trained feature subset only)

Architecture boundaries
-----------------------
Current Deployment Model
    url_feature_extractor → inference.py → FNN / DNN / Wide & Deep / TabNet → Prediction
    (original trained feature subset — unchanged)

Future Hybrid Research Dataset
    url_feature_extractor ┐
                          ├─→ unified_feature_pipeline → Unified Feature Record
    html_feature_extractor┘                           → Hybrid Feature Dataset
    (for analysis and future hybrid model training — NOT used for prediction yet)
"""

# ── Submodule 3.5 public API ──────────────────────────────────────────────────
from utils.unified_feature_pipeline import (  # noqa: F401
    HybridFeatureSchema,
    HybridResult,
    UnifiedFeaturePipeline,
    build_feature_dictionary,
    export_batch_dataframe,
    export_dataframe,
    export_json,
    generate_unified_report,
    load_hybrid_schema,
    scale_url_features,
)

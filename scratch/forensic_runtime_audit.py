"""
scratch/forensic_runtime_audit.py
=================================
Forensic runtime audit script covering Steps 1 through 8.
Monkey-patches StandardScaler.transform and model.predict to trace execution,
verify module filepaths, object IDs, call stacks, and DataFrame values.
"""
import sys
import inspect
import json
import traceback
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle


def run_forensic_audit():
    print("=" * 80)
    print("FORENSIC RUNTIME AUDIT")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # STEP 1: Every place where build_feature_key_mapping() is called
    # -----------------------------------------------------------------------
    print("\n--- STEP 1: Calls to build_feature_key_mapping() ---")
    py_files = list(_PROJECT_ROOT.rglob("*.py"))
    step1_calls = []
    for fpath in py_files:
        if ".venv" in fpath.parts or "__pycache__" in fpath.parts:
            continue
        content = fpath.read_text(encoding="utf-8", errors="ignore")
        for idx, line in enumerate(content.splitlines(), 1):
            if "build_feature_key_mapping(" in line:
                rel = fpath.relative_to(_PROJECT_ROOT)
                step1_calls.append((str(rel), idx, line.strip()))
                print(f"  File: {rel}:{idx}\n    Line: {line.strip()}")

    # -----------------------------------------------------------------------
    # STEP 2: Every place where URLSimilarityIndex or DomainTitleMatchScore is assigned
    # -----------------------------------------------------------------------
    print("\n--- STEP 2: Assignments to URLSimilarityIndex or DomainTitleMatchScore ---")
    step2_assigns = []
    for fpath in py_files:
        if ".venv" in fpath.parts or "__pycache__" in fpath.parts:
            continue
        content = fpath.read_text(encoding="utf-8", errors="ignore")
        for idx, line in enumerate(content.splitlines(), 1):
            if ("URLSimilarityIndex" in line or "DomainTitleMatchScore" in line) and ("=" in line or ":" in line):
                rel = fpath.relative_to(_PROJECT_ROOT)
                step2_assigns.append((str(rel), idx, line.strip()))
                print(f"  File: {rel}:{idx}\n    Line: {line.strip()}")

    # -----------------------------------------------------------------------
    # STEP 3: Every place where the dataframe for inference is created
    # -----------------------------------------------------------------------
    print("\n--- STEP 3: Inference DataFrame creation locations ---")
    for fpath in py_files:
        if ".venv" in fpath.parts or "__pycache__" in fpath.parts:
            continue
        content = fpath.read_text(encoding="utf-8", errors="ignore")
        for idx, line in enumerate(content.splitlines(), 1):
            if "pd.DataFrame(" in line and ("feat_dict" in line or "top20" in line or "columns=" in line):
                rel = fpath.relative_to(_PROJECT_ROOT)
                print(f"  File: {rel}:{idx}\n    Line: {line.strip()}")

    # -----------------------------------------------------------------------
    # STEP 7 & 8: Import Verification and Object IDs
    # -----------------------------------------------------------------------
    print("\n--- STEP 7: Module __file__ locations ---")
    import utils.html_feature_extractor as hfe
    import utils.unified_feature_pipeline as ufp
    import backend.routes.predict as pr
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf

    print(f"  html_feature_extractor  : {getattr(hfe, '__file__', 'None')}")
    print(f"  unified_feature_pipeline: {getattr(ufp, '__file__', 'None')}")
    print(f"  backend.routes.predict  : {getattr(pr, '__file__', 'None')}")

    print("\n--- STEP 8: Object IDs ---")
    print(f"  id(build_feature_key_mapping): {id(pr.build_feature_key_mapping)}")
    print(f"  id(StandardScaler)           : {id(StandardScaler)}")

    models_dir = _PROJECT_ROOT / "models"
    model = tf.keras.models.load_model(models_dir / "fnn_phase2_v2.keras")
    print(f"  id(model)                    : {id(model)}")

    # -----------------------------------------------------------------------
    # STEP 4 & 5: Monkey patch StandardScaler.transform and model.predict
    # -----------------------------------------------------------------------
    print("\n--- STEP 4 & 5: Setting up Monkey Patches & Call Stack Tracing ---")

    orig_transform = StandardScaler.transform

    def patched_transform(self, X, **kwargs):
        print("\n" + "=" * 60)
        print("MONKEY PATCH: StandardScaler.transform() ENTERED")
        print("=" * 60)
        if isinstance(X, pd.DataFrame):
            print(f"  Input DataFrame Shape  : {X.shape}")
            print(f"  Input DataFrame Columns: {list(X.columns)}")
            print("  Raw Values before transform:")
            for col in X.columns:
                print(f"    {col:<30} = {X[col].iloc[0]}")
        else:
            print(f"  Input Array Shape: {getattr(X, 'shape', 'unknown')}")

        print("\n  CALL STACK AT SCALER TRANSFORM:")
        for frame_info in inspect.stack()[1:7]:
            print(f"    -> {frame_info.filename}:{frame_info.lineno} in {frame_info.function}()")

        res = orig_transform(self, X, **kwargs)
        print("\n  Scaled Array Output (res):")
        print(f"    Shape: {res.shape}")
        print(f"    Values: {res[0].tolist()}")
        print("=" * 60)
        return res

    StandardScaler.transform = patched_transform

    orig_predict = tf.keras.Model.predict

    def patched_predict(self, x, **kwargs):
        print("\n" + "=" * 60)
        print("MONKEY PATCH: model.predict() ENTERED")
        print("=" * 60)
        print(f"  Input ndarray type : {type(x)}")
        print(f"  Input ndarray shape: {getattr(x, 'shape', None)}")
        print(f"  Input ndarray dtype: {getattr(x, 'dtype', None)}")
        print(f"  Input values       : {x}")

        print("\n  CALL STACK AT MODEL PREDICT:")
        for frame_info in inspect.stack()[1:7]:
            print(f"    -> {frame_info.filename}:{frame_info.lineno} in {frame_info.function}()")

        res = orig_predict(self, x, **kwargs)
        print(f"\n  Raw Sigmoid Output : {res[0][0]:.18e} (float: {float(res[0][0])})")
        print("=" * 60)
        return res

    tf.keras.Model.predict = patched_predict

    # -----------------------------------------------------------------------
    # STEP 6: Execute via Flask test_client to trace full request stack
    # -----------------------------------------------------------------------
    print("\n--- STEP 6: Executing Flask Request Trace for https://kongu.ac.in ---")
    from backend.app import create_app
    app = create_app()

    with app.test_client() as client:
        url = "https://kongu.ac.in"
        resp = client.post("/predict", json={"url": url})
        print(f"\nFlask Response Status Code: {resp.status_code}")
        print("Flask Response Payload:")
        print(json.dumps(resp.get_json(), indent=2))


if __name__ == "__main__":
    run_forensic_audit()

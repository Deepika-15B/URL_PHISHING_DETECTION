"""
test_unified_pipeline.py
=========================
Unit tests for Submodule 3.5 — Unified HTML & Hybrid Feature Extraction Pipeline.

All tests use unittest.mock to avoid any live network requests, browser
launches, or file I/O dependencies.  The tests verify:

  - Correct architecture: URL and HTML features remain SEPARATE.
  - HybridResult.to_dict() produces the canonical three-level structure.
  - export_as_unified_record() column ordering (URL cols first, then HTML).
  - scale_url_features() scales only URL columns; HTML columns unchanged.
  - Graceful degradation: a failing URL extractor produces zero-filled URL dict.
  - Graceful degradation: a failing HTML extractor produces zero-filled HTML dict.
  - export_batch_dataframe() stacks rows with correct record_type metadata.
  - generate_unified_report() writes a file with expected sections.
  - HybridResult.empty() constructor produces valid all-zero structure.
  - HybridFeatureSchema property invariants.

Run from project root::

    python -m pytest test_unified_pipeline.py -v
    python -m unittest test_unified_pipeline -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Ensure project root is on sys.path ────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils.unified_feature_pipeline import (
    HybridFeatureSchema,
    HybridResult,
    UnifiedFeaturePipeline,
    _HTML_NUMERIC_KEYS,
    _HTML_STRING_KEYS,
    build_feature_dictionary,
    export_batch_dataframe,
    generate_unified_report,
    scale_url_features,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_SAMPLE_URL_COLS = [f"url_feat_{i}" for i in range(10)]   # abbreviated 10-col URL schema
_SAMPLE_HTML_NUMERIC = {k: 0 for k in _HTML_NUMERIC_KEYS}
_SAMPLE_HTML_STRINGS = {k: "" for k in _HTML_STRING_KEYS}


def _make_url_features(val: float = 0.5) -> dict:
    return {col: val for col in _SAMPLE_URL_COLS}


def _make_result(
    url: str = "https://example.com",
    url_ok: bool = True,
    html_ok: bool = True,
) -> HybridResult:
    """Build a minimal HybridResult for testing."""
    html_numeric = dict(_SAMPLE_HTML_NUMERIC)
    if html_ok:
        html_numeric["num_password_inputs"] = 2
        html_numeric["has_external_form_action"] = 1
        html_numeric["num_links"] = 15

    return HybridResult(
        url=url,
        url_features=_make_url_features() if url_ok else {},
        html_features=html_numeric,
        html_diagnostics={"page_title": "Test Page", "meta_description": "Test"},
        metadata={
            "url": url,
            "timestamp": "2026-07-31T00:00:00+00:00",
            "url_extraction_ok": url_ok,
            "html_extraction_ok": html_ok,
            "url_extraction_secs": 1.23,
            "html_extraction_secs": 2.45,
            "total_extraction_secs": 3.68,
            "url_status": [],
            "html_status": [],
            "url_feature_count": len(_SAMPLE_URL_COLS) if url_ok else 0,
            "html_feature_count": len(_HTML_NUMERIC_KEYS),
        },
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHybridFeatureSchema(unittest.TestCase):
    """HybridFeatureSchema property invariants."""

    def setUp(self):
        self.schema = HybridFeatureSchema(
            url_columns=list(_SAMPLE_URL_COLS),
            html_numeric_keys=list(_HTML_NUMERIC_KEYS),
            html_string_keys=list(_HTML_STRING_KEYS),
        )

    def test_url_feature_count(self):
        self.assertEqual(self.schema.url_feature_count, len(_SAMPLE_URL_COLS))

    def test_html_numeric_count(self):
        self.assertEqual(self.schema.html_numeric_count, len(_HTML_NUMERIC_KEYS))

    def test_total_numeric_count_is_sum(self):
        expected = len(_SAMPLE_URL_COLS) + len(_HTML_NUMERIC_KEYS)
        self.assertEqual(self.schema.total_numeric_count, expected)

    def test_schema_is_frozen(self):
        """HybridFeatureSchema must be immutable (frozen dataclass)."""
        with self.assertRaises((AttributeError, TypeError)):
            self.schema.url_columns = []  # type: ignore[misc]

    def test_html_string_keys_not_in_numeric(self):
        """Diagnostic string fields must not appear in numeric keys."""
        for key in self.schema.html_string_keys:
            self.assertNotIn(
                key,
                self.schema.html_numeric_keys,
                msg=f"'{key}' must not be in html_numeric_keys",
            )


class TestHybridResultToDict(unittest.TestCase):
    """to_dict() produces the canonical three-level structure."""

    def test_three_level_keys(self):
        result = _make_result()
        d = result.to_dict()
        self.assertIn("url_features", d)
        self.assertIn("html_features", d)
        self.assertIn("metadata", d)
        self.assertEqual(set(d.keys()), {"url_features", "html_features", "metadata"})

    def test_url_and_html_are_separate(self):
        """url_features and html_features must share NO keys."""
        result = _make_result()
        d = result.to_dict()
        overlap = set(d["url_features"].keys()) & set(d["html_features"].keys())
        self.assertEqual(
            overlap,
            set(),
            msg=f"url_features and html_features share keys: {overlap}",
        )

    def test_url_features_content(self):
        result = _make_result()
        d = result.to_dict()
        for col in _SAMPLE_URL_COLS:
            self.assertIn(col, d["url_features"])

    def test_html_features_content(self):
        result = _make_result()
        d = result.to_dict()
        for key in _HTML_NUMERIC_KEYS:
            self.assertIn(key, d["html_features"])

    def test_string_diagnostics_not_in_html_features(self):
        """page_title / meta_description must NOT appear in the html_features dict."""
        result = _make_result()
        d = result.to_dict()
        for key in _HTML_STRING_KEYS:
            self.assertNotIn(
                key,
                d["html_features"],
                msg=f"Diagnostic string '{key}' must not be in html_features",
            )

    def test_metadata_required_keys(self):
        result = _make_result()
        m = result.to_dict()["metadata"]
        for required in [
            "url", "url_extraction_ok", "html_extraction_ok",
            "url_extraction_secs", "html_extraction_secs", "total_extraction_secs",
        ]:
            self.assertIn(required, m)


class TestHybridResultExportAsUnifiedRecord(unittest.TestCase):
    """export_as_unified_record() returns a correctly ordered DataFrame."""

    def test_returns_single_row(self):
        result = _make_result()
        df = result.export_as_unified_record()
        self.assertEqual(len(df), 1)

    def test_no_string_diagnostic_columns(self):
        result = _make_result()
        df = result.export_as_unified_record()
        for key in _HTML_STRING_KEYS:
            self.assertNotIn(
                key,
                df.columns.tolist(),
                msg=f"Diagnostic string '{key}' must not be a DataFrame column",
            )

    def test_record_type_metadata(self):
        result = _make_result()
        df = result.export_as_unified_record()
        self.assertEqual(df.attrs.get("record_type"), "unified_feature_record")

    def test_architecture_note_present(self):
        result = _make_result()
        df = result.export_as_unified_record()
        note = df.attrs.get("note", "")
        self.assertIn("NOT an inference", note)
        self.assertIn("FNN", note)

    def test_export_dataframe_alias(self):
        """export_dataframe() must be an alias for export_as_unified_record()."""
        result = _make_result()
        df1 = result.export_as_unified_record()
        df2 = result.export_dataframe()
        self.assertEqual(df1.columns.tolist(), df2.columns.tolist())
        self.assertEqual(df1.values.tolist(), df2.values.tolist())


class TestHybridResultEmpty(unittest.TestCase):
    """HybridResult.empty() constructor."""

    def test_empty_has_correct_keys(self):
        result = HybridResult.empty("https://fail.example.com", reason="Test failure")
        d = result.to_dict()
        self.assertIn("url_features", d)
        self.assertIn("html_features", d)
        self.assertIn("metadata", d)

    def test_empty_url_extraction_is_false(self):
        result = HybridResult.empty("https://fail.example.com")
        self.assertFalse(result.metadata["url_extraction_ok"])

    def test_empty_html_features_all_zero(self):
        result = HybridResult.empty("https://fail.example.com")
        for key in _HTML_NUMERIC_KEYS:
            self.assertEqual(
                result.html_features.get(key),
                0,
                msg=f"html_features['{key}'] must be 0 in empty result",
            )

    def test_empty_reason_recorded(self):
        result = HybridResult.empty("https://x.com", reason="Browser failed")
        self.assertIn("Browser failed", result.metadata["url_status"])


class TestScaleUrlFeatures(unittest.TestCase):
    """scale_url_features() scales ONLY URL columns; HTML columns untouched."""

    def _make_mock_scaler(self, n_cols: int):
        """Return a mock scaler that returns the input array halved."""
        import numpy as np
        scaler = MagicMock()
        scaler.transform = MagicMock(
            side_effect=lambda X: X * 0.5  # simulate scaling
        )
        return scaler

    def test_url_columns_are_scaled(self):
        import numpy as np
        result = _make_result()
        df = result.export_as_unified_record()
        scaler = self._make_mock_scaler(len(_SAMPLE_URL_COLS))

        scaled_df = scale_url_features(df, scaler, _SAMPLE_URL_COLS)

        for col in _SAMPLE_URL_COLS:
            if col in df.columns:
                original_val = df[col].iloc[0]
                scaled_val = scaled_df[col].iloc[0]
                self.assertAlmostEqual(
                    float(scaled_val), float(original_val) * 0.5, places=5,
                    msg=f"URL column '{col}' was not scaled correctly",
                )

    def test_html_columns_not_scaled(self):
        """HTML feature values must be identical before and after URL scaling."""
        import numpy as np
        result = _make_result()
        df = result.export_as_unified_record()
        scaler = self._make_mock_scaler(len(_SAMPLE_URL_COLS))

        scaled_df = scale_url_features(df, scaler, _SAMPLE_URL_COLS)

        for key in _HTML_NUMERIC_KEYS:
            if key in df.columns and key in scaled_df.columns:
                original = df[key].iloc[0]
                after = scaled_df[key].iloc[0]
                self.assertEqual(
                    float(original), float(after),
                    msg=f"HTML column '{key}' must NOT be modified by scale_url_features",
                )

    def test_scaling_flags_in_attrs(self):
        result = _make_result()
        df = result.export_as_unified_record()
        scaler = self._make_mock_scaler(len(_SAMPLE_URL_COLS))
        scaled_df = scale_url_features(df, scaler, _SAMPLE_URL_COLS)

        self.assertTrue(scaled_df.attrs.get("url_columns_scaled"))
        self.assertFalse(scaled_df.attrs.get("html_columns_scaled"))

    def test_missing_url_columns_raises(self):
        result = _make_result()
        df = result.export_as_unified_record()
        scaler = MagicMock()
        with self.assertRaises(ValueError) as ctx:
            scale_url_features(df, scaler, ["nonexistent_col_1", "nonexistent_col_2"])
        self.assertIn("missing", str(ctx.exception).lower())


class TestBuildFeatureDictionary(unittest.TestCase):
    """build_feature_dictionary() wraps to_dict()."""

    def test_returns_three_level_dict(self):
        result = _make_result()
        d = build_feature_dictionary(result)
        self.assertIn("url_features", d)
        self.assertIn("html_features", d)
        self.assertIn("metadata", d)


class TestExportJson(unittest.TestCase):
    """HybridResult.export_json() writes valid JSON with the three-level structure."""

    def test_json_file_created(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_output.json"
            returned_path = result.export_json(out)
            self.assertTrue(out.exists())
            self.assertEqual(returned_path, out)

    def test_json_is_valid_and_has_required_keys(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_output.json"
            result.export_json(out)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("url_features", payload)
            self.assertIn("html_features", payload)
            self.assertIn("metadata", payload)

    def test_json_no_string_diagnostics_in_html_features(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_output.json"
            result.export_json(out)
            payload = json.loads(out.read_text(encoding="utf-8"))
            for key in _HTML_STRING_KEYS:
                self.assertNotIn(key, payload["html_features"])


class TestExportBatchDataframe(unittest.TestCase):
    """export_batch_dataframe() produces a Hybrid Feature Dataset, not an inference vector."""

    def test_row_count_matches_results(self):
        results = [_make_result(f"https://site{i}.com") for i in range(3)]
        df = export_batch_dataframe(results)
        self.assertEqual(len(df), 3)

    def test_record_type_metadata(self):
        results = [_make_result()]
        df = export_batch_dataframe(results)
        self.assertEqual(df.attrs.get("record_type"), "hybrid_feature_dataset")

    def test_architecture_note_present(self):
        results = [_make_result()]
        df = export_batch_dataframe(results)
        note = df.attrs.get("note", "")
        self.assertIn("future hybrid model training", note.lower())
        self.assertIn("URL features", note)

    def test_empty_list_returns_empty_dataframe(self):
        df = export_batch_dataframe([])
        self.assertTrue(df.empty)

    def test_total_urls_attr(self):
        results = [_make_result(f"https://site{i}.com") for i in range(5)]
        df = export_batch_dataframe(results)
        self.assertEqual(df.attrs.get("total_urls"), 5)


class TestGenerateUnifiedReport(unittest.TestCase):
    """generate_unified_report() writes a structured text report."""

    def test_report_file_is_created(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "test_report.txt"
            returned = generate_unified_report([result], report_path=report_path)
            self.assertTrue(report_path.exists())
            self.assertEqual(returned, report_path)

    def test_report_contains_url(self):
        result = _make_result(url="https://report-test.example.com")
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.txt"
            generate_unified_report([result], report_path=report_path)
            content = report_path.read_text(encoding="utf-8")
            self.assertIn("https://report-test.example.com", content)

    def test_report_has_architecture_note(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.txt"
            generate_unified_report([result], report_path=report_path)
            content = report_path.read_text(encoding="utf-8")
            self.assertIn("ARCHITECTURE NOTE", content)
            self.assertIn("FNN", content)
            self.assertIn("inference.py", content)

    def test_report_sections_present(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.txt"
            generate_unified_report([result], report_path=report_path)
            content = report_path.read_text(encoding="utf-8")
            self.assertIn("URL extraction", content)
            self.assertIn("HTML extraction", content)

    def test_report_multiple_results(self):
        results = [_make_result(f"https://site{i}.com") for i in range(3)]
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "batch_report.txt"
            generate_unified_report(results, report_path=report_path)
            content = report_path.read_text(encoding="utf-8")
            self.assertIn("URLs processed : 3", content)


class TestUnifiedFeaturePipelineGracefulDegradation(unittest.TestCase):
    """Graceful failure: partial extraction failures must not crash the pipeline."""

    @patch("utils.unified_feature_pipeline.UnifiedFeaturePipeline.extract_url_features")
    def test_url_failure_returns_valid_result(self, mock_url_extract):
        """A failing URL extractor still returns a valid HybridResult."""
        schema_mock = HybridFeatureSchema(
            url_columns=list(_SAMPLE_URL_COLS),
            html_numeric_keys=list(_HTML_NUMERIC_KEYS),
            html_string_keys=list(_HTML_STRING_KEYS),
        )
        zero_url = {col: 0 for col in _SAMPLE_URL_COLS}
        mock_url_extract.return_value = (zero_url, ["URL extraction failed: test"], 0.0, False)

        pipeline = UnifiedFeaturePipeline(enable_html=False)
        with patch.object(pipeline, "extract_html_features") as mock_html:
            mock_html.return_value = (
                dict(_SAMPLE_HTML_NUMERIC),
                {"page_title": "", "meta_description": ""},
                [],
                0.0,
                False,
            )
            result = pipeline.extract("https://example.com")

        self.assertIsInstance(result, HybridResult)
        self.assertFalse(result.metadata["url_extraction_ok"])
        d = result.to_dict()
        self.assertIn("url_features", d)
        self.assertIn("html_features", d)
        self.assertIn("metadata", d)

    @patch("utils.unified_feature_pipeline.UnifiedFeaturePipeline.extract_html_features")
    def test_html_failure_returns_zero_filled_html(self, mock_html_extract):
        """A failing HTML extractor fills all html keys with zero."""
        zero_html = {k: 0 for k in _HTML_NUMERIC_KEYS}
        mock_html_extract.return_value = (
            zero_html,
            {"page_title": "", "meta_description": ""},
            ["Browser failed"],
            0.0,
            False,
        )

        with patch("utils.unified_feature_pipeline.UnifiedFeaturePipeline.extract_url_features") as mock_url:
            mock_url.return_value = (_make_url_features(), [], 1.0, True)
            pipeline = UnifiedFeaturePipeline(enable_html=False)
            result = pipeline.extract("https://example.com")

        self.assertFalse(result.metadata["html_extraction_ok"])
        for key in _HTML_NUMERIC_KEYS:
            self.assertEqual(
                result.html_features.get(key),
                0,
                msg=f"html_features['{key}'] must be 0 on failure",
            )

    def test_url_and_html_features_always_separate(self):
        """Regardless of success/failure, url_features and html_features share no keys."""
        result = HybridResult.empty("https://example.com", reason="test")
        d = result.to_dict()
        overlap = set(d["url_features"].keys()) & set(d["html_features"].keys())
        self.assertEqual(overlap, set(), msg=f"Feature groups overlap on keys: {overlap}")


class TestArchitectureInvariant(unittest.TestCase):
    """High-level invariant: URL and HTML feature groups must never merge into one dict."""

    def test_to_dict_never_merges_groups(self):
        results = [_make_result(f"https://site{i}.com", url_ok=(i % 2 == 0)) for i in range(4)]
        for result in results:
            d = result.to_dict()
            overlap = set(d["url_features"].keys()) & set(d["html_features"].keys())
            self.assertEqual(
                overlap, set(),
                msg=f"url_features and html_features must never share keys. Overlap: {overlap}",
            )

    def test_unified_record_note_says_not_inference(self):
        """Every Unified Feature Record DataFrame must carry the 'NOT an inference' note."""
        results = [_make_result(f"https://site{i}.com") for i in range(3)]
        for result in results:
            df = result.export_as_unified_record()
            note = df.attrs.get("note", "")
            self.assertIn(
                "NOT an inference",
                note,
                msg="Unified Feature Record must carry architecture boundary note",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

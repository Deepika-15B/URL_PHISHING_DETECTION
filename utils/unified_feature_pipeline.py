"""
utils/unified_feature_pipeline.py
==================================
Submodule 3.5 — Unified HTML & Hybrid Feature Extraction Pipeline
------------------------------------------------------------------
Implements a modular, future-ready pipeline that extracts URL lexical/network
features and rendered-HTML content features independently, then assembles them
into a single structured report object.

IMPORTANT DESIGN INVARIANT
---------------------------
This pipeline is for EXTRACTION and REPORTING only.  It does NOT modify,
retrain, or feed any new features into the existing trained models (FNN, DNN,
Wide & Deep, TabNet).  Those models continue to receive exactly the feature
subset they were trained on.

The structured output is designed so that a future hybrid model can consume
both feature groups from the returned object without any architectural changes
to the extraction layer.

Output structure
-----------------
{
    "url_features":  { <feature_name>: <value>, ... },   # 101-column URL schema
    "html_features": { <feature_name>: <value>, ... },   # 30+ numeric HTML signals
    "metadata": {
        "url":                  str,
        "timestamp":            str (ISO-8601),
        "url_extraction_ok":    bool,
        "html_extraction_ok":   bool,
        "url_extraction_secs":  float,
        "html_extraction_secs": float,
        "total_extraction_secs":float,
        "url_status":           list[str],    # fallback / error notes
        "html_status":          list[str],
        "url_feature_count":    int,
        "html_feature_count":   int,
    }
}

Usage (single URL)
-------------------
    from utils.unified_feature_pipeline import UnifiedFeaturePipeline

    with UnifiedFeaturePipeline() as pipeline:
        result = pipeline.extract("https://example.com")
        print(result.to_dict())
        result.export_json("output.json")

Usage (batch)
--------------
    with UnifiedFeaturePipeline() as pipeline:
        results = pipeline.extract_batch(["https://a.com", "https://b.com"])

Author  : Phishing Detection IEEE Team
Version : 1.0.0  (Submodule 3.5)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ── Project paths ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent           # .../utils/
_PROJECT_ROOT = _HERE.parent                      # .../phishing_detection_ieee/
_MODELS_DIR = _PROJECT_ROOT / "models"
_REPORTS_DIR = _PROJECT_ROOT / "reports"
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── HTML numeric feature keys (ordered, authoritative list) ───────────────────
# These match exactly the keys emitted by HTMLFeatureExtractor.extract_all_html_features()
# minus the two diagnostic string fields (page_title, meta_description).
_HTML_NUMERIC_KEYS: list[str] = [
    # Basic page structure
    "number_of_forms",
    "number_of_images",
    "number_of_javascript_files",
    "number_of_css_files",
    "number_of_hyperlinks",
    # Form & credential signals
    "num_forms",
    "num_password_inputs",
    "has_external_form_action",
    "has_empty_or_blank_action",
    "has_relative_form_action",
    "num_hidden_inputs",
    "num_text_inputs",
    "num_submit_inputs",
    "has_external_action_password_form",
    # Link & anchor signals
    "num_links",
    "num_external_links",
    "num_internal_links",
    "num_null_self_links",
    "ratio_external_links",
    "ratio_internal_links",
    "ratio_null_self_links",
    "num_suspicious_anchor_text",
    "has_mismatch_link_text",
    # Anti-analysis & JS obfuscation
    "has_right_click_disabled",
    "has_text_selection_disabled",
    "num_iframes",
    "num_hidden_iframes",
    "has_popup_script",
    "has_obfuscated_js",
    # Metadata & DOM structure
    "has_external_favicon",
    "has_meta_refresh",
    "title_matches_domain",
    "title_domain_similarity_score",
    "num_meta_tags",
    "dom_depth",
    "num_total_dom_elements",
]

# Diagnostic string fields kept separately (not in numeric vector)
_HTML_STRING_KEYS: list[str] = ["page_title", "meta_description", "extracted_brand_name"]


# ── Schema dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HybridFeatureSchema:
    """Immutable descriptor for both feature groups in the hybrid pipeline.

    Attributes
    ----------
    url_columns : list[str]
        Ordered 101-column URL schema loaded from preprocessed_feature_names.pkl.
        These columns are the ONLY features consumed by the existing trained models.
    html_numeric_keys : list[str]
        Ordered list of numeric HTML signal keys extracted independently.
        These are NOT fed into any existing trained model.
    html_string_keys : list[str]
        Diagnostic string fields kept in html_diagnostics, excluded from all
        numeric feature arrays.
    """

    url_columns: list[str]
    html_numeric_keys: list[str]
    html_string_keys: list[str]

    @property
    def url_feature_count(self) -> int:
        return len(self.url_columns)

    @property
    def html_numeric_count(self) -> int:
        return len(self.html_numeric_keys)

    @property
    def total_numeric_count(self) -> int:
        """Combined feature count for Hybrid Feature Dataset construction.

        This count represents the full Unified Feature Record width available
        for FUTURE hybrid model training only.  The existing deployed models
        (FNN, DNN, Wide & Deep, TabNet) continue to use url_feature_count
        columns exclusively via the unchanged inference.py pipeline.
        """
        return self.url_feature_count + self.html_numeric_count


# Module-level schema singleton (lazy-loaded)
_schema_cache: HybridFeatureSchema | None = None


def load_hybrid_schema() -> HybridFeatureSchema:
    """Load and cache the hybrid feature schema.

    The URL column list is read from the project's persisted pickle artifact.
    HTML keys are defined statically in this module.

    Returns
    -------
    HybridFeatureSchema
        Immutable schema descriptor.

    Raises
    ------
    FileNotFoundError
        If the URL feature name pickle cannot be found.
    """
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache

    feature_names_path = _MODELS_DIR / "preprocessed_feature_names.pkl"
    if not feature_names_path.exists():
        raise FileNotFoundError(
            f"URL feature schema not found: {feature_names_path}\n"
            "Run utils/preprocessing.py first to generate preprocessed_feature_names.pkl."
        )

    import pickle
    with feature_names_path.open("rb") as fh:
        url_columns: list[str] = pickle.load(fh)

    if not isinstance(url_columns, list) or not url_columns:
        raise ValueError("preprocessed_feature_names.pkl must contain a non-empty list.")

    _schema_cache = HybridFeatureSchema(
        url_columns=url_columns,
        html_numeric_keys=list(_HTML_NUMERIC_KEYS),
        html_string_keys=list(_HTML_STRING_KEYS),
    )
    return _schema_cache


# ── Extraction result ─────────────────────────────────────────────────────────

@dataclass
class HybridResult:
    """Container for a single URL's Unified Feature Record.

    URL features and HTML features are stored as SEPARATE dictionaries and
    are never concatenated into a single inference vector.

    Architecture boundaries
    -----------------------
    Current Deployment Model
        url_features → existing scaler → original trained feature subset
                     → FNN / DNN / Wide & Deep / TabNet → Prediction

    Future Hybrid Research Dataset
        url_features + html_features → Unified Feature Record → export_as_unified_record()
                                     → future hybrid model training (not yet implemented)

    HTML features are reported independently as security intelligence.
    They are NOT scaled and NOT fed into any existing trained model.
    """

    url: str
    url_features: dict[str, Any]
    html_features: dict[str, Any]      # numeric keys only
    html_diagnostics: dict[str, str]   # page_title, meta_description
    metadata: dict[str, Any]
    raw_html: str = ""
    soup: Any = None

    # ── Convenience constructors ───────────────────────────────────────────

    @classmethod
    def empty(cls, url: str, reason: str = "") -> "HybridResult":
        """Return an all-zero HybridResult when extraction completely fails."""
        return cls(
            url=url,
            url_features={},
            html_features={k: 0 for k in _HTML_NUMERIC_KEYS},
            html_diagnostics={"page_title": "", "meta_description": ""},
            metadata={
                "url": url,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "url_extraction_ok": False,
                "html_extraction_ok": False,
                "url_extraction_secs": 0.0,
                "html_extraction_secs": 0.0,
                "total_extraction_secs": 0.0,
                "url_status": [reason] if reason else [],
                "html_status": [reason] if reason else [],
                "url_feature_count": 0,
                "html_feature_count": 0,
            },
        )

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical three-level dictionary representation.

        Structure
        ---------
        {
            "url_features":  { ... },   # raw URL lexical + network features
            "html_features": { ... },   # numeric HTML security signals
            "metadata":      { ... },
        }
        """
        return {
            "url_features": dict(self.url_features),
            "html_features": dict(self.html_features),
            "metadata": dict(self.metadata),
        }

    def export_json(self, path: str | Path, indent: int = 2) -> Path:
        """Serialise the result to a JSON file and return the resolved path.

        Non-serialisable values (e.g. numpy scalars) are cast to float/int
        before dumping so the output is always valid JSON.
        """
        output_path = Path(path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _coerce(obj: Any) -> Any:
            """Recursively make values JSON-safe."""
            if isinstance(obj, dict):
                return {k: _coerce(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_coerce(v) for v in obj]
            try:
                import numpy as np
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
            except ImportError:
                pass
            return obj

        payload = _coerce(self.to_dict())
        output_path.write_text(json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8")
        return output_path

    def export_as_unified_record(self) -> pd.DataFrame:
        """Return a single-row Unified Feature Record DataFrame.

        PURPOSE
        -------
        Analysis, visualisation, and future Hybrid Feature Dataset construction.
        This DataFrame is NOT an inference vector and is NOT passed to any
        existing trained model (FNN, DNN, Wide & Deep, TabNet).

        Column ordering
        ---------------
        URL schema columns first (101 cols), then HTML numeric columns (35 cols).
        String diagnostic fields (page_title, meta_description) are excluded
        from the DataFrame but remain accessible via self.html_diagnostics.

        HTML feature values are raw (un-scaled).  If URL-column scaling is
        required for downstream analysis, pass the URL portion of this frame
        through scale_url_features() separately.
        """
        combined: dict[str, Any] = {}
        combined.update(self.url_features)
        combined.update(self.html_features)
        frame = pd.DataFrame([combined])
        # Ensure URL columns come first if schema is available
        try:
            schema = load_hybrid_schema()
            ordered_cols = (
                [c for c in schema.url_columns if c in frame.columns]
                + [c for c in schema.html_numeric_keys if c in frame.columns]
                + [c for c in frame.columns
                   if c not in schema.url_columns and c not in schema.html_numeric_keys]
            )
            frame = frame[ordered_cols]
        except Exception:
            pass  # Return columns in insertion order if schema unavailable
        frame.attrs["url"] = self.url
        frame.attrs["record_type"] = "unified_feature_record"
        frame.attrs["url_extraction_ok"] = self.metadata.get("url_extraction_ok", False)
        frame.attrs["html_extraction_ok"] = self.metadata.get("html_extraction_ok", False)
        frame.attrs["note"] = (
            "Unified Feature Record - for analysis and future hybrid model training only. "
            "NOT an inference vector. "
            "Existing FNN/DNN/Wide&Deep/TabNet models use url_features exclusively."
        )
        return frame

    def export_dataframe(self) -> pd.DataFrame:
        """Alias for export_as_unified_record() for backwards compatibility."""
        return self.export_as_unified_record()

    def extract_18_features(self) -> dict[str, Any]:
        """Extract the exact 18 leakage-free features using PhiUSIILFeatureAdapter."""
        adapter = PhiUSIILFeatureAdapter(self.url, self.raw_html, self.soup)
        return adapter.extract()


# ── Pipeline ──────────────────────────────────────────────────────────────────

class UnifiedFeaturePipeline:
    """Modular pipeline that extracts URL features and HTML features independently.

    The pipeline is designed as a context manager so that the headless Chromium
    browser is launched once and reused across all URLs in a batch, then closed
    cleanly on exit.

    Parameters
    ----------
    timeout_ms : int
        Navigation timeout in milliseconds passed to HTMLFeatureExtractor.
    enable_html : bool
        If False, skip HTML rendering entirely (useful for URL-only benchmarks).
    enable_url : bool
        If False, skip URL network extraction (useful for offline HTML analysis).

    Examples
    --------
    Single URL::

        with UnifiedFeaturePipeline() as pipeline:
            result = pipeline.extract("https://example.com")

    Batch::

        with UnifiedFeaturePipeline() as pipeline:
            results = pipeline.extract_batch(["https://a.com", "https://b.com"])
    """

    def __init__(
        self,
        timeout_ms: int = 30_000,
        enable_html: bool = True,
        enable_url: bool = True,
    ) -> None:
        self.timeout_ms = timeout_ms
        self.enable_html = enable_html
        self.enable_url = enable_url
        self._html_extractor: Any = None  # HTMLFeatureExtractor | None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def open(self) -> "UnifiedFeaturePipeline":
        """Launch the headless browser.  Called automatically by __enter__."""
        if self.enable_html and self._html_extractor is None:
            try:
                from utils.html_feature_extractor import HTMLFeatureExtractor
                self._html_extractor = HTMLFeatureExtractor(timeout_ms=self.timeout_ms)
                self._html_extractor.launch_browser()
            except Exception as exc:
                self._html_extractor = None
                self._html_browser_error = str(exc)
                import traceback
                traceback.print_exc()
                raise  # Re-raise to prevent silent fallback
        return self

    def close(self) -> None:
        """Close the headless browser.  Called automatically by __exit__."""
        if self._html_extractor is not None:
            try:
                self._html_extractor.close_browser()
            except Exception:
                pass
            finally:
                self._html_extractor = None

    def __enter__(self) -> "UnifiedFeaturePipeline":
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Individual extraction steps ────────────────────────────────────────

    def extract_url_features(
        self, url: str
    ) -> tuple[dict[str, Any], list[str], float, bool, dict[str, bool]]:
        """Run URL lexical and network feature extraction.

        Returns
        -------
        tuple of (features_dict, status_list, elapsed_seconds, success_flag, flags_dict)
        """
        if not self.enable_url:
            return {}, ["URL extraction disabled"], 0.0, False, {"dns_extraction_ok": False, "whois_extraction_ok": False}

        t0 = time.perf_counter()
        status: list[str] = []
        try:
            from utils.url_feature_extractor import extract_all_features
            frame = extract_all_features(url)
            features = {col: frame[col].iloc[0] for col in frame.columns}
            status.extend(frame.attrs.get("status", []))
            elapsed = time.perf_counter() - t0
            
            dns_ok = frame.attrs.get("dns_extraction_ok", False)
            whois_ok = frame.attrs.get("whois_extraction_ok", False)
            flags = {"dns_extraction_ok": dns_ok, "whois_extraction_ok": whois_ok}
            
            return features, status, elapsed, True, flags
        except Exception as exc:
            status.append(f"URL extraction failed: {exc}")
            return {}, status, time.perf_counter() - t0, False, {"dns_extraction_ok": False, "whois_extraction_ok": False}

    def extract_html_features(
        self, url: str
    ) -> tuple[dict[str, Any], dict[str, str], list[str], float, bool, str, Any]:
        """Render the URL with Chromium and extract all HTML feature signals.

        Returns
        -------
        tuple of (numeric_features, string_diagnostics, status_list, elapsed_seconds, success_flag)
        """
        zero_numeric = {k: 0 for k in _HTML_NUMERIC_KEYS}
        zero_strings = {"page_title": "", "meta_description": ""}

        if not self.enable_html:
            return zero_numeric, zero_strings, ["HTML extraction disabled"], 0.0, False, "", None

        if self._html_extractor is None:
            err = getattr(self, "_html_browser_error", "Browser not launched")
            return zero_numeric, zero_strings, [f"HTML skipped: {err}"], 0.0, False, "", None

        t0 = time.perf_counter()
        status: list[str] = []
        try:
            from utils.html_feature_extractor import HTMLFeatureExtractionError
            html_str = self._html_extractor.fetch_rendered_html(url)
            soup = self._html_extractor.parse_html(html_str)
            raw = self._html_extractor.extract_all_html_features(soup, page_url=url)

            # ── STEP 4 FORENSIC: value in raw dict immediately after extractor returns ──
            print("========== STEP4: raw from extract_all_html_features ==========")
            print(f"[STEP4-A] raw['title_domain_similarity_score'] = {raw.get('title_domain_similarity_score')!r}")
            print(f"[STEP4-A] raw['title_matches_domain']          = {raw.get('title_matches_domain')!r}")
            print(f"[STEP4-A] raw['page_title']                    = {raw.get('page_title')!r}")
            print("==============================================================")

            numeric: dict[str, Any] = {}
            diagnostics: dict[str, str] = {}
            for key in _HTML_NUMERIC_KEYS:
                raw_val = raw.get(key)  # None if key is absent
                if key == "title_domain_similarity_score":
                    print("PIPELINE raw contains key:", key in raw)
                    print("PIPELINE raw value:", raw_val)
                # Only store the value if the extractor explicitly returned it.
                # None means the extractor did not provide a value for this key.
                # 0 means the extractor explicitly returned 0 (e.g. no submit buttons found).
                if raw_val is not None:
                    numeric[key] = raw_val
                # If raw_val is None, do NOT store the key in numeric - this
                # allows Pass 3 (alias map) to compute it dynamically.
            for key in _HTML_STRING_KEYS:
                diagnostics[key] = str(raw.get(key, ""))

            # ── STEP 4 FORENSIC: value AFTER key-loop ──
            print("========== STEP4: numeric after key-loop ==========")
            print(f"[STEP4-B] numeric.get('title_domain_similarity_score') = {numeric.get('title_domain_similarity_score')!r}")
            print(f"[STEP4-B] 'title_domain_similarity_score' in numeric = {'title_domain_similarity_score' in numeric}")
            print("===================================================")

            elapsed = time.perf_counter() - t0

            print("========== PIPELINE STAGE 2 ==========")
            print(f"html_numeric['title_domain_similarity_score']: {numeric.get('title_domain_similarity_score')}")
            print(f"html_numeric['title_matches_domain']:          {numeric.get('title_matches_domain')}")
            print("======================================")

            return numeric, diagnostics, status, elapsed, True, html_str, soup

        except Exception as exc:
            import traceback
            traceback.print_exc()
            status.append(f"HTML extraction failed: {exc}")
            raise  # Re-raise to prevent silent fallback

    # ── Unified extraction ─────────────────────────────────────────────────

    def extract(self, url: str) -> HybridResult:
        """Extract both URL and HTML features for a single URL.

        Both extraction stages run independently.  A failure in one does not
        prevent the other from running.

        Parameters
        ----------
        url : str
            Absolute HTTP or HTTPS URL to analyse.

        Returns
        -------
        HybridResult
            Structured extraction result.  Call .to_dict(), .export_json(), or
            .export_dataframe() to consume the output.
        """
        wall_start = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        url_feats, url_status, url_secs, url_ok, url_flags = self.extract_url_features(url)
        html_numeric, html_diag, html_status, html_secs, html_ok, raw_html, soup = self.extract_html_features(url)

        total_secs = time.perf_counter() - wall_start

        metadata: dict[str, Any] = {
            "url": url,
            "timestamp": timestamp,
            "url_extraction_ok": url_ok,
            "html_extraction_ok": html_ok,
            "url_extraction_secs": round(url_secs, 4),
            "html_extraction_secs": round(html_secs, 4),
            "total_extraction_secs": round(total_secs, 4),
            "url_status": url_status,
            "html_status": html_status,
            "url_feature_count": len(url_feats),
            "html_feature_count": len(html_numeric),
        }
        metadata.update(url_flags)

        return HybridResult(
            url=url,
            url_features=url_feats,
            html_features=html_numeric,
            html_diagnostics=html_diag,
            metadata=metadata,
            raw_html=raw_html,
            soup=soup,
        )

    def extract_batch(
        self,
        urls: list[str],
        on_progress: Any = None,
    ) -> list[HybridResult]:
        """Extract features for multiple URLs, reusing the warm browser context.

        Parameters
        ----------
        urls : list[str]
            List of absolute HTTP or HTTPS URLs.
        on_progress : callable, optional
            Called after each URL as on_progress(index, total, result).
            Useful for progress bars or live logging.

        Returns
        -------
        list[HybridResult]
            One result per URL, in the same order as the input list.
        """
        results: list[HybridResult] = []
        total = len(urls)
        for idx, url in enumerate(urls, start=1):
            result = self.extract(url)
            results.append(result)
            if callable(on_progress):
                try:
                    on_progress(idx, total, result)
                except Exception:
                    pass
        return results

    def extract_18_features(self, url: str) -> dict[str, Any]:
        """Extract the exact 18 leakage-free features for a given URL."""
        res = self.extract(url)
        return res.extract_18_features()


# ── Helper utilities ──────────────────────────────────────────────────────────

def build_feature_dictionary(result: HybridResult) -> dict[str, Any]:
    """Return the canonical three-level feature dictionary from a HybridResult.

    This is the primary output contract for Submodule 3.5:

        {
            "url_features":  { <name>: <value>, ... },
            "html_features": { <name>: <value>, ... },
            "metadata":      { ... },
        }

    Parameters
    ----------
    result : HybridResult
        Populated result from UnifiedFeaturePipeline.extract().

    Returns
    -------
    dict[str, Any]
        Three-level dictionary.
    """
    return result.to_dict()


def generate_unified_report(
    results: list[HybridResult],
    report_path: Path | str | None = None,
) -> Path:
    """Write a human-readable extraction report for one or more HybridResults.

    The report format is consistent with existing project reports
    (preprocessing_report.txt, feature_selection_report.txt).

    Parameters
    ----------
    results : list[HybridResult]
        One or more extraction results.
    report_path : Path or str, optional
        Output file.  Defaults to reports/unified_pipeline_report.txt.

    Returns
    -------
    Path
        Resolved path to the written report.
    """
    if report_path is None:
        report_path = _REPORTS_DIR / "unified_pipeline_report.txt"
    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "=" * 72,
        "IEEE PHISHING DETECTION PROJECT — UNIFIED HYBRID EXTRACTION REPORT",
        f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"URLs processed : {len(results)}",
        "=" * 72,
    ]

    for i, result in enumerate(results, start=1):
        m = result.metadata
        lines += [
            "",
            f"[{i}/{len(results)}] {result.url}",
            "-" * 72,
            f"  Timestamp            : {m.get('timestamp', 'N/A')}",
            f"  URL extraction       : {'OK' if m.get('url_extraction_ok') else 'FAILED'}  "
            f"({m.get('url_extraction_secs', 0):.3f}s, {m.get('url_feature_count', 0)} features)",
            f"  HTML extraction      : {'OK' if m.get('html_extraction_ok') else 'FAILED'}  "
            f"({m.get('html_extraction_secs', 0):.3f}s, {m.get('html_feature_count', 0)} features)",
            f"  Total elapsed        : {m.get('total_extraction_secs', 0):.3f}s",
        ]

        url_status = m.get("url_status", [])
        html_status = m.get("html_status", [])
        all_status = [f"  [URL]  {s}" for s in url_status] + [f"  [HTML] {s}" for s in html_status]

        if all_status:
            lines.append("  Fallbacks / Errors:")
            lines.extend(all_status)

        # Page diagnostics
        page_title = result.html_diagnostics.get("page_title", "")
        meta_desc = result.html_diagnostics.get("meta_description", "")
        if page_title:
            lines.append(f"  Page Title           : {page_title[:80]}")
        if meta_desc:
            lines.append(f"  Meta Description     : {meta_desc[:80]}")

        # Top HTML signals (non-zero only)
        html_signals = {k: v for k, v in result.html_features.items() if v not in (0, 0.0, "")}
        if html_signals:
            lines.append("  HTML Security Signals (non-zero):")
            for key, val in sorted(html_signals.items()):
                lines.append(f"    {key:<40}: {val}")
        else:
            lines.append("  HTML Security Signals : all zero / extraction failed")

    lines += [
        "",
        "=" * 72,
        "ARCHITECTURE NOTE",
        "-" * 72,
        "URL features and HTML features are stored separately.",
        "The existing trained FNN/DNN/Wide&Deep/TabNet models continue to",
        "receive ONLY their original training feature subset via inference.py.",
        "This report is for analysis and future hybrid model training ONLY.",
        "=" * 72,
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def export_json(result: HybridResult, path: str | Path) -> Path:
    """Convenience wrapper — serialise a single HybridResult to JSON."""
    return result.export_json(path)


def scale_url_features(
    unified_record: pd.DataFrame,
    scaler: Any,
    url_columns: list[str],
) -> pd.DataFrame:
    """Apply the saved MinMaxScaler to URL columns ONLY within a Unified Feature Record.

    HTML features are intentionally excluded from scaling because:
    - They were not part of the original training distribution.
    - No HTML-specific fitted scaler exists.
    - Scaling HTML features would produce semantically incorrect inputs.

    This function is provided as a utility for analysis and research.  It is
    NOT called during inference — the existing inference.py pipeline handles
    scaling internally using only the original URL feature subset.

    Parameters
    ----------
    unified_record : pd.DataFrame
        Single-row Unified Feature Record from export_as_unified_record().
    scaler : sklearn MinMaxScaler
        The fitted scaler loaded from models/scaler.pkl.
    url_columns : list[str]
        Ordered list of URL feature column names the scaler was fitted on.

    Returns
    -------
    pd.DataFrame
        Copy of unified_record with URL columns scaled in-place;
        HTML columns remain raw and un-scaled.

    Raises
    ------
    ValueError
        If required URL columns are missing from unified_record.
    """
    import numpy as np

    missing = [c for c in url_columns if c not in unified_record.columns]
    if missing:
        raise ValueError(
            f"scale_url_features: {len(missing)} URL column(s) missing from "
            f"Unified Feature Record: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    result = unified_record.copy()
    url_block = result[url_columns].to_numpy(dtype=np.float32)
    scaled_block = scaler.transform(url_block)
    result[url_columns] = scaled_block
    result.attrs["url_columns_scaled"] = True
    result.attrs["html_columns_scaled"] = False  # HTML features remain raw — by design
    return result


def export_dataframe(result: HybridResult) -> pd.DataFrame:
    """Convenience wrapper — return a single-row Unified Feature Record DataFrame.

    The returned DataFrame is suitable for analysis and future Hybrid Feature
    Dataset construction.  It is NOT an inference input for existing models.
    """
    return result.export_as_unified_record()


def export_batch_dataframe(results: list[HybridResult]) -> pd.DataFrame:
    """Stack multiple Unified Feature Records into a Hybrid Feature Dataset DataFrame.

    Each row is one URL's Unified Feature Record.  The resulting multi-row
    DataFrame is designed for:
    - Exploratory analysis of URL and HTML signals together.
    - Building a labelled Hybrid Feature Dataset for future hybrid model training.

    It is NOT an inference input for the existing trained models (FNN, DNN,
    Wide & Deep, TabNet).  Those models continue to use the original URL-only
    feature subset via the unchanged inference.py pipeline.

    Parameters
    ----------
    results : list[HybridResult]
        Results from UnifiedFeaturePipeline.extract_batch().

    Returns
    -------
    pd.DataFrame
        Hybrid Feature Dataset: one row per URL, URL columns first then HTML.
    """
    frames = [r.export_as_unified_record() for r in results]
    if not frames:
        return pd.DataFrame()
    dataset = pd.concat(frames, ignore_index=True)
    dataset.attrs["record_type"] = "hybrid_feature_dataset"
    dataset.attrs["total_urls"] = len(results)
    dataset.attrs["note"] = (
        "Hybrid Feature Dataset - for future hybrid model training only. "
        "NOT an inference vector. "
        "Existing deployed models use URL features exclusively."
    )
    return dataset


# ── Self-test (run directly) ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.google.com"
    print(f"[Submodule 3.5 Self-Test] URL: {test_url}")

    try:
        schema = load_hybrid_schema()
        print(f"Schema loaded  — URL columns (current deployment): {schema.url_feature_count}")
        print(f"                 HTML numeric signals (research)  : {schema.html_numeric_count}")
        print(f"                 Unified Feature Record width     : {schema.total_numeric_count} "
              f"(for future hybrid model training only)")
    except FileNotFoundError as e:
        print(f"Schema load skipped: {e}")

    # Run extraction (URL only if HTML deps unavailable)
    with UnifiedFeaturePipeline(enable_html=False) as pipeline:
        result = pipeline.extract(test_url)

    d = result.to_dict()
    print(f"URL features   : {len(d['url_features'])} keys")
    print(f"HTML features  : {len(d['html_features'])} keys")
    print(f"Metadata keys  : {list(d['metadata'].keys())}")
    print(f"URL OK         : {d['metadata']['url_extraction_ok']}")
    print(f"HTML OK        : {d['metadata']['html_extraction_ok']}")

    report = generate_unified_report([result])
    print(f"Report written : {report}")
    print("Self-test complete.")


class FeatureOrderError(Exception):
    pass

class PhiUSIILFeatureAdapter:
    """Adapter to generate the exact 18 leakage-free PhiUSIIL features."""
    
    def __init__(self, url: str, raw_html: str = "", soup: Any = None):
        self.url = url
        self.raw_html = raw_html
        self.soup = soup
        
        # Load expected feature order
        import joblib
        from pathlib import Path
        model_dir = Path(__file__).resolve().parent.parent / "models" / "leakage_free"
        self.expected_features = joblib.load(model_dir / "features_18.pkl")

    def extract(self) -> dict[str, Any]:
        from utils.url_feature_extractor import extract_phiusiil_url_features
        url_feats = extract_phiusiil_url_features(self.url)

        html_feats: dict[str, Any] = {}
        if self.soup is not None and self.raw_html:
            from utils.html_feature_extractor import HTMLFeatureExtractor
            html_feats = HTMLFeatureExtractor.extract_phiusiil_html_features(self.soup, self.raw_html, self.url)
        else:
            # HTML signals genuinely cannot be extracted -> return None, do not fabricate 0s
            html_keys = [
                "LineOfCode", "LargestLineLength", "NoOfImage", "NoOfJS", "NoOfCSS",
                "HasDescription", "IsResponsive", "HasSubmitButton", "HasSocialNet",
                "HasCopyrightInfo", "NoOfExternalRef", "NoOfSelfRef", "DomainTitleMatchScore"
            ]
            html_feats = {k: None for k in html_keys}
        
        # Merge results
        merged = {**html_feats, **url_feats}
        
        # Apply type casting and validation according to spec
        validated: dict[str, Any] = {}
        for k, v in merged.items():
            if v is None:
                validated[k] = None
                continue
            if k in {"NoOfExternalRef", "LineOfCode", "NoOfSelfRef", "NoOfImage", 
                     "NoOfJS", "NoOfCSS", "NoOfOtherSpecialCharsInURL", 
                     "LargestLineLength", "NoOfDegitsInURL", "URLLength"}:
                validated[k] = int(v)
                if validated[k] < 0:
                    raise ValueError(f"{k} must be >= 0, got {validated[k]}")
            elif k in {"HasSocialNet", "HasCopyrightInfo", "HasDescription", 
                       "IsResponsive", "HasSubmitButton"}:
                validated[k] = int(v)
                if validated[k] not in {0, 1}:
                    raise ValueError(f"{k} must be 0 or 1, got {validated[k]}")
            elif k in {"DegitRatioInURL", "DomainTitleMatchScore", "SpacialCharRatioInURL"}:
                validated[k] = float(v)
                if validated[k] < 0:
                    raise ValueError(f"{k} must be >= 0, got {validated[k]}")
                if k == "DomainTitleMatchScore" and validated[k] > 100:
                    raise ValueError(f"{k} must be <= 100, got {validated[k]}")
            else:
                validated[k] = v

        # Verify exactly the expected features are present
        extracted_keys = list(validated.keys())
        missing = [f for f in self.expected_features if f not in extracted_keys]
        if missing:
            raise FeatureOrderError(f"Missing features: {missing}")
            
        # Order the dictionary correctly according to 18-feature schema
        ordered_result = {k: validated[k] for k in self.expected_features}
        
        if list(ordered_result.keys()) != self.expected_features:
            raise FeatureOrderError("Feature order mismatch")
            
        return ordered_result

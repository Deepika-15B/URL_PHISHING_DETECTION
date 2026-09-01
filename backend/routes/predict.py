"""backend/routes/predict.py
========================
Phase 4 Prediction Route for IEEE Phishing Detection Web Application.

Accepts user URL, extracts URL + HTML features via UnifiedFeaturePipeline,
captures browser screenshots when page loading succeeds, selects trained
top20 URL feature subset, scales with scaler_phase2_v2.pkl, runs inference
with fnn_phase2_v2.keras, computes Threat Score, and returns structured JSON.

Bug Fixes (2026-08-01):
  Bug 1 - Feature Key Mismatch: PHIUSIIL top20 feature names (camelCase) were
           looked up only in url_feats (snake_case URL lexical features), yielding
           0.0 for every feature.

           Fixed via a three-pass resolver:
             Pass 1 - Exact match in combined url_feats + html_feats (136 keys).
             Pass 2 - Normalization match: case / underscore / camelCase insensitive.
             Pass 3 - Alias-map match: loads models/feature_alias_map.json which
                      maps each PHIUSIIL feature to an extractor key or a typed
                      dynamic computation spec. Feature names appear ONLY in the
                      JSON config file, never hardcoded in this Python module.

           If any feature is still unresolved after all three passes, inference
           is aborted and the error is returned in the Flask JSON response.

  Bug 2 - Premature Page Closure: fetch_rendered_html() closes its page before
           capture_page_screenshot() runs, leaving extractor._context.pages empty.

           Fixed by opening a fresh page on the still-alive BrowserContext inside
           capture_page_screenshot(), navigating to the URL, waiting for networkidle,
           capturing the screenshot, then closing ONLY that temporary page.
           Context and browser stay open until UnifiedFeaturePipeline.close().
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Blueprint, current_app, jsonify, request

predict_bp = Blueprint("predict_bp", __name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"

_ALIAS_MAP_PATH = _MODELS_DIR / "feature_alias_map.json"
_SCREENSHOTS_DIR = _PROJECT_ROOT / "reports" / "screenshots"
_SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

_PHASE2_MODEL_FILE = "fnn_phase2_v2.keras"
_PHASE2_SCALER_FILE = "scaler_phase2_v2.pkl"
_PHASE2_FEATURES_FILE = "top20_features.pkl"


def _exception_type_names(exc: BaseException) -> str:
    names: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        names.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    return " ".join(names)


def _bot_markers_from_exc(exc: BaseException) -> list[str]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        markers = getattr(current, "markers", None)
        if markers:
            return list(markers)
        current = current.__cause__ or current.__context__
    return []


def _log_phase2_runtime(message: str, **fields: Any) -> None:
    """Diagnostic runtime log for live Phase 2 inference (not shown in UI)."""
    parts = [message] + [f"{k}={v}" for k, v in fields.items()]
    logging.info("PHASE2_RUNTIME | %s", " | ".join(parts))
    try:
        current_app.logger.info("PHASE2_RUNTIME | %s", " | ".join(parts))
    except Exception:
        pass


def interpret_phase2_probability(raw_prob: float) -> tuple[int, str, float, float, str]:
    """Interpret the Phase 2 v2 sigmoid output without changing its thresholds.

    The Phase 2 v2 training report names class 0 ``Phishing`` and class 1
    ``Legitimate``. Therefore the sigmoid output is P(Legitimate), while the
    complementary probability is P(Phishing).
    """
    legitimate_probability = float(raw_prob)
    phishing_probability = 1.0 - legitimate_probability
    model_class = int(legitimate_probability >= 0.5)

    if legitimate_probability >= 0.75:
        prediction_label = "Legitimate"
    elif legitimate_probability <= 0.30:
        prediction_label = "Phishing"
    else:
        prediction_label = "Suspicious"

    if prediction_label == "Legitimate":
        confidence = round(legitimate_probability * 100, 2)
    elif prediction_label == "Phishing":
        confidence = round(phishing_probability * 100, 2)
    else:
        confidence = round(max(legitimate_probability, phishing_probability) * 100, 2)

    risk_level = (
        "High" if phishing_probability >= 0.8
        else "Medium" if phishing_probability >= 0.4
        else "Low"
    )
    return model_class, prediction_label, confidence, phishing_probability, risk_level


# ---------------------------------------------------------------------------
# Prediction Trace Logging Helper
# ---------------------------------------------------------------------------

def _write_prediction_trace(trace_data: dict, payload: dict):
    import json
    from pathlib import Path
    
    trace_path = Path(__file__).resolve().parent.parent.parent / "reports" / "flask_prediction_trace.txt"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    out.append("=== FLASK PREDICTION TRACE ===")
    out.append(f"URL: {trace_data.get('url', 'Unknown')}")
    out.append(f"Unreachable Fallback Executed: {trace_data.get('is_unreachable', False)}")
    out.append(f"Normal Inference Executed: {trace_data.get('normal_inference', False)}")
    out.append(f"Model File Loaded: {trace_data.get('model_file', 'N/A')}")
    out.append(f"Scaler File Loaded: {trace_data.get('scaler_file', 'N/A')}")
    out.append(f"Top20 Features File Loaded: {trace_data.get('top20_file', 'N/A')}")
    
    raw_feats = trace_data.get('raw_features', {})
    if raw_feats:
        out.append("\nRaw Top-20 Feature Values:")
        for k, v in raw_feats.items():
            out.append(f"  {k}: {v}")
            
    scaled_feats = trace_data.get('scaled_features', [])
    if len(scaled_feats) > 0:
        out.append("\nScaled Feature Values:")
        for i, v in enumerate(scaled_feats):
            out.append(f"  [{i}]: {v}")
            
    out.append(f"\nRaw FNN Output: {trace_data.get('raw_prob', 'N/A')}")
    out.append(f"Final Prediction: {trace_data.get('prediction', 'N/A')}")
    out.append("\nFinal JSON Payload returned to frontend:")
    out.append(json.dumps(payload, indent=2))
    
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

# ---------------------------------------------------------------------------
# URL validation and filename helpers
# ---------------------------------------------------------------------------

def validate_url(url: str) -> tuple[bool, str]:
    """Validate HTTP/HTTPS URL and return (is_valid, error_message)."""
    if not isinstance(url, str) or not url.strip():
        return False, "Missing URL: Please provide a valid HTTP or HTTPS URL."
    url_clean = url.strip()
    parsed = urlparse(url_clean)
    if parsed.scheme not in {"http", "https"}:
        return False, "Invalid URL scheme: URL must start with http:// or https://."
    if not parsed.hostname:
        return False, "Invalid URL: Missing host domain name."
    return True, ""


def sanitize_filename(hostname: str) -> str:
    """Sanitize hostname for screenshot file naming."""
    clean = re.sub(r"[^\w\.-]", "_", hostname)
    clean = re.sub(r"^www\.", "", clean, flags=re.IGNORECASE)
    return clean or "page"


# ---------------------------------------------------------------------------
# Bug 1 Fix - Pass 1 & 2: name-normalization key mapper
# ---------------------------------------------------------------------------

def _normalize_key(name: str) -> str:
    """Produce a canonical form for fuzzy name comparison.

    Steps applied in order:
    1. Split acronym boundaries: ``URLLength`` -> ``URL Length``
    2. Split camelCase boundaries: ``LineOfCode`` -> ``Line Of Code``
    3. Lowercase the entire string.
    4. Remove all underscores, hyphens, and whitespace.

    This makes pairs like ``URLLength`` / ``url_length`` / ``urllength`` equivalent.
    """
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)   # acronym split
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)           # camelCase split
    return re.sub(r"[\s_\-]+", "", s).lower()


# ---------------------------------------------------------------------------
# Bug 1 Fix - Pass 3: alias-map loader and dynamic computation engine
# ---------------------------------------------------------------------------

def _load_alias_map() -> dict[str, Any]:
    """Load models/feature_alias_map.json.

    The JSON file maps each PHIUSIIL top20 feature name to either:
    - A direct extractor output key  (``{"type": "key", "key": "num_external_links"}``)
    - A typed dynamic computation spec (``{"type": "url_https"}``)

    Feature names live only in the JSON; this function and the resolver below
    contain NO hardcoded feature names.

    Returns an empty dict if the file is absent (resolver skips Pass 3).
    """
    if not _ALIAS_MAP_PATH.exists():
        logging.warning("Alias map not found: %s - Pass 3 will be skipped.", _ALIAS_MAP_PATH)
        return {}
    try:
        with _ALIAS_MAP_PATH.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        # Strip metadata keys (prefixed with '_')
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    except Exception as exc:
        logging.warning("Could not load alias map: %s", exc)
        return {}


def _resolve_alias_entry(
    spec: dict[str, Any],
    url: str,
    combined_feats: dict[str, Any],
    html_diagnostics: dict[str, str],
) -> float | None:
    """Resolve one alias-map entry to a numeric value.

    Supported spec types (all documented in feature_alias_map.json):
    - key              : combined_feats[spec["key"]]
    - key_gt0          : 1 if combined_feats[spec["key"]] > 0 else 0
    - key_gt           : 1 if combined_feats[spec["key"]] > spec["threshold"] else 0
    - url_len          : len(url)
    - url_https        : 1 if url starts with "https://" else 0
    - url_digit_count  : count of digit chars in url
    - url_digit_ratio  : digit_count / url_len
    - url_special_count: count of non-alphanumeric chars not in ". / : ? = & - _"
    - url_special_ratio: url_special_count / url_len
    - html_diag_nonempty: 1 if html_diagnostics[spec["key"]] is non-empty
    - const            : spec["value"]

    Returns None if the spec type is unknown or a key is missing.
    """
    spec_type = spec.get("type", "")

    if spec_type == "key":
        key = spec.get("key", "")
        raw = combined_feats.get(key)
        _val = float(raw) if raw is not None else None
        return _val

    if spec_type == "key_gt0":
        key = spec.get("key", "")
        return float(int(combined_feats.get(key, 0) > 0))

    if spec_type == "key_gt":
        key = spec.get("key", "")
        threshold = spec.get("threshold", 0)
        return float(int(combined_feats.get(key, 0) > threshold))

    if spec_type == "url_len":
        return float(len(url))

    if spec_type == "url_https":
        return float(int(url.lower().startswith("https://")))

    if spec_type == "url_digit_count":
        return float(sum(1 for c in url if c.isdigit()))

    if spec_type == "url_digit_ratio":
        url_len = len(url)
        digit_count = sum(1 for c in url if c.isdigit())
        return float(digit_count / url_len) if url_len > 0 else 0.0

    if spec_type == "url_special_count":
        structural = set("./: ?=&-_")
        return float(sum(1 for c in url if not c.isalnum() and c not in structural))

    if spec_type == "url_special_ratio":
        structural = set("./: ?=&-_")
        url_len = len(url)
        special = sum(1 for c in url if not c.isalnum() and c not in structural)
        return float(special / url_len) if url_len > 0 else 0.0

    if spec_type == "html_diag_nonempty":
        key = spec.get("key", "")
        return float(int(bool(html_diagnostics.get(key, "").strip())))

    if spec_type == "const":
        return float(spec.get("value", 0))

    # key_pct: converts a binary or fractional extractor value to a 0-100 percentage
    # to match PHIUSIIL features whose training distribution is on a 0-100 scale
    # (e.g. URLSimilarityIndex mean=78.6, DomainTitleMatchScore mean=50.3).
    if spec_type == "key_pct":
        key = spec.get("key", "")
        raw = combined_feats.get(key, None)
        _val = float(raw) * 100.0 if raw is not None else None
        return _val

    # dom_to_loc: approximates PHIUSIIL LineOfCode (actual HTML source line count,
    # training mean=1145) from num_total_dom_elements.
    # Empirical approximation: ~8 HTML lines per rendered DOM element.
    if spec_type == "dom_to_loc":
        key = spec.get("key", "num_total_dom_elements")
        raw = combined_feats.get(key)
        dom_elements = float(raw) if raw is not None else 0.0
        return dom_elements * 8.0

    return None


# ---------------------------------------------------------------------------
# Main feature-key mapping: Passes 1, 2, 3
# ---------------------------------------------------------------------------

def build_feature_key_mapping(
    top20_features: list[str],
    combined_feats: dict[str, Any],
    url: str = "",
    html_diagnostics: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve every top20 feature to a numeric value using three passes.

    Pass 1 - Exact match:        feature_name in combined_feats
    Pass 2 - Normalized match:   canonical(feature_name) == canonical(extractor_key)
    Pass 3 - Alias map:          loads models/feature_alias_map.json and resolves
                                 each feature via a typed computation spec

    Diagnostics printed and logged:
    - All URL extractor keys (101)
    - All HTML extractor keys (35)
    - All top20 feature names (20)
    - Per-feature normalization canonical forms
    - Matched Features: name -> how it was resolved -> value
    - Missing Features: name (normalized) with explanation

    Raises ValueError and aborts inference if any feature is still missing after
    all three passes.

    Parameters
    ----------
    top20_features:
        Ordered list from top20_features.pkl.
    combined_feats:
        Merged url_feats + html_feats (136 keys at runtime).
    url:
        Full URL string - used by Pass 3 for URL-derived computations.
    html_diagnostics:
        Dict with page_title and meta_description from HybridResult.

    Returns
    -------
    dict mapping each top20 feature name to its resolved numeric value.
    """
    if html_diagnostics is None:
        html_diagnostics = {}


    logging.info(
        "Feature mapping: combined_feats=%d, top20=%d", len(combined_feats), len(top20_features)
    )

    # -- Build canonical lookup for Pass 2 --------------------------------
    canonical_to_key: dict[str, str] = {
        _normalize_key(k): k for k in combined_feats
    }

    # -- Load alias map for Pass 3 -----------------------------------------
    alias_map = _load_alias_map()

    # -- Three-pass resolution --------------------------------------------
    resolved: dict[str, Any] = {}
    matched_log: list[str] = []
    missing: list[str] = []

    for feat in top20_features:
        canon = _normalize_key(feat)
        value: float | None = None
        how = ""

        # Pass 1: exact match - only if value is not None
        if feat in combined_feats and combined_feats[feat] is not None:
            value = float(combined_feats[feat])
            how = "Pass 1 (exact match)"

        # Pass 2: normalized match - only if value is not None
        elif canon in canonical_to_key:
            actual_key = canonical_to_key[canon]
            raw_val = combined_feats[actual_key]
            if raw_val is not None:
                value = float(raw_val)
                how = f"Pass 2 (normalized: '{actual_key}')"
                logging.info("Normalized match: '%s' -> '%s'", feat, actual_key)

        elif feat in alias_map:
            spec = alias_map[feat]
            value = _resolve_alias_entry(spec, url, combined_feats, html_diagnostics)
            if value is not None:
                note = spec.get("_note", "")
                how = f"Pass 3 (alias map type='{spec.get('type')}'{(', ' + note) if note else ''})"
            else:
                how = f"Pass 3 attempted (alias map) but spec failed: {spec}"

        if value is not None:
            resolved[feat] = value
            matched_log.append(f"  {feat} = {value}  [{how}]")
        else:
            missing.append(feat)



    logging.info(
        "Feature resolution: %d matched, %d missing | missing=%s",
        len(resolved), len(missing), missing,
    )

    if missing:
        raise ValueError(
            f"Inference aborted: {len(missing)}/{len(top20_features)} Top20 features "
            f"remain unresolvable after all 3 passes (exact, normalization, alias map). "
            f"Missing: {missing}. "
            f"Add entries to {_ALIAS_MAP_PATH.name} to resolve them."
        )

    return resolved




# ---------------------------------------------------------------------------
# Bug 2 Fix - Screenshot capture on fresh page within the live context
# ---------------------------------------------------------------------------

def capture_page_screenshot(pipeline: Any, url: str) -> str | None:
    """Capture a Playwright screenshot AFTER feature extraction.

    Root cause of original bug
    --------------------------
    HTMLFeatureExtractor.fetch_rendered_html() closes its page in a finally
    block. By the time capture_page_screenshot ran, the context had zero open
    pages, so the ``if pages:`` guard was always False and no screenshot was
    ever saved.

    Fix
    ---
    The BrowserContext (extractor._context) remains alive until
    pipeline.close() is called. We open a *new* page on that still-live
    context, navigate, wait for networkidle, take the screenshot, then close
    ONLY that temporary page. Context and browser stay open.

    Enforced workflow
    -----------------
    Browser launch (already done by UnifiedFeaturePipeline.open())
    -> fresh page = context.new_page()
    -> page.goto(url, wait_until="networkidle")
    -> page.screenshot(path=<domain>_<timestamp>.png)
    -> page.close()
    -> return "/reports/screenshots/<filename>"

    Parameters
    ----------
    pipeline:
        The UnifiedFeaturePipeline instance (still inside its with block,
        so context is open).
    url:
        URL to navigate to for the screenshot.

    Returns
    -------
    Relative path string "/reports/screenshots/<filename>" on success, else None.
    """
    try:
        extractor = getattr(pipeline, "_html_extractor", None)
        if extractor is None:
            logging.warning("Screenshot skipped: html_extractor not available for %s", url)
            logging.info("Screenshot skipped: html_extractor is not available")
            return None

        context = getattr(extractor, "_context", None)
        if context is None:
            logging.warning("Screenshot skipped: browser context is None for %s", url)
            logging.info("Screenshot skipped: browser context is not available")
            return None

        parsed = urlparse(url)
        domain = sanitize_filename(parsed.hostname or "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{domain}_{timestamp}.png"
        filepath = _SCREENSHOTS_DIR / filename

        logging.info("Screenshot browser active; navigating to %s", url)

        # Open a fresh page on the still-live BrowserContext
        page = context.new_page()
        try:
            try:
                page.goto(url, wait_until="networkidle", timeout=15000)
                logging.info("Screenshot page loaded (networkidle)")
            except Exception:
                page.goto(url, wait_until="domcontentloaded", timeout=10000)
                page.wait_for_timeout(500)
                logging.info("Screenshot page loaded (domcontentloaded fallback)")
            try:
                from utils.html_feature_extractor import wait_for_rendered_content
                wait_for_rendered_content(page, timeout_ms=8000)
            except Exception:
                page.wait_for_timeout(500)

            page.screenshot(path=str(filepath), full_page=False, timeout=8000)
            logging.info("Screenshot saved: %s (exists=%s)", filepath, filepath.exists())
            logging.info("Screenshot saved: %s (exists=%s)", filepath, filepath.exists())
            return f"/reports/screenshots/{filename}"
        finally:
            page.close()

    except Exception as exc:
        tb = traceback.format_exc()
        logging.warning("Screenshot capture failed for %s: %s\n%s", url, exc, tb)
        logging.warning("Screenshot capture failed for %s: %s\n%s", url, exc, tb)
        return None


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

@predict_bp.post("/predict")
@predict_bp.post("/api/predict")
def predict_endpoint():
    """End-to-End Prediction Route."""
    t_start = time.perf_counter()
    import uuid as _uuid
    _req_id = str(_uuid.uuid4())[:8]


    # 1. Input parsing & validation

    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    if not payload or not isinstance(payload, dict):
        return jsonify({"error": "Invalid request payload. Expected JSON body or form data with 'url'."}), 400

    raw_url = payload.get("url", "")
    is_valid, err_msg = validate_url(raw_url)
    if not is_valid:
        return jsonify({"error": err_msg}), 400

    url = raw_url.strip()

    # 2. Extract features using UnifiedFeaturePipeline
    #    Screenshot captured INSIDE the with-block so the BrowserContext is still live.
    try:
        from utils.unified_feature_pipeline import UnifiedFeaturePipeline
        with UnifiedFeaturePipeline(timeout_ms=15000) as pipeline:
            result = pipeline.extract(url)
            # Bug 2 Fix: capture screenshot before pipeline.close() is called
            screenshot_rel_path = capture_page_screenshot(pipeline, url)
    except Exception as exc:
        logging.exception("UnifiedFeaturePipeline failed for %s", url)
        import traceback
        tb_str = traceback.format_exc()
        
        # Collect Playwright state
        browser_state = "Not available"
        context_state = "Not available"
        page_state = "Not available"
        
        try:
            if 'pipeline' in locals() and hasattr(pipeline, '_html_extractor') and pipeline._html_extractor is not None:
                extractor = pipeline._html_extractor
                browser_state = "Alive" if getattr(extractor, '_browser', None) is not None else "None/Dead"
                context = getattr(extractor, '_context', None)
                if context is not None:
                    context_state = f"Alive (pages: {len(context.pages)})"
                    page_state = f"{len(context.pages)} pages open"
                else:
                    context_state = "None/Dead"
        except Exception as state_exc:
            browser_state = f"Error: {state_exc}"
            
        debug_path = Path(_PROJECT_ROOT) / "reports" / "flask_playwright_debug.txt"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_text = f"URL: {url}\nException Type: {type(exc).__name__}\n"
        debug_text += f"Browser State: {browser_state}\n"
        debug_text += f"Context State: {context_state}\n"
        debug_text += f"Page State: {page_state}\n"
        debug_text += f"Screenshot State: Failed before screenshot capture\n\n"
        debug_text += f"Stack Trace:\n{tb_str}\n"
        debug_path.write_text(debug_text, encoding="utf-8")

        exc_str = str(exc)
        network_errors = [
            "ERR_NAME_NOT_RESOLVED", 
            "ERR_CONNECTION_REFUSED", 
            "ERR_CONNECTION_TIMED_OUT", 
            "ERR_INTERNET_DISCONNECTED", 
            "ERR_SSL_PROTOCOL_ERROR",
            "net::ERR_"
        ]
        is_unreachable = any(err in exc_str for err in network_errors)
        
        if "AntiBotProtectionError" in _exception_type_names(exc):
            markers = _bot_markers_from_exc(exc)
            html_length = None
            http_status = None
            try:
                from utils.html_feature_extractor import HTMLFeatureExtractor as _HX
                html_length = getattr(_HX, "_last_html_length", None)
                http_status = getattr(_HX, "_last_http_status", None)
            except Exception:
                pass
            _log_phase2_runtime(
                "bot_protection_abort",
                URL=url,
                MODEL_PATH=_PHASE2_MODEL_FILE,
                SCALER_PATH=_PHASE2_SCALER_FILE,
                FEATURE_FILE=_PHASE2_FEATURES_FILE,
                FEATURE_COUNT=0,
                BROWSER_STATUS="Blocked",
                HTML_EXTRACTION_STATUS="Failed",
                BOT_PROTECTION_STATUS="DETECTED",
                DETECTED_MARKERS=markers,
                MODEL_PREDICTION_CALLED="NO",
                MODEL_CLASS="NOT_CALLED",
                FINAL_PREDICTION="Unknown",
            )
            payload = {
                "status": "BOT_PROTECTION_PAGE",
                "prediction": "Unknown",
                "confidence": 0,
                "threat_score": 0,
                "reason": ["Website is protected by anti-bot mechanisms. Unable to analyse actual webpage."],
                "browser_status": "Blocked",
                "screenshot": None,
                "risk_level": "Unknown",
                "bot_protection_detected": True,
                "detected_markers": markers,
                "model_predict_called": False,
                "html_length": html_length,
                "http_status": http_status,
            }
            _write_prediction_trace({
                "url": url,
                "is_unreachable": False,
                "normal_inference": False,
                "prediction": "Unknown",
                "model_predict_called": False,
            }, payload)
            return jsonify(payload), 200

        if "CDNErrorPageError" in type(exc).__name__:
            payload = {
                "status": "CDN_ERROR_PAGE",
                "prediction": "Unknown",
                "confidence": 0,
                "threat_score": 0,
                "reason": ["CDN/CloudFront error page detected. Actual webpage could not be analysed."],
                "browser_status": "Blocked",
                "screenshot": None,
                "risk_level": "Unknown"
            }
            _write_prediction_trace({
                "url": url,
                "is_unreachable": False,
                "normal_inference": False,
                "prediction": "Unknown"
            }, payload)
            return jsonify(payload), 200

        if is_unreachable:
            payload = {
                "status": "unreachable",
                "prediction": "Unknown",
                "confidence": 0,
                "threat_score": 0,
                "reason": [
                    "Domain could not be reached.",
                    "DNS resolution failed or website is unavailable.",
                    "HTML analysis could not be performed."
                ],
                "browser_status": "Unavailable",
                "screenshot": None,
                "risk_level": "Unknown"
            }
            _write_prediction_trace({
                "url": url,
                "is_unreachable": True,
                "normal_inference": False,
                "prediction": "Unknown"
            }, payload)
            return jsonify(payload), 200

        return jsonify({
            "error": f"Feature extraction failed: {type(exc).__name__} - {exc}",
            "url": url,
            "status": "failed",
        }), 500

    url_feats = result.url_features
    html_feats = result.html_features
    html_diagnostics = result.html_diagnostics
    meta = result.metadata

    if screenshot_rel_path:
        meta["screenshot_path"] = screenshot_rel_path
        meta["screenshot_captured"] = True
    else:
        meta["screenshot_captured"] = False

    def _safe_get_url_feat(key: str) -> Any:
        import math
        val = url_feats.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "N/A"
        return _to_native(val)

    def _to_native(val):
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        return val

    if not meta.get("html_extraction_ok"):
        _log_phase2_runtime(
            "partial_extraction_abort",
            URL=url,
            MODEL_PATH=_PHASE2_MODEL_FILE,
            SCALER_PATH=_PHASE2_SCALER_FILE,
            FEATURE_FILE=_PHASE2_FEATURES_FILE,
            FEATURE_COUNT=0,
            BROWSER_STATUS="Failed",
            HTML_EXTRACTION_STATUS="Failed",
            BOT_PROTECTION_STATUS="NO",
            MODEL_PREDICTION_CALLED="NO",
            FINAL_PREDICTION="Unknown",
        )
        payload = {
            "status": "PARTIAL_EXTRACTION",
            "prediction": "Unknown",
            "confidence": 0,
            "threat_score": 0,
            "reason": ["HTML extraction failed. Required features cannot be computed reliably."],
            "browser_status": "Failed",
            "screenshot": None,
            "risk_level": "Unknown",
            "model_predict_called": False,
            "url_feature_summary": {
                "domain": urlparse(url).hostname,
                "tls_ssl_certificate": _safe_get_url_feat("tls_ssl_certificate"),
                "qty_ip_resolved": _safe_get_url_feat("qty_ip_resolved"),
                "qty_nameservers": _safe_get_url_feat("qty_nameservers"),
                "time_response": _safe_get_url_feat("time_response"),
                "domain_spf": _safe_get_url_feat("domain_spf"),
                "qty_redirects": _safe_get_url_feat("qty_redirects"),
            },
            "system_info": {
                "url_extraction_secs": round(meta.get("url_extraction_secs", 0.0), 3),
                "html_extraction_secs": round(meta.get("html_extraction_secs", 0.0), 3),
                "extraction_time": round(meta.get("total_extraction_secs", 0.0), 3),
                "prediction_time": 0.0,
                "total_processing_time": 0.0,
                "browser_status": "Failed",
                "url_feature_count": _to_native(meta.get("url_feature_count", 0)),
                "html_feature_count": _to_native(meta.get("html_feature_count", 0)),
                "url_extraction_status": "Success" if meta.get("url_extraction_ok") else "Failed",
                "html_extraction_status": "Success" if meta.get("html_extraction_ok") else "Failed",
                "dns_extraction_status": "Success" if meta.get("dns_extraction_ok") else "Failed",
                "whois_extraction_status": "Success" if meta.get("whois_extraction_ok") else "Failed",
            }
        }
        _write_prediction_trace({
            "url": url,
            "is_unreachable": False,
            "normal_inference": False,
            "prediction": "Unknown"
        }, payload)
        return jsonify(payload), 200

    # Stub / Incomplete page check
    title_stripped = html_diagnostics.get("page_title", "").strip().lower()
    dom_elements = html_feats.get("num_total_dom_elements", 0)
    num_css = html_feats.get("number_of_css_files", 0)
    num_images = html_feats.get("number_of_images", 0)
    num_js = html_feats.get("number_of_javascript_files", 0)
    num_links = html_feats.get("num_links", 0)

    is_empty_title = not title_stripped or any(term in title_stripped for term in [
        "loading...", "please wait...", "just a moment...", "403 forbidden", 
        "site maintenance", "unsupported browser", "checking your browser"
    ])
    is_low_dom = dom_elements < 50
    is_low_resources = (num_links + num_images + num_css + num_js) < 10

    interstitial_haystack = " ".join(
        [
            str(html_diagnostics.get("page_title", "")),
            str(html_diagnostics.get("meta_description", "")),
            str(html_diagnostics.get("html_excerpt", "")),
        ]
    )
    interstitial_markers = []
    try:
        from utils.html_feature_extractor import detect_bot_protection_markers

        interstitial_markers = detect_bot_protection_markers(
            str(html_diagnostics.get("page_title", "")),
            interstitial_haystack,
        )
    except Exception:
        interstitial_markers = []

    if interstitial_markers:
        _log_phase2_runtime(
            "bot_protection_from_diagnostics",
            URL=url,
            MODEL_PATH=_PHASE2_MODEL_FILE,
            SCALER_PATH=_PHASE2_SCALER_FILE,
            FEATURE_FILE=_PHASE2_FEATURES_FILE,
            FEATURE_COUNT=0,
            BROWSER_STATUS="Blocked",
            HTML_EXTRACTION_STATUS="Success",
            BOT_PROTECTION_STATUS="DETECTED",
            DETECTED_MARKERS=interstitial_markers,
            MODEL_PREDICTION_CALLED="NO",
            FINAL_PREDICTION="Unknown",
        )
        payload = {
            "status": "BOT_PROTECTION_PAGE",
            "prediction": "Unknown",
            "confidence": 0,
            "threat_score": 0,
            "reason": ["Website is protected by anti-bot mechanisms. Unable to analyse actual webpage."],
            "browser_status": "Blocked",
            "screenshot": None,
            "risk_level": "Unknown",
            "bot_protection_detected": True,
            "detected_markers": interstitial_markers,
            "model_predict_called": False,
            "url_feature_summary": {
                "domain": urlparse(url).hostname,
                "tls_ssl_certificate": _safe_get_url_feat("tls_ssl_certificate"),
                "qty_ip_resolved": _safe_get_url_feat("qty_ip_resolved"),
                "qty_nameservers": _safe_get_url_feat("qty_nameservers"),
                "time_response": _safe_get_url_feat("time_response"),
                "domain_spf": _safe_get_url_feat("domain_spf"),
                "qty_redirects": _safe_get_url_feat("qty_redirects"),
            },
            "system_info": {
                "url_extraction_secs": round(meta.get("url_extraction_secs", 0.0), 3),
                "html_extraction_secs": round(meta.get("html_extraction_secs", 0.0), 3),
                "extraction_time": round(meta.get("total_extraction_secs", 0.0), 3),
                "prediction_time": 0.0,
                "total_processing_time": 0.0,
                "browser_status": "Blocked",
                "url_feature_count": _to_native(meta.get("url_feature_count", 0)),
                "html_feature_count": _to_native(meta.get("html_feature_count", 0)),
                "url_extraction_status": "Success" if meta.get("url_extraction_ok") else "Failed",
                "html_extraction_status": "Success" if meta.get("html_extraction_ok") else "Failed",
                "dns_extraction_status": "Success" if meta.get("dns_extraction_ok") else "Failed",
                "whois_extraction_status": "Success" if meta.get("whois_extraction_ok") else "Failed",
                "model_predict_called": False,
            },
        }
        _write_prediction_trace({
            "url": url,
            "is_unreachable": False,
            "normal_inference": False,
            "prediction": "Unknown",
            "model_predict_called": False,
        }, payload)
        return jsonify(payload), 200

    is_unresolved_shell = (
        int(num_links or 0) == 0
        and int(num_images or 0) <= 2
        and int(dom_elements or 0) < 150
    )

    if (is_empty_title and is_low_dom and is_low_resources) or is_unresolved_shell:
        _log_phase2_runtime(
            "incomplete_page_abort",
            URL=url,
            MODEL_PATH=_PHASE2_MODEL_FILE,
            SCALER_PATH=_PHASE2_SCALER_FILE,
            FEATURE_FILE=_PHASE2_FEATURES_FILE,
            FEATURE_COUNT=0,
            BROWSER_STATUS="Failed",
            HTML_EXTRACTION_STATUS="Success",
            BOT_PROTECTION_STATUS="NO",
            MODEL_PREDICTION_CALLED="NO",
            FINAL_PREDICTION="Unknown",
        )
        payload = {
            "status": "PARTIAL_EXTRACTION",
            "prediction": "Unknown",
            "confidence": 0,
            "threat_score": 0,
            "reason": [
                "Incomplete browser/interstitial page detected. Real website content could not be reliably analysed."
            ],
            "browser_status": "Failed",
            "screenshot": None,
            "risk_level": "Unknown",
            "model_predict_called": False,
            "url_feature_summary": {
                "domain": urlparse(url).hostname,
                "tls_ssl_certificate": _safe_get_url_feat("tls_ssl_certificate"),
                "qty_ip_resolved": _safe_get_url_feat("qty_ip_resolved"),
                "qty_nameservers": _safe_get_url_feat("qty_nameservers"),
                "time_response": _safe_get_url_feat("time_response"),
                "domain_spf": _safe_get_url_feat("domain_spf"),
                "qty_redirects": _safe_get_url_feat("qty_redirects"),
            },
            "system_info": {
                "url_extraction_secs": round(meta.get("url_extraction_secs", 0.0), 3),
                "html_extraction_secs": round(meta.get("html_extraction_secs", 0.0), 3),
                "extraction_time": round(meta.get("total_extraction_secs", 0.0), 3),
                "prediction_time": 0.0,
                "total_processing_time": 0.0,
                "browser_status": "Failed",
                "url_feature_count": _to_native(meta.get("url_feature_count", 0)),
                "html_feature_count": _to_native(meta.get("html_feature_count", 0)),
                "url_extraction_status": "Success" if meta.get("url_extraction_ok") else "Failed",
                "html_extraction_status": "Success" if meta.get("html_extraction_ok") else "Failed",
                "dns_extraction_status": "Success" if meta.get("dns_extraction_ok") else "Failed",
                "whois_extraction_status": "Success" if meta.get("whois_extraction_ok") else "Failed",
            }
        }
        _write_prediction_trace({
            "url": url,
            "is_unreachable": False,
            "normal_inference": False,
            "prediction": "Unknown"
        }, payload)
        return jsonify(payload), 200

    # 3. Model Inference
    t_pred_start = time.perf_counter()
    try:
        model = current_app.config.get("INFERENCE_MODEL")
        scaler = current_app.config.get("INFERENCE_SCALER")
        top20_features = current_app.config.get("TOP20_FEATURES")

        if model is None or scaler is None or top20_features is None:
            import pickle
            with open(_MODELS_DIR / "top20_features.pkl", "rb") as f:
                top20_features = pickle.load(f)
            with open(_MODELS_DIR / "scaler_phase2_v2.pkl", "rb") as f:
                scaler = pickle.load(f)
            model = tf.keras.models.load_model(_MODELS_DIR / "fnn_phase2_v2.keras")

        if len(top20_features) != 20:
            raise ValueError(
                f"Phase 2 feature list must contain 20 names, got {len(top20_features)}"
            )
        model_in = getattr(model, "input_shape", None)
        if model_in is not None:
            last_dim = model_in[-1] if isinstance(model_in, tuple) else None
            if last_dim not in (None, 20):
                raise ValueError(
                    f"Phase 2 model input last dim must be 20, got {model_in}"
                )

        # Bug 1 Fix: combine url_feats + html_feats and resolve using 3-pass mapper
        combined_feats: dict[str, Any] = {**url_feats, **html_feats}



        try:
            feat_dict = build_feature_key_mapping(
                top20_features,
                combined_feats,
                url=url,
                html_diagnostics=html_diagnostics,
            )
        except ValueError as mapping_err:
            logging.error("Feature mapping failed for %s: %s", url, mapping_err)
            payload = {
                "status": "PARTIAL_EXTRACTION",
                "prediction": "Unknown",
                "confidence": 0,
                "threat_score": 0,
                "reason": [f"A required Top20 feature could not be computed reliably: {mapping_err}"],
                "browser_status": "Failed",
                "screenshot": None,
                "risk_level": "Unknown",
                "url_feature_summary": {
                    "domain": urlparse(url).hostname,
                    "tls_ssl_certificate": _safe_get_url_feat("tls_ssl_certificate"),
                    "qty_ip_resolved": _safe_get_url_feat("qty_ip_resolved"),
                    "qty_nameservers": _safe_get_url_feat("qty_nameservers"),
                    "time_response": _safe_get_url_feat("time_response"),
                    "domain_spf": _safe_get_url_feat("domain_spf"),
                    "qty_redirects": _safe_get_url_feat("qty_redirects"),
                },
                "system_info": {
                    "url_extraction_secs": round(meta.get("url_extraction_secs", 0.0), 3),
                    "html_extraction_secs": round(meta.get("html_extraction_secs", 0.0), 3),
                    "extraction_time": round(meta.get("total_extraction_secs", 0.0), 3),
                    "prediction_time": 0.0,
                    "total_processing_time": 0.0,
                    "browser_status": "Active" if meta.get("html_extraction_ok") else "Failed",
                    "url_feature_count": _to_native(meta.get("url_feature_count", 0)),
                    "html_feature_count": _to_native(meta.get("html_feature_count", 0)),
                    "url_extraction_status": "Success" if meta.get("url_extraction_ok") else "Failed",
                    "html_extraction_status": "Success" if meta.get("html_extraction_ok") else "Failed",
                    "dns_extraction_status": "Success" if meta.get("dns_extraction_ok") else "Failed",
                    "whois_extraction_status": "Success" if meta.get("whois_extraction_ok") else "Failed",
                }
            }
            _write_prediction_trace({
                "url": url,
                "is_unreachable": False,
                "normal_inference": False,
                "prediction": "Unknown"
            }, payload)
            return jsonify(payload), 200

        input_df = pd.DataFrame([feat_dict], columns=top20_features).astype(np.float32)

        # === STRICT RUNTIME ASSERTIONS (Phase 2) ===
        if list(input_df.columns) != list(top20_features):
            raise ValueError(
                f"Feature column mismatch: expected {top20_features}, got {list(input_df.columns)}"
            )
        if input_df.shape != (1, 20):
            raise ValueError(
                f"Feature DataFrame shape mismatch: expected (1, 20), got {input_df.shape}"
            )

        X_scaled = scaler.transform(input_df)

        if tuple(X_scaled.shape) != (1, 20):
            raise ValueError(
                f"Scaled input shape mismatch: expected (1, 20), got {X_scaled.shape}"
            )

        raw_prob = float(model.predict(X_scaled, verbose=0)[0][0])
        _log_phase2_runtime(
            "model_inference",
            URL=url,
            MODEL_PATH=_PHASE2_MODEL_FILE,
            SCALER_PATH=_PHASE2_SCALER_FILE,
            FEATURE_FILE=_PHASE2_FEATURES_FILE,
            FEATURE_COUNT=len(top20_features),
            FEATURE_ORDER=list(top20_features),
            RAW_FEATURES=dict(zip(top20_features, input_df.iloc[0].tolist())),
            SCALED_INPUT_SHAPE=tuple(X_scaled.shape),
            BROWSER_STATUS="Active",
            HTML_EXTRACTION_STATUS="Success",
            BOT_PROTECTION_STATUS="NO",
            MODEL_PREDICTION_CALLED="YES",
            RAW_MODEL_PROBABILITY=raw_prob,
        )
        t_pred_end = time.perf_counter()
        prediction_time = t_pred_end - t_pred_start

        model_class, prediction_label, confidence, phishing_probability, risk_level = (
            interpret_phase2_probability(raw_prob)
        )
        threat_score = round(phishing_probability * 100, 1)

        # Generate Explainable Feature Contributions
        feature_contributions = {"phishing": [], "legitimate": []}
        try:
            contributions = []
            for i in range(len(top20_features)):
                x_mod = X_scaled.copy()
                x_mod[0, i] = 0.5  # Neutral baseline
                p_mod = float(model.predict(x_mod, verbose=0)[0][0])
                phish_mod = 1.0 - p_mod
                contrib = phishing_probability - phish_mod
                contributions.append((top20_features[i], contrib))
            
            contributions.sort(key=lambda x: x[1], reverse=True)
            
            phishing_reasons = []
            for feat, val in contributions:
                if val > 0.01:
                    phishing_reasons.append(f"{feat} +{round(val * 100, 1)}%")
            
            legitimate_reasons = []
            for feat, val in reversed(contributions):
                if val < -0.01:
                    legitimate_reasons.append(f"{feat} {round(val * 100, 1)}%")
                    
            feature_contributions["phishing"] = phishing_reasons[:5]
            feature_contributions["legitimate"] = legitimate_reasons[:5]
        except Exception as exc:
            logging.error(f"Failed to generate feature contributions: {exc}")

    except Exception as exc:
        logging.exception("Model inference failed for %s", url)
        return jsonify({
            "error": f"Model inference error: {exc}",
            "url": url,
            "status": "failed",
        }), 500

    total_processing_time = time.perf_counter() - t_start

    # 4. Construct response JSON

    response_payload = {
        "url": url,
        "prediction": prediction_label,
        "model_class": model_class,
        "probability": round(raw_prob, 4),
        "phishing_probability": round(phishing_probability, 4),
        "confidence": confidence,
        "threat_score": threat_score,
        "risk_level": risk_level,
        "screenshot_captured": meta.get("screenshot_captured", False),
        "screenshot_path": meta.get("screenshot_path", None),
        "html_security_report": {
            "forms": {
                "number_of_forms": _to_native(html_feats.get("num_forms", 0)),
                "num_password_inputs": _to_native(html_feats.get("num_password_inputs", 0)),
                "has_external_form_action": "Yes" if html_feats.get("has_external_form_action") == 1 else "No",
                "has_empty_or_blank_action": "Yes" if html_feats.get("has_empty_or_blank_action") == 1 else "No",
                "has_relative_form_action": "Yes" if html_feats.get("has_relative_form_action") == 1 else "No",
                "num_hidden_inputs": _to_native(html_feats.get("num_hidden_inputs", 0)),
                "has_external_action_password_form": "Yes" if html_feats.get("has_external_action_password_form") == 1 else "No",
            },
            "links": {
                "num_links": _to_native(html_feats.get("num_links", 0)),
                "num_external_links": _to_native(html_feats.get("num_external_links", 0)),
                "num_internal_links": _to_native(html_feats.get("num_internal_links", 0)),
                "num_null_self_links": _to_native(html_feats.get("num_null_self_links", 0)),
                "ratio_external_links": _to_native(html_feats.get("ratio_external_links", 0.0)),
                "num_suspicious_anchor_text": _to_native(html_feats.get("num_suspicious_anchor_text", 0)),
                "has_mismatch_link_text": "Yes" if html_feats.get("has_mismatch_link_text") == 1 else "No",
            },
            "scripts": {
                "has_obfuscated_js": "Detected" if html_feats.get("has_obfuscated_js") == 1 else "Not Detected",
                "num_iframes": _to_native(html_feats.get("num_iframes", 0)),
                "num_hidden_iframes": _to_native(html_feats.get("num_hidden_iframes", 0)),
                "has_popup_script": "Yes" if html_feats.get("has_popup_script") == 1 else "No",
                "has_right_click_disabled": "Yes" if html_feats.get("has_right_click_disabled") == 1 else "No",
                "has_text_selection_disabled": "Yes" if html_feats.get("has_text_selection_disabled") == 1 else "No",
            },
            "metadata": {
                "page_title": result.html_diagnostics.get("page_title", ""),
                "meta_description": result.html_diagnostics.get("meta_description", ""),
                "brand_name": result.html_diagnostics.get("extracted_brand_name", ""),
                "page_domain": result.html_diagnostics.get("extracted_page_domain", ""),
                "registered_domain": result.html_diagnostics.get("extracted_registered_domain", ""),
                "has_external_favicon": "Yes" if html_feats.get("has_external_favicon") == 1 else "No",
                "has_meta_refresh": "Yes" if html_feats.get("has_meta_refresh") == 1 else "No",
                "title_matches_domain": "Yes" if html_feats.get("title_matches_domain") == 1 else "No",
                "num_meta_tags": _to_native(html_feats.get("num_meta_tags", 0)),
                "dom_depth": _to_native(html_feats.get("dom_depth", 0)),
                "num_total_dom_elements": _to_native(html_feats.get("num_total_dom_elements", 0)),
            },
        },
        "url_feature_summary": {
            "domain": urlparse(url).hostname,
            "tls_ssl_certificate": _safe_get_url_feat("tls_ssl_certificate"),
            "qty_ip_resolved": _safe_get_url_feat("qty_ip_resolved"),
            "qty_nameservers": _safe_get_url_feat("qty_nameservers"),
            "time_response": _safe_get_url_feat("time_response"),
            "domain_spf": _safe_get_url_feat("domain_spf"),
            "qty_redirects": _safe_get_url_feat("qty_redirects"),
        },
        "system_info": {
            "url_extraction_secs": round(meta.get("url_extraction_secs", 0.0), 3),
            "html_extraction_secs": round(meta.get("html_extraction_secs", 0.0), 3),
            "extraction_time": round(meta.get("total_extraction_secs", 0.0), 3),
            "prediction_time": round(prediction_time, 4),
            "total_processing_time": round(total_processing_time, 3),
            "browser_status": "Active" if meta.get("html_extraction_ok") else "Skipped/Failed",
            "url_feature_count": _to_native(meta.get("url_feature_count", 0)),
            "html_feature_count": _to_native(meta.get("html_feature_count", 0)),
            "url_extraction_status": "Success" if meta.get("url_extraction_ok") else "Failed",
            "html_extraction_status": "Success" if meta.get("html_extraction_ok") else "Failed",
            "dns_extraction_status": "Success" if meta.get("dns_extraction_ok") else "Failed",
            "whois_extraction_status": "Success" if meta.get("whois_extraction_ok") else "Failed",
            "model_predict_called": True,
            "feature_count": 20,
        },
        "top20_features": feat_dict,
        "feature_contributions": feature_contributions,
        "model_predict_called": True,
        "bot_protection_detected": False,
        "status": "success",
    }

    _log_phase2_runtime(
        "final_prediction",
        URL=url,
        MODEL_PREDICTION_CALLED="YES",
        RAW_MODEL_PROBABILITY=raw_prob,
        MODEL_CLASS=model_class,
        FINAL_PREDICTION=prediction_label,
        CONFIDENCE=confidence,
        RISK_LEVEL=risk_level,
    )

    _write_prediction_trace({
        "url": url,
        "is_unreachable": False,
        "normal_inference": True,
        "model_file": "fnn_phase2_v2.keras",
        "scaler_file": "scaler_phase2_v2.pkl",
        "top20_file": "top20_features.pkl",
        "raw_features": dict(zip(top20_features, input_df.iloc[0].tolist())),
        "scaled_features": X_scaled[0].tolist(),
        "raw_prob": raw_prob,
        "prediction": prediction_label
    }, response_payload)

    return jsonify(response_payload), 200

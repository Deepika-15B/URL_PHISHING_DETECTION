"""
phase3_hybrid_pipeline_demo.py
================================
Command-line demonstration for Submodule 3.5 — Unified HTML & Hybrid Feature
Extraction Pipeline.

This script runs the unified pipeline on one or more URLs and prints a
structured extraction summary to the terminal.  It also writes a machine-
readable JSON file and the standard text report.

DESIGN INVARIANT
----------------
This script is for EXTRACTION and REPORTING only.  It does NOT call the
inference pipeline.  Prediction must continue to use the unchanged
inference.py → FNN flow with the original feature subset.

Usage — single URL
-------------------
    python phase3_hybrid_pipeline_demo.py https://example.com

Usage — batch file
-------------------
    python phase3_hybrid_pipeline_demo.py --batch-file urls.txt

Usage — URL features only (no browser)
----------------------------------------
    python phase3_hybrid_pipeline_demo.py https://example.com --no-html

Options
-------
    --timeout-ms INT      Navigation timeout in ms  (default: 30000)
    --no-html             Skip HTML rendering (URL features only)
    --no-url              Skip URL network extraction (HTML only)
    --report-path PATH    Override default report output path
    --json-path PATH      Override default JSON output path
    --batch-file PATH     Path to a plain-text file with one URL per line
    --quiet               Suppress per-feature terminal output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Ensure project root is on sys.path when run directly ─────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils.unified_feature_pipeline import (
    HybridResult,
    UnifiedFeaturePipeline,
    generate_unified_report,
    load_hybrid_schema,
)

_SECTION = "=" * 68
_SUBSECTION = "-" * 68


def _print_result(result: HybridResult, quiet: bool = False) -> None:
    """Pretty-print a single HybridResult to stdout."""
    m = result.metadata
    url_ok = m.get("url_extraction_ok", False)
    html_ok = m.get("html_extraction_ok", False)

    print(_SECTION)
    print(f"  URL: {result.url}")
    print(_SECTION)
    print(f"  Timestamp            : {m.get('timestamp', 'N/A')}")
    print(f"  Total elapsed        : {m.get('total_extraction_secs', 0):.3f}s")
    print()

    # ── URL Feature Summary ────────────────────────────────────────────────
    print(f"  URL FEATURES  {'[OK]' if url_ok else '[FAILED]'}  "
          f"({m.get('url_extraction_secs', 0):.3f}s)")
    print(_SUBSECTION)
    if url_ok and not quiet:
        url_feats = result.url_features
        print(f"  Total URL features extracted : {len(url_feats)}")
        # Show a concise grouped summary rather than all 101 columns
        groups = {
            "URL structure": [k for k in url_feats if k.endswith("_url")],
            "Domain":        [k for k in url_feats if k.endswith("_domain")],
            "Directory":     [k for k in url_feats if k.endswith("_directory")],
            "File":          [k for k in url_feats if k.endswith("_file")],
            "Params":        [k for k in url_feats if k.endswith("_params")],
            "Security/DNS":  [k for k in url_feats if k in (
                "tls_ssl_certificate", "qty_ip_resolved", "qty_nameservers",
                "qty_mx_servers", "domain_spf", "ttl_hostname",
            )],
            "WHOIS":         [k for k in url_feats if "domain_activ" in k or "domain_expir" in k],
            "HTTP/Network":  [k for k in url_feats if k in (
                "time_response", "qty_redirects", "asn_ip",
            )],
        }
        for group_name, keys in groups.items():
            if keys:
                print(f"\n  [{group_name}]")
                for key in sorted(keys)[:8]:   # cap at 8 per group for readability
                    print(f"    {key:<40}: {url_feats[key]}")
                if len(keys) > 8:
                    print(f"    ... (+{len(keys) - 8} more)")
    elif not url_ok:
        for note in m.get("url_status", [])[:5]:
            print(f"    [!] {note}")
    else:
        print(f"  Feature count : {m.get('url_feature_count', 0)}")

    print()

    # ── HTML Security Analysis ─────────────────────────────────────────────
    print(f"  HTML SECURITY ANALYSIS  {'[OK]' if html_ok else '[FAILED/SKIPPED]'}  "
          f"({m.get('html_extraction_secs', 0):.3f}s)")
    print(_SUBSECTION)

    page_title = result.html_diagnostics.get("page_title", "")
    meta_desc  = result.html_diagnostics.get("meta_description", "")
    if page_title:
        print(f"  Page Title       : {page_title[:80]}")
    if meta_desc:
        print(f"  Meta Description : {meta_desc[:80]}")

    if html_ok and not quiet:
        h = result.html_features
        print()
        print("  [Basic Structure]")
        for k in ["number_of_forms", "number_of_images", "number_of_javascript_files",
                  "number_of_css_files", "number_of_hyperlinks"]:
            print(f"    {k:<40}: {h.get(k, 0)}")

        print()
        print("  [Form & Credential Signals]")
        for k in ["num_forms", "num_password_inputs", "has_external_form_action",
                  "has_empty_or_blank_action", "has_relative_form_action",
                  "num_hidden_inputs", "num_submit_inputs",
                  "has_external_action_password_form"]:
            val = h.get(k, 0)
            flag = "  ⚠" if (k.startswith("has_") and val == 1) else ""
            print(f"    {k:<40}: {val}{flag}")

        print()
        print("  [Link & Anchor Analysis]")
        for k in ["num_links", "num_external_links", "num_internal_links",
                  "num_null_self_links", "ratio_external_links",
                  "num_suspicious_anchor_text", "has_mismatch_link_text"]:
            val = h.get(k, 0)
            flag = "  ⚠" if (k.startswith("has_") and val == 1) else ""
            print(f"    {k:<40}: {val}{flag}")

        print()
        print("  [Anti-Analysis & Obfuscation]")
        for k in ["has_right_click_disabled", "has_text_selection_disabled",
                  "num_iframes", "num_hidden_iframes",
                  "has_popup_script", "has_obfuscated_js"]:
            val = h.get(k, 0)
            flag = "  ⚠" if (k.startswith("has_") and val == 1) else ""
            print(f"    {k:<40}: {val}{flag}")

        print()
        print("  [Metadata & DOM Structure]")
        for k in ["has_external_favicon", "has_meta_refresh", "title_matches_domain",
                  "num_meta_tags", "dom_depth", "num_total_dom_elements"]:
            val = h.get(k, 0)
            flag = "  ⚠" if (k in ("has_external_favicon", "has_meta_refresh") and val == 1) else ""
            print(f"    {k:<40}: {val}{flag}")

    elif not html_ok:
        for note in m.get("html_status", [])[:3]:
            print(f"    [!] {note}")
    else:
        print(f"  Feature count : {m.get('html_feature_count', 0)}")

    print()


def _progress(idx: int, total: int, result: HybridResult) -> None:
    u = "OK" if result.metadata.get("url_extraction_ok") else "FAIL"
    h = "OK" if result.metadata.get("html_extraction_ok") else "FAIL"
    t = result.metadata.get("total_extraction_secs", 0)
    print(f"  [{idx:>3}/{total}]  URL={u}  HTML={h}  {t:.2f}s  {result.url}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="phase3_hybrid_pipeline_demo",
        description="Submodule 3.5 — Unified HTML & Hybrid Feature Extraction Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url", nargs="?", help="Single absolute HTTP/HTTPS URL to analyse")
    parser.add_argument("--batch-file", metavar="PATH",
                        help="Plain-text file with one URL per line (batch mode)")
    parser.add_argument("--timeout-ms", type=int, default=30_000,
                        help="Browser navigation timeout in ms (default: 30000)")
    parser.add_argument("--no-html", action="store_true",
                        help="Skip HTML rendering; extract URL features only")
    parser.add_argument("--no-url", action="store_true",
                        help="Skip URL network extraction; render HTML only")
    parser.add_argument("--report-path", metavar="PATH",
                        help="Override report output path")
    parser.add_argument("--json-path", metavar="PATH",
                        help="Write extraction output as JSON to this path")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-feature terminal output")
    args = parser.parse_args()

    if not args.url and not args.batch_file:
        parser.error("Provide a URL positional argument or --batch-file.")

    # ── Collect URLs ───────────────────────────────────────────────────────
    urls: list[str] = []
    if args.batch_file:
        batch_path = Path(args.batch_file)
        if not batch_path.exists():
            print(f"[ERROR] Batch file not found: {batch_path}", file=sys.stderr)
            return 1
        urls = [line.strip() for line in batch_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")]
        if not urls:
            print("[ERROR] Batch file contains no valid URLs.", file=sys.stderr)
            return 1
    if args.url:
        urls.insert(0, args.url)

    # ── Schema summary ─────────────────────────────────────────────────────
    try:
        schema = load_hybrid_schema()
        print()
        print(_SECTION)
        print("  SUBMODULE 3.5 — Unified HTML & Hybrid Feature Extraction Pipeline")
        print(_SECTION)
        print(f"  Current Deployment (URL features only)")
        print(f"    URL feature schema     : {schema.url_feature_count} columns  →  FNN / DNN / Wide&Deep / TabNet")
        print(f"    (from models/preprocessed_feature_names.pkl)")
        print()
        print(f"  Future Hybrid Research Dataset")
        print(f"    URL numeric features   : {schema.url_feature_count} signals")
        print(f"    HTML numeric signals   : {schema.html_numeric_count} signals  (un-scaled, stored separately)")
        print(f"    Unified Feature Record : {schema.total_numeric_count} total  (NOT an inference vector)")
        print()
        print(f"  HTML rendering         : {'enabled' if not args.no_html else 'disabled'}")
        print(f"  URL extraction         : {'enabled' if not args.no_url else 'disabled'}")
        print(f"  URLs to process        : {len(urls)}")
        print(_SECTION)
        print()
    except FileNotFoundError:
        print("[NOTE] preprocessed_feature_names.pkl not found — URL schema unavailable.")
        print()

    # ── Run pipeline ───────────────────────────────────────────────────────
    is_batch = len(urls) > 1

    with UnifiedFeaturePipeline(
        timeout_ms=args.timeout_ms,
        enable_html=not args.no_html,
        enable_url=not args.no_url,
    ) as pipeline:
        if is_batch:
            print(f"  Processing {len(urls)} URLs (warm browser reuse enabled)...")
            print()
            results = pipeline.extract_batch(urls, on_progress=_progress if not args.quiet else None)
        else:
            results = [pipeline.extract(urls[0])]

    # ── Print results ──────────────────────────────────────────────────────
    if not is_batch or not args.quiet:
        for result in results:
            _print_result(result, quiet=args.quiet)

    # ── Generate report ────────────────────────────────────────────────────
    report_path = Path(args.report_path) if args.report_path else None
    saved_report = generate_unified_report(results, report_path=report_path)
    print(f"  Report written  → {saved_report}")

    # ── JSON export ────────────────────────────────────────────────────────
    if args.json_path:
        if len(results) == 1:
            results[0].export_json(args.json_path)
        else:
            import json
            payload = [r.to_dict() for r in results]
            out = Path(args.json_path).resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  JSON exported   → {args.json_path}")

    # ── Batch summary ──────────────────────────────────────────────────────
    if is_batch:
        url_ok  = sum(1 for r in results if r.metadata.get("url_extraction_ok"))
        html_ok = sum(1 for r in results if r.metadata.get("html_extraction_ok"))
        total_t = sum(r.metadata.get("total_extraction_secs", 0) for r in results)
        print()
        print(_SECTION)
        print(f"  BATCH SUMMARY — {len(results)} URLs")
        print(_SUBSECTION)
        print(f"  URL  extraction  : {url_ok}/{len(results)} succeeded")
        print(f"  HTML extraction  : {html_ok}/{len(results)} succeeded")
        print(f"  Total time       : {total_t:.2f}s  (avg {total_t/len(results):.2f}s/URL)")
        print(_SECTION)

    print()
    print("  [DESIGN NOTE] Prediction unchanged — the existing FNN/DNN/TabNet/Wide&Deep")
    print("  models continue to use their original trained feature subsets via inference.py.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

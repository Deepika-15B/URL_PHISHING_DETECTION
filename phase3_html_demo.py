"""Command-line demonstration for the standalone Phase 3 HTML extractor."""
from __future__ import annotations

import argparse

from utils.html_feature_extractor import HTMLFeatureExtractionError, HTMLFeatureExtractor


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a URL and show its basic HTML information.")
    parser.add_argument("url", help="Absolute HTTP or HTTPS URL to render")
    parser.add_argument("--timeout-ms", type=int, default=30_000, help="Navigation timeout (default: 30000)")
    args = parser.parse_args()
    try:
        with HTMLFeatureExtractor(timeout_ms=args.timeout_ms) as extractor:
            html = extractor.fetch_rendered_html(args.url)
            soup = extractor.parse_html(html)
            info = extractor.extract_basic_information(soup)
            form_info = extractor.extract_form_features(soup, page_url=args.url)
            link_info = extractor.extract_link_features(soup, page_url=args.url)
            security_info = extractor.extract_security_script_features(soup)
            meta_dom_info = extractor.extract_metadata_dom_features(soup, page_url=args.url)
            all_html_features = extractor.extract_all_html_features(soup, page_url=args.url)
    except HTMLFeatureExtractionError as error:
        parser.error(str(error))
        return 2

    print("=" * 60)
    print("PAGE BASIC INFORMATION")
    print("=" * 60)
    basic_labels = {
        "page_title": "Page Title", "meta_description": "Meta Description",
        "number_of_forms": "Number of Forms", "number_of_images": "Number of Images",
        "number_of_javascript_files": "Number of JavaScript Files", "number_of_css_files": "Number of CSS Files",
        "number_of_hyperlinks": "Number of Hyperlinks",
    }
    for key, label in basic_labels.items():
        print(f"  {label:<30}: {info[key]}")

    print("\n" + "=" * 60)
    print("FORM & CREDENTIAL SECURITY FEATURES")
    print("=" * 60)
    form_labels = {
        "num_forms": "Total Forms Count",
        "num_password_inputs": "Password Inputs Count",
        "has_external_form_action": "Has External Form Action",
        "has_empty_or_blank_action": "Has Empty/Blank Action",
        "has_relative_form_action": "Has Relative Form Action",
        "num_hidden_inputs": "Hidden Inputs Count",
        "num_text_inputs": "Text/Email Inputs Count",
        "num_submit_inputs": "Submit Inputs/Buttons Count",
        "has_external_action_password_form": "Has External Action on Password Form",
    }
    for key, label in form_labels.items():
        print(f"  {label:<35}: {form_info[key]}")

    print("\n" + "=" * 60)
    print("LINK & ANCHOR ANALYSIS FEATURES")
    print("=" * 60)
    link_labels = {
        "num_links": "Total Links Count",
        "num_external_links": "External Links Count",
        "num_internal_links": "Internal Links Count",
        "num_null_self_links": "Null/Self (#, javascript:) Count",
        "ratio_external_links": "Ratio External Links",
        "ratio_internal_links": "Ratio Internal Links",
        "ratio_null_self_links": "Ratio Null/Self Links",
        "num_suspicious_anchor_text": "Suspicious Anchor Text Count",
        "has_mismatch_link_text": "Has Link Text/URL Domain Mismatch",
    }
    for key, label in link_labels.items():
        print(f"  {label:<35}: {link_info[key]}")

    print("\n" + "=" * 60)
    print("SECURITY & SCRIPT SIGNALS")
    print("=" * 60)
    sec_labels = {
        "has_right_click_disabled": "Has Right Click Disabled",
        "has_text_selection_disabled": "Has Text Selection Disabled",
        "num_iframes": "Total iFrames Count",
        "num_hidden_iframes": "Hidden iFrames Count",
        "has_popup_script": "Has Popup Script Detected",
        "has_obfuscated_js": "Has Obfuscated JS Detected",
    }
    for key, label in sec_labels.items():
        print(f"  {label:<35}: {security_info[key]}")

    print("\n" + "=" * 60)
    print("METADATA & DOM STRUCTURE FEATURES")
    print("=" * 60)
    meta_labels = {
        "has_external_favicon": "Has External Favicon",
        "has_meta_refresh": "Has Meta Refresh Auto-Redirect",
        "title_matches_domain": "Title Contains Brand Domain",
        "num_meta_tags": "Total Meta Tags Count",
        "dom_depth": "Maximum DOM Tree Depth",
        "num_total_dom_elements": "Total DOM Elements Count",
    }
    for key, label in meta_labels.items():
        print(f"  {label:<35}: {meta_dom_info[key]}")

    print("\n" + "=" * 60)
    print(f"CONSOLIDATED HTML FEATURE VECTOR ({len(all_html_features)} total signals)")
    print("=" * 60)
    print("  Feature Keys:", list(all_html_features.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json
import logging
import time
import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")

DOMAINS = [
    "https://www.youtube.com",
    "https://stackoverflow.com",
    "https://www.google.com",
    "https://github.com",
    "https://www.microsoft.com",
    "https://www.amazon.in",
    "https://www.hdfcbank.com",
    "https://www.irctc.co.in",
    "https://www.onlinesbi.sbi",
    "https://www.tn.gov.in",
    "https://kongu.ac.in",
]

API_URL = "http://localhost:5000/predict"

results_table = []

def test_domain(url: str):
    logging.info(f"Testing {url} ...")
    try:
        response = requests.post(API_URL, json={"url": url}, timeout=60)
        data = response.json()
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return

    print("=" * 80)
    print(f"URL: {url}")
    print("=" * 80)
    
    sys_info = data.get("system_info", {})
    url_summary = data.get("url_feature_summary", {})
    top20 = data.get("top20_features", {})
    html_report = data.get("html_security_report", {}).get("metadata", {})
    
    # Check if FNN was executed
    # FNN is executed if status is "success" and top20 features are populated (non-empty)
    fnn_executed = "YES" if data.get("status") == "success" and top20 else "NO"
    
    # Save table row info
    row = {
        "URL": url,
        "status": data.get("status", "N/A"),
        "page_title": html_report.get("page_title", "N/A") if html_report else "N/A",
        "html_extraction_status": sys_info.get("html_extraction_status", "N/A") if sys_info else "N/A",
        "URLSimilarityIndex": top20.get("URLSimilarityIndex", "N/A") if fnn_executed == "YES" else "N/A",
        "DomainTitleMatchScore": top20.get("DomainTitleMatchScore", "N/A") if fnn_executed == "YES" else "N/A",
        "fnn_executed": fnn_executed,
        "prediction": data.get("prediction", "N/A"),
        "confidence": f"{data.get('confidence', 0)}%",
        "reason": ", ".join(data.get("reason", [])) if data.get("reason") else "N/A"
    }
    results_table.append(row)

    # Required outputs
    print(f"normalized domain         : {url_summary.get('domain', 'N/A')}")
    print(f"URL extraction status     : {sys_info.get('url_extraction_status', 'N/A')}")
    print(f"DNS status                : {sys_info.get('dns_extraction_status', 'N/A')}")
    print(f"WHOIS status              : {sys_info.get('whois_extraction_status', 'N/A')}")
    print(f"SSL status                : {url_summary.get('tls_ssl_certificate', 'N/A')}")
    print(f"HTML extraction status    : {sys_info.get('html_extraction_status', 'N/A')}")
    print(f"page title                : {html_report.get('page_title', 'N/A') if html_report else 'N/A'}")
    
    if data.get("status") in ["PARTIAL_EXTRACTION", "BOT_PROTECTION_PAGE", "unreachable"]:
        print(f"\nFinal status              : {data.get('status')}")
        print(f"Prediction                : {data.get('prediction')}")
        print(f"Confidence                : {data.get('confidence')}%")
        print(f"Reason                    : {data.get('reason', [])}")
        print("\n")
        return

    print(f"DomainTitleMatchScore     : {top20.get('DomainTitleMatchScore', 'N/A')}")
    print(f"URLSimilarityIndex        : {top20.get('URLSimilarityIndex', 'N/A')}")
    
    print("\n--- Top20 Features ---")
    for k, v in top20.items():
        print(f"{k:30}: {v}")
    
    print("\n--- Final Prediction ---")
    print(f"Final status              : {data.get('status')}")
    print(f"Prediction                : {data.get('prediction')}")
    print(f"Confidence                : {data.get('confidence')}%")
    print("\n")
    
    # Store complete raw JSON for the user to review
    domain_name = url_summary.get('domain') or "domain"
    with open(f"reports/predictions/{domain_name}.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    import os
    os.makedirs("reports/predictions", exist_ok=True)
    
    # Test sequentially to avoid overloading the local Playwright browser
    for url in DOMAINS:
        test_domain(url)

    # Print summary table in Markdown format at the end
    print("\n" + "="*80)
    print("CONCISE SUMMARY TABLE")
    print("="*80)
    headers = ["URL", "status", "page_title", "HTML status", "URLSim", "DomTitle", "FNN Executed?", "prediction", "confidence", "reason"]
    # Adjust widths for readable console output
    row_format = "{:<26} | {:<18} | {:<25} | {:<11} | {:<6} | {:<8} | {:<13} | {:<10} | {:<10} | {:<30}"
    print(row_format.format(*headers))
    print("-" * 175)
    for r in results_table:
        # truncate title/reason to fit cleanly
        title = r["page_title"]
        if len(title) > 22:
            title = title[:22] + "..."
        reason = r["reason"]
        if len(reason) > 27:
            reason = reason[:27] + "..."
        print(row_format.format(
            r["URL"][:26],
            r["status"][:18],
            title,
            r["html_extraction_status"][:11],
            str(r["URLSimilarityIndex"])[:6],
            str(r["DomainTitleMatchScore"])[:8],
            r["fnn_executed"],
            r["prediction"][:10],
            r["confidence"][:10],
            reason
        ))
    print("="*175)

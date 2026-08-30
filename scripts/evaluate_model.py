"""
Re-run evaluation for remaining/failed URLs.
Deletes previously failed results and re-runs them.
Also runs phishing URLs that haven't been processed.
"""
import sys, os, time, json, urllib.request, urllib.error
from urllib.parse import urlparse
from pathlib import Path

PROJECT_ROOT = Path("E:/phishing_project_ieee/phishing_detection_ieee")
PREDICTIONS_DIR = PROJECT_ROOT / "reports" / "predictions"
API_URL = "http://127.0.0.1:5000/predict"

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Delete previously failed/timed-out results so they get re-tried
for f in PREDICTIONS_DIR.glob("*.json"):
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        status = data.get("status", "")
        if status in ("unreachable", "failed_http", "failed"):
            print(f"Deleting failed result: {f.name}")
            f.unlink()
    except:
        pass

LEGIT_CSV = PROJECT_ROOT / "data" / "evaluation" / "legitimate.csv"
PHISH_CSV = PROJECT_ROOT / "data" / "evaluation" / "live_phishing.csv"

def get_domain(url):
    parsed = urlparse(url)
    domain = parsed.netloc if parsed.netloc else url.replace("https://", "").replace("http://", "")
    domain = domain.split("/")[0].replace("www.", "")
    return domain

def evaluate_urls(csv_file, label_type):
    if not csv_file.exists():
        print(f"Skipping {label_type}, file not found: {csv_file}")
        return
    with open(csv_file, "r", encoding="utf-8-sig") as f:
        urls = [line.strip() for line in f if line.strip()]
    print(f"\nEvaluating {len(urls)} {label_type} URLs...")
    for url in urls:
        domain = get_domain(url)
        output_file = PREDICTIONS_DIR / f"{domain}.json"
        if output_file.exists():
            continue
        print(f"  Processing {url} ...")
        payload = json.dumps({"url": url}).encode("utf-8")
        req = urllib.request.Request(API_URL, data=payload, headers={'Content-Type': 'application/json'})
        try:
            response = urllib.request.urlopen(req, timeout=90)
            data = json.loads(response.read().decode('utf-8'))
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(data, out_f, indent=2)
            print(f"    -> Saved: {output_file.name} | Prediction: {data.get('prediction', 'N/A')}")
        except urllib.error.HTTPError as e:
            err_data = e.read().decode('utf-8')
            print(f"    -> HTTP Error {e.code}")
            try:
                err_json = json.loads(err_data)
            except:
                err_json = {"error": err_data, "url": url, "status": "failed_http"}
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump(err_json, out_f, indent=2)
        except Exception as e:
            print(f"    -> Failed: {e}")
            with open(output_file, "w", encoding="utf-8") as out_f:
                json.dump({"error": str(e), "url": url, "status": "unreachable"}, out_f, indent=2)

if __name__ == "__main__":
    time.sleep(3)  # Wait for Flask server
    print("=" * 60)
    print("BATCH EVALUATION - RETRY FAILED + PHISHING")
    print("=" * 60)
    evaluate_urls(LEGIT_CSV, "Legitimate")
    evaluate_urls(PHISH_CSV, "Live Phishing")
    print("\nDONE.")

import os
import sys
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app import create_app

def main():
    print("Initializing Flask test client...")
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    url_to_test = "https://thisdomain-does-not-exist-999.com"
    print(f"Sending POST request to /predict for {url_to_test}...")
    
    response = client.post(
        "/predict",
        json={"url": url_to_test}
    )
    
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(response.get_json())
    
    debug_file = _PROJECT_ROOT / "reports" / "flask_playwright_debug.txt"
    if debug_file.exists():
        print(f"\n--- {debug_file.name} ---")
        print(debug_file.read_text(encoding="utf-8"))
    else:
        print(f"\n{debug_file.name} was not generated.")

if __name__ == "__main__":
    main()

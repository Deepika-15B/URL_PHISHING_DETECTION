"""Flask REST API for the IEEE Phishing URL Detection project.

This backend contains only JSON endpoints. It delegates feature extraction and
prediction to the existing utility modules and never retrains or modifies them.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, current_app, jsonify, request
from flask_cors import CORS

# Support both ``python backend/app.py`` and package-based imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.inference import load_artifacts, load_model, predict
from utils.url_feature_extractor import extract_all_features

_LOGS_DIR = _PROJECT_ROOT / "logs"
_LOG_FILE = _LOGS_DIR / "app.log"


def _configure_logging(app: Flask) -> None:
    """Create the application log directory and install one rotating file handler."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(_LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    app.logger.setLevel(logging.INFO)
    app.logger.handlers.clear()
    app.logger.addHandler(handler)
    app.logger.propagate = False


def validate_request() -> tuple[str | None, tuple[dict, int] | None]:
    """Validate /predict JSON body and return a normalized HTTP(S) URL or JSON error."""
    if not request.is_json:
        return None, ({"error": "Invalid JSON: Content-Type must be application/json."}, 400)
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return None, ({"error": "Invalid JSON request body."}, 400)
    url = payload.get("url")
    if not isinstance(url, str) or not url.strip():
        return None, ({"error": "Missing URL: provide a non-empty 'url' field."}, 400)

    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None, ({"error": "Invalid URL: provide an absolute http:// or https:// URL."}, 400)
    return url, None


def log_prediction(url: str, result: dict, processing_time: float) -> None:
    """Record the requested audit fields for each successful API prediction."""
    current_app.logger.info(
        "url=%s | prediction=%s | probability=%.4f | confidence=%.2f | processing_time=%.3fs",
        url, result["prediction"], result["probability"], result["confidence"], processing_time,
    )


def predict_url(url: str, app: Flask) -> dict:
    """Extract the ordered 101-feature row and obtain its existing FNN prediction."""
    try:
        feature_frame = extract_all_features(url)
    except Exception as error:
        app.logger.exception("Feature extraction failed for url=%s", url)
        raise RuntimeError(f"Feature Extraction Error: {error}") from error

    try:
        result = predict(
            feature_frame,
            artifacts=app.config["INFERENCE_ARTIFACTS"],
            model=app.config["INFERENCE_MODEL"],
        )
    except Exception as error:
        app.logger.exception("Inference failed for url=%s", url)
        raise RuntimeError(f"Inference Error: {error}") from error

    if not isinstance(result, dict):
        raise RuntimeError("Inference Error: expected one prediction dictionary for one URL.")
    return result


def home():
    """Return backend health metadata for frontend or deployment checks."""
    return jsonify({"project": "IEEE Phishing URL Detection", "status": "Backend Running", "version": "1.0"})


def create_app() -> Flask:
    """Create a CORS-enabled Flask API and load immutable inference artifacts once."""
    app = Flask(__name__)
    CORS(app)
    _configure_logging(app)

    # Loading once at startup avoids model deserialization for every prediction.
    try:
        app.config["INFERENCE_MODEL"] = load_model()
        app.config["INFERENCE_ARTIFACTS"] = load_artifacts()
    except Exception as error:
        app.logger.exception("Backend startup failed while loading inference artifacts")
        raise RuntimeError(f"Unable to initialize inference artifacts: {error}") from error

    app.add_url_rule("/", view_func=home, methods=["GET"])

    @app.post("/predict")
    def predict_route():
        url, error_response = validate_request()
        if error_response:
            return jsonify(error_response[0]), error_response[1]

        started = time.perf_counter()
        try:
            result = predict_url(url, app)
            processing_time = time.perf_counter() - started
            log_prediction(url, result, processing_time)
            return jsonify({"url": url, **result})
        except RuntimeError as error:
            message = str(error)
            status = 500
            return jsonify({"error": message}), status
        except Exception as error:  # Defensive final boundary for JSON API safety.
            app.logger.exception("Unexpected error for url=%s", url)
            return jsonify({"error": f"Unexpected Error: {error}"}), 500

    return app


def _verify_routes(app: Flask) -> None:
    """Exercise GET / and one real POST /predict without starting a public server."""
    with app.test_client() as client:
        home_response = client.get("/")
        if home_response.status_code != 200:
            raise RuntimeError(f"GET / verification failed: {home_response.status_code}")
        predict_response = client.post("/predict", json={"url": "https://example.com"})
        if predict_response.status_code != 200:
            raise RuntimeError(f"POST /predict verification failed: {predict_response.get_json()}")
    if not _LOG_FILE.exists():
        raise RuntimeError("Logging verification failed: logs/app.log was not created.")
    print("Backend Started Successfully")
    print("Routes Verified")
    print("Prediction Successful")
    print("Logs Created")


if __name__ == "__main__":
    application = create_app()
    _verify_routes(application)
    application.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)


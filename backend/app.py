"""Flask REST API for the IEEE Phishing URL Detection project.

Provides web dashboard UI (GET /) and JSON prediction endpoint (POST /predict).
Delegates feature extraction and prediction to the modular utility modules.
"""
from __future__ import annotations

import logging
import os
import pickle
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlparse

import tensorflow as tf
from flask import Flask, current_app, jsonify, render_template, render_template_string, request, send_from_directory
from flask_cors import CORS

# Support both ``python backend/app.py`` and package-based imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.routes.predict import predict_bp

_LOGS_DIR = _PROJECT_ROOT / "logs"
_LOG_FILE = _LOGS_DIR / "app.log"
_REPORTS_DIR = _PROJECT_ROOT / "reports"
_TEMPLATES_DIR = _PROJECT_ROOT / "templates"
_BACKEND_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


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


def create_app() -> Flask:
    """Create Flask application and pre-load model artifacts."""
    template_folder = str(_TEMPLATES_DIR) if _TEMPLATES_DIR.exists() else str(_BACKEND_TEMPLATES_DIR)
    app = Flask(__name__, template_folder=template_folder, static_folder=str(_PROJECT_ROOT / "backend" / "static"))
    CORS(app)
    _configure_logging(app)

    # Pre-load Phase 2 v2 artifacts once at startup
    models_dir = _PROJECT_ROOT / "models"
    try:
        with open(models_dir / "top20_features.pkl", "rb") as f:
            top20_feats = pickle.load(f)
        with open(models_dir / "scaler_phase2_v2.pkl", "rb") as f:
            scaler = pickle.load(f)
        model = tf.keras.models.load_model(models_dir / "fnn_phase2_v2.keras")

        app.config["TOP20_FEATURES"] = top20_feats
        app.config["INFERENCE_SCALER"] = scaler
        app.config["INFERENCE_MODEL"] = model
        app.logger.info("Loaded Phase 2 v2 model artifacts successfully.")
    except Exception as exc:
        app.logger.warning("Could not pre-load Phase 2 v2 artifacts: %s", exc)

    # Register routes blueprint
    app.register_blueprint(predict_bp)

    @app.get("/")
    def index():
        if request.headers.get("Accept") == "application/json":
            return jsonify({"project": "IEEE Phishing URL Detection", "status": "Backend Running", "version": "1.0"})
        index_path = _TEMPLATES_DIR / "index.html"
        if not index_path.exists():
            index_path = _BACKEND_TEMPLATES_DIR / "index.html"
        if index_path.exists():
            return index_path.read_text(encoding="utf-8"), 200, {"Content-Type": "text/html"}
        return jsonify({"project": "IEEE Phishing URL Detection", "status": "Backend Running"}), 200

    @app.route("/reports/<path:filename>")
    def serve_report_file(filename):
        return send_from_directory(_REPORTS_DIR, filename)

    return app


if __name__ == "__main__":
    print(f"[FLASK SERVER STARTUP] Process PID: {os.getpid()}")
    application = create_app()
    application.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)

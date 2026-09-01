"""Regression tests for the Phase 2 v2 sigmoid-label contract.

Phase 2 v2 trains with 0=Phishing and 1=Legitimate, so its sigmoid output is
P(Legitimate). These tests intentionally avoid live-network/model calls.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from flask import Flask

from backend.routes.predict import interpret_phase2_probability, predict_bp
from utils.html_feature_extractor import AntiBotProtectionError, HTMLFeatureExtractor


class _BotProtectedPipeline:
    """Raises before feature construction, as a real challenge page does."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def __enter__(self):
        raise AntiBotProtectionError("Challenge page", markers=["awswaf"])

    def __exit__(self, *_args) -> None:
        return None


class Phase2ProbabilityInterpretationTests(unittest.TestCase):
    def test_known_legitimate_probability_maps_to_label_one_and_low_risk(self) -> None:
        model_class, prediction, confidence, phishing_probability, risk = interpret_phase2_probability(0.9999952912330627)
        self.assertEqual((model_class, prediction, risk), (1, "Legitimate", "Low"))
        self.assertEqual(confidence, 100.0)
        self.assertAlmostEqual(phishing_probability, 0.0000047087669373)

    def test_known_phishing_probability_maps_to_label_zero_and_high_risk(self) -> None:
        model_class, prediction, confidence, phishing_probability, risk = interpret_phase2_probability(0.01)
        self.assertEqual((model_class, prediction, confidence, risk), (0, "Phishing", 99.0, "High"))
        self.assertAlmostEqual(phishing_probability, 0.99)

    def test_existing_suspicious_band_and_confidence_are_preserved(self) -> None:
        model_class, prediction, confidence, phishing_probability, risk = interpret_phase2_probability(0.50)
        self.assertEqual((model_class, prediction, confidence, risk), (1, "Suspicious", 50.0, "Medium"))
        self.assertEqual(phishing_probability, 0.50)

    def test_bot_protection_returns_unknown_without_model_inference(self) -> None:
        app = Flask(__name__)
        app.register_blueprint(predict_bp)
        with patch("utils.unified_feature_pipeline.UnifiedFeaturePipeline", _BotProtectedPipeline):
            response = app.test_client().post("/predict", json={"url": "https://www.amazon.in"})
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["prediction"], "Unknown")
        self.assertFalse(payload["model_predict_called"])
        self.assertTrue(payload["bot_protection_detected"])

    def test_browser_cleanup_does_not_require_a_console_handle(self) -> None:
        class _Browser:
            def close(self) -> None:
                pass

        extractor = HTMLFeatureExtractor()
        extractor._browser = _Browser()
        with patch("builtins.print", side_effect=OSError(22, "Invalid argument")):
            extractor.close_browser()

    def test_browser_cleanup_does_not_mask_an_oserror(self) -> None:
        class _Context:
            def close(self) -> None:
                raise OSError(22, "Invalid argument")

        extractor = HTMLFeatureExtractor()
        extractor._context = _Context()
        extractor.close_browser()
        self.assertIsNone(extractor._context)

    def test_context_creation_does_not_override_extra_http_headers(self) -> None:
        class _Context:
            def add_init_script(self, _script: str) -> None:
                pass

            def close(self) -> None:
                pass

        class _Browser:
            def __init__(self) -> None:
                self.context_kwargs = None

            def new_context(self, **kwargs):
                self.context_kwargs = kwargs
                return _Context()

            def close(self) -> None:
                pass

        class _Playwright:
            def __init__(self, browser) -> None:
                self.chromium = type("Chromium", (), {"launch": lambda _self, **_kwargs: browser})()

            def stop(self) -> None:
                pass

        browser = _Browser()
        playwright = _Playwright(browser)
        with patch("utils.html_feature_extractor.sync_playwright") as factory:
            factory.return_value.start.return_value = playwright
            extractor = HTMLFeatureExtractor()
            extractor.launch_browser()
        self.assertNotIn("extra_http_headers", browser.context_kwargs)
        extractor.close_browser()


if __name__ == "__main__":
    unittest.main()

"""Optional approved-artifact smoke test; synthetic observations, never Test data."""

import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.schemas.import_file import FEATURES
from app.services.inference import load_bundle
from main import create_app

ROOT = Path(
    os.environ.get("SEWS_AI_REPO_ROOT", Path(__file__).resolve().parents[2] / "AI")
)
HAS_ARTIFACTS = all(
    (ROOT / name).is_file()
    for name in (
        "models/lstm_full_pw10_best.pt",
        "models/lstm_preprocessing_stats.json",
        "reports/final_evaluation_policy_locked.json",
    )
)


class FrozenModelTests(unittest.TestCase):
    @unittest.skipUnless(
        HAS_ARTIFACTS,
        "Approved frozen weights/statistics are not present; live-model verification blocked",
    )
    def test_frozen_cpu_loader_and_synthetic_score(self):
        bundle = load_bundle(ROOT)
        rows = [
            {"ICULOS": float(hour), **{f: None for f in FEATURES}}
            for hour in range(1, 25)
        ]
        result = bundle.predict(rows)
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["score_threshold"], 0.2)
        self.assertEqual(result["alert"], result["risk_score"] >= 0.2)
        self.assertIsNone(result["calibrated_probability"])
        self.assertGreaterEqual(result["risk_score"], 0)
        self.assertLessEqual(result["risk_score"], 1)

    def test_incomplete_artifacts_degrade_real_loader_without_synthetic_fallback(self):
        # Real loader, deliberately invalid root. No stand-in predictor.
        with (
            tempfile.TemporaryDirectory() as temp,
            TestClient(
                create_app(Settings(Path(temp) / "db", ai_root=Path(temp)))
            ) as client,
        ):
            self.assertFalse(client.get("/health").json()["model_ready"])
            response = client.post(
                "/predict",
                json={"patient_id": "synthetic", "observations": [{"ICULOS": 1}]},
            )
            self.assertEqual(response.status_code, 503)
            self.assertEqual(client.get("/patients").json()["summary"]["total"], 0)


if __name__ == "__main__":
    unittest.main()

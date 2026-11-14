"""Contract tests: synthetic records only; no model or patient data is loaded."""

import unittest

from fastapi.testclient import TestClient

from main import app


def fake_predict(observations, model, stats):
    n = len(observations)
    return {
        "status": "scored" if n >= 24 else "insufficient_history",
        "risk_score": 0.25 if n >= 24 else None,
        "alert": n >= 24,
        "score_threshold": 0.2,
        "records_received": n,
        "minimum_records": 24,
        "calibrated_probability": None,
    }


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        # TestClient without context manager intentionally skips startup and
        # uses a deterministic fake. Live model tests use the actual API server.
        app.state.inference = (object(), {}, fake_predict)
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        app.state.inference = None

    @staticmethod
    def patient(patient_id, hours):
        return {
            "patient_id": patient_id,
            "observations": [{"ICULOS": float(t)} for t in range(1, hours + 1)],
        }

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["model_loaded"])

    def test_predict(self):
        r = self.client.post("/predict", json=self.patient("synthetic-1", 24))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["patient_id"], "synthetic-1")
        self.assertTrue(r.json()["alert"])
        self.assertIsNone(r.json()["calibrated_probability"])

    def test_short_history(self):
        r = self.client.post("/predict", json=self.patient("synthetic-1", 23))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "insufficient_history")
        self.assertIsNone(r.json()["risk_score"])

    def test_batch(self):
        r = self.client.post(
            "/predict_batch",
            json={"patients": [self.patient("a", 23), self.patient("b", 24)]},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()), 2)

    def test_duplicate_batch_id(self):
        r = self.client.post(
            "/predict_batch",
            json={"patients": [self.patient("same", 23), self.patient("same", 24)]},
        )
        self.assertEqual(r.status_code, 422)

    def test_missing_input(self):
        r = self.client.post("/predict", json={"patient_id": "a", "observations": []})
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()

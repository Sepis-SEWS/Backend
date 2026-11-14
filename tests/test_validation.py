"""Boundary tests for file validation, model lifecycle, and safe serving errors."""

import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.schemas.import_file import FEATURES, ImportFile
from app.services.inference import ModelBundle
from main import create_app


def file_payload():
    return {
        "encounter_id": "example",
        "patient_id": "synthetic",
        "bed": "B1",
        "assignee": "R1",
        "icu_id": "ICU-A",
        "admission_anchor": "2020-01-01T00:00:00+00:00",
        "observations": [{"ICULOS": 1, **{f: None for f in FEATURES}}],
    }


class ValidationTests(unittest.TestCase):
    def test_rejects_missing_features_bad_numbers_and_metadata(self):
        for feature_value in (True, "80", float("inf"), float("nan")):
            data = file_payload()
            data["observations"][0]["HR"] = feature_value
            with self.subTest(value=feature_value), self.assertRaises(ValidationError):
                ImportFile.model_validate(data)
        for key in ("HR", "ICULOS"):
            data = file_payload()
            del data["observations"][0][key]
            with self.assertRaises(ValidationError):
                ImportFile.model_validate(data)
        for key, value in (
            ("encounter_id", "slash/id"),
            ("bed", " "),
            ("assignee", ""),
            ("icu_id", "ICU-B"),
            ("unit_profile", "demo-units"),
        ):
            data = file_payload()
            data[key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError):
                ImportFile.model_validate(data)
        data = file_payload()
        data["observations"][0]["SepsisLabel"] = 0
        with self.assertRaises(ValidationError):
            ImportFile.model_validate(data)

    def test_hourly_cadence_and_time_exclusivity(self):
        for hour in (0, 0.5, 2, 1e300):
            data = file_payload()
            data["observations"][0]["ICULOS"] = hour
            with self.subTest(hour=hour), self.assertRaises(ValidationError):
                ImportFile.model_validate(data)
        data = file_payload()
        data["admission_anchor"] = "9999-12-31T23:00:00Z"
        with self.assertRaises(ValidationError):
            ImportFile.model_validate(data)
        data = file_payload()
        data["observations"][0]["observed_at"] = "2020-01-01T01:00:00Z"
        with self.assertRaises(ValidationError):
            ImportFile.model_validate(data)
        del data["admission_anchor"]
        self.assertEqual(ImportFile.model_validate(data).times()[0].hour, 1)
        for stamp in (None, "2020-01-01T01:00:00"):
            data["observations"][0]["observed_at"] = stamp
            with self.assertRaises(ValidationError):
                ImportFile.model_validate(data)
        data = file_payload()
        del data["admission_anchor"]
        data["observations"][0]["observed_at"] = "2020-01-01T10:00:00+09:00"
        second = copy.deepcopy(data["observations"][0])
        second.update(ICULOS=2, observed_at="2020-01-01T11:00:00+09:00")
        data["observations"].append(second)
        self.assertEqual(ImportFile.model_validate(data).times()[0].hour, 1)
        data["observations"][1]["observed_at"] = "2020-01-01T12:00:00+09:00"
        with self.assertRaises(ValidationError):
            ImportFile.model_validate(data)

    def test_loader_once_and_missing_model_is_not_fake(self):
        calls = []

        def load(root):
            calls.append(root)
            raise FileNotFoundError("private path must not appear")

        with tempfile.TemporaryDirectory() as temp:
            app = create_app(Settings(Path(temp) / "db"), load)
            with TestClient(app) as client:
                for _ in range(3):
                    response = client.get("/health")
                    self.assertEqual(response.json()["status"], "degraded")
                    self.assertFalse(response.json()["model_ready"])
                failed = client.post(
                    "/predict",
                    json={"patient_id": "synthetic", "observations": [{"ICULOS": 1}]},
                )
                self.assertEqual(failed.status_code, 503)
                self.assertNotIn("private", failed.text)
            self.assertEqual(len(calls), 1)
            self.assertIsNone(app.state.inference)

    def test_health_tracks_store_failure_without_losing_model_readiness(self):
        bundle = ModelBundle((object(), {}, lambda *args: None), "test", "test")
        with tempfile.TemporaryDirectory() as temp:
            app = create_app(Settings(Path(temp) / "db"), lambda root: bundle)
            with TestClient(app) as client:
                healthy = client.get("/health").json()
                self.assertEqual(healthy["status"], "ok")
                self.assertTrue(healthy["store_ready"])
                with patch.object(
                    app.state.store,
                    "connect",
                    side_effect=sqlite3.OperationalError("private-db-path"),
                ):
                    failed = client.get("/health")
                self.assertEqual(failed.status_code, 200)
                self.assertEqual(failed.json()["status"], "unavailable")
                self.assertTrue(failed.json()["model_ready"])
                self.assertFalse(failed.json()["store_ready"])
                self.assertNotIn("private-db-path", failed.text)
                self.assertEqual(client.get("/health").json()["status"], "ok")
                with app.state.store.connect(write=True) as connection:
                    connection.execute("DROP TABLE records")
                self.assertEqual(client.get("/health").json()["status"], "unavailable")
                self.assertTrue(client.get("/health").json()["model_ready"])

    def test_serving_input_exception_does_not_reflect_data(self):
        def predict(*args):
            raise ValueError("sensitive-observation")

        with tempfile.TemporaryDirectory() as temp:
            bundle = ModelBundle((object(), {}, predict), "test", "test")
            with TestClient(
                create_app(Settings(Path(temp) / "db"), lambda root: bundle)
            ) as client:
                response = client.post(
                    "/predict",
                    json={"patient_id": "synthetic", "observations": [{"ICULOS": 1}]},
                )
                self.assertEqual(response.status_code, 422)
                self.assertNotIn("sensitive-observation", response.text)
                nonfinite = client.post(
                    "/predict",
                    content='{"patient_id":"synthetic","observations":[{"ICULOS":Infinity}]}',
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(nonfinite.status_code, 422)


if __name__ == "__main__":
    unittest.main()

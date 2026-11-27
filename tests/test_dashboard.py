"""Synthetic API integration; the actual frontend consumer validates responses."""

import base64
import json
import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.config import Settings
from app.schemas.import_file import FEATURES, ImportFile
from app.services.import_run import import_encounter
from app.services.inference import ModelBundle
from app.store.sqlite import ALERT_COLUMNS, SCHEMA
from main import create_app

FRONTEND = Path(__file__).resolve().parents[2] / "Frontend"
sys.path.insert(0, str(FRONTEND))
from config import Settings as FrontendSettings
from domain.schemas import (
    Detail,
    Health,
    Listing,
    ManualAlertDraft,
    PatientSummary,
    RecordDraft,
)
from services.api_client import ApiGateway
from services.gateway import ServiceError as FrontendError

sys.path.remove(str(FRONTEND))


def fake_predict(rows, model, stats):
    n = len(rows)
    score = None if n < 24 else 0.25
    return {
        "status": "insufficient_history" if n < 24 else "scored",
        "risk_score": score,
        "alert": n >= 24,
        "score_threshold": 0.2,
        "records_received": n,
        "minimum_records": 24,
        "calibrated_probability": None,
    }


def payload(identifier="episode-a", count=26):
    return {
        "encounter_id": identifier,
        "patient_id": "source-person",
        "bed": "Bed-1",
        "assignee": "Researcher",
        "icu_id": "ICU-A",
        "admission_anchor": "2020-01-01T00:00:00Z",
        "observations": [
            {"ICULOS": hour, **{f: (80.0 if f == "HR" else None) for f in FEATURES}}
            for hour in range(1, count + 1)
        ],
    }


def unavailable(root):
    raise RuntimeError("no local artifacts")


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(database=Path(self.temp.name) / "store.sqlite")
        self.bundle = ModelBundle(
            (object(), {}, fake_predict), "test-checkpoint", "test-policy"
        )
        self.app = create_app(self.settings, model_loader=lambda root: self.bundle)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.addCleanup(self.temp.cleanup)
        import_encounter(
            ImportFile.model_validate(payload()), self.app.state.store, self.bundle
        )

    def frontend_gateway(self):
        def handle(request):
            response = self.client.request(
                request.method,
                str(request.url),
                headers=request.headers,
                content=request.content,
            )
            return httpx.Response(
                response.status_code, headers=response.headers, content=response.content
            )

        return ApiGateway(
            FrontendSettings(mode="api", api_base_url="http://testserver"),
            transport=httpx.MockTransport(handle),
        )

    def test_real_frontend_gateway_workflow_and_restart(self):
        gateway = self.frontend_gateway()
        self.assertTrue(gateway.get_health().model_ready)
        patients = gateway.list_patients({})
        self.assertEqual(patients.summary.high, 1)
        self.assertEqual(patients.summary.unacknowledged_alert_count, 3)
        detail = gateway.get_patient("episode-a")
        self.assertEqual(detail.item.id, "episode-a")
        self.assertTrue(detail.item.data_delayed)
        self.assertEqual(detail.item.delay_reason, "replay_historical")
        self.assertIsNone(detail.item.sofa.total)
        self.assertFalse(detail.item.explanation.available)
        events = gateway.list_alerts({"patient_id": "episode-a"}).items
        alert = events[0]
        draft = RecordDraft(assessment=" observed ", action=" reviewed ")
        with self.assertRaises(FrontendError) as rejected:
            gateway.save_record(alert.id, draft, "care-key")
        self.assertEqual(rejected.exception.status, 409)
        acknowledged = gateway.acknowledge_alert(alert.id)
        self.assertEqual(
            gateway.acknowledge_alert(alert.id).acknowledged_at,
            acknowledged.acknowledged_at,
        )
        self.assertEqual(acknowledged.acknowledged_at.year, datetime.now(UTC).year)
        saved = gateway.save_record(alert.id, draft, "care-key")
        self.assertEqual(gateway.save_record(alert.id, draft, "care-key").id, saved.id)
        restarted = create_app(self.settings, model_loader=unavailable)
        with TestClient(restarted) as client:
            result = client.get("/alerts", params={"patient_id": "episode-a"}).json()
            self.assertEqual(result["items"][0]["records"][0]["id"], saved.id)
            self.assertEqual(client.get("/health").json()["status"], "degraded")

    def test_manual_alert_for_patient_without_model_alert_and_restart(self):
        import_encounter(
            ImportFile.model_validate(payload("episode-b", 23)),
            self.app.state.store,
            self.bundle,
        )
        gateway = self.frontend_gateway()
        self.assertEqual(gateway.list_alerts({"patient_id": "episode-b"}).items, [])
        draft = ManualAlertDraft(
            message=" manual note ",
            record=RecordDraft(assessment="assessment", action="action"),
        )
        created = gateway.create_manual_alert("episode-b", draft, "manual-key")
        retry = gateway.create_manual_alert("episode-b", draft, "manual-key")
        self.assertEqual(retry, created)
        self.assertEqual(created.message, "manual note")
        self.assertEqual(created.created_by, self.settings.actor)
        self.assertIsNone(created.risk_score)
        self.assertEqual(created.status, "acknowledged")
        self.assertEqual([record.action for record in created.records], ["action"])
        self.assertEqual(
            gateway.get_patient("episode-b").item.latest_alert_id, created.id
        )

        restarted = create_app(self.settings, model_loader=unavailable)
        with TestClient(restarted) as client:
            response = client.get("/alerts", params={"patient_id": "episode-b"}).json()
            self.assertEqual([item["id"] for item in response["items"]], [created.id])
            self.assertEqual(
                response["items"][0]["records"][0]["id"], created.records[0].id
            )
            self.assertEqual(
                client.get("/alerts").json()["items"][0]["source"], "manual"
            )

    def test_manual_alert_retry_validation_and_unacknowledged_counts(self):
        url = "/patients/episode-a/alerts"
        headers = {"Idempotency-Key": "manual-key"}
        with ThreadPoolExecutor(max_workers=4) as pool:
            writes = list(
                pool.map(
                    lambda _: self.client.post(
                        url, json={"message": "note"}, headers=headers
                    ),
                    range(4),
                )
            )
        self.assertEqual([response.status_code for response in writes], [201] * 4)
        created = writes[0].json()
        self.assertEqual(
            {response.json()["id"] for response in writes}, {created["id"]}
        )
        self.assertEqual(created["status"], "unacknowledged")
        self.assertEqual(
            self.client.post(
                url, json={"message": "different"}, headers=headers
            ).status_code,
            409,
        )
        self.assertEqual(
            self.client.post(url, json={"message": " "}, headers=headers).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(url, json={"message": "note"}).status_code, 422
        )
        self.assertEqual(
            self.client.post(
                "/patients/missing/alerts", json={"message": "note"}, headers=headers
            ).status_code,
            404,
        )
        summary = self.client.get("/patients").json()["summary"]
        self.assertEqual(summary["unacknowledged_alert_count"], 4)
        self.assertEqual(summary["unacknowledged_high_count"], 3)
        base = f"/alerts/{created['id']}"
        self.assertEqual(self.client.post(base + "/acknowledgement").status_code, 200)
        self.assertEqual(
            self.client.post(
                base + "/records",
                json={"assessment": "review", "action": "document"},
                headers=headers,
            ).status_code,
            200,
        )

    def test_legacy_database_keeps_alerts_and_care_records(self):
        model_alert = self.client.get("/alerts").json()["items"][0]
        url = f"/alerts/{model_alert['id']}"
        self.client.post(url + "/acknowledgement")
        saved = self.client.post(
            url + "/records",
            json={"assessment": "legacy", "action": "review"},
            headers={"Idempotency-Key": "legacy"},
        ).json()
        legacy_path = Path(self.temp.name) / "legacy.sqlite"
        legacy_alert_columns = """(
            id TEXT PRIMARY KEY, encounter_id TEXT NOT NULL, hour INTEGER NOT NULL,
            snapshot TEXT NOT NULL, acknowledged_by TEXT, acknowledged_at TEXT,
            UNIQUE(encounter_id,hour),
            FOREIGN KEY(encounter_id,hour) REFERENCES observations(encounter_id,hour)
        )"""
        with sqlite3.connect(legacy_path) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(
                SCHEMA.replace(ALERT_COLUMNS, legacy_alert_columns)
            )
            connection.execute(
                "ATTACH DATABASE ? AS source", (str(self.settings.database),)
            )
            for table in ("encounters", "observations", "predictions"):
                connection.execute(f"INSERT INTO {table} SELECT * FROM source.{table}")
            connection.execute(
                "INSERT INTO alerts SELECT id,encounter_id,hour,snapshot,acknowledged_by,acknowledged_at FROM source.alerts"
            )
            connection.execute("INSERT INTO records SELECT * FROM source.records")

        migrated = create_app(Settings(database=legacy_path), model_loader=unavailable)
        with TestClient(migrated) as client:
            events = client.get("/alerts").json()["items"]
            self.assertEqual(len(events), 3)
            self.assertEqual(events[0]["source"], "model")
            self.assertEqual(events[0]["records"][0]["id"], saved["id"])
            with migrated.state.store.connect() as connection:
                self.assertIsNone(
                    connection.execute("PRAGMA foreign_key_check").fetchone()
                )

    def test_sparse_projection_horizon_units_and_event_clock(self):
        response = self.client.get("/patients/episode-a")
        detail = Detail.model_validate(response.json()).item
        self.assertEqual(len(detail.series), 24)
        self.assertEqual(detail.series[0].observed_at.hour, 3)
        self.assertIsNone(detail.series[0].risk_score)
        self.assertIsNone(detail.series[0].predicted_at)
        observations = {o.feature_name: o for o in detail.observations}
        self.assertIsNone(observations["Lactate"].value)
        self.assertIsNone(observations["Lactate"].observed_at)
        self.assertEqual(observations["Lactate"].unit, "mg/dL")
        self.assertEqual(observations["FiO2"].unit, "%")
        self.assertEqual(detail.prediction.predicted_at.year, 2020)
        self.assertEqual(observations["HR"].observed_at, detail.last_observed_at)
        with self.app.state.store.connect() as connection:
            row = connection.execute(
                "SELECT computed_at FROM predictions ORDER BY hour DESC LIMIT 1"
            ).fetchone()
        self.assertNotEqual(
            datetime.fromisoformat(row[0]).year, detail.prediction.predicted_at.year
        )

    def test_filters_pagination_and_full_aggregate(self):
        import_encounter(
            ImportFile.model_validate(payload("episode-b", 23)),
            self.app.state.store,
            self.bundle,
        )
        first = self.client.get("/patients", params={"limit": 1}).json()
        self.assertEqual(first["summary"]["total"], 2)
        self.assertEqual(first["summary"]["unavailable_count"], 1)
        self.assertEqual(first["items"][0]["id"], "episode-a")
        second = self.client.get(
            "/patients", params={"limit": 1, "cursor": first["next_cursor"]}
        ).json()
        self.assertEqual(second["items"][0]["id"], "episode-b")
        self.assertIsNone(second["next_cursor"])
        self.assertEqual(
            self.client.get(
                "/patients", params={"cursor": first["next_cursor"], "query": "x"}
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.get("/patients", params={"cursor": "!bad"}).status_code, 422
        )
        self.assertEqual(
            self.client.get("/patients", params={"risk_level": "warning"}).json()[
                "summary"
            ]["total"],
            0,
        )
        self.assertEqual(
            self.client.get(
                "/patients", params={"query": "BED", "assignee": "Researcher"}
            ).json()["summary"]["total"],
            2,
        )
        alert_ids = []
        cursor = None
        for _ in range(3):
            params = {"limit": 1, **({"cursor": cursor} if cursor else {})}
            page = self.client.get("/alerts", params=params).json()
            self.assertEqual(page["summary"]["total"], 3)
            alert_ids.append(page["items"][0]["id"])
            cursor = page["next_cursor"]
        self.assertIsNone(cursor)
        self.assertEqual(len(set(alert_ids)), 3)
        start = "2020-01-02T00:00:00Z"
        end = "2020-01-02T01:00:00Z"
        self.assertEqual(
            self.client.get("/alerts", params={"from": start, "to": end}).json()[
                "summary"
            ]["total"],
            1,
        )
        self.assertEqual(
            self.client.get(
                "/alerts", params={"from": "2020-01-01T00:00:00"}
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.get("/alerts", params={"from": end, "to": start}).status_code,
            422,
        )

    def test_stateless_routes_do_not_write(self):
        before = self.client.get("/patients").json()["summary"]["total"]
        result = self.client.post(
            "/predict",
            json={"patient_id": "not-imported", "observations": [{"ICULOS": 1}]},
        )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(
            self.client.get("/patients").json()["summary"]["total"], before
        )
        self.assertEqual(self.client.get("/patients/not-imported").status_code, 404)

    def test_concurrent_care_writes_and_key_conflict(self):
        alert = self.client.get("/alerts").json()["items"][0]
        path = f"/alerts/{alert['id']}"
        with ThreadPoolExecutor(max_workers=4) as executor:
            acknowledged = list(
                executor.map(
                    lambda _: self.client.post(path + "/acknowledgement").json(),
                    range(4),
                )
            )
        self.assertEqual(len({a["acknowledged_at"] for a in acknowledged}), 1)
        body = {"assessment": " checked ", "action": " reviewed "}

        def save(_):
            return self.client.post(
                path + "/records", json=body, headers={"Idempotency-Key": "repeat"}
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(executor.map(save, range(4)))
        self.assertTrue(all(r.status_code == 200 for r in responses))
        self.assertEqual(len({r.json()["id"] for r in responses}), 1)
        self.assertEqual(responses[0].json()["assessment"], "checked")
        conflict = self.client.post(
            path + "/records",
            json={"assessment": "other", "action": "reviewed"},
            headers={"Idempotency-Key": "repeat"},
        )
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.json()["code"], "idempotency_conflict")
        self.assertEqual(
            self.client.post(path + "/records", json=body).status_code, 422
        )
        self.assertEqual(
            self.client.post(
                path + "/records", json=body, headers={"Idempotency-Key": " "}
            ).status_code,
            422,
        )

    def test_sensitive_errors_and_token_protection(self):
        secret = "patient-body-must-not-leak"
        with TestClient(
            create_app(
                Settings(self.settings.database, token="dev-secret"), unavailable
            )
        ) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/patients").status_code, 401)
            headers = {"Authorization": "Bearer dev-secret"}
            self.assertEqual(client.get("/patients", headers=headers).status_code, 200)
            invalid = client.post(
                "/predict",
                json={"patient_id": secret, "observations": []},
                headers=headers,
            )
            self.assertEqual(invalid.status_code, 422)
            self.assertNotIn(secret, invalid.text)
            self.assertEqual(set(invalid.json()), {"code", "message", "request_id"})
            self.assertEqual(
                invalid.headers["x-request-id"], invalid.json()["request_id"]
            )
            unavailable_response = client.post(
                "/predict",
                json={"patient_id": secret, "observations": [{"ICULOS": 1}]},
                headers=headers,
            )
            self.assertEqual(unavailable_response.status_code, 503)
        with patch(
            "app.services.query.utc_now", side_effect=sqlite3.OperationalError(secret)
        ):
            failed = self.client.get("/patients")
            self.assertEqual(failed.status_code, 503)
            self.assertNotIn(secret, failed.text)

    def test_frontend_schemas_validate_empty_and_short_history(self):
        self.assertTrue(
            Health.model_validate(self.client.get("/health").json()).model_ready
        )
        import_encounter(
            ImportFile.model_validate(payload("short", 1)),
            self.app.state.store,
            self.bundle,
        )
        result = Listing[PatientSummary].model_validate(
            self.client.get("/patients", params={"risk_level": "unavailable"}).json()
        )
        self.assertEqual(
            result.items[0].prediction.unavailable_reason, "insufficient_history"
        )
        self.assertIsNone(result.items[0].prediction.risk_score)
        self.assertEqual(
            self.client.get("/patients", params={"limit": 0}).status_code, 422
        )
        bad = base64.urlsafe_b64encode(
            json.dumps({"key": [], "signature": "wrong"}).encode()
        ).decode()
        self.assertEqual(
            self.client.get("/patients", params={"cursor": bad}).status_code, 422
        )


if __name__ == "__main__":
    unittest.main()

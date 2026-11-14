"""Synthetic-only import replay tests; no model artifacts or clinical files."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

import import_cli
from app.errors import ServiceError
from app.schemas.import_file import FEATURES, ImportFile
from app.services.import_run import import_encounter
from app.services.inference import ModelBundle
from app.store.sqlite import Store

ANCHOR = datetime(2020, 1, 1, tzinfo=UTC)
NOW = datetime(2021, 1, 1, tzinfo=UTC)


def source(hours=24, encounter="synthetic-encounter"):
    return {
        "encounter_id": encounter,
        "patient_id": "synthetic-source",
        "bed": "synthetic-bed",
        "assignee": "synthetic-owner",
        "icu_id": "ICU-A",
        "admission_anchor": ANCHOR.isoformat(),
        "observations": [
            dict(
                {feature: None for feature in FEATURES},
                ICULOS=float(hour),
                HR=60.0 + hour,
            )
            for hour in range(1, hours + 1)
        ],
    }


def result_for(count, score=0.25):
    return {
        "status": "scored" if count >= 24 else "insufficient_history",
        "risk_score": score if count >= 24 else None,
        "alert": count >= 24 and score >= 0.2,
        "score_threshold": 0.2,
        "records_received": count,
        "minimum_records": 24,
        "calibrated_probability": None,
    }


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "synthetic.sqlite")
        self.calls = []

        def predict(rows, model, stats):
            self.calls.append(copy.deepcopy(rows))
            return result_for(len(rows))

        self.bundle = ModelBundle(
            (object(), {}, predict), "checkpoint-test", "policy-test"
        )

    def run_import(self, data, bundle=None, store=None):
        return import_encounter(
            ImportFile.model_validate(data),
            store or self.store,
            bundle or self.bundle,
            NOW,
        )

    def rows(self, table):
        with self.store.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")
            ]

    def snapshot(self):
        return {
            table: self.rows(table)
            for table in (
                "encounters",
                "observations",
                "predictions",
                "alerts",
                "records",
            )
        }

    def test_exact_short_append_preserve_ids_and_causal_full_history(self):
        data = source(25)
        self.assertEqual(
            self.run_import(data),
            {
                "encounter_id": data["encounter_id"],
                "records_received": 25,
                "records_added": 25,
                "alerts_added": 2,
            },
        )
        before = self.snapshot()
        self.assertEqual(self.run_import(data)["records_added"], 0)
        self.assertEqual(self.run_import(source(10))["records_added"], 0)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(len(self.calls), 25)
        self.assertEqual(self.run_import(source(27))["alerts_added"], 2)
        expected = source(27)["observations"]
        for hour, rows in enumerate(self.calls, 1):
            self.assertEqual(rows, expected[:hour])
        self.assertEqual(self.rows("predictions")[:25], before["predictions"])
        self.assertEqual(self.rows("alerts")[:2], before["alerts"])

    def test_all_positive_hours_and_exact_results_and_logical_times(self):
        self.run_import(source(28))
        self.assertEqual(
            [row["hour"] for row in self.rows("alerts")], list(range(24, 29))
        )
        for prediction in self.rows("predictions"):
            self.assertEqual(
                json.loads(prediction["result"]), result_for(prediction["hour"])
            )
            self.assertGreater(datetime.fromisoformat(prediction["computed_at"]), NOW)
            self.assertGreater(datetime.fromisoformat(prediction["imported_at"]), NOW)
        for alert in self.rows("alerts"):
            value = json.loads(alert["snapshot"])
            self.assertEqual(value["id"], alert["id"])
            self.assertEqual(value["patient_id"], "synthetic-encounter")
            self.assertEqual(
                value["occurred_at"],
                (ANCHOR + timedelta(hours=alert["hour"])).isoformat(),
            )
            self.assertEqual(value["risk_level"], "high")
            self.assertEqual(value["status"], "unacknowledged")
            self.assertEqual(value["repeat_count"], 0)
            self.assertEqual(value["records"], [])

    def test_changed_envelope_prefix_and_versions_write_nothing(self):
        self.run_import(source())
        before = self.snapshot()
        variants = []
        for key in ("patient_id", "bed", "assignee"):
            value = source(25)
            value[key] = "different"
            variants.append(value)
        value = source(25)
        value["admission_anchor"] = (ANCHOR + timedelta(hours=1)).isoformat()
        variants.append(value)
        for replacement in (None, 99.0):
            value = source(25)
            value["observations"][0]["HR"] = replacement
            variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ServiceError) as caught:
                self.run_import(value)
            self.assertEqual(caught.exception.code, "import_conflict")
            self.assertEqual(self.snapshot(), before)
        for bundle in (
            ModelBundle(self.bundle.state, "changed", "policy-test"),
            ModelBundle(self.bundle.state, "checkpoint-test", "changed"),
        ):
            with self.assertRaises(ServiceError):
                self.run_import(source(25), bundle)
            self.assertEqual(self.snapshot(), before)

    def test_explicit_row_times_and_time_mode_are_immutable(self):
        data = source(24)
        del data["admission_anchor"]
        for hour, row in enumerate(data["observations"], 1):
            row["observed_at"] = (ANCHOR + timedelta(hours=hour)).isoformat()
        self.run_import(data)
        envelope = json.loads(self.rows("encounters")[0]["envelope"])
        self.assertEqual(envelope["anchor"], ANCHOR.isoformat())
        self.assertEqual(envelope["time_mode"], "rows")
        with self.assertRaises(ServiceError):
            self.run_import(source())
        for row in data["observations"]:
            row["observed_at"] = (
                datetime.fromisoformat(row["observed_at"])
                .astimezone(timezone(timedelta(hours=9)))
                .isoformat()
            )
        self.assertEqual(self.run_import(data)["records_added"], 0)

    def test_future_and_invalid_input_never_score(self):
        data = source()
        with self.assertRaises(ServiceError):
            import_encounter(
                ImportFile.model_validate(data), self.store, self.bundle, ANCHOR
            )
        variants = []
        for key, value in (
            ("icu_id", "ICU-B"),
            ("unit_profile", "other"),
            ("patient_id", ""),
            ("admission_anchor", "2020-01-01T00:00:00"),
        ):
            invalid = source()
            invalid[key] = value
            variants.append(invalid)
        for offset in (2.0, 1.5):
            invalid = source()
            invalid["observations"][0]["ICULOS"] = offset
            variants.append(invalid)
        for key, value in (("HR", float("nan")), ("SepsisLabel", 1)):
            invalid = source()
            invalid["observations"][0][key] = value
            variants.append(invalid)
        invalid = source()
        del invalid["observations"][0]["Temp"]
        variants.append(invalid)
        for invalid in variants:
            with self.assertRaises(ValidationError):
                ImportFile.model_validate(invalid)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.store.path.exists())

    def test_broken_runtime_is_atomic_and_safe(self):
        self.run_import(source(23))
        before = self.snapshot()
        malformed = [
            {"score_threshold": 0.3},
            {"calibrated_probability": 0.5},
            {"records_received": 1},
            {"minimum_records": 23},
            {"alert": False},
            {"status": "insufficient_history"},
            {"risk_score": float("nan")},
            {"risk_score": True},
            {"risk_score": 2.0},
            {"calibration_available": True},
        ]
        for change in malformed:

            def predict(rows, model, stats, change=change):
                result = result_for(len(rows))
                if len(rows) == 25:
                    result.update(change)
                return result

            bundle = ModelBundle((None, {}, predict), "checkpoint-test", "policy-test")
            with self.assertRaises(ServiceError):
                self.run_import(source(25), bundle)
            self.assertEqual(self.snapshot(), before)

        def broken(rows, model, stats):
            raise RuntimeError("sensitive source or feature value")

        with self.assertRaises(ServiceError) as caught:
            self.run_import(
                source(25),
                ModelBundle((None, {}, broken), "checkpoint-test", "policy-test"),
            )
        self.assertNotIn("sensitive", str(caught.exception))
        self.assertEqual(self.snapshot(), before)

    def test_score_boundaries(self):
        for score in (0.0, 0.19, 0.2, 1.0):

            def predict(rows, model, stats, score=score):
                return result_for(len(rows), score)

            bundle = ModelBundle((None, {}, predict), "checkpoint-test", "policy-test")
            summary = self.run_import(
                source(24, f"boundary-{str(score).replace('.', '_')}"), bundle
            )
            self.assertEqual(summary["alerts_added"], int(score >= 0.2))

    def test_commit_rechecks_identical_and_conflicting_race(self):
        for conflict in (False, True):
            encounter = "race-conflict" if conflict else "race-identical"
            data = source(25, encounter)
            fired = False

            def racing(rows, model, stats, data=data, conflict=conflict):
                nonlocal fired
                if not fired:
                    fired = True
                    competing = copy.deepcopy(data)
                    if conflict:
                        competing["observations"][0]["HR"] = 999.0
                    self.run_import(competing)
                return result_for(len(rows))

            bundle = ModelBundle((None, {}, racing), "checkpoint-test", "policy-test")
            if conflict:
                with self.assertRaises(ServiceError) as caught:
                    self.run_import(data, bundle)
                self.assertEqual(caught.exception.code, "import_conflict")
            else:
                self.assertEqual(self.run_import(data, bundle)["records_added"], 0)
            rows = [
                row
                for row in self.rows("predictions")
                if row["encounter_id"] == encounter
            ]
            self.assertEqual(len(rows), 25)

    def test_independent_encounters_restart_ack_and_record_survive(self):
        self.run_import(source(25, "first"))
        self.run_import(source(24, "second"))
        alert = self.rows("alerts")[0]
        with self.store.connect(write=True) as connection:
            connection.execute(
                "UPDATE alerts SET acknowledged_by=?,acknowledged_at=? WHERE id=?",
                ("synthetic-actor", NOW.isoformat(), alert["id"]),
            )
            connection.execute(
                "INSERT INTO records VALUES(?,?,?,?,?,?)",
                (
                    "synthetic-record",
                    alert["id"],
                    "synthetic-actor",
                    "synthetic-key",
                    '{"assessment":"synthetic","action":"synthetic"}',
                    NOW.isoformat(),
                ),
            )
        before = self.snapshot()
        restarted = Store(self.store.path)
        self.assertEqual(
            self.run_import(source(20, "first"), store=restarted)["records_added"], 0
        )
        self.assertEqual(
            self.run_import(source(24, "second"), store=restarted)["records_added"], 0
        )
        self.assertEqual(before, self.snapshot())
        self.run_import(source(26, "first"), store=restarted)
        self.assertEqual(self.rows("alerts")[:3], before["alerts"])
        self.assertEqual(self.rows("records"), before["records"])

    def test_cli_safe_errors_once_loaded_and_per_file_commit(self):
        good = Path(self.temp.name) / "synthetic-good.json"
        bad = Path(self.temp.name) / "secret-filename.json"
        good.write_text(json.dumps(source()), encoding="utf-8")
        bad.write_text('{"patient_id":"secret-patient"}', encoding="utf-8")
        output, errors = io.StringIO(), io.StringIO()
        with (
            patch("import_cli.load_bundle", return_value=self.bundle) as loader,
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            status = import_cli.main(
                [
                    str(good),
                    str(bad),
                    "--database",
                    str(self.store.path),
                    "--ai-root",
                    self.temp.name,
                ]
            )
        self.assertEqual(status, 1)
        loader.assert_called_once_with(Path(self.temp.name).resolve())
        self.assertEqual(len(self.rows("observations")), 24)
        self.assertEqual(json.loads(output.getvalue())["records_added"], 24)
        self.assertNotIn("secret", errors.getvalue())
        self.assertEqual(
            json.loads(errors.getvalue())["error"]["code"], "invalid_import"
        )

    def test_cli_unavailable_model_writes_nothing(self):
        output, errors = io.StringIO(), io.StringIO()
        with (
            patch("import_cli.load_bundle", side_effect=RuntimeError("secret-value")),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            status = import_cli.main(
                ["secret-filename", "--database", str(self.store.path)]
            )
        self.assertEqual(status, 1)
        self.assertFalse(self.store.path.exists())
        self.assertEqual(output.getvalue(), "")
        self.assertNotIn("secret", errors.getvalue())

    def test_cli_argument_errors_are_safe_json(self):
        errors = io.StringIO()
        with redirect_stderr(errors):
            status = import_cli.main(["secret-filename", "--secret-option"])
        self.assertEqual(status, 2)
        self.assertNotIn("secret", errors.getvalue())
        self.assertEqual(
            json.loads(errors.getvalue())["error"]["code"], "invalid_arguments"
        )

    def test_explicit_timestamp_gap_is_rejected(self):
        data = source(24)
        del data["admission_anchor"]
        for hour, row in enumerate(data["observations"], 1):
            row["observed_at"] = (
                ANCHOR + timedelta(hours=hour + int(hour > 12))
            ).isoformat()
        with self.assertRaises(ValidationError):
            ImportFile.model_validate(data)
        self.assertFalse(self.store.path.exists())


if __name__ == "__main__":
    unittest.main()

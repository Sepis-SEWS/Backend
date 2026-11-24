import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

import import_cli
from app.schemas.import_file import FEATURES, ImportFile
from app.services.import_run import import_encounter
from app.services.inference import ModelBundle
from app.services.query import list_patients
from app.store.sqlite import Store
from tools.physionet2019 import convert, run, validate
from tools.physionet2019.common import (
    DEFAULT_REPLAY_BASE,
    SOURCE_COLUMNS,
    ConversionError,
    build_import,
    check_import,
    read_psv,
)

NOW = datetime(2027, 1, 1, tzinfo=UTC)


def psv(hours, start=1, label=False):
    lines = ["|".join(SOURCE_COLUMNS)]
    for offset in range(hours):
        row = dict.fromkeys(SOURCE_COLUMNS, "NaN")
        row.update(
            HR=str(70 + offset),
            FiO2="0.4" if offset == 0 else "NaN",
            Age="60",
            Gender="1",
            HospAdmTime="-0.02",
            ICULOS=str(start + offset),
            SepsisLabel="1" if label else "0",
        )
        lines.append("|".join(row[name] for name in SOURCE_COLUMNS))
    return "\n".join(lines) + "\n"


def result_for(rows, model, stats):
    count = len(rows)
    return {
        "status": "scored" if count >= 24 else "insufficient_history",
        "risk_score": 0.3 if count >= 24 else None,
        "alert": count >= 24,
        "score_threshold": 0.2,
        "records_received": count,
        "minimum_records": 24,
        "calibrated_probability": None,
    }


class PhysioNetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for name in ("training_setA", "training_setB"):
            (self.root / "training" / name).mkdir(parents=True)
        self.bundle = ModelBundle((object(), {}, result_for), "ckpt", "policy")

    def write(self, name, text, source_set="training_setA"):
        path = self.root / "training" / source_set / f"{name}.psv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_label_dropped_nan_to_null_values_unchanged_replay_marked(self):
        path = self.write("p999001", psv(25, label=True))
        source = read_psv(path, "training_setA")
        data = build_import(source, DEFAULT_REPLAY_BASE)
        for row in data["observations"]:
            self.assertEqual(set(row), {"ICULOS", *FEATURES})
        self.assertNotIn("SepsisLabel", json.dumps(data))
        first = data["observations"][0]
        self.assertEqual((first["HR"], first["FiO2"], first["Temp"]), (70.0, 0.4, None))
        self.assertEqual(data["provenance"]["time_basis"], "research_replay")
        self.assertEqual(check_import(data, NOW, source), [])
        times = ImportFile.model_validate(data).times()
        self.assertEqual(times[0], DEFAULT_REPLAY_BASE + timedelta(hours=1))

    def test_late_start_keeps_source_iculos_and_replay_offset(self):
        source = read_psv(self.write("p999004", psv(5, start=4)), "training_setA")
        data = build_import(source, DEFAULT_REPLAY_BASE)
        self.assertEqual(
            [r["ICULOS"] for r in data["observations"]], [4.0, 5.0, 6.0, 7.0, 8.0]
        )
        self.assertEqual(check_import(data, NOW, source), [])
        times = ImportFile.model_validate(data).times()
        self.assertEqual(times[0], DEFAULT_REPLAY_BASE + timedelta(hours=4))

    def test_bad_grid_and_cells_rejected(self):
        for text, code in (
            (psv(5, start=0), "bad_iculos"),
            (psv(5).replace("|1|0\n", "|1.5|0\n", 1), "bad_iculos"),
            (psv(5).replace("70|", "inf|", 1), "non_finite_value"),
            ("HR|ICULOS\n1|1\n", "unexpected_header"),
        ):
            with self.subTest(code=code), self.assertRaises(ConversionError) as caught:
                read_psv(self.write("p999002", text), "training_setA")
            self.assertEqual(caught.exception.code, code)
        lines = psv(5).splitlines()
        lines[3], lines[4] = lines[4], lines[3]
        with self.assertRaises(ConversionError) as caught:
            read_psv(self.write("p999003", "\n".join(lines)), "training_setA")
        self.assertEqual(caught.exception.code, "iculos_not_contiguous")

    def test_validator_catches_label_future_and_value_drift(self):
        source = read_psv(self.write("p999001", psv(3)), "training_setA")
        good = build_import(source, DEFAULT_REPLAY_BASE)
        labelled = copy.deepcopy(good)
        labelled["observations"][0]["SepsisLabel"] = 0.0
        self.assertTrue(check_import(labelled, NOW))
        self.assertTrue(check_import(good, DEFAULT_REPLAY_BASE))  # in the future
        drift = copy.deepcopy(good)
        drift["observations"][1]["HR"] = 1.0
        self.assertIn(
            "feature values differ from the source file",
            check_import(drift, NOW, source),
        )
        mismatch = copy.deepcopy(good)
        mismatch["provenance"]["replay_base"] = "2025-01-01T00:00:00+00:00"
        with self.assertRaises(ValidationError):
            ImportFile.model_validate(mismatch)

    def test_convert_all_validate_import_and_dashboard_replay_flag(self):
        self.write("p999001", psv(25))
        self.write("p999002", psv(3, start=7))
        self.write("p100001", "HR|ICULOS\n1|1\n", "training_setB")
        (self.root / "training/training_setA/._p999001.psv").write_bytes(b"\0")
        out = self.root / "out"
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            status = convert.main(["--source-root", str(self.root), "--out", str(out)])
            self.assertEqual(status, 1)  # the malformed record fails, others convert
            self.assertEqual(
                validate.main([str(out), "--source-root", str(self.root)]), 0
            )
        report = json.loads((out / "_conversion_report.json").read_text("utf-8"))
        self.assertEqual(report["summary"]["converted"], 2)
        self.assertEqual(report["summary"]["failure_codes"], {"unexpected_header": 1})
        self.assertEqual(report["summary"]["iculos_start_after_1"], 1)
        self.assertEqual(report["dataset"]["version"], "1.0.0")
        self.assertEqual(len(report["dataset"]["source_listing_sha256"]), 64)

        store = Store(self.root / "db.sqlite")
        stdout = io.StringIO()
        with (
            patch("import_cli.load_bundle", return_value=self.bundle),
            redirect_stdout(stdout),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(
                import_cli.main([str(out), "--database", str(store.path)]), 0
            )
        summary = json.loads(stdout.getvalue().splitlines()[-1])["import_summary"]
        self.assertEqual((summary["files_imported"], summary["records_added"]), (2, 28))
        with store.connect() as connection:
            envelope = json.loads(
                connection.execute(
                    "SELECT envelope FROM encounters WHERE id='pn19-p999002'"
                ).fetchone()[0]
            )
        self.assertEqual(envelope["provenance"]["time_basis"], "research_replay")
        self.assertEqual(envelope["anchor"], DEFAULT_REPLAY_BASE.isoformat())
        later = DEFAULT_REPLAY_BASE + timedelta(hours=26)
        with patch("app.services.query.utc_now", return_value=later):
            items = list_patients(store, {}, None, 10).items
        self.assertTrue(all(item.data_delayed for item in items))
        self.assertEqual({item.delay_reason for item in items}, {"replay_historical"})

    def test_one_command_pipeline_reports_and_stops_without_model(self):
        self.write("p999001", psv(25))
        self.write("p100001", psv(3, start=5), "training_setB")
        out = self.root / "import" / "physionet2019"
        database = self.root / "run.sqlite"
        common = [
            "--source-root",
            str(self.root),
            "--out",
            str(out),
            "--database",
            str(database),
            "--ai-root",
            str(self.root),
        ]
        printed = io.StringIO()
        with (
            patch("import_cli.load_bundle", return_value=self.bundle),
            redirect_stdout(printed),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(run.main(common), 0)
        text = printed.getvalue()
        self.assertIn("/patients        total 2", text)
        self.assertIn("replay-time marked: 2/2", text)
        report = json.loads((out.parent / "_import_report.json").read_text("utf-8"))
        self.assertEqual(report["summary"]["files_imported"], 2)
        self.assertEqual(report["summary"]["records_added"], 28)
        errors = io.StringIO()
        with (
            patch("import_cli.load_bundle", side_effect=RuntimeError("no model")),
            redirect_stdout(io.StringIO()),
            redirect_stderr(errors),
        ):
            self.assertEqual(run.main(common + ["--skip-convert"]), 1)
        self.assertIn("model_unavailable", errors.getvalue())
        self.assertFalse((out.parent / "_import_report.json").exists())

    def test_files_without_provenance_keep_their_envelope_hash(self):
        source = read_psv(self.write("p999001", psv(2)), "training_setA")
        data = build_import(source, DEFAULT_REPLAY_BASE)
        del data["provenance"]
        store = Store(self.root / "plain.sqlite")
        import_encounter(ImportFile.model_validate(data), store, self.bundle, NOW)
        with store.connect() as connection:
            envelope = json.loads(
                connection.execute("SELECT envelope FROM encounters").fetchone()[0]
            )
        self.assertNotIn("provenance", envelope)


if __name__ == "__main__":
    unittest.main()

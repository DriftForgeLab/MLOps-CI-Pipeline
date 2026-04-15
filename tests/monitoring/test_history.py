"""Tests for src.monitoring.history."""
from __future__ import annotations

import json
from pathlib import Path

from src.monitoring.history import (
    append_history_entry,
    iter_history,
    load_latest_drift,
)


def _snapshot(severity: str = "low", **overrides) -> dict:
    payload = {
        "schema_version": "1.0.0",
        "drift_type": "tabular",
        "generated_at": "2026-04-14T00:00:00+00:00",
        "overall": {
            "dataset_drift_detected": severity != "low",
            "drift_share": 0.1,
            "drifted_feature_count": 1,
            "total_feature_count": 4,
            "severity": severity,
        },
        "features": {},
    }
    payload.update(overrides)
    return payload


class TestMissingOrEmpty:
    def test_missing_outputs_root_returns_none(self, tmp_path: Path):
        assert load_latest_drift("iris", tmp_path / "does_not_exist") is None

    def test_missing_model_dir_returns_none(self, tmp_path: Path):
        (tmp_path / "other_model").mkdir()
        assert load_latest_drift("iris", tmp_path) is None

    def test_empty_model_dir_returns_none(self, tmp_path: Path):
        (tmp_path / "iris").mkdir()
        assert load_latest_drift("iris", tmp_path) is None


class TestSnapshotFallback:
    def test_single_snapshot_returned(self, tmp_path: Path):
        model_dir = tmp_path / "iris"
        model_dir.mkdir()
        path = model_dir / "20260414T000000Z.json"
        path.write_text(json.dumps(_snapshot("medium")), encoding="utf-8")
        result = load_latest_drift("iris", tmp_path)
        assert result is not None
        assert result["overall"]["severity"] == "medium"

    def test_newest_by_mtime_wins(self, tmp_path: Path):
        model_dir = tmp_path / "iris"
        model_dir.mkdir()
        older = model_dir / "20260101T000000Z.json"
        newer = model_dir / "20260414T000000Z.json"
        older.write_text(json.dumps(_snapshot("low")), encoding="utf-8")
        newer.write_text(json.dumps(_snapshot("high")), encoding="utf-8")
        # Force older mtime on the older file.
        import os
        os.utime(older, (1_700_000_000, 1_700_000_000))
        os.utime(newer, (1_800_000_000, 1_800_000_000))
        result = load_latest_drift("iris", tmp_path)
        assert result is not None
        assert result["overall"]["severity"] == "high"

    def test_decision_sidecars_ignored(self, tmp_path: Path):
        model_dir = tmp_path / "iris"
        model_dir.mkdir()
        (model_dir / "20260414T000000Z.json").write_text(
            json.dumps(_snapshot("low")), encoding="utf-8"
        )
        decision = model_dir / "20260414T000000Z_decision.json"
        decision.write_text(json.dumps({"decision": "retrain"}), encoding="utf-8")
        result = load_latest_drift("iris", tmp_path)
        assert result is not None
        assert result["overall"]["severity"] == "low"

    def test_unreadable_json_returns_none(self, tmp_path: Path):
        model_dir = tmp_path / "iris"
        model_dir.mkdir()
        (model_dir / "20260414T000000Z.json").write_text(
            "{not valid json", encoding="utf-8"
        )
        assert load_latest_drift("iris", tmp_path) is None


class TestJsonlIndex:
    def test_jsonl_last_entry_wins(self, tmp_path: Path):
        model_dir = tmp_path / "iris"
        model_dir.mkdir()
        old_snap = model_dir / "20260101T000000Z.json"
        new_snap = model_dir / "20260414T000000Z.json"
        old_snap.write_text(json.dumps(_snapshot("low")), encoding="utf-8")
        new_snap.write_text(json.dumps(_snapshot("high")), encoding="utf-8")

        history = model_dir / "history.jsonl"
        with history.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"json_path": str(old_snap)}) + "\n")
            fh.write(json.dumps({"json_path": str(new_snap)}) + "\n")

        result = load_latest_drift("iris", tmp_path)
        assert result is not None
        assert result["overall"]["severity"] == "high"

    def test_jsonl_relative_json_path_resolved(self, tmp_path: Path):
        model_dir = tmp_path / "iris"
        model_dir.mkdir()
        snap = model_dir / "20260414T000000Z.json"
        snap.write_text(json.dumps(_snapshot("medium")), encoding="utf-8")

        history = model_dir / "history.jsonl"
        history.write_text(
            json.dumps({"json_path": "20260414T000000Z.json"}) + "\n",
            encoding="utf-8",
        )

        result = load_latest_drift("iris", tmp_path)
        assert result is not None
        assert result["overall"]["severity"] == "medium"

    def test_jsonl_malformed_line_skipped(self, tmp_path: Path):
        model_dir = tmp_path / "iris"
        model_dir.mkdir()
        snap = model_dir / "20260414T000000Z.json"
        snap.write_text(json.dumps(_snapshot("medium")), encoding="utf-8")

        history = model_dir / "history.jsonl"
        with history.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"json_path": str(snap)}) + "\n")
            fh.write("{truncated\n")  # malformed trailing line

        result = load_latest_drift("iris", tmp_path)
        assert result is not None
        assert result["overall"]["severity"] == "medium"

    def test_jsonl_without_json_path_falls_back_to_snapshot_scan(
        self, tmp_path: Path
    ):
        model_dir = tmp_path / "iris"
        model_dir.mkdir()
        snap = model_dir / "20260414T000000Z.json"
        snap.write_text(json.dumps(_snapshot("high")), encoding="utf-8")

        history = model_dir / "history.jsonl"
        history.write_text(
            json.dumps({"severity": "high"}) + "\n", encoding="utf-8"
        )

        result = load_latest_drift("iris", tmp_path)
        assert result is not None
        assert result["overall"]["severity"] == "high"


class TestAppendHistoryEntry:
    def test_append_creates_directory_and_file(self, tmp_path: Path):
        result = _snapshot("medium")
        json_path = tmp_path / "iris" / "20260414T000000Z.json"
        entry = append_history_entry(
            model_name="iris",
            result=result,
            json_path=json_path,
            outputs_root=tmp_path,
        )
        history_path = tmp_path / "iris" / "history.jsonl"
        assert history_path.is_file()
        lines = history_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["overall_severity"] == "medium"
        assert parsed["drift_type"] == "tabular"
        assert parsed["drift_share"] == 0.1
        assert parsed["drifted_feature_count"] == 1
        assert parsed["json_path"] == str(json_path)
        assert parsed["timestamp"] == "2026-04-14T00:00:00+00:00"
        assert entry == parsed

    def test_append_is_appending(self, tmp_path: Path):
        json1 = tmp_path / "iris" / "a.json"
        json2 = tmp_path / "iris" / "b.json"
        append_history_entry("iris", _snapshot("low"), json1, tmp_path)
        append_history_entry("iris", _snapshot("high"), json2, tmp_path)
        lines = (tmp_path / "iris" / "history.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["overall_severity"] == "low"
        assert json.loads(lines[1])["overall_severity"] == "high"

    def test_append_includes_mlflow_run_id_when_given(self, tmp_path: Path):
        json_path = tmp_path / "iris" / "a.json"
        append_history_entry(
            "iris",
            _snapshot("low"),
            json_path,
            tmp_path,
            mlflow_run_id="run-abc-123",
        )
        lines = (tmp_path / "iris" / "history.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        assert json.loads(lines[0])["mlflow_run_id"] == "run-abc-123"

    def test_append_mlflow_run_id_absent_when_not_given(self, tmp_path: Path):
        json_path = tmp_path / "iris" / "a.json"
        append_history_entry("iris", _snapshot("low"), json_path, tmp_path)
        entry = json.loads(
            (tmp_path / "iris" / "history.jsonl").read_text(encoding="utf-8")
        )
        assert entry.get("mlflow_run_id") is None

    def test_append_handles_image_result_without_drift_share(self, tmp_path: Path):
        image_result = {
            "schema_version": "1.0.0",
            "drift_type": "image_statistical",
            "generated_at": "2026-04-14T00:00:00+00:00",
            "task_type": "image_classification_cnn",
            "overall": {
                "drift_score": 0.15,
                "dataset_drift_detected": True,
                "severity": "medium",
            },
            "channels": {},
        }
        json_path = tmp_path / "images" / "a.json"
        append_history_entry("images", image_result, json_path, tmp_path)
        entry = json.loads(
            (tmp_path / "images" / "history.jsonl").read_text(encoding="utf-8")
        )
        assert entry["overall_severity"] == "medium"
        assert entry["drift_type"] == "image_statistical"
        assert entry.get("drift_share") is None
        assert entry.get("drifted_feature_count") is None


class TestIterHistory:
    def test_iter_missing_returns_empty(self, tmp_path: Path):
        assert list(iter_history("iris", tmp_path)) == []

    def test_iter_yields_entries_in_order(self, tmp_path: Path):
        append_history_entry(
            "iris", _snapshot("low"),
            tmp_path / "iris" / "a.json", tmp_path,
        )
        append_history_entry(
            "iris", _snapshot("high"),
            tmp_path / "iris" / "b.json", tmp_path,
        )
        entries = list(iter_history("iris", tmp_path))
        assert len(entries) == 2
        assert entries[0]["overall_severity"] == "low"
        assert entries[1]["overall_severity"] == "high"

    def test_iter_skips_malformed_lines(self, tmp_path: Path):
        model_dir = tmp_path / "iris"
        model_dir.mkdir()
        (model_dir / "history.jsonl").write_text(
            json.dumps({"overall_severity": "low"}) + "\n"
            + "{broken\n"
            + json.dumps({"overall_severity": "high"}) + "\n",
            encoding="utf-8",
        )
        entries = list(iter_history("iris", tmp_path))
        assert [e["overall_severity"] for e in entries] == ["low", "high"]


class TestLoadLatestFromAppendedHistory:
    def test_load_latest_after_append(self, tmp_path: Path):
        json_path = tmp_path / "iris" / "20260414T000000Z.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = _snapshot("high")
        json_path.write_text(json.dumps(snapshot), encoding="utf-8")
        append_history_entry("iris", snapshot, json_path, tmp_path)
        result = load_latest_drift("iris", tmp_path)
        assert result is not None
        assert result["overall"]["severity"] == "high"

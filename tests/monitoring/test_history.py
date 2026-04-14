"""Tests for src.monitoring.history.load_latest_drift."""
from __future__ import annotations

import json
from pathlib import Path

from src.monitoring.history import load_latest_drift


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

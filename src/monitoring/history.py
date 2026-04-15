"""Drift history helpers: read the most recent drift result for a model.

This module loads the newest entry regardless of which shape exists yet:

1. If ``history.jsonl`` is present, parse the last valid line and read the
   JSON file it references (``json_path``).
2. Otherwise, fall back to the newest ``*.json`` snapshot in the model
   directory (excluding ``*_decision.json`` sidecars).
3. If neither exists, return ``None``.

All I/O failures (missing dirs, truncated lines, unreadable JSON) are
swallowed and reported as ``None`` — this is a governance read-back, not
a pipeline-critical data path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

DEFAULT_OUTPUTS_ROOT = Path("outputs/drift_monitoring")
_HISTORY_FILENAME = "history.jsonl"
_DECISION_SUFFIX = "_decision.json"


def append_history_entry(
    model_name: str,
    result: dict,
    json_path: Path | str,
    outputs_root: Path | str = DEFAULT_OUTPUTS_ROOT,
    mlflow_run_id: str | None = None,
) -> dict:
    """Append one index entry for ``result`` to ``history.jsonl``.

    Creates ``<outputs_root>/<model_name>/`` and an empty ``history.jsonl``
    if neither exists. Returns the entry that was written so callers can
    log it or forward it to other sinks.

    Args:
        model_name:    Model directory name under ``outputs_root``.
        result:        Full drift result dict (tabular or image shape).
        json_path:     Path to the full JSON snapshot this entry indexes.
        outputs_root:  Root of the drift monitoring tree.
        mlflow_run_id: Optional MLflow run ID to associate with this entry.

    Returns:
        The dict that was serialised as a JSONL line.
    """
    model_dir = Path(outputs_root) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    history_path = model_dir / _HISTORY_FILENAME

    overall = result.get("overall") or {}
    entry: dict = {
        "timestamp": result.get("generated_at"),
        "dataset_version_id": result.get("dataset_version_id"),
        "drift_type": result.get("drift_type"),
        "overall_severity": overall.get("severity"),
        "drift_share": overall.get("drift_share"),
        "drifted_feature_count": overall.get("drifted_feature_count"),
        "drift_score": overall.get("drift_score"),
        "json_path": str(Path(json_path).resolve()),
        "mlflow_run_id": mlflow_run_id,
    }

    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    return entry


def iter_history(
    model_name: str,
    outputs_root: Path | str = DEFAULT_OUTPUTS_ROOT,
) -> Iterator[dict]:
    """Yield each parseable entry from ``history.jsonl`` in append order.

    Silently skips malformed lines so partial writes or manual edits don't
    break downstream trending tools. Returns an empty iterator when the
    model directory or the history file is missing.
    """
    model_dir = Path(outputs_root) / model_name
    history_path = model_dir / _HISTORY_FILENAME
    if not history_path.is_file():
        return

    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Failed to read %s: %s", history_path, exc)
        return

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def load_latest_drift(
    model_name: str,
    outputs_root: Path | str = DEFAULT_OUTPUTS_ROOT,
) -> dict | None:
    """Load the most recent drift result for ``model_name``.

    Args:
        model_name:   Model name used as the subdirectory under ``outputs_root``.
        outputs_root: Root directory for drift monitoring outputs. Defaults to
                      ``outputs/drift_monitoring``.

    Returns:
        The drift result dict (same schema as ``build_drift_result``), or
        ``None`` when no history exists or the stored data cannot be read.
    """
    model_dir = Path(outputs_root) / model_name
    if not model_dir.is_dir():
        return None

    payload = _load_from_jsonl(model_dir)
    if payload is not None:
        return payload

    return _load_latest_snapshot(model_dir)


def _load_from_jsonl(model_dir: Path) -> dict | None:
    jsonl_path = model_dir / _HISTORY_FILENAME
    if not jsonl_path.is_file():
        return None

    try:
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Failed to read %s: %s", jsonl_path, exc)
        return None

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        json_path = entry.get("json_path")
        if not json_path:
            continue
        snapshot_path = Path(json_path)
        if not snapshot_path.is_absolute():
            snapshot_path = (model_dir / snapshot_path).resolve()

        snapshot = _read_json(snapshot_path)
        if snapshot is not None:
            return snapshot
    return None


def _load_latest_snapshot(model_dir: Path) -> dict | None:
    candidates = [
        p for p in model_dir.glob("*.json")
        if not p.name.endswith(_DECISION_SUFFIX)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return _read_json(candidates[-1])


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read drift snapshot %s: %s", path, exc)
        return None

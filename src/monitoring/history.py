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

logger = logging.getLogger(__name__)

DEFAULT_OUTPUTS_ROOT = Path("outputs/drift_monitoring")
_HISTORY_FILENAME = "history.jsonl"
_DECISION_SUFFIX = "_decision.json"


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

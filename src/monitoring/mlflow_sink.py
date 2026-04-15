"""MLflow sink for runtime drift results.

Runtime drift is produced by the monitor CLIs (``monitor-drift``,
``monitor-drift-image``) — long after training is complete. Training runs
are immutable records of what was fit; runtime drift is a separate
observation over live data. Mixing them into the same MLflow run muddies
both provenance and trending.

This sink therefore opens a standalone MLflow run (never nested under
an active training run) in the configured experiment and logs the drift
result under the ``drift.*`` namespace. All failures are swallowed — a
broken tracking server must never break the monitoring CLI.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def _to_mlflow_uri(raw_uri: str) -> str:
    """Convert a raw URI or local path to a form MLflow accepts."""
    parsed = urlparse(raw_uri)
    if not parsed.scheme or len(parsed.scheme) == 1:
        return Path(raw_uri).resolve().as_uri()
    return raw_uri


def log_runtime_drift_to_mlflow(
    model_name: str,
    drift_result: dict,
    tracking_uri: str | None,
    experiment_name: str,
) -> str | None:
    """Log a runtime drift result as its own MLflow run.

    Uses ``MlflowClient`` directly so the call is independent of any
    training run that may be active on the thread-local — drift is a
    runtime observation and must not nest inside an immutable training run.

    Args:
        model_name:      Model the drift was observed for; recorded as the
                         ``drift.model`` tag so the run is filterable in the UI.
        drift_result:    Full drift result dict (tabular or image shape).
        tracking_uri:    MLflow tracking URI / local path. None falls back to
                         the library default (respects ``MLFLOW_TRACKING_URI``).
        experiment_name: Target experiment. Callers typically pass the same name
                         used by the training pipeline so drift and training
                         live side-by-side in one experiment.

    Returns:
        The newly created MLflow run ID, or ``None`` if logging failed.
    """
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(_to_mlflow_uri(tracking_uri))

        # Ensure experiment exists; get its id.
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"drift-{ts}"

        client = MlflowClient()
        run = client.create_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags={"mlflow.runName": run_name},
        )
        run_id = run.info.run_id

        _write_tags(client, run_id, model_name, drift_result)
        _write_metrics(client, run_id, drift_result)

        client.set_terminated(run_id, status="FINISHED")
        return run_id
    except Exception as exc:
        logger.warning(
            "Failed to log runtime drift to MLflow (model=%s): %s",
            model_name, exc,
        )
        return None


def _write_tags(
    client: MlflowClient,
    run_id: str,
    model_name: str,
    drift_result: dict,
) -> None:
    overall = drift_result.get("overall") or {}
    drift_type = str(drift_result.get("drift_type") or "")
    severity = str(overall.get("severity") or "")
    detected = "true" if overall.get("dataset_drift_detected") else "false"

    tags = {
        "drift.runtime": "true",
        "drift.model": model_name,
        "drift.type": drift_type,
        "drift.overall_severity": severity,
        "drift.dataset_drift_detected": detected,
    }
    if "drift_share" in overall:
        tags["drift.drift_share"] = str(overall.get("drift_share"))
    if "drifted_feature_count" in overall:
        tags["drift.drifted_feature_count"] = str(
            overall.get("drifted_feature_count")
        )
    dataset_version_id = drift_result.get("dataset_version_id")
    if dataset_version_id:
        tags["drift.dataset_version_id"] = str(dataset_version_id)

    for key, value in tags.items():
        client.set_tag(run_id, key, value)


def _write_metrics(
    client: MlflowClient,
    run_id: str,
    drift_result: dict,
) -> None:
    drift_type = str(drift_result.get("drift_type") or "")
    overall = drift_result.get("overall") or {}
    metrics: dict[str, float] = {}

    if drift_type == "tabular":
        for name, data in (drift_result.get("features") or {}).items():
            score = data.get("drift_score")
            if isinstance(score, (int, float)):
                metrics[f"drift.{name}.score"] = float(score)
    else:
        if isinstance(overall.get("drift_score"), (int, float)):
            metrics["drift.image.overall_score"] = float(overall["drift_score"])
        for ch, data in (drift_result.get("channels") or {}).items():
            score = data.get("drift_score")
            if isinstance(score, (int, float)):
                metrics[f"drift.image.{ch}_score"] = float(score)

    ts_ms = int(time.time() * 1000)
    for name, value in metrics.items():
        client.log_metric(run_id, name, value, timestamp=ts_ms, step=0)

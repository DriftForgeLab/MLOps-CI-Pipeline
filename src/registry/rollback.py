"""
src/registry/rollback.py — Rollback production model to a previous registry version.

CLI usage:
    rollback-model --config src/config/pipeline.yaml
    rollback-model --config src/config/pipeline.yaml --version 2
    rollback-model --config src/config/pipeline.yaml --version 2 --reason "accuracy regression" --yes
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow as _mlflow

from src.config.loader import load_config, PipelineConfig
from src.common.io import atomic_write_json
from src.registry.model_registry import (
    resolve_model_name,
    get_mlflow_client,
    promote_to_production,
    resolve_tracking_uri,
)

logger = logging.getLogger(__name__)


def _get_cli_actor() -> str:
    """Best-effort OS-level actor identity for audit records.

    NOTE: This is NOT authenticated identity. It records the OS username as a
    lightweight audit hint. Do not treat as authoritative attribution until a
    proper identity system is in place.
    """
    username = os.environ.get("USER") or os.environ.get("USERNAME") or "unidentified"
    return f"cli:{username}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll back a production model to a previous registry version."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to pipeline.yaml config file.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Target registry version number to roll back to.",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default=None,
        help="Audit reason for the rollback.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Skip confirmation prompt.",
    )
    return parser.parse_args()


def _get_version_tag(version, tag_key: str, fallback: str = "") -> str:
    """Return the value of a tag on a model version, or fallback if absent."""
    tags = version.tags or {}
    return tags.get(tag_key, fallback)


def _print_versions_table(versions: list, model_name: str) -> None:
    """Print a formatted table of model versions to stdout."""
    print(f"\nAvailable versions for '{model_name}':")
    header_version = "VERSION"
    header_stage = "STAGE"
    header_trained = "TRAINED AT"
    header_algo = "ALGORITHM"

    col_version = max(len(header_version), max((len(str(v.version)) for v in versions), default=0))
    col_stage = max(len(header_stage), max((len(v.current_stage or "") for v in versions), default=0))
    col_trained = max(
        len(header_trained),
        max(
            (len(_get_version_tag(v, "lineage.trained_at") or v.run_id or "") for v in versions),
            default=0,
        ),
    )
    col_algo = max(
        len(header_algo),
        max(
            (len(_get_version_tag(v, "lineage.algorithm") or "") for v in versions),
            default=0,
        ),
    )

    def _row(version_str, stage_str, trained_str, algo_str):
        return (
            f"{version_str:<{col_version}}  "
            f"{stage_str:<{col_stage}}  "
            f"{trained_str:<{col_trained}}  "
            f"{algo_str:<{col_algo}}"
        )

    print(_row(header_version, header_stage, header_trained, header_algo))
    print(_row("-" * col_version, "-" * col_stage, "-" * col_trained, "-" * col_algo))

    for v in versions:
        trained_at = _get_version_tag(v, "lineage.trained_at") or v.run_id or ""
        algorithm = _get_version_tag(v, "lineage.algorithm") or ""
        print(_row(str(v.version), v.current_stage or "", trained_at, algorithm))

    print()


def _find_current_production(versions: list):
    """Return the version currently in Production stage, or None.

    Warns if the single-Production invariant is violated (should not happen
    given archive_existing_versions=True, but registry state can be manually
    altered). Returns the highest-numbered Production version if multiple exist.
    """
    prod = [v for v in versions if v.current_stage == "Production"]
    if len(prod) > 1:
        logger.warning(
            "Registry invariant violated: %d versions in Production (%s). "
            "Using highest version number. Investigate registry state manually.",
            len(prod),
            [v.version for v in prod],
        )
        return max(prod, key=lambda v: int(v.version))
    return prod[0] if prod else None


def _build_decision_data(
    model_name: str,
    target_version,
    previous_production,
    reason: str,
    actor: str,
    event_ts: datetime,
    status: str,
) -> dict:
    """Single source of truth for the rollback audit record dict."""
    return {
        "event_type": "rollback",
        "status": status,
        "model_name": model_name,
        "from_version": int(previous_production.version) if previous_production else None,
        "to_version": int(target_version.version),
        "reason": reason,
        "executed_at": event_ts.isoformat(),
        "executed_by": actor,
        "from_run_id": previous_production.run_id if previous_production else None,
        "to_run_id": target_version.run_id or "",
    }


def _write_rollback_intent(
    config: PipelineConfig,
    model_name: str,
    target_version,
    previous_production,
    reason: str,
    actor: str,
    event_ts: datetime,
    timestamp_str: str,
) -> Path:
    """Write the pre-promotion intent record with status='initiated'.

    Called BEFORE promote_to_production so an audit trail exists even if the
    promotion or subsequent steps fail.
    """
    output_dir = Path(config.output_dir)
    decision_path = output_dir / f"rollback_decision_{timestamp_str}.json"
    decision_data = _build_decision_data(
        model_name, target_version, previous_production, reason, actor, event_ts,
        status="initiated",
    )
    atomic_write_json(decision_path, decision_data)
    logger.info("Rollback intent written (pre-promotion): %s", decision_path)
    return decision_path


def _complete_rollback_audit(
    config: PipelineConfig,
    model_name: str,
    target_version,
    previous_production,
    reason: str,
    actor: str,
    event_ts: datetime,
    timestamp_str: str,
    decision_path: Path,
) -> None:
    """Post-promotion audit: set governance tags, finalise intent record, create governance run.

    Policy: this function is REQUIRED after a successful promotion. Any exception
    raised here must be treated as a hard error by the caller (exit non-zero),
    because the registry state has already changed and an incomplete audit trail
    requires operator attention.
    """
    client = get_mlflow_client(config)
    version_str = str(target_version.version)
    previous_version_number = int(previous_production.version) if previous_production else None

    # Overwrite promoted_by: promote_to_production hardcodes "pipeline", which is
    # wrong for operator-initiated rollbacks.
    client.set_model_version_tag(model_name, version_str, "promotion.promoted_by", "rollback-cli")

    # Replace decision_run_id: promote_to_production sets it to the training run_id,
    # which is not the decision authority for a rollback. Use a rollback-specific ID.
    rollback_id = f"rollback:{timestamp_str}"
    client.set_model_version_tag(model_name, version_str, "promotion.decision_run_id", rollback_id)

    # Rollback-specific governance tags
    client.set_model_version_tag(model_name, version_str, "governance.last_rollback_at", event_ts.isoformat())
    client.set_model_version_tag(model_name, version_str, "governance.rollback_reason", reason)
    client.set_model_version_tag(model_name, version_str, "governance.rollback_executed_by", actor)

    # Update intent JSON to completed status (same path, atomic overwrite)
    decision_data = _build_decision_data(
        model_name, target_version, previous_production, reason, actor, event_ts,
        status="completed",
    )
    atomic_write_json(decision_path, decision_data)

    # Create searchable governance MLflow run.
    # This run is the cross-model queryable audit record in the MLflow UI.
    # It is required, not best-effort. Failure here propagates to the caller.
    tracking_uri = resolve_tracking_uri(config)
    _mlflow.set_tracking_uri(tracking_uri)
    _mlflow.set_experiment(f"{model_name}-governance")
    with _mlflow.start_run(run_name=f"rollback-v{version_str}"):
        _mlflow.set_tags({
            "governance.event_type": "rollback",
            "governance.model_name": model_name,
            "governance.from_version": str(previous_version_number) if previous_version_number is not None else "none",
            "governance.to_version": version_str,
            "governance.reason": reason,
            "governance.executed_at": event_ts.isoformat(),
            "governance.executed_by": actor,
        })
        if decision_path.exists():
            _mlflow.log_artifact(str(decision_path))

    logger.info("Rollback audit completed: %s", decision_path)


def _execute_rollback(
    config: PipelineConfig,
    version: int | None,
    reason: str | None,
    yes: bool,
) -> None:
    """Validate, confirm, and execute the rollback with correct audit ordering."""
    model_name = resolve_model_name(config)
    client = get_mlflow_client(config)
    actor = _get_cli_actor()

    # Single version fetch — main() no longer pre-fetches
    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as exc:
        print(f"Error: Could not fetch model versions for '{model_name}': {exc}")
        sys.exit(1)

    if not all_versions:
        print(f"Error: No versions found for model '{model_name}'. Does the model exist in the registry?")
        sys.exit(1)

    all_versions = sorted(all_versions, key=lambda v: int(v.version), reverse=True)
    _print_versions_table(all_versions, model_name)

    # Resolve target version interactively if not provided via CLI
    if version is not None:
        target_version_str = str(version)
    else:
        try:
            target_version_str = input("Enter version number to roll back to: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(1)
        if not target_version_str:
            print("Error: No version number entered.")
            sys.exit(1)

    try:
        target_version_number = int(target_version_str)
    except (ValueError, TypeError):
        print(f"Error: Invalid version number '{target_version_str}'.")
        sys.exit(1)

    if target_version_number <= 0:
        print(f"Error: Version number must be positive, got {target_version_number}.")
        sys.exit(1)

    # Resolve reason interactively if not provided via CLI
    if not reason:
        try:
            reason = input("Enter rollback reason (or press Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            reason = ""
    reason = reason or "no reason provided"

    # Find the target version object
    target_version = next(
        (v for v in all_versions if int(v.version) == target_version_number),
        None,
    )
    if target_version is None:
        print(f"Error: Version {target_version_number} does not exist for model '{model_name}'.")
        sys.exit(1)

    # Validate: status must be READY
    if target_version.status != "READY":
        print(
            f"Error: Version {target_version_number} has status '{target_version.status}'. "
            "Only READY versions can be promoted."
        )
        sys.exit(1)

    # Recovery path: detect incomplete governance from a prior interrupted rollback.
    # _complete_rollback_audit overwrites promotion.promoted_by to "rollback-cli" as its
    # first tag write. If that tag is set but governance.last_rollback_at is absent, a
    # previous rollback completed the promotion and started the audit but failed mid-flight.
    if target_version.current_stage == "Production":
        tags = target_version.tags or {}
        rollback_partial = (
            tags.get("promotion.promoted_by") == "rollback-cli"
            and "governance.last_rollback_at" not in tags
        )
        if rollback_partial:
            print(
                f"\nWARNING: Version {target_version_number} is already in Production but "
                "governance.last_rollback_at is missing despite promotion.promoted_by=rollback-cli.\n"
                "A previous rollback likely succeeded at promotion but failed mid-audit.\n"
                "The registry state is potentially inconsistent. Review:\n"
                f"  - rollback_decision_*.json files in '{config.output_dir}'\n"
                f"  - MLflow tags for model '{model_name}' version {target_version_number}\n"
                "Manual remediation is required."
            )
            sys.exit(2)
        print(
            f"Error: Version {target_version_number} is already in Production stage. "
            "Nothing to roll back to."
        )
        sys.exit(1)

    # Identify current Production
    current_production = _find_current_production(all_versions)

    # Build and print confirmation summary
    target_algo = _get_version_tag(target_version, "lineage.algorithm") or "unknown"
    target_run = target_version.run_id or "unknown"

    if current_production:
        prod_algo = _get_version_tag(current_production, "lineage.algorithm") or "unknown"
        prod_run = current_production.run_id or "unknown"
        print(
            f"Current Production: version {current_production.version} "
            f"(run: {prod_run}, algorithm: {prod_algo})"
        )
        print(
            f"Target for rollback: version {target_version_number} "
            f"(run: {target_run}, algorithm: {target_algo})"
        )
        print()
        print(
            f"WARNING: This will archive version {current_production.version} "
            f"and promote version {target_version_number} to Production."
        )
    else:
        print("Current Production: none")
        print(
            f"Target for rollback: version {target_version_number} "
            f"(run: {target_run}, algorithm: {target_algo})"
        )
        print()
        print(f"WARNING: This will promote version {target_version_number} to Production.")

    print("This action will be logged as a governance decision.")
    print()

    if yes:
        print("(--yes flag: skipping confirmation prompt)")
    else:
        answer = input("Confirm rollback? [y/N]: ").strip().lower()
        if answer != "y":
            print("Rollback cancelled.")
            sys.exit(0)

    # Single event timestamp — propagated to all audit records
    event_ts = datetime.now(timezone.utc)
    timestamp_str = event_ts.strftime("%Y%m%dT%H%M%SZ")

    # Guard against None run_id (programmatically created versions without an associated run)
    run_id = target_version.run_id
    if not run_id:
        logger.warning(
            "Version %s has no associated run_id. Proceeding, but promotion.decision_run_id "
            "will not reference a training run.",
            target_version_number,
        )
        run_id = ""

    # Step 1: Write pre-promotion intent — audit trail exists before state changes
    decision_path = _write_rollback_intent(
        config=config,
        model_name=model_name,
        target_version=target_version,
        previous_production=current_production,
        reason=reason,
        actor=actor,
        event_ts=event_ts,
        timestamp_str=timestamp_str,
    )

    # Step 2: Execute production promotion (irreversible registry state change)
    promote_to_production(config, target_version_number, run_id)

    # Step 3: Complete post-promotion audit — mandatory, not best-effort.
    # If this fails, registry state has changed but governance trail is incomplete.
    try:
        _complete_rollback_audit(
            config=config,
            model_name=model_name,
            target_version=target_version,
            previous_production=current_production,
            reason=reason,
            actor=actor,
            event_ts=event_ts,
            timestamp_str=timestamp_str,
            decision_path=decision_path,
        )
    except Exception as exc:
        logger.error(
            "GOVERNANCE INCOMPLETE: Production promotion succeeded (version %s is now Production) "
            "but post-promotion audit failed: %s",
            target_version_number,
            exc,
        )
        try:
            actual_status = json.loads(decision_path.read_text()).get("status", "unknown")
        except Exception:
            actual_status = "unknown"
        print(
            f"\nERROR: Rollback executed but governance audit failed: {exc}\n"
            f"Version {target_version_number} IS now Production.\n"
            f"Audit record at '{decision_path}' has status='{actual_status}'.\n"
            "Inspect the file and MLflow registry tags to assess what requires manual remediation."
        )
        sys.exit(2)

    print(f"\nRollback complete: version {target_version_number} is now Production.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    args = _parse_args()

    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading config: {exc}")
        sys.exit(1)

    _execute_rollback(
        config=config,
        version=args.version,
        reason=args.reason,
        yes=args.yes,
    )

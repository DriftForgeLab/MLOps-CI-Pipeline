"""CLI entry point for online image drift monitoring.

Compares a new production batch against the training reference (train.npz)
and reports the statistical distance between the two image distributions.

    reference = stored training data (train.npz from the trained model version)
    batch     = new, incoming production images

Detection method is auto-selected based on pipeline type:
    Standard JPG/PNG pipelines  → embedding (CNN feature MMD)  [model auto-resolved]
    Raw DNG / ISP pipelines     → statistical (Wasserstein per channel)

ONE-STEP usage — standard JPG/PNG (preprocessing + monitoring in one command):
    monitor-drift-image \\
        --input-dir data/incoming/my_new_images \\
        --config src/config/pipeline_image.yaml

ONE-STEP usage — raw-image pipeline with ISP scenario interpretation:
    monitor-drift-image \\
        --input-dir data/incoming/drone_batch_001 \\
        --config src/config/pipeline_raw_image.yaml

    ISP scenario dir and sensitivity report are auto-resolved from
    data/drift_scenarios/<dataset>/<version_id>/ when not specified.

TWO-STEP usage (if batch NPZ already exists from prepare-image-batch):
    monitor-drift-image \\
        --batch-npz data/batches/my_batch.npz \\
        --config src/config/pipeline_image.yaml

Exactly one of --input-dir or --batch-npz must be given.

Auto-resolution:
    dataset name   — from 'dataset' field in pipeline config
    dataset version — most recently preprocessed version (or --dataset-version)
    model.pt        — artifacts/runs/<version>/model/model.pt (standard pipelines)

What is measured:
    Standard pipelines (embedding):
        CNN feature MMD between batch and reference feature distributions.
        Catches semantic and structural drift that pixel histograms miss.

    Raw-image pipelines (statistical):
        Per-channel Wasserstein distance between batch and reference pixel
        distributions. ISP-induced drift is systematic and channel-specific,
        making Wasserstein a well-motivated detector here.

        With --drift-scenarios-dir: the observed drift profile is matched
        against pre-computed ISP scenario signatures (e.g. "warm white balance
        shift"). This is interpretation of detected drift, not a separate
        detection step.

Output:
    <output_dir>/<timestamp>.json           — monitoring result
    <output_dir>/<timestamp>_decision.json  — user decision (if interactive)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config.loader import load_config, load_drift_config
from src.common.io import atomic_write_json, atomic_write_npz
from src.monitoring.history import append_history_entry
from src.monitoring.image_drift_monitor import monitor_image_batch
from src.monitoring.mlflow_sink import log_runtime_drift_to_mlflow
from src.monitoring.drift_decision import EXIT_CODE_DRIFT_GATE, should_trip_ci_gate
from src.drift.interpret import _SEVERITY_ORD
from src.data.prepare_batch import resolve_latest_version

logger = logging.getLogger(__name__)


# =============================================================================
# Argument parsing
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run online image batch drift monitoring against the training reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Batch input (mutually exclusive: --input-dir or --batch-npz) ---
    batch_group = parser.add_mutually_exclusive_group(required=True)
    batch_group.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=(
            "Directory containing new production images (flat or ImageFolder). "
            "Preprocesses images on the fly using the same pipeline as training, "
            "then runs drift monitoring — no separate prepare-image-batch step needed."
        ),
    )
    batch_group.add_argument(
        "--batch-npz",
        type=str,
        default=None,
        help=(
            "Path to a pre-existing preprocessed batch NPZ file. "
            "Use this if you already ran prepare-image-batch separately."
        ),
    )

    # --- Reference (all optional — auto-resolved by default) ---
    parser.add_argument(
        "--reference-npz",
        type=str,
        default=None,
        help=(
            "Explicit path to the training reference NPZ (train.npz). "
            "If not provided, auto-resolved from --dataset-name and --dataset-version."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name. Defaults to the 'dataset' field in the pipeline config.",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help=(
            "Dataset version ID (content-addressed hash). "
            "Defaults to the most recently preprocessed version."
        ),
    )

    # --- Config ---
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to pipeline config YAML (e.g. src/config/pipeline_image.yaml).",
    )
    parser.add_argument(
        "--drift-config",
        type=str,
        default=None,
        help="Optional path to an override drift config YAML.",
    )

    # --- Detection method ---
    parser.add_argument(
        "--method",
        choices=["statistical", "embedding"],
        default=None,
        help=(
            "Drift detection method. Auto-selected based on pipeline type if not given.\n"
            "  'statistical': per-channel Wasserstein distance on pixel distributions. "
            "Default for raw-image (DNG) pipelines — ISP changes produce systematic, "
            "channel-specific pixel shifts. Enables ISP scenario interpretation.\n"
            "  'embedding': CNN feature MMD. Default for standard (JPG/PNG) pipelines — "
            "detects semantic and structural drift that pixel histograms miss. "
            "Requires model.pt (auto-resolved or via --model-pt)."
        ),
    )
    parser.add_argument(
        "--model-pt",
        type=str,
        default=None,
        help=(
            "Path to model.pt (PyTorch CNN). Used by --method=embedding. "
            "Auto-resolved from --artifact-dir/<version>/model/model.pt when not given."
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="artifacts/runs",
        help=(
            "Base directory for model artifacts used to auto-resolve model.pt. "
            "Defaults to artifacts/runs."
        ),
    )

    # --- ISP scenario interpretation (raw-image pipelines only) ---
    parser.add_argument(
        "--drift-scenarios-dir",
        type=str,
        default=None,
        help=(
            "Path to drift scenarios directory (data/drift_scenarios/). "
            "Enables ISP scenario interpretation for raw-image pipelines: "
            "the observed drift profile is matched against pre-computed ISP signatures."
        ),
    )
    parser.add_argument(
        "--sensitivity-report",
        type=str,
        default=None,
        help=(
            "Path to sensitivity_report.json (from the offline ISP sensitivity analysis). "
            "Annotates the matched scenario with estimated accuracy drop and gradient sensitivity."
        ),
    )

    # --- Output ---
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for monitoring output. Defaults to outputs/drift_monitoring/images/.",
    )

    return parser.parse_args()


# =============================================================================
# Helpers (extracted for testability)
# =============================================================================

def _resolve_method(
    explicit_method: str | None,
    is_raw_isp_pipeline: bool,
    model_pt_available: bool,
) -> str:
    """Select drift detection method, logging the rationale.

    Args:
        explicit_method:      Value of --method if given by the user, else None.
        is_raw_isp_pipeline:  True when preprocessing config has raw_input=True and isp block.
        model_pt_available:   True when a model.pt path is known (explicit or auto-resolved).

    Returns:
        "statistical" or "embedding".
    """
    if explicit_method is not None:
        if is_raw_isp_pipeline and explicit_method == "embedding":
            logger.warning(
                "Embedding method on a raw-image pipeline disables ISP scenario "
                "interpretation (no per-channel breakdown). Consider --method statistical."
            )
        elif not is_raw_isp_pipeline and explicit_method == "statistical":
            logger.warning(
                "Statistical (Wasserstein) method on a standard JPG/PNG pipeline only "
                "detects pixel-level shifts — semantic drift may go undetected. "
                "Consider --method embedding (model auto-resolved or use --model-pt)."
            )
        return explicit_method

    if is_raw_isp_pipeline:
        logger.info(
            "Auto-selected method=statistical for raw-image pipeline "
            "(ISP changes produce channel-specific pixel shifts; enables scenario matching)."
        )
        return "statistical"

    if model_pt_available:
        logger.info(
            "Auto-selected method=embedding for standard JPG/PNG pipeline "
            "(CNN feature MMD detects semantic and structural drift, not just pixel histograms)."
        )
        return "embedding"

    logger.warning(
        "Standard JPG/PNG pipeline: falling back to method=statistical "
        "(no model available for embedding). Run the pipeline first so a model artifact "
        "exists, or supply --model-pt <path> explicitly."
    )
    return "statistical"


def _auto_resolve_model_pt(
    artifact_dir: Path,
    version_id: str | None,
    is_raw_isp_pipeline: bool,
) -> Path | None:
    """Try to locate model.pt from the artifact directory.

    Only attempted for standard (non-ISP) pipelines when version_id is known.

    Args:
        artifact_dir:        Base directory for model artifacts (e.g. artifacts/runs).
        version_id:          Dataset/run version hash, or None if not resolved.
        is_raw_isp_pipeline: Skip auto-resolution for raw-ISP pipelines.

    Returns:
        Path to model.pt if found, else None.
    """
    if is_raw_isp_pipeline or version_id is None:
        return None
    candidate = artifact_dir / version_id / "model" / "model.pt"
    if candidate.exists():
        logger.info("Auto-resolved model: %s", candidate)
        return candidate
    logger.warning(
        "Model not found at auto-resolved path '%s'. "
        "Supply --model-pt <path> to enable embedding-based drift detection.",
        candidate,
    )
    return None


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()

    # --- Load pipeline config ---
    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load pipeline config: %s", e)
        sys.exit(1)

    drift_config_path = (
        Path(args.drift_config) if args.drift_config else Path(config.configs.drift)
    )
    try:
        drift_config = load_drift_config(drift_config_path.resolve())
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load drift config: %s", e)
        sys.exit(1)

    # --- Determine pipeline type ---
    from src.config.loader import load_preprocessing_config
    prep_config_path = Path(config.configs.preprocessing).resolve()
    try:
        prep_config = load_preprocessing_config(prep_config_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load preprocessing config: %s", e)
        sys.exit(1)

    is_raw_isp_pipeline = bool(
        prep_config.image and prep_config.image.raw_input and prep_config.image.isp
    )

    # --- Preprocess from directory if --input-dir given ---
    # Resolves version_id, loads training stats, preprocesses images, and saves a
    # batch NPZ — equivalent to running prepare-image-batch then monitor-drift-image.
    version_id: str | None = None
    if args.input_dir:
        from src.data.prepare_batch import load_training_stats, prepare_image_batch

        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            logger.error("Input directory not found: %s", input_dir)
            sys.exit(1)

        dataset_name = args.dataset_name or config.dataset
        processed_dir = Path(config.data.processed)

        if args.dataset_version:
            version_id = args.dataset_version
            logger.info("Dataset: %s  version: %s (explicit)", dataset_name, version_id)
        else:
            try:
                version_id = resolve_latest_version(processed_dir, dataset_name)
            except FileNotFoundError as e:
                logger.error("%s", e)
                sys.exit(1)
            logger.info(
                "Dataset: %s  version: %s (auto-resolved, newest)", dataset_name, version_id
            )

        try:
            norm_mean, norm_std, expected_formats = load_training_stats(
                processed_dir, dataset_name, version_id
            )
        except FileNotFoundError as e:
            logger.error("%s", e)
            sys.exit(1)

        logger.info("Preprocessing images from '%s'...", input_dir)
        try:
            X_batch = prepare_image_batch(
                input_dir=input_dir,
                prep_config=prep_config,
                norm_mean=norm_mean,
                norm_std=norm_std,
                expected_formats=expected_formats,
            )
        except (ValueError, ImportError) as e:
            logger.error("Batch preprocessing failed: %s", e)
            sys.exit(1)

        timestamp_npz = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        npz_dir = Path("data/batches")
        npz_dir.mkdir(parents=True, exist_ok=True)
        batch_npz_path = npz_dir / f"{timestamp_npz}.npz"
        atomic_write_npz(batch_npz_path, X=X_batch, version_id=np.array(version_id))
        logger.info(
            "Batch preprocessed: %d images → %s", len(X_batch), batch_npz_path
        )
        args.batch_npz = str(batch_npz_path)

    # --- Load batch ---
    batch_path = Path(args.batch_npz)
    if not batch_path.exists():
        logger.error("Batch NPZ not found: %s", batch_path)
        sys.exit(1)
    try:
        batch_data = np.load(batch_path)
        batch = batch_data["X"]
    except Exception as e:
        logger.error("Failed to load batch NPZ '%s': %s", batch_path, e)
        sys.exit(1)
    logger.info("Loaded batch: %s  shape=%s", batch_path.name, batch.shape)

    # Extract version_id embedded in NPZ by prepare-image-batch or --input-dir mode
    batch_version_id: str | None = None
    if "version_id" in batch_data:
        try:
            batch_version_id = str(batch_data["version_id"])
        except Exception:
            pass

    # --- Load reference ---
    if args.reference_npz:
        ref_path = Path(args.reference_npz)
        if not ref_path.exists():
            logger.error("Reference NPZ not found: %s", ref_path)
            sys.exit(1)
        try:
            reference = np.load(ref_path)["X"]
        except Exception as e:
            logger.error("Failed to load reference NPZ: %s", e)
            sys.exit(1)
        logger.info("Loaded reference: %s  shape=%s", ref_path.name, reference.shape)
    else:
        dataset_name = args.dataset_name or config.dataset
        processed_dir = Path(config.data.processed)

        # version_id may already be set by --input-dir preprocessing above
        if version_id is None:
            if args.dataset_version:
                version_id = args.dataset_version
                logger.info("Dataset: %s  version: %s (explicit)", dataset_name, version_id)
            else:
                try:
                    version_id = resolve_latest_version(processed_dir, dataset_name)
                except FileNotFoundError as e:
                    logger.error("%s", e)
                    sys.exit(1)
                logger.info(
                    "Dataset: %s  version: %s (auto-resolved, newest)",
                    dataset_name, version_id,
                )

        if batch_version_id is not None and batch_version_id != version_id:
            logger.warning(
                "Version mismatch: batch NPZ was prepared with version '%s' but "
                "reference is version '%s'. Drift scores may be invalid — "
                "re-run with the correct dataset version.",
                batch_version_id,
                version_id,
            )

        from src.monitoring.image_drift_monitor import load_reference_images
        try:
            reference = load_reference_images(
                dataset_name=dataset_name,
                version_id=version_id,
                processed_dir=processed_dir,
                split="train",
            )
        except FileNotFoundError as e:
            logger.error("Reference data not found: %s", e)
            sys.exit(1)
        logger.info("Loaded reference: shape=%s", reference.shape)

    # --- Auto-resolve model.pt (standard pipelines, version_id known) ---
    if not args.model_pt:
        resolved = _auto_resolve_model_pt(
            artifact_dir=Path(args.artifact_dir),
            version_id=version_id,
            is_raw_isp_pipeline=is_raw_isp_pipeline,
        )
        if resolved:
            args.model_pt = str(resolved)

    # --- Select drift detection method ---
    args.method = _resolve_method(
        explicit_method=args.method,
        is_raw_isp_pipeline=is_raw_isp_pipeline,
        model_pt_available=bool(args.model_pt),
    )

    # --- Load model ---
    model = None
    if args.method == "embedding":
        if not args.model_pt:
            logger.error(
                "--model-pt is required when --method=embedding and could not be auto-resolved."
            )
            sys.exit(1)
        pt_path = Path(args.model_pt)
        if not pt_path.exists():
            logger.error("Model file not found: %s", pt_path)
            sys.exit(1)
        try:
            import torch
            from src.common.device import resolve_device
            device = resolve_device()
            model = torch.load(pt_path, weights_only=False, map_location="cpu")
            if hasattr(model, "to"):
                model.to(device)
        except Exception as e:
            logger.error("Failed to load model from '%s': %s", pt_path, e)
            sys.exit(1)
        logger.info("Loaded model for embedding drift: %s (device=%s)", pt_path.name, device)

    # --- Resolve optional ISP scenario matching ---
    # For raw-ISP pipelines, auto-resolve the versioned drift_scenarios dir when
    # not explicitly provided: data/drift_scenarios/<dataset>/<version_id>/
    drift_scenarios_dir: Path | None = None
    if args.drift_scenarios_dir:
        drift_scenarios_dir = Path(args.drift_scenarios_dir)
    elif is_raw_isp_pipeline and version_id is not None:
        dataset_name = args.dataset_name or config.dataset
        candidate = Path(config.data.drift_scenarios) / dataset_name / version_id
        if candidate.exists():
            drift_scenarios_dir = candidate
            logger.info(
                "Auto-resolved drift scenarios dir: %s", drift_scenarios_dir
            )
        else:
            logger.info(
                "No pre-computed ISP scenarios found at '%s' — "
                "running without scenario interpretation. "
                "Run the pipeline with model_analysis to generate scenarios.",
                candidate,
            )

    sensitivity_report_path: Path | None = None
    if args.sensitivity_report:
        sensitivity_report_path = Path(args.sensitivity_report)
    elif drift_scenarios_dir is not None:
        candidate_report = drift_scenarios_dir / "sensitivity_report.json"
        if candidate_report.exists():
            sensitivity_report_path = candidate_report
            logger.info("Auto-resolved sensitivity report: %s", sensitivity_report_path)

    is_raw_isp = drift_scenarios_dir is not None
    if is_raw_isp:
        logger.info(
            "ISP scenario interpretation enabled — drift_scenarios_dir=%s", drift_scenarios_dir
        )

    # --- Run drift monitoring ---
    drift_result = monitor_image_batch(
        batch=batch,
        reference=reference,
        drift_config=drift_config,
        method=args.method,
        model=model,
        task_type=config.task_type,
        drift_scenarios_dir=drift_scenarios_dir,
        sensitivity_report_path=sensitivity_report_path,
    )

    if drift_result is None:
        logger.info(
            "Batch has fewer than min_batch_size=%d images — no drift analysis performed.",
            drift_config.monitoring.min_batch_size,
        )
        sys.exit(0)

    # --- Print summary ---
    overall = drift_result["overall"]
    ref_info   = drift_result.get("reference", {})
    batch_info = drift_result.get("batch", {})
    print("\n" + "=" * 60)
    print("  IMAGE DRIFT MONITORING RESULT")
    print("=" * 60)
    print(f"\n  Method:        {drift_result.get('method', '?')}")
    print(f"  Reference:     {ref_info.get('size', '?')} training images")
    print(f"  Batch:         {batch_info.get('size', '?')} images")
    print(f"  Drift score:   {overall.get('drift_score', '?'):.6f}")
    print(f"  Severity:      {overall.get('severity', '?').upper()}")
    print(f"  Drift detected: {overall.get('dataset_drift_detected', '?')}")

    channels = drift_result.get("channels", {})
    if channels:
        print("\n  Per-channel Wasserstein scores:")
        for ch, data in channels.items():
            print(f"    {ch}: {data['drift_score']:.6f}  ({data['severity']})")

    scenario_interp = drift_result.get("scenario_interpretation")
    if scenario_interp:
        matched_name = scenario_interp['matched_scenario']
        print(f"\n  Closest ISP scenario (interpretation):")
        print(f"    Scenario:       {matched_name}")
        desc = scenario_interp.get("description", "")
        if desc:
            print(f"    Description:    {desc}")
        print(f"    Match distance: {scenario_interp['match_distance']:.4f}  "
              f"(confidence: {scenario_interp['confidence']})")
        drop = scenario_interp.get("estimated_accuracy_drop")
        if drop is not None:
            print(f"    Est. acc. drop: {drop:+.4f}")

        if sensitivity_report_path and Path(sensitivity_report_path).exists():
            try:
                import json as _json
                with open(sensitivity_report_path) as _f:
                    sens_report = _json.load(_f)
                matched_scenario_data = next(
                    (s for s in sens_report.get("scenarios", []) if s["name"] == matched_name),
                    None,
                )
                if matched_scenario_data:
                    grad_sens = matched_scenario_data.get("channel_gradient_sensitivity")
                    if grad_sens:
                        grad_str = "  |  ".join(
                            f"{ch}: {v:.4f}" for ch, v in grad_sens.items()
                        )
                        most_sensitive_ch = max(grad_sens, key=lambda k: grad_sens[k])
                        print(f"    Model gradient sensitivity (per channel): {grad_str}")
                        print(f"    Most sensitive channel: {most_sensitive_ch} "
                              f"— the model is most vulnerable to drift in this channel "
                              f"under the '{matched_name}' ISP configuration.")
            except Exception:
                pass

        print("    Note: this interpretation identifies a plausible cause of drift,")
        print("          not a separate detection — the Wasserstein score is the measurement.")
    elif is_raw_isp:
        print("\n  ISP scenario interpretation: none (pattern does not resemble known scenarios)")

    print("\n" + "=" * 60)

    # --- Resolve model name (same as MLflow registry name used by promotion) ---
    from src.registry.model_registry import resolve_model_name
    registry_model_name = resolve_model_name(config)

    # --- Save results ---
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs/drift_monitoring") / registry_model_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{timestamp}.json"
    atomic_write_json(output_path, drift_result)
    logger.info("Drift result written to %s", output_path)

    # --- Persistence: MLflow runtime-drift run + JSONL index ---
    experiment_name = config.mlflow.experiment_name or config.project.name
    mlflow_run_id = log_runtime_drift_to_mlflow(
        model_name=registry_model_name,
        drift_result=drift_result,
        tracking_uri=config.mlflow.tracking_uri,
        experiment_name=experiment_name,
    )
    if mlflow_run_id:
        logger.info("Runtime drift logged to MLflow run %s", mlflow_run_id)

    append_history_entry(
        model_name=registry_model_name,
        result=drift_result,
        json_path=output_path,
        outputs_root=output_dir.parent,
        mlflow_run_id=mlflow_run_id,
    )

    # --- Decision gate ---
    overall_severity = overall["severity"]
    alert_severity = drift_config.monitoring.alert_severity
    is_interactive = sys.stdin.isatty()
    if is_interactive and (
        _SEVERITY_ORD.get(overall_severity, 0) >= _SEVERITY_ORD.get(alert_severity, 0)
    ):
        from src.monitoring.drift_decision import request_drift_decision
        decision = request_drift_decision(
            drift_result,
            is_image_isp=is_raw_isp,
            is_image_cnn=True,
            drift_report_linked=output_path.name,
            config_path=args.config,
        )
        if decision is not None:
            decision_path = output_dir / f"{timestamp}_decision.json"
            atomic_write_json(decision_path, decision.to_dict())
            logger.info("Drift decision written to %s", decision_path)

    # --- CI exit-code gate (non-interactive only) ---
    fail_on = drift_config.monitoring.fail_on_severity
    if should_trip_ci_gate(overall_severity, fail_on, is_interactive):
        logger.error(
            "Drift gate tripped: severity=%s >= fail_on_severity=%s (exit %d)",
            overall_severity, fail_on, EXIT_CODE_DRIFT_GATE,
        )
        sys.exit(EXIT_CODE_DRIFT_GATE)

    sys.exit(0)


if __name__ == "__main__":
    main()

"""CLI entry point for online image drift monitoring.

Compares a new production batch (NPZ) against the training reference (train.npz)
and reports the statistical distance between the two image distributions.

    reference = stored training data (train.npz from the trained model version)
    batch     = new, incoming production images (preprocessed to NPZ)

Minimal usage (auto-resolves dataset name and latest version from config):
    monitor-drift-image \\
        --batch-npz data/batches/my_batch.npz \\
        --config src/config/pipeline_image.yaml

Usage — raw-image pipeline with ISP scenario interpretation:
    monitor-drift-image \\
        --batch-npz data/batches/drone_batch.npz \\
        --config src/config/pipeline_raw_image.yaml \\
        --drift-scenarios-dir data/drift_scenarios/ \\
        --sensitivity-report data/drift_scenarios/sensitivity_report.json

Usage — explicit dataset version:
    monitor-drift-image \\
        --batch-npz data/batches/my_batch.npz \\
        --config src/config/pipeline_image.yaml \\
        --dataset-version abc123hash

Usage — explicit reference path:
    monitor-drift-image \\
        --batch-npz data/batches/my_batch.npz \\
        --reference-npz data/processed/my_dataset/abc123/preprocessed/train.npz \\
        --config src/config/pipeline_image.yaml

Typical workflow:
    1. Collect new production images in data/incoming/
    2. Run: prepare-image-batch --input-dir data/incoming/my_batch --config <config>
    3. Run: monitor-drift-image --batch-npz data/batches/<timestamp>.npz --config <config>

What is measured:
    Per-channel Wasserstein distance between the batch pixel distribution and
    the training reference. A high score means the incoming images are
    statistically different from what the model was trained on.

    For raw-image pipelines, --drift-scenarios-dir enables ISP scenario
    interpretation: the observed drift profile is compared against pre-computed
    ISP scenario signatures to find a plausible physical explanation (e.g.
    "warm white balance shift"). This is interpretation of detected drift,
    not a second detection step.

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
from src.common.io import atomic_write_json
from src.monitoring.image_drift_monitor import monitor_image_batch
from src.drift.interpret import _SEVERITY_ORD
from src.data.prepare_batch import resolve_latest_version

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run online image batch drift monitoring against the training reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Batch input ---
    parser.add_argument(
        "--batch-npz",
        type=str,
        required=True,
        help=(
            "Path to a preprocessed batch NPZ file. "
            "Create one with: prepare-image-batch --input-dir <dir> --config <config>"
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
        help=(
            "Dataset name. Defaults to the 'dataset' field in the pipeline config."
        ),
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
        default="statistical",
        help=(
            "Drift detection method. 'statistical' (default) uses per-channel "
            "Wasserstein distance — fast, no model needed. 'embedding' uses "
            "CNN feature MMD — requires --model-pt."
        ),
    )
    parser.add_argument(
        "--model-pt",
        type=str,
        default=None,
        help="Path to model.pt (PyTorch CNN). Required when --method=embedding.",
    )

    # --- ISP scenario interpretation (raw-image pipelines only) ---
    parser.add_argument(
        "--drift-scenarios-dir",
        type=str,
        default=None,
        help=(
            "Path to drift scenarios directory (data/drift_scenarios/). "
            "When provided, the monitor attempts to interpret the observed drift "
            "by matching it against pre-computed ISP scenario signatures. "
            "Only meaningful for raw-image pipelines."
        ),
    )
    parser.add_argument(
        "--sensitivity-report",
        type=str,
        default=None,
        help=(
            "Path to sensitivity_report.json (from the offline ISP sensitivity analysis). "
            "Used to annotate the matched scenario with its estimated accuracy drop."
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

    # --- Load reference ---
    if args.reference_npz:
        # Explicit path
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
        # Auto-resolve dataset name from config if not given
        dataset_name = args.dataset_name or config.dataset
        processed_dir = Path(config.data.processed)

        # Auto-resolve version to the newest if not given
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

    # --- Load model for embedding method ---
    model = None
    if args.method == "embedding":
        if not args.model_pt:
            logger.error("--model-pt is required when --method=embedding.")
            sys.exit(1)
        pt_path = Path(args.model_pt)
        if not pt_path.exists():
            logger.error("Model file not found: %s", pt_path)
            sys.exit(1)
        try:
            import torch
            model = torch.load(pt_path, weights_only=False, map_location="cpu")
        except Exception as e:
            logger.error("Failed to load model from '%s': %s", pt_path, e)
            sys.exit(1)
        logger.info("Loaded model for embedding drift: %s", pt_path.name)

    # --- Resolve optional ISP scenario matching ---
    drift_scenarios_dir = (
        Path(args.drift_scenarios_dir) if args.drift_scenarios_dir else None
    )
    sensitivity_report_path = (
        Path(args.sensitivity_report) if args.sensitivity_report else None
    )

    is_raw_isp = drift_scenarios_dir is not None
    if is_raw_isp:
        logger.info(
            "ISP scenario interpretation enabled — drift_scenarios_dir=%s", drift_scenarios_dir
        )
    else:
        logger.info(
            "Running image drift monitoring (statistical). "
            "Pass --drift-scenarios-dir to enable ISP scenario interpretation."
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
        print(f"\n  Closest ISP scenario (interpretation):")
        print(f"    Scenario:       {scenario_interp['matched_scenario']}")
        desc = scenario_interp.get("description", "")
        if desc:
            print(f"    Description:    {desc}")
        print(f"    Match distance: {scenario_interp['match_distance']:.4f}  "
              f"(confidence: {scenario_interp['confidence']})")
        drop = scenario_interp.get("estimated_accuracy_drop")
        if drop is not None:
            print(f"    Est. acc. drop: {drop:+.4f}")
        print("    Note: this interpretation identifies a plausible cause of drift,")
        print("          not a separate detection — the Wasserstein score is the measurement.")
    elif is_raw_isp:
        print("\n  ISP scenario interpretation: none (pattern does not resemble known scenarios)")

    print("\n" + "=" * 60)

    # --- Save results ---
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs/drift_monitoring/images")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{timestamp}.json"
    atomic_write_json(output_path, drift_result)
    logger.info("Drift result written to %s", output_path)

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
            drift_report_linked=output_path.name,
        )
        if decision is not None:
            decision_path = output_dir / f"{timestamp}_decision.json"
            atomic_write_json(decision_path, decision.to_dict())
            logger.info("Drift decision written to %s", decision_path)

    sys.exit(0)


if __name__ == "__main__":
    main()

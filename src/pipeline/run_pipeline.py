# =============================================================================
# src/pipeline/run_pipeline.py — Pipeline entrypoint
# =============================================================================
# This is the single point of entry for the entire MLOps pipeline.
# It is registered as a CLI command via [project.scripts] in pyproject.toml.
#
# Responsibility: Parse CLI arguments, load config, and orchestrate pipeline
# stages. This file should remain THIN — it coordinates, it does not contain
# business logic. All actual work is delegated to other modules.
#
# Sprint 1 scope: Parse args, load config, confirm success, exit cleanly.
# Later sprints will add actual pipeline steps (training, evaluation, etc.)
# =============================================================================

import argparse
import sys
import logging
import uuid
import yaml

import mlflow
from pathlib import Path
from src.config.loader import load_config
from src.pipeline.steps import execute_stage, StageResult, run_drift_adaptation_eval
from src.pipeline.report import compute_config_hash, build_run_report, write_run_report
from src.pipeline.mlflow_logger import configure_mlflow, log_isp_scenario_artifacts
from src.common.io import atomic_write_json
from src.data.create_dataset_yaml import detect_and_generate
from src.data.versioning import create_dataset_version
from src.data.validate import validate_dataset
from src.data.split import split_dataset

logger = logging.getLogger(__name__)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Lightweight MLOps pipeline entrypoint"
    )
    parser.add_argument(
        "--config",
        type = str,
        required = True,
        help = "Path to pipeline config file (e.g., src/config/pipeline_tabular.yaml)"
    )
    parser.add_argument(
        "--fine-tune",
        action = "store_true",
        default = False,
        help = (
            "Fine-tune the existing Production model instead of training from scratch. "
            "Loads Production weights from the MLflow Registry and continues training "
            "with the fine_tune hyperparameters defined in training_image_cnn.yaml "
            "(fewer epochs, lower learning rate). Only applies to CNN image pipelines; "
            "ignored for tabular (random forest) pipelines.\n\n"
            "DRIFT-ADAPTIVE FINE-TUNING (recommended workflow):\n"
            "  When responding to detected drift, run prepare-drift-training BEFORE "
            "this command. Organise your drifted batch into class subdirectories "
            "(e.g. data/batches/images/drifted/cats/, dogs/) then run:\n\n"
            "    prepare-drift-training --drifted-dir data/batches/images/drifted --config <config>\n"
            "    run-pipeline --config <config> --fine-tune\n\n"
            "The pipeline will then automatically evaluate the fine-tuned model on the "
            "held-out drifted images and print a before/after performance comparison. "
            "Run 'prepare-drift-training --help' for setup instructions."
        ),
    )
    return parser.parse_args()

def _setup_logging(level_name: str) -> None:
    logging.basicConfig(
        level = getattr(logging, level_name),
        format = "%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S", 
    )

def _print_drift_adaptation_eval(result: dict) -> None:
    """Print a before/after drift adaptation evaluation summary."""
    baseline = result.get("baseline", {})
    after = result.get("after_finetuning", {})
    delta = result.get("delta", {})
    improved = result.get("improved", False)
    n = result.get("n_holdout_images", "?")

    def _fmt(val) -> str:
        return f"{val:.4f}" if isinstance(val, float) else str(val)

    def _fmt_delta(key: str) -> str:
        v = delta.get(key)
        if v is None:
            return "N/A"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.4f}"

    print()
    print("=" * 62)
    print("  DRIFT ADAPTATION EVALUATION")
    print("=" * 62)
    print(f"\n  Held-out drifted images evaluated: {n}")
    print()
    print(f"  {'Metric':<12}  {'Before':>10}  {'After':>10}  {'Delta':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
    for key, label in [
        ("accuracy",  "Accuracy"),
        ("f1_score",  "F1 score"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
    ]:
        b = _fmt(baseline.get(key, "N/A"))
        a = _fmt(after.get(key, "N/A"))
        d = _fmt_delta(key)
        print(f"  {label:<12}  {b:>10}  {a:>10}  {d:>10}")

    print()
    verdict = "IMPROVED" if improved else "NO IMPROVEMENT"
    detail = (
        "fine-tuning improved performance on drifted images."
        if improved
        else "fine-tuning did not improve performance on the drifted holdout."
    )
    print(f"  Result: {verdict} — {detail}")
    print()
    print("=" * 62)


def main() -> None:
    _setup_logging("INFO")
    args = _parse_args()
    config_path = Path(args.config).resolve()

    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        logger.error("Config file not found: %s", e)
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error("Failed to parse YAML config file: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)

    logging.root.setLevel(getattr(logging, config.log_level))

    try:
        detect_and_generate(non_interactive=not sys.stdin.isatty())
        version_result = create_dataset_version(config.dataset)
        version_id = version_result.name
        validate_dataset(config.dataset, version_id)
        split_dataset(config.dataset, version_id, config.random_seed)
    except Exception as e:
        logger.error("Data preparation failed: %s", e)
        sys.exit(1)

    logger.info(
        "\nConfig loaded: project = %s v%s, tasktype = %s, seed = %d",
        config.project.name,
        config.project.version,
        config.task_type,
        config.random_seed,
    )

    fine_tune = args.fine_tune

    config_hash = compute_config_hash(config_path)
    pipeline_execution_id = uuid.uuid4().hex

    try:
        configure_mlflow(config, pipeline_execution_id, config_hash)
        mlflow.set_tag("pipeline.dataset_version_id", version_id)
        mlflow.set_tag("pipeline.fine_tune", "true" if fine_tune else "false")
    except Exception as e:
        logger.warning("MLflow setup failed — tracking disabled for this run: %s", e)

    stage_results: list[StageResult] = []
    overall_status = "completed"
    try:
        for stage_name in config.pipeline_stages:
            # On fine-tune runs, evaluate the model on the held-out drifted images
            # BEFORE the promotion stage so the user sees before/after performance
            # when making the promotion decision.
            if stage_name == "promotion" and fine_tune:
                drift_eval = run_drift_adaptation_eval(config, version_id)
                if drift_eval is not None:
                    eval_path = Path(config.output_dir) / "drift_adaptation_eval.json"
                    atomic_write_json(eval_path, drift_eval)
                    logger.info("Drift adaptation eval written to %s", eval_path)
                    if mlflow.active_run():
                        for key, val in drift_eval.get("delta", {}).items():
                            mlflow.log_metric(f"drift_adapt.delta_{key}", val)
                        mlflow.log_metric(
                            "drift_adapt.after_accuracy",
                            drift_eval["after_finetuning"].get("accuracy", 0.0),
                        )
                        mlflow.log_artifact(str(eval_path), artifact_path="outputs")

            result = execute_stage(stage_name, config, version_id, fine_tune=fine_tune)
            stage_results.append(result)

            if result.status == "blocked":
                overall_status = "blocked"
                break
            if result.status == "failed":
                logger.error("Stage '%s' failed - aborting remaining steps", stage_name)
                overall_status = "failed"
                break

        artifact_path = None
        model_dir = Path("artifacts/runs") / version_id / "model"
        if model_dir.exists():
            artifact_path = str(model_dir.resolve())

        if mlflow.active_run():
            mlflow.log_artifact(str(config_path), artifact_path="config")
            analysis_ran = any(
                r.stage == "model_analysis" and r.status == "completed"
                for r in stage_results
            )
            if analysis_ran:
                log_isp_scenario_artifacts(Path(config.data.drift_scenarios) / config.dataset / version_id)

        mlflow_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

        report = build_run_report(
            project_name=config.project.name,
            project_version=config.project.version,
            config_hash=config_hash,
            task_type=config.task_type,
            random_seed=config.random_seed,
            dataset_version_id=version_id,
            stage_results=stage_results,
            artifact_path=artifact_path,
            pipeline_execution_id=pipeline_execution_id,
            mlflow_run_id=mlflow_run_id,
        )
        output_report_path = write_run_report(report, config.output_dir)

        if mlflow.active_run():
            mlflow.set_tag("pipeline.overall_status", overall_status)
            mlflow.log_artifact(str(output_report_path), artifact_path="outputs")

        # --- Drift adaptation evaluation (fine-tune runs only) ---
        # Evaluates the fine-tuned model on the held-out drifted images saved
        # by prepare-drift-training, and prints a before/after comparison.
        # Skips silently if prepare-drift-training was not run beforehand.
        if fine_tune and overall_status == "completed":
            drift_eval = run_drift_adaptation_eval(config, version_id)
            if drift_eval is not None:
                _print_drift_adaptation_eval(drift_eval)
                eval_path = Path(config.output_dir) / "drift_adaptation_eval.json"
                atomic_write_json(eval_path, drift_eval)
                logger.info("Drift adaptation eval written to %s", eval_path)
                if mlflow.active_run():
                    for key, val in drift_eval.get("delta", {}).items():
                        mlflow.log_metric(f"drift_adapt.delta_{key}", val)
                    mlflow.log_metric(
                        "drift_adapt.after_accuracy",
                        drift_eval["after_finetuning"].get("accuracy", 0.0),
                    )
                    mlflow.log_artifact(str(eval_path), artifact_path="outputs")
    finally:
        mlflow.end_run()

    if overall_status == "blocked":
        blocked_result = next(r for r in stage_results if r.status == "blocked")
        print("\n" + "=" * 60)
        print("  PIPELINE STOPPED — MODEL DID NOT MEET PROMOTION CRITERIA")
        print("=" * 60)
        for line in blocked_result.error.splitlines():
            print(f"  {line}")
        print()
        print("  The model has been trained and evaluated but will NOT be")
        print("  promoted to production. Improve the model or dataset and")
        print("  re-run the pipeline.")
        print("=" * 60 + "\n")
        sys.exit(2)

    if overall_status == "failed":
        logger.error("Pipeline finished with failures")
        sys.exit(1)

    logger.info("Pipeline completed successfully")
    sys.exit(0)
    
if __name__ == "__main__":
    main()


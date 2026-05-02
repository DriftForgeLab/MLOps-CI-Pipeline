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
        help = "Path to pipeline config file (e.g., src/config/pipeline_tabular_classification.yaml)"
    )
    parser.add_argument(
        "--fine-tune",
        action = "store_true",
        default = False,
        help = (
            "For CNN image pipelines: load Production weights from the MLflow Registry "
            "and continue training with fine_tune hyperparameters (fewer epochs, lower LR). "
            "For tabular pipelines this flag has no special effect — retraining always "
            "starts from scratch on whatever data is in data/raw/<dataset>/data.csv.\n\n"
            "DRIFT-ADAPTIVE RETRAINING (recommended workflow after drift is detected):\n\n"
            "  Image pipelines:\n"
            "    prepare-drift-training --drifted-dir data/batches/images/drifted --config <config>\n"
            "    run-pipeline --config <config> --fine-tune\n\n"
            "  Tabular pipelines:\n"
            "    prepare-drift-training-tabular --drifted-csv data/batches/tabular/drifted.csv --config <config>\n"
            "    run-pipeline --config <config>\n\n"
            "The pipeline automatically evaluates the retrained model on the held-out "
            "drifted data and prints a before/after comparison at the promotion gate."
        ),
    )
    return parser.parse_args()

def _setup_logging(level_name: str) -> None:
    logging.basicConfig(
        level = getattr(logging, level_name),
        format = "%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S", 
    )

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
            # Before the promotion stage, evaluate the retrained model on any
            # held-out drifted data so the user sees before/after performance
            # when making the promotion decision.
            # Image pipelines: only run on --fine-tune (weights loaded from production).
            # Tabular pipelines: always run (retraining is always from scratch;
            # the holdout file's existence gates whether anything is computed).
            from src.config.schema import IMAGE_TASK_TYPES as _IMAGE_TASK_TYPES
            _is_image = config.task_type in _IMAGE_TASK_TYPES
            if stage_name == "promotion" and (fine_tune or not _is_image):
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


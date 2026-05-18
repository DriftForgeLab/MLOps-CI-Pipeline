# =============================================================================
# src/pipeline/steps.py — Pipeline stage execution
# =============================================================================
# Responsibility: Execute individual pipeline stages with structured logging
# and timing. Each stage is wrapped with START/END markers for observability.
#
# Each stage is a real implementation: preprocessing, training, evaluation,
# model_analysis (image pipelines only), promotion, and deployment (manifest
# generation). The execution framework wraps each call with START/END
# markers, timing, and status tracking.
#
# Design: Each stage returns a StageResult dataclass capturing what happened.
# This feeds into the run report for traceability.
# =============================================================================
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config.loader import PipelineConfig, load_promotion_config, load_drift_config, load_preprocessing_config
from src.config.schema import IMAGE_TASK_TYPES
from src.data.preprocess import run_preprocessing, PREPROCESSED_SUBDIR

from src.training.classification.train import run_training as run_classification_training
from src.training.regression.train import run_training as run_regression_training
from src.registry.metadata import save_model_artifact

from src.evaluation.evaluate import evaluate
from src.common.io import atomic_write_json
from src.pipeline.mlflow_logger import (
    log_training_to_mlflow,
    log_training_artifacts_to_mlflow,
    log_evaluation_to_mlflow,
    log_comparison_to_mlflow,
    log_promotion_decision_to_mlflow,
    log_isp_versioning_to_mlflow,
)

from src.promotion.rules import run_promotion_rules
from src.promotion.approval import request_approval, ApprovalUnavailableError
from src.monitoring.history import DEFAULT_OUTPUTS_ROOT, load_latest_drift
from src.registry.model_registry import PROMOTION_ARTIFACT_SUBDIR, resolve_model_name
import json
import mlflow

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Captures the result of executing a single pipeline stage.

    Attributes:
        stage:            Name of the stage that was executed (ex. "training")
        status:           Execution status — one of:
                            "completed" — stage finished successfully
                            "failed"    — engineering breakage / unexpected exception
                            "blocked"   — governance rejection (rules failed)
                            "cancelled" — non-interactive environment could not
                                          obtain a human decision at the approval gate
        started_at:       ISO 8601 timestamp when stage began
        ended_at:         ISO 8601 timestamp when stage ended
        duration_seconds: Wall-clock time elapsed during execution
        error:            Exception message if status is "failed", "blocked",
                          or "cancelled"; else None
    """
    stage: str
    status: str
    started_at: str
    ended_at: str
    duration_seconds: float
    error: str | None = None

def _preprocessing_stage(config: PipelineConfig, version_id: str) -> None:
    prep_config_path = Path(config.configs.preprocessing)
    run_preprocessing(
        dataset_name=config.dataset,
        version_id=version_id,
        prep_config_path=prep_config_path,
        processed_dir=Path(config.data.processed),
        random_seed=config.random_seed,
    )

    # For raw-image tasks: log ISP pipeline version and key config params to MLflow.
    # The preprocess_hash (written by run_image_preprocessing) covers the full ISP
    # config + input manifest, so the same hash guarantees identical outputs.
    if config.task_type in IMAGE_TASK_TYPES:
        prep_config = load_preprocessing_config(prep_config_path)
        if prep_config.image and prep_config.image.raw_input:
            preprocessed_dir = (
                Path(config.data.processed) / config.dataset / version_id / PREPROCESSED_SUBDIR
            )
            meta_path = preprocessed_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                log_isp_versioning_to_mlflow(meta, prep_config.image.isp)


def _training_stage(config: PipelineConfig, version_id: str, fine_tune: bool = False) -> None:
    if config.task_type == "classification":
        result = run_classification_training(config, version_id)
    elif config.task_type == "regression":
        result = run_regression_training(config, version_id)
    elif config.task_type == "image_classification_cnn":
        from src.training.image_classification_cnn.train import run_training as run_cnn_training
        result = run_cnn_training(config, version_id, fine_tune=fine_tune)
    else:
        raise ValueError(f"Unsupported task_type: '{config.task_type}'")

    artifact_path = save_model_artifact(result, run_id=version_id, task_type=config.task_type)
    log_training_to_mlflow(result)
    preprocessed_dir = (
        Path(config.data.processed) / config.dataset / version_id / PREPROCESSED_SUBDIR
    )
    log_training_artifacts_to_mlflow(model_dir=artifact_path, preprocessed_dir=preprocessed_dir)
    
    
def _evaluation_stage(config: PipelineConfig, version_id: str) -> None:
    report = evaluate(config, version_id)
    report_path = Path(config.output_dir) / "evaluation_report.json"
    atomic_write_json(report_path, report)
    logger.info("  Evaluation report written to: %s", report_path.resolve())
    official_test_report = report.get("official_test_report")
    if official_test_report is not None:
        official_test_path = Path(config.output_dir) / "official_test_evaluation_report.json"
        atomic_write_json(official_test_path, official_test_report)
        logger.info("  Official test evaluation report written to: %s", official_test_path.resolve())
    log_evaluation_to_mlflow(report)
    log_comparison_to_mlflow(report, output_dir=Path(config.output_dir))
    

_MIN_ANALYSIS_ACCURACY = 0.6  # Models at or below this haven't learned enough for meaningful analysis


class PromotionBlockedError(Exception):
    """Raised when a model fails promotion rules. Not a system error — handled cleanly."""
    pass


def _model_analysis_stage(config: PipelineConfig, version_id: str) -> None:
    """Offline model analysis: ISP sensitivity or augmentation robustness.

    Dispatches to the appropriate analysis based on task configuration:
      - Raw DNG images (raw_input=True):  ISP-simulation + sensitivity analysis.
        Systematically re-processes the training images with modified ISP
        parameters to pre-compute drift signatures used by monitor-drift-image.
      - Standard JPG/PNG images (raw_input=False): augmentation robustness
        analysis. Applies geometric and photometric transformations to the
        validation split and measures how much each transformation degrades
        model accuracy before deployment.

    Neither branch is drift detection. Drift detection — comparing training
    data against new, real production batches over time — runs separately via
    the monitoring CLIs (monitor-drift, monitor-drift-image).

    Two conditions can skip this stage:
      1. Model accuracy is at or below _MIN_ANALYSIS_ACCURACY — the model has
         not learned meaningful patterns, making the analysis uninformative.
      2. The user declines when prompted interactively.
    """
    import sys

    drift_config = load_drift_config(Path(config.configs.drift))
    if not drift_config.enabled:
        logger.info("  Offline model analysis disabled — skipping.")
        return

    # --- Quality gate: skip if model hasn't learned ---
    report_path = Path(config.output_dir) / "evaluation_report.json"
    if report_path.exists():
        with open(report_path) as f:
            eval_report = json.load(f)
        accuracy = eval_report.get("metrics", {}).get("accuracy", 1.0)
        if accuracy <= _MIN_ANALYSIS_ACCURACY:
            logger.warning(
                "  Model accuracy=%.4f is at or below minimum threshold=%.2f "
                "(model has not learned meaningful patterns). "
                "Skipping model analysis — re-run after improving the model or dataset.",
                accuracy,
                _MIN_ANALYSIS_ACCURACY,
            )
            return

    # --- Interactive prompt: give user the option to skip ---
    if sys.stdin.isatty():
        print("\n" + "=" * 60)
        print("  MODEL ANALYSIS")
        print("=" * 60)
        print("\n  Augmentation robustness / ISP sensitivity analysis provides")
        print("  insight into how the model handles image perturbations.")
        print("  For raw image datasets this can take significant time.")
        print()
        try:
            answer = input("  Run model analysis? [Y/n]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            answer = "n"
        print()
        if answer in ("n", "no"):
            logger.info("  Model analysis skipped by user.")
            return

    _run_image_model_analysis_stage(config, version_id, drift_config)


def _run_image_model_analysis_stage(config: PipelineConfig, version_id: str, drift_config) -> None:
    """Dispatch to ISP sensitivity or augmentation robustness based on preprocessing config."""
    prep_config_path = Path(config.configs.preprocessing)
    prep_config = load_preprocessing_config(prep_config_path)
    # Versioned output dir: data/drift_scenarios/<dataset>/<version_id>/
    drift_dir = Path(config.data.drift_scenarios) / config.dataset / version_id
    drift_dir.mkdir(parents=True, exist_ok=True)
    is_raw_isp = bool(
        prep_config.image
        and prep_config.image.raw_input
        and prep_config.image.isp
    )

    if is_raw_isp:
        _run_isp_simulation_and_sensitivity(config, version_id, drift_config, prep_config, drift_dir)
    else:
        _run_augmentation_robustness(config, version_id, drift_dir)


def _run_isp_simulation_and_sensitivity(
    config: PipelineConfig,
    version_id: str,
    drift_config,
    prep_config,
    drift_dir: Path,
) -> Path | None:
    """Run ISP drift simulation and sensitivity analysis.

    Returns the path to sensitivity_report.json if successful, else None.
    Called only when raw_input=True and an isp: block is present.
    """
    from src.data.image_drift_simulator import (
        run_drift_simulation,
        load_image_scenarios_from_drift_yaml,
    )
    from src.evaluation.image_drift_analysis import (
        run_sensitivity_analysis,
        save_sensitivity_report_json,
        save_sensitivity_report_html,
    )

    scenarios, parameter_bounds = load_image_scenarios_from_drift_yaml(
        Path(config.configs.drift)
    )
    if not scenarios:
        logger.info("  No image drift scenarios configured in drift.yaml — skipping ISP simulation.")
        return None

    img_cfg = prep_config.image
    results = run_drift_simulation(
        dataset_name=config.dataset,
        version_id=version_id,
        baseline_isp_config=img_cfg.isp,
        target_size=img_cfg.target_size,
        color_mode=img_cfg.color_mode,
        normalize=img_cfg.normalize,
        scenarios=scenarios,
        parameter_bounds=parameter_bounds,
        drift_scenarios_dir=drift_dir,
        processed_dir=Path(config.data.processed),
    )
    logger.info("  ISP drift simulation complete: %d scenario(s) generated", len(results))

    if not results:
        return None

    baseline_report_path = Path(config.output_dir) / "evaluation_report.json"
    sensitivity_report = run_sensitivity_analysis(
        version_id=version_id,
        task_type=config.task_type,
        drift_scenarios_dir=drift_dir,
        baseline_report_path=baseline_report_path,
    )

    json_path = save_sensitivity_report_json(sensitivity_report, drift_dir)
    html_path = save_sensitivity_report_html(sensitivity_report, drift_dir)
    logger.info("  Sensitivity report written: %s", json_path.resolve())

    if mlflow.active_run():
        mlflow.set_tags({
            "analysis.type": "isp_sensitivity",
            "analysis.scenarios_generated": str(len(results)),
            "analysis.scenario_names": ",".join(r["name"] for r in results),
            "analysis.most_sensitive_scenario": sensitivity_report.get("most_sensitive_scenario", ""),
            "analysis.least_sensitive_scenario": sensitivity_report.get("least_sensitive_scenario", ""),
        })
        for scenario in sensitivity_report.get("scenarios", []):
            name = scenario["name"]
            acc = scenario["metrics"].get("accuracy")
            f1  = scenario["metrics"].get("f1_score")
            delta_acc = scenario["delta"].get("accuracy")
            if acc is not None:
                mlflow.log_metric(f"analysis.scenario.{name}.accuracy", acc)
            if f1 is not None:
                mlflow.log_metric(f"analysis.scenario.{name}.f1_score", f1)
            if delta_acc is not None:
                mlflow.log_metric(f"analysis.scenario.{name}.delta_accuracy", delta_acc)
        mlflow.log_artifact(str(json_path), artifact_path="model_analysis")
        mlflow.log_artifact(str(html_path), artifact_path="model_analysis")

    return json_path


def _run_augmentation_robustness(
    config: PipelineConfig,
    version_id: str,
    drift_dir: Path,
) -> None:
    """Augmentation robustness analysis for standard JPG/PNG image pipelines.

    Applies 8 fixed augmentation scenarios (horizontal flip, rotation, brightness
    shifts, contrast reduction, Gaussian noise, Gaussian blur) to the validation
    split and measures model accuracy under each transformation.

    This is an offline stresstest of the model before deployment. It answers:
      "If incoming images are slightly darker, blurrier, rotated, or noisier,
       how much does accuracy drop?"

    This is NOT drift detection. Augmented images are synthetic — they are
    not real production batches. Production drift detection runs via the
    monitoring CLI (monitor-drift-image) after real batches are collected.
    """
    from src.evaluation.standard_image_robustness import (
        run_robustness_analysis,
        save_robustness_report_json,
        save_robustness_report_html,
    )

    preprocessed_dir = (
        Path(config.data.processed) / config.dataset / version_id / PREPROCESSED_SUBDIR
    )
    baseline_report_path = Path(config.output_dir) / "evaluation_report.json"

    robustness_report = run_robustness_analysis(
        version_id=version_id,
        task_type=config.task_type,
        preprocessed_dir=preprocessed_dir,
        baseline_report_path=baseline_report_path,
    )

    json_path = save_robustness_report_json(robustness_report, drift_dir)
    html_path = save_robustness_report_html(robustness_report, drift_dir)
    logger.info("  Augmentation robustness report written: %s", json_path.resolve())

    if mlflow.active_run():
        mlflow.set_tags({
            "analysis.type": "augmentation_robustness",
            "analysis.most_sensitive": robustness_report.get("most_sensitive_augmentation", ""),
            "analysis.least_sensitive": robustness_report.get("least_sensitive_augmentation", ""),
        })
        for scenario in robustness_report.get("scenarios", []):
            delta_acc = scenario["delta"].get("accuracy")
            if delta_acc is not None:
                mlflow.log_metric(
                    f"analysis.robustness.{scenario['name']}.delta_accuracy", delta_acc
                )
        mlflow.log_artifact(str(json_path), artifact_path="model_analysis")
        mlflow.log_artifact(str(html_path), artifact_path="model_analysis")


def _drift_provenance(drift: dict | None, model_name: str) -> dict:
    """Summarise the drift read-back for the promotion decision record."""
    if drift is None:
        return {"available": False, "model_name": model_name}
    overall = drift.get("overall", {})
    return {
        "available": True,
        "model_name": model_name,
        "generated_at": drift.get("generated_at"),
        "severity": overall.get("severity"),
        "dataset_drift_detected": overall.get("dataset_drift_detected"),
        "drifted_feature_count": overall.get("drifted_feature_count"),
        "total_feature_count": overall.get("total_feature_count"),
    }


def _offer_drift_rollback_on_rejection(config: PipelineConfig) -> None:
    """If a recent drift attempt is recorded, offer to roll it back interactively.

    Called when a promotion is rejected. Skips silently if there are no
    recorded attempts, if there is no TTY (non-interactive run), or if the
    user declines. Never raises — rollback is best-effort.
    """
    import sys

    try:
        from src.data.drift_adaptation import (
            latest_drift_attempt,
            rollback_drift_attempt,
        )
    except Exception:  # pragma: no cover - defensive
        return

    raw_dataset_dir = Path(config.data.raw) / config.dataset
    attempt = latest_drift_attempt(raw_dataset_dir)
    if attempt is None:
        return

    attempt_id = attempt.get("attempt_id", "?")
    n_files = attempt.get("n_files_added", 0)
    drifted_dir = attempt.get("drifted_dir", "?")

    if not sys.stdin.isatty():
        logger.warning(
            "Promotion rejected — drift attempt '%s' added %d file(s) from '%s' "
            "to the raw dataset. Run `rollback-drift-training --config %s --latest` "
            "to remove them.",
            attempt_id, n_files, drifted_dir, config.configs.preprocessing,
        )
        return

    print()
    print("=" * 62)
    print("  DRIFT ROLLBACK")
    print("=" * 62)
    print()
    print(f"  Latest drift attempt:   {attempt_id}")
    print(f"  Files added to dataset: {n_files}")
    print(f"  Source drifted_dir:     {drifted_dir}")
    print()
    print("  Promotion was rejected. The drifted images added by this attempt")
    print("  are still in the raw dataset and will be picked up by the next")
    print("  pipeline run. Roll them back now?")
    print()
    try:
        answer = input("  Roll back this drift attempt? [y/N]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        answer = "n"

    if answer not in ("y", "yes"):
        print()
        print("  Skipped. To roll back later, run:")
        print(f"    rollback-drift-training --latest --config <pipeline-config>")
        print()
        return

    try:
        result = rollback_drift_attempt(Path(attempt["manifest_path"]), remove_holdout=False)
    except Exception as exc:
        logger.warning("Drift rollback failed: %s", exc)
        return

    print()
    print(f"  Removed {result['removed']} file(s); {result['missing']} already missing.")
    print()


def _promotion_stage(config: PipelineConfig, version_id: str) -> None:
    # Load evaluation report
    report_path = Path(config.output_dir) / "evaluation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Evaluation report not found at '{report_path}'. "
            "Run the evaluation stage before promotion."
        )
    with open(report_path) as f:
        report = json.load(f)

    # Run threshold rules against the configured evaluation split(s).
    promotion_config = load_promotion_config(Path(config.configs.promotion))
    eval_split = promotion_config.promotion_evaluation_split
    test_report = report.get("official_test_report") or {}

    splits_to_check: list[tuple[str, dict]] = []
    if eval_split in ("val", "both"):
        splits_to_check.append(("val", report["metrics"]))
    if eval_split in ("test", "both"):
        test_metrics = test_report.get("metrics")
        if test_metrics is None:
            raise RuntimeError(
                f"promotion_evaluation_split='{eval_split}' requires a held-out "
                "test-split report, but none was produced — the test/ split is "
                "empty or missing. Set promotion_evaluation_split to 'val' or "
                "ensure the dataset produces a populated test split."
            )
        splits_to_check.append(("test", test_metrics))

    violations: list[dict] = []
    for split_name, split_metrics in splits_to_check:
        for v in run_promotion_rules(split_metrics, config.task_type, promotion_config):
            violations.append({**v, "split": split_name})

    if violations:
        violation_lines = "\n  ".join(
            f"[{v['split']}:{v['rule_id']}] {v['metric']}={v['observed']} "
            f"(required {v['operator']} {v['threshold']}): {v['description']}"
            for v in violations
        )
        raise PromotionBlockedError(
            f"Promotion blocked — {len(violations)} rule(s) failed "
            f"(evaluation split: {eval_split}):\n  {violation_lines}"
        )

    logger.info("  Promotion rules evaluated on split: %s", eval_split)

    # Log comparison summary
    comparison = report.get("comparison", {})
    overall_verdict = comparison.get("overall_verdict", "no_baseline")
    logger.info(
        "  Comparison verdict: %s (has_production_model=%s)",
        overall_verdict.upper(),
        comparison.get("has_production_model", False),
    )

    logger.info("  All promotion rules passed — candidate is eligible for promotion.")

    model_name = resolve_model_name(config)
    try:
        drift = load_latest_drift(model_name, DEFAULT_OUTPUTS_ROOT)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load latest drift for %s: %s", model_name, exc)
        drift = None
    if drift is None:
        logger.info("  No drift history found for model '%s'.", model_name)
    else:
        logger.info(
            "  Loaded latest drift for '%s': severity=%s",
            model_name,
            drift.get("overall", {}).get("severity", "unknown"),
        )

    # Load drift adaptation eval if present — written just before this stage
    # runs so the before/after holdout comparison is available to the user when
    # they make the promotion decision. Guard on task_type to prevent a stale
    # file from a prior run of a different pipeline type being picked up.
    drift_eval: dict | None = None
    drift_eval_path = Path(config.output_dir) / "drift_adaptation_eval.json"
    if drift_eval_path.exists():
        try:
            with open(drift_eval_path) as f:
                candidate = json.load(f)
            eval_task_type = candidate.get("task_type")
            if eval_task_type == config.task_type:
                drift_eval = candidate
            else:
                logger.debug(
                    "drift_adaptation_eval.json is from a '%s' run — ignoring for current '%s' run.",
                    eval_task_type, config.task_type,
                )
        except Exception as _de_exc:
            logger.debug("Could not load drift_adaptation_eval.json: %s", _de_exc)

    result = request_approval(report, drift=drift, drift_eval=drift_eval)

    mlflow_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
    decision = {
        "outcome":            "approved" if result.approved else "rejected",
        "reason":             result.reason,
        "run_id":             mlflow_run_id,
        "dataset_version_id": version_id,
        "evaluation_split":   eval_split,
        "metrics":            report.get("metrics", {}),
        "comparison":         report.get("comparison", {}),
        "drift":              _drift_provenance(drift, model_name),
    }

    decision_path = Path(config.output_dir) / "promotion_decision.json"
    atomic_write_json(decision_path, decision)
    logger.info("  Promotion decision written to: %s", decision_path.resolve())

    log_promotion_decision_to_mlflow(decision)

    if mlflow.active_run():
        mlflow.log_artifact(str(decision_path), artifact_path=PROMOTION_ARTIFACT_SUBDIR)

    if not result.approved:
        _offer_drift_rollback_on_rejection(config)
        reason_msg = f" Reason: {result.reason}" if result.reason else " No reason provided."
        raise ValueError(f"Promotion rejected by user.{reason_msg}")

    # Register and promote the approved model in MLflow Model Registry.
    # Lineage tags are built (and validated) BEFORE any registry mutation so
    # that a missing required tag aborts the promotion before a new version
    # is registered and the previous Production is archived.
    from src.registry.model_registry import (
        register_approved_model,
        promote_to_production,
        build_lineage_tags,
        write_lineage_tags,
        get_mlflow_client,
    )
    client = get_mlflow_client(config)
    run = client.get_run(mlflow_run_id)
    tags = build_lineage_tags(config, run, report, decision)

    model_version = register_approved_model(config, mlflow_run_id)
    promote_to_production(config, model_version.version, mlflow_run_id)
    write_lineage_tags(config, model_version.version, tags)


def run_drift_adaptation_eval(
    config: PipelineConfig,
    version_id: str,
) -> dict | None:
    """Evaluate the retrained/fine-tuned model on the held-out drifted data.

    Dispatches to an image or tabular evaluation based on config.task_type.
    Looks for a holdout set at data/evaluation/drifted_holdout/<dataset>/ and
    baseline metrics from a prior prepare-drift-training[/tabular] run. If both
    are present, evaluates the newly trained model and returns a before/after
    comparison report.

    Skips silently (returns None) when no holdout or baseline exists.

    Args:
        config:     Pipeline config for the current run.
        version_id: Dataset/model version hash for the retrained model.

    Returns:
        Dict with keys "task_type", "baseline", "after_finetuning", "delta",
        and "improved" (bool), or None if the evaluation was skipped.
    """
    holdout_dir = Path(config.data.evaluation) / "drifted_holdout" / config.dataset
    if not holdout_dir.exists():
        logger.debug(
            "No drifted holdout directory found at '%s' — skipping drift adaptation eval.",
            holdout_dir,
        )
        return None

    if config.task_type in IMAGE_TASK_TYPES:
        return _run_image_drift_adaptation_eval(config, version_id, holdout_dir)
    else:
        return _run_tabular_drift_adaptation_eval(config, version_id, holdout_dir)


def _run_image_drift_adaptation_eval(
    config: PipelineConfig,
    version_id: str,
    holdout_dir: Path,
) -> dict | None:
    """Image CNN drift adaptation evaluation (holdout images → .pt model)."""
    from src.data.drift_adaptation import (
        evaluate_on_holdout_dir,
        load_baseline_metrics,
        load_class_to_index,
        load_normalization_stats,
    )
    from src.config.loader import load_preprocessing_config
    from src.data.prepare_batch import load_training_stats

    baseline_payload = load_baseline_metrics(holdout_dir)
    if baseline_payload is None:
        logger.debug(
            "No baseline metrics found in '%s' — skipping image drift adaptation eval.",
            holdout_dir,
        )
        return None

    model_pt = Path("artifacts/runs") / version_id / "model" / "model.pt"
    if not model_pt.exists():
        logger.warning(
            "Fine-tuned model not found at '%s' — skipping drift adaptation eval.",
            model_pt,
        )
        return None

    try:
        import torch
        model = torch.load(str(model_pt), weights_only=False, map_location="cpu")
    except Exception as e:
        logger.warning("Could not load fine-tuned model for drift eval: %s", e)
        return None

    try:
        prep_config = load_preprocessing_config(Path(config.configs.preprocessing))
        processed_dir = Path(config.data.processed)
        saved_mean, saved_std = load_normalization_stats(holdout_dir)
        if saved_mean is not None:
            norm_mean, norm_std = saved_mean, saved_std
        else:
            norm_mean, norm_std, _ = load_training_stats(processed_dir, config.dataset, version_id)
        class_to_index = load_class_to_index(processed_dir, config.dataset, version_id)
    except Exception as e:
        logger.warning("Could not load preprocessing info for drift eval: %s", e)
        return None

    try:
        after_metrics = evaluate_on_holdout_dir(
            model, holdout_dir, prep_config, norm_mean, norm_std, class_to_index
        )
    except Exception as e:
        logger.warning("Holdout evaluation failed: %s", e)
        return None

    baseline_metrics = baseline_payload["metrics"]
    delta = {
        key: round(after_metrics[key] - baseline_metrics[key], 4)
        for key in ("accuracy", "f1_score", "precision", "recall")
        if key in after_metrics and key in baseline_metrics
    }
    return {
        "task_type": config.task_type,
        "holdout_dir": str(holdout_dir),
        "n_holdout_samples": after_metrics["n_samples"],
        "baseline": baseline_metrics,
        "after_finetuning": after_metrics,
        "delta": delta,
        "improved": delta.get("accuracy", 0.0) > 0.0,
    }


def _run_tabular_drift_adaptation_eval(
    config: PipelineConfig,
    version_id: str,
    holdout_dir: Path,
) -> dict | None:
    """Tabular drift adaptation evaluation (holdout.csv → .joblib model)."""
    from src.data.tabular_drift_adaptation import (
        load_baseline_metrics,
        load_holdout_csv,
        load_feature_contract,
        evaluate_on_holdout_tabular,
    )

    baseline_payload = load_baseline_metrics(holdout_dir)
    if baseline_payload is None:
        logger.debug(
            "No baseline metrics found in '%s' — skipping tabular drift adaptation eval.",
            holdout_dir,
        )
        return None

    holdout_df = load_holdout_csv(holdout_dir)
    if holdout_df is None or len(holdout_df) == 0:
        logger.debug(
            "Holdout CSV not found or empty in '%s' — skipping tabular drift adaptation eval.",
            holdout_dir,
        )
        return None

    model_path = Path("artifacts/runs") / version_id / "model" / "model.joblib"
    if not model_path.exists():
        logger.warning(
            "Retrained model not found at '%s' — skipping drift adaptation eval.",
            model_path,
        )
        return None

    try:
        import joblib
        model = joblib.load(model_path)
    except Exception as e:
        logger.warning("Could not load retrained model for drift eval: %s", e)
        return None

    # Use the NEW version's pipeline so the retrained model is evaluated with
    # the preprocessing it was trained on (fitted on original + drifted data).
    try:
        feature_contract = load_feature_contract(
            Path(config.data.processed), config.dataset, version_id
        )
    except Exception as e:
        logger.warning("Could not load feature contract for drift eval: %s", e)
        return None

    try:
        after_metrics = evaluate_on_holdout_tabular(
            model, holdout_df, feature_contract, config.task_type
        )
    except Exception as e:
        logger.warning("Tabular holdout evaluation failed: %s", e)
        return None

    baseline_metrics = baseline_payload["metrics"]
    metric_keys = set(after_metrics.keys()) & set(baseline_metrics.keys()) - {"n_samples"}
    delta = {
        key: round(after_metrics[key] - baseline_metrics[key], 4)
        for key in metric_keys
    }

    # Primary improvement signal: accuracy for classification, r2 for regression
    primary = "accuracy" if config.task_type == "classification" else "r2"
    improved = delta.get(primary, 0.0) > 0.0

    return {
        "task_type": config.task_type,
        "holdout_dir": str(holdout_dir),
        "n_holdout_samples": after_metrics["n_samples"],
        "baseline": baseline_metrics,
        "after_finetuning": after_metrics,
        "delta": delta,
        "improved": improved,
    }


def _deployment_stage(config: PipelineConfig, version_id: str) -> None:
    """Emit a deterministic deployment manifest describing the deployable state.

    Reads the active MLflow run, the promotion decision, the registry's
    current Production model (if any), and the deployment.yaml service
    config. Writes outputs/deployment_manifest.json. Does NOT provision
    infrastructure, build images, or call external services.
    """
    from src.config.loader import load_deployment_config
    from src.config.schema import DeploymentConfig
    from src.registry.model_registry import resolve_tracking_uri

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    # 1. Deployment config (safe-default on any error)
    try:
        deploy_cfg = load_deployment_config(Path(config.configs.deployment))
    except Exception as exc:
        logger.warning("deployment_config_load_failed: %s — using defaults", exc)
        deploy_cfg = DeploymentConfig()
        warnings.append("deployment_config_missing")

    # 2. Registry name + tracking URI
    model_name = resolve_model_name(config)
    tracking_uri = resolve_tracking_uri(config)

    # 3. Read promotion_decision.json if present
    promotion_path = output_dir / "promotion_decision.json"
    promotion_record: dict = {}
    if promotion_path.exists():
        try:
            promotion_record = json.loads(promotion_path.read_text())
        except Exception as exc:
            logger.warning("promotion_decision_unreadable: %s", exc)
            warnings.append("promotion_decision_unreadable")

    # 4. Active MLflow run (orchestrator opens one before stages run)
    execution_id = None
    experiment_name = None
    active = mlflow.active_run()
    if active is not None:
        execution_id = active.info.run_id
        try:
            experiment_name = mlflow.get_experiment(active.info.experiment_id).name
        except Exception:
            experiment_name = None

    # 5. Query registry for current Production model (safe-default on error)
    registry_version = None
    registry_run_id = None
    algorithm = None
    trained_at = None
    dataset_version_id_from_tag = None
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        versions = client.get_latest_versions(model_name, stages=[deploy_cfg.model.allowed_stage])
        if versions:
            mv = versions[0]
            registry_version = mv.version
            registry_run_id = mv.run_id
            try:
                run = client.get_run(mv.run_id)
                tags = run.data.tags or {}
                algorithm = tags.get("lineage.algorithm")
                trained_at = tags.get("lineage.trained_at")
                dataset_version_id_from_tag = tags.get("lineage.dataset_version_id")
            except Exception as exc:
                logger.warning("registry_run_lookup_failed: %s", exc)
                warnings.append("registry_run_lookup_failed")
    except Exception as exc:
        logger.warning("mlflow_registry_unreachable: %s", exc)
        warnings.append("mlflow_registry_unreachable")

    # 6. Local artifact path
    local_artifact_path = None
    resolved_run_id = registry_run_id or promotion_record.get("run_id")
    if resolved_run_id:
        candidate = Path("artifacts") / "runs" / resolved_run_id / "model"
        if candidate.exists():
            local_artifact_path = str(candidate)
        else:
            warnings.append("local_artifact_missing")

    # 7. Readiness
    if "mlflow_registry_unreachable" in warnings:
        readiness_status = "mlflow_unavailable"
    elif registry_version is None:
        readiness_status = "no_production_model"
    elif promotion_record.get("outcome") != "approved":
        readiness_status = "promotion_not_approved"
    else:
        readiness_status = "ready"

    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": {
            "task_type": config.task_type,
            "dataset_version_id": version_id,
            "config_dataset": config.dataset,
            "output_dir": config.output_dir,
            "execution_id": execution_id,
            "mlflow_experiment_name": experiment_name,
            "tracking_uri": tracking_uri,
        },
        "model": {
            "registry_name": model_name,
            "registry_version": registry_version,
            "stage": deploy_cfg.model.allowed_stage if registry_version else None,
            "run_id": resolved_run_id,
            "algorithm": algorithm,
            "trained_at": trained_at,
            "dataset_version_id": dataset_version_id_from_tag or promotion_record.get("dataset_version_id"),
            "metrics": promotion_record.get("metrics"),
            "local_artifact_path": local_artifact_path,
        },
        "promotion": {
            "outcome": promotion_record.get("outcome"),
            "run_id": promotion_record.get("run_id"),
            "reason": promotion_record.get("reason"),
            "comparison_verdict": (promotion_record.get("comparison") or {}).get("overall_verdict"),
        },
        "service": {
            "app_import": "src.deployment.app:app",
            "entrypoint_cli": "run-api",
            "endpoints": ["GET /health", "GET /models", "POST /predict"],
            "host": deploy_cfg.server.host,
            "port": deploy_cfg.server.port,
            "log_level": deploy_cfg.server.log_level,
            "allowed_stage": deploy_cfg.model.allowed_stage,
            "require_production_model": deploy_cfg.model.require_production_model,
            "startup_timeout_seconds": deploy_cfg.model.startup_timeout_seconds,
        },
        "container": {
            "dockerfile": "docker/Dockerfile",
            "compose_file": "docker/docker-compose.yml",
            "compose_service": "prediction-api",
            "image_name": "mlops-prediction-api",
            "suggested_image_tag": f"{model_name}:{version_id}",
            "exposed_port": 8000,
        },
        "env_template": [
            f"MLFLOW_TRACKING_URI={tracking_uri}",
            f"API_PORT={deploy_cfg.server.port}",
            f"MODEL_STAGE={deploy_cfg.model.allowed_stage}",
            "MODEL_DEVICE=auto",
            f"LOG_LEVEL={deploy_cfg.server.log_level.upper()}",
        ],
        "outputs": {
            "evaluation_report": str(Path(config.output_dir) / "evaluation_report.json"),
            "promotion_decision": str(Path(config.output_dir) / "promotion_decision.json"),
            "run_report": str(Path(config.output_dir) / "run_report.json"),
        },
        "readiness": {
            "status": readiness_status,
            "warnings": warnings,
        },
        "notes": [
            "This manifest declares deployment metadata only. It does not provision infrastructure, build or push container images, start Docker containers, or call external services.",
            "Live serving is performed out-of-band by the FastAPI application started via 'run-api' or 'docker compose -f docker/docker-compose.yml up --build'.",
        ],
    }

    manifest_path = output_dir / "deployment_manifest.json"
    atomic_write_json(manifest_path, manifest)
    logger.info(
        "  Deployment manifest written: %s — readiness=%s, model=%s%s",
        manifest_path,
        readiness_status,
        model_name,
        f" v{registry_version}" if registry_version else "",
    )


_STAGE_REGISTRY: dict[str, Callable] = {
    "preprocessing":  _preprocessing_stage,
    "training":       _training_stage,
    "evaluation":     _evaluation_stage,
    "model_analysis": _model_analysis_stage,
    "promotion":      _promotion_stage,
    "deployment":     _deployment_stage,
}


def execute_stage(
    stage_name: str,
    config: PipelineConfig,
    version_id: str,
    fine_tune: bool = False,
) -> StageResult:
    """Execute a single pipeline stage with START/END markers and timing.

    Looks up the stage function in _STAGE_REGISTRY, calls it with config,
    and wraps the call in structured logging and timing. Exceptions are
    caught and recorded — the END marker is always logged.

    Args:
        stage_name: Must be a key in _STAGE_REGISTRY (e.g. "training")
        config:     Validated, immutable PipelineConfig
        version_id: Dataset version identifier for this run.
        fine_tune:  If True, the training stage loads existing Production model
                    weights and continues training with fine-tune hyperparameters
                    instead of training from scratch. Ignored for non-CNN tasks.

    Returns:
        StageResult capturing timing, status, and optional error info.

    Raises:
        KeyError: If stage_name is not found in _STAGE_REGISTRY.
    """
    if stage_name not in _STAGE_REGISTRY:
        raise KeyError(
            f"Unknown pipeline stage: '{stage_name}'. "
            f"Registered stages: {', '.join(sorted(_STAGE_REGISTRY))}"
        )

    start = datetime.now(timezone.utc)
    logger.info("===== START: %s =====", stage_name)

    try:
        stage_fn = _STAGE_REGISTRY[stage_name]
        if stage_name == "training":
            stage_fn(config, version_id, fine_tune=fine_tune)
        else:
            stage_fn(config, version_id)
        status = "completed"
        error = None
    except PromotionBlockedError as e:
        status = "blocked"
        error = str(e)
        logger.warning("  %s", e)
    except ApprovalUnavailableError as e:
        status = "cancelled"
        error = str(e)
        logger.warning("  Stage '%s' cancelled: %s", stage_name, e)
    except Exception as e:
        status = "failed"
        error = str(e)
        logger.error("  Stage '%s' failed: %s", stage_name, e, exc_info=True)
        
    end = datetime.now(timezone.utc)
    duration = (end - start).total_seconds()
    
    logger.info("===== END: %s (%s, %.3fs) =====", stage_name, status, duration)
    
    return StageResult(
        stage=stage_name,
        status=status,
        started_at=start.isoformat(),
        ended_at=end.isoformat(),
        duration_seconds=duration,
        error=error,
    )
    

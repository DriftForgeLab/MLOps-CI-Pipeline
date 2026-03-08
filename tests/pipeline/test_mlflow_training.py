import pytest
import mlflow
from src.pipeline.mlflow_logger import log_training_to_mlflow

# TrainingResult is defined in src/training/__init__.py
from src.training import TrainingResult


def _make_result(**kwargs):
    defaults = dict(
        model=None,
        algorithm="random_forest",
        hyperparameters={"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced"},
        dataset_version_id="d32f10723d05",
        random_seed=42,
        trained_at="2026-03-04T11:00:00+00:00",
        train_rows=104,
    )
    defaults.update(kwargs)
    return TrainingResult(**defaults)


def test_training_params_logged(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        log_training_to_mlflow(_make_result())
    data = mlflow.get_run(run.info.run_id).data
    assert data.params["algorithm"] == "random_forest"
    assert data.params["dataset_version_id"] == "d32f10723d05"
    assert data.params["random_seed"] == "42"
    assert data.params["train_rows"] == "104"
    assert data.params["hp.n_estimators"] == "100"
    assert data.params["hp.max_depth"] == "null"   # None → "null"
    assert data.params["hp.class_weight"] == "balanced"


def test_training_tags_logged(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        log_training_to_mlflow(_make_result())
    data = mlflow.get_run(run.info.run_id).data
    assert data.tags["training.algorithm"] == "random_forest"
    assert "training.trained_at" in data.tags


from src.pipeline.mlflow_logger import log_training_artifacts_to_mlflow


def test_training_artifacts_logged(tmp_path):
    model_dir = tmp_path / "artifacts/runs/abc/model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.joblib").write_bytes(b"fake")
    (model_dir / "metadata.json").write_text('{"algorithm":"random_forest"}')

    prep_dir = tmp_path / "preprocessed"
    prep_dir.mkdir()
    (prep_dir / "feature_map.json").write_text('{"output_features":["a"]}')
    (prep_dir / "metadata.json").write_text('{}')

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        log_training_artifacts_to_mlflow(model_dir=model_dir, preprocessed_dir=prep_dir)

    client = mlflow.tracking.MlflowClient()
    top_level = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "model" in top_level, f"Expected 'model' dir in artifacts, got {top_level}"
    assert "preprocessing" in top_level, f"Expected 'preprocessing' dir, got {top_level}"

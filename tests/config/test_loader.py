import pytest
from pathlib import Path
from src.config.loader import load_config, load_training_config, PipelineConfig

CONFIG_DIR = Path(__file__).parent.parent.parent / "src" / "config"

# ---------------------------------------------------------------------------
# load_config integration tests
# ---------------------------------------------------------------------------

def test_load_config_valid():
    config = load_config(CONFIG_DIR / "pipeline.yaml")
    assert config.task_type == "classification"
    assert config.random_seed == 42
    assert config.log_level == "INFO"
    assert config.project.name == "lightweight-mlops-pipeline"
    assert config.pipeline_stages == ("preprocessing", "training", "evaluation", "promotion")
    assert isinstance(config, PipelineConfig)

def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("nonexistent.yaml"))

def test_load_training_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_training_config(Path("nonexistent.yaml"))

def test_load_config_invalid_task_type(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "project:\n  name: x\n  version: '1'\n"
        "task_type: clustering\nrandom_seed: 1\n"
        "pipeline_stages: [preprocessing]\noutput_dir: out\n"
        "data:\n  raw: r\n  processed: p\n  evaluation: e\n  drift_scenarios: d\n"
        "configs:\n  training: t\n  evaluation: ev\n  deployment: dep\n"
        "log_level: INFO\ndataset: iris\n"
    )
    with pytest.raises(ValueError, match="task_type"):
        load_config(cfg)

def test_load_config_invalid_log_level(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "project:\n  name: x\n  version: '1'\n"
        "task_type: classification\nrandom_seed: 1\n"
        "pipeline_stages: [preprocessing]\noutput_dir: out\n"
        "data:\n  raw: r\n  processed: p\n  evaluation: e\n  drift_scenarios: d\n"
        "configs:\n  training: t\n  evaluation: ev\n  deployment: dep\n"
        "log_level: VERBOSE\ndataset: iris\n"
    )
    with pytest.raises(ValueError, match="log_level"):
        load_config(cfg)

def test_load_config_missing_top_level_key(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("project:\n  name: x\n  version: '1'\n")
    with pytest.raises(ValueError, match="Missing required top-level keys"):
        load_config(cfg)

def test_load_config_project_not_dict(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "project: not_a_dict\n"
        "task_type: classification\nrandom_seed: 1\n"
        "pipeline_stages: [preprocessing]\noutput_dir: out\n"
        "data:\n  raw: r\n  processed: p\n  evaluation: e\n  drift_scenarios: d\n"
        "configs:\n  training: t\n  evaluation: ev\n  deployment: dep\n"
        "log_level: INFO\ndataset: iris\n"
    )
    with pytest.raises(ValueError, match="mapping"):
        load_config(cfg)

# ---------------------------------------------------------------------------
# load_training_config integration tests
# ---------------------------------------------------------------------------

def test_load_training_config_classification():
    config = load_training_config(CONFIG_DIR / "training_classification.yaml")
    assert config.model.algorithm == "random_forest"
    assert config.model.hyperparameters.n_estimators == 100

def test_load_training_config_regression():
    config = load_training_config(CONFIG_DIR / "training_regression.yaml")
    assert config.model.algorithm == "random_forest"
    assert config.model.hyperparameters.n_estimators == 100

def test_load_training_config_invalid_algorithm(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "model:\n  algorithm: svm\n  hyperparameters:\n    n_estimators: 10\n"
    )
    with pytest.raises(ValueError, match="algorithm"):
        load_training_config(cfg)

def test_load_training_config_bad_n_estimators(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "model:\n  algorithm: random_forest\n"
        "  hyperparameters:\n    n_estimators: -5\n"
        "    max_depth: null\n    min_samples_split: 2\n    class_weight: null\n"
    )
    with pytest.raises(ValueError, match="n_estimators"):
        load_training_config(cfg)

# ---------------------------------------------------------------------------
# _validate_section unit tests
# ---------------------------------------------------------------------------
from src.config.loader import _validate_section

class TestValidateSection:
    def test_valid_section_returns_dict_and_no_errors(self):
        raw = {"data": {"raw": "r", "processed": "p", "evaluation": "e", "drift_scenarios": "d"}}
        errors = []
        result = _validate_section(raw, "data", {"raw", "processed", "evaluation", "drift_scenarios"}, errors)
        assert errors == []
        assert result == raw["data"]

    def test_not_a_dict_appends_error_and_returns_none(self):
        raw = {"data": "not_a_dict"}
        errors = []
        result = _validate_section(raw, "data", {"raw"}, errors)
        assert result is None
        assert len(errors) == 1
        assert "'data' must be a mapping" in errors[0]

    def test_missing_required_key_appends_error(self):
        raw = {"data": {"raw": "r", "processed": "p"}}
        errors = []
        _validate_section(raw, "data", {"raw", "processed", "evaluation", "drift_scenarios"}, errors)
        assert any("evaluation" in e or "drift_scenarios" in e for e in errors)

    def test_extra_keys_do_not_cause_errors(self):
        raw = {"data": {"raw": "r", "processed": "p", "evaluation": "e",
                        "drift_scenarios": "d", "unexpected": "x"}}
        errors = []
        _validate_section(raw, "data", {"raw", "processed", "evaluation", "drift_scenarios"}, errors)
        assert errors == []

# ---------------------------------------------------------------------------
# _validate_enum unit tests
# ---------------------------------------------------------------------------
from src.config.loader import _validate_enum

class TestValidateEnum:
    def test_valid_value_no_error(self):
        errors = []
        _validate_enum("classification", {"classification", "regression"}, "task_type", errors)
        assert errors == []

    def test_invalid_value_appends_error(self):
        errors = []
        _validate_enum("clustering", {"classification", "regression"}, "task_type", errors)
        assert len(errors) == 1
        assert "clustering" in errors[0]
        assert "task_type" in errors[0]

    def test_error_message_lists_valid_options(self):
        errors = []
        _validate_enum("bad", {"lbfgs", "saga", "liblinear"}, "solver", errors)
        assert "lbfgs" in errors[0] or "saga" in errors[0]

    def test_case_sensitive(self):
        errors = []
        _validate_enum("Classification", {"classification", "regression"}, "task_type", errors)
        assert len(errors) == 1

# ---------------------------------------------------------------------------
# _validate_positive_int unit tests
# ---------------------------------------------------------------------------
from src.config.loader import _validate_positive_int

class TestValidatePositiveInt:
    def test_valid_positive_int(self):
        errors = []
        _validate_positive_int(10, "n_estimators", errors)
        assert errors == []

    def test_zero_is_invalid(self):
        errors = []
        _validate_positive_int(0, "n_estimators", errors)
        assert len(errors) == 1
        assert "n_estimators" in errors[0]

    def test_negative_is_invalid(self):
        errors = []
        _validate_positive_int(-1, "n_estimators", errors)
        assert len(errors) == 1

    def test_bool_true_is_invalid(self):
        errors = []
        _validate_positive_int(True, "n_estimators", errors)
        assert len(errors) == 1

    def test_bool_false_is_invalid(self):
        errors = []
        _validate_positive_int(False, "n_estimators", errors)
        assert len(errors) == 1

    def test_float_is_invalid(self):
        errors = []
        _validate_positive_int(1.5, "n_estimators", errors)
        assert len(errors) == 1

    def test_none_disallowed_by_default(self):
        errors = []
        _validate_positive_int(None, "n_estimators", errors)
        assert len(errors) == 1

    def test_none_allowed_when_allow_null_true(self):
        errors = []
        _validate_positive_int(None, "max_depth", errors, allow_null=True)
        assert errors == []

    def test_min_val_enforced(self):
        errors = []
        _validate_positive_int(1, "min_samples_split", errors, min_val=2)
        assert len(errors) == 1

    def test_exactly_min_val_is_valid(self):
        errors = []
        _validate_positive_int(2, "min_samples_split", errors, min_val=2)
        assert errors == []


# ---------------------------------------------------------------------------
# Image classification config tests
# ---------------------------------------------------------------------------

def test_load_config_image_classification():
    config = load_config(CONFIG_DIR / "pipeline_image_classification.yaml")
    assert config.task_type == "image_classification"
    assert config.dataset == "sample_images"


def test_load_preprocessing_config_with_image_section():
    from src.config.loader import load_preprocessing_config
    prep = load_preprocessing_config(CONFIG_DIR / "preprocessing_image.yaml")
    assert prep.image is not None
    assert prep.image.target_size == (64, 64)
    assert prep.image.color_mode == "rgb"
    assert prep.image.normalize is True
    assert prep.image.flatten is True
    assert prep.image.augmentation.enabled is False


def test_image_config_invalid_color_mode(tmp_path):
    from src.config.loader import load_preprocessing_config
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        _PREP_CONFIG_YAML + "\nimage:\n  target_size: [64, 64]\n  color_mode: cmyk\n"
        "  normalize: true\n  flatten: true\n"
        "  augmentation:\n    enabled: false\n    horizontal_flip: false\n"
        "    rotation_degrees: 0\n    augmentation_factor: 1\n"
    )
    with pytest.raises(ValueError, match="color_mode"):
        load_preprocessing_config(cfg)


def test_image_config_invalid_target_size(tmp_path):
    from src.config.loader import load_preprocessing_config
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        _PREP_CONFIG_YAML + "\nimage:\n  target_size: [64]\n  color_mode: rgb\n"
        "  normalize: true\n  flatten: true\n"
        "  augmentation:\n    enabled: false\n    horizontal_flip: false\n"
        "    rotation_degrees: 0\n    augmentation_factor: 1\n"
    )
    with pytest.raises(ValueError, match="target_size"):
        load_preprocessing_config(cfg)


# Reuse the prep config yaml base from test_preprocess for image config tests
_PREP_CONFIG_YAML = """\
fail_on_nulls: true
min_rows: 1
validate_types: false
validate_labels: true
validate_on_skip: false
numeric_features: null
categorical_features: null
encoding:
  enabled: false
  strategy: onehot
  handle_unknown: ignore
  min_frequency: null
scaling:
  enabled: false
  strategy: standard
missing_values:
  policy: passthrough
  numeric_strategy: mean
  categorical_strategy: most_frequent
  fill_value: null
"""

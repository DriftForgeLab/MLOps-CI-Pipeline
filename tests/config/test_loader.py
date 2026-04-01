import pytest
from pathlib import Path
from src.config.loader import load_config, load_training_config, PipelineConfig

CONFIG_DIR = Path(__file__).parent.parent.parent / "src" / "config"

# ---------------------------------------------------------------------------
# load_config integration tests
# ---------------------------------------------------------------------------

def test_load_config_valid():
    config = load_config(CONFIG_DIR / "pipeline_tabular.yaml")
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
from src.config.validation import _validate_section

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
from src.config.validation import _validate_enum

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
from src.config.validation import _validate_positive_int

class TestValidatePositiveInt:
    @pytest.mark.parametrize("value", [1, 10, 100, 999])
    def test_valid_positive_int(self, value):
        errors = []
        _validate_positive_int(value, "n_estimators", errors)
        assert errors == []

    @pytest.mark.parametrize(
        "value, label",
        [
            (0, "zero"),
            (-1, "negative"),
            (True, "bool_true"),
            (False, "bool_false"),
            (1.5, "float"),
            (None, "none"),
        ],
        ids=lambda x: x if isinstance(x, str) else repr(x),
    )
    def test_invalid_values_rejected(self, value, label):
        errors = []
        _validate_positive_int(value, "n_estimators", errors)
        assert len(errors) == 1
        assert "n_estimators" in errors[0]

    def test_none_allowed_when_allow_null_true(self):
        errors = []
        _validate_positive_int(None, "max_depth", errors, allow_null=True)
        assert errors == []

    @pytest.mark.parametrize(
        "value, min_val, should_pass",
        [
            (1, 2, False),
            (2, 2, True),
            (3, 2, True),
        ],
    )
    def test_min_val_boundary(self, value, min_val, should_pass):
        errors = []
        _validate_positive_int(value, "min_samples_split", errors, min_val=min_val)
        if should_pass:
            assert errors == []
        else:
            assert len(errors) == 1


# ---------------------------------------------------------------------------
# Image classification config tests
# ---------------------------------------------------------------------------

def test_load_config_image_classification():
    config = load_config(CONFIG_DIR / "pipeline_image.yaml")
    assert config.task_type == "image_classification_cnn"
    assert config.dataset == "sample_images"


def test_load_preprocessing_config_with_image_section():
    from src.config.loader import load_preprocessing_config
    prep = load_preprocessing_config(CONFIG_DIR / "preprocessing_image.yaml")
    assert prep.image is not None
    assert prep.image.target_size == (64, 64)
    assert prep.image.color_mode == "rgb"
    assert prep.image.normalize is True
    assert prep.image.flatten is False
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


# ---------------------------------------------------------------------------
# ISP config loading tests
# ---------------------------------------------------------------------------

def test_load_preprocessing_raw_cnn_roundtrip():
    """preprocessing_raw_cnn.yaml must load cleanly and produce a valid ISPConfig."""
    from src.config.loader import load_preprocessing_config
    prep = load_preprocessing_config(CONFIG_DIR / "preprocessing_raw_cnn.yaml")

    assert prep.image is not None
    assert prep.image.raw_input is True
    assert prep.image.isp is not None

    isp = prep.image.isp
    assert isp.black_level_correction.enabled is True
    assert isp.black_level_correction.black_level is None   # reads from DNG
    assert isp.demosaicing.algorithm == "bilinear"
    assert isp.white_balance.r_gain is None                 # reads from DNG
    assert isp.white_balance.g_gain is None
    assert isp.white_balance.b_gain is None
    assert isp.color_correction.enabled is True
    assert isp.color_correction.matrix is None              # reads from DNG
    assert isp.denoising.algorithm == "gaussian"
    assert isp.denoising.strength == 0.5
    assert isp.sharpening.algorithm == "unsharp_mask"
    assert isp.sharpening.radius == 1.0
    assert isp.sharpening.amount == 1.0
    assert isp.gamma_correction.gamma == 2.2


def test_raw_input_true_without_isp_block_raises(tmp_path):
    """raw_input: true without an isp: block must fail validation."""
    from src.config.loader import load_preprocessing_config
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        _PREP_CONFIG_YAML
        + "\nimage:\n  target_size: [64, 64]\n  color_mode: rgb\n"
        "  normalize: true\n  flatten: false\n  raw_input: true\n"
        "  augmentation:\n    enabled: false\n    horizontal_flip: false\n"
        "    rotation_degrees: 0\n    augmentation_factor: 1\n"
    )
    with pytest.raises(ValueError, match="raw_input"):
        load_preprocessing_config(cfg)


def test_isp_gamma_out_of_range_raises(tmp_path):
    from src.config.loader import load_preprocessing_config
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        _PREP_CONFIG_YAML
        + "\nimage:\n  target_size: [64, 64]\n  color_mode: rgb\n"
        "  normalize: true\n  flatten: false\n  raw_input: true\n"
        "  augmentation:\n    enabled: false\n    horizontal_flip: false\n"
        "    rotation_degrees: 0\n    augmentation_factor: 1\n"
        "  isp:\n    gamma_correction:\n      gamma: 5.0\n"
    )
    with pytest.raises(ValueError, match="gamma"):
        load_preprocessing_config(cfg)


def test_isp_invalid_demosaicing_algorithm_raises(tmp_path):
    from src.config.loader import load_preprocessing_config
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        _PREP_CONFIG_YAML
        + "\nimage:\n  target_size: [64, 64]\n  color_mode: rgb\n"
        "  normalize: true\n  flatten: false\n  raw_input: true\n"
        "  augmentation:\n    enabled: false\n    horizontal_flip: false\n"
        "    rotation_degrees: 0\n    augmentation_factor: 1\n"
        "  isp:\n    demosaicing:\n      algorithm: nearest_neighbor\n"
    )
    with pytest.raises(ValueError, match="demosaicing"):
        load_preprocessing_config(cfg)


def test_isp_white_balance_gain_out_of_range_raises(tmp_path):
    from src.config.loader import load_preprocessing_config
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        _PREP_CONFIG_YAML
        + "\nimage:\n  target_size: [64, 64]\n  color_mode: rgb\n"
        "  normalize: true\n  flatten: false\n  raw_input: true\n"
        "  augmentation:\n    enabled: false\n    horizontal_flip: false\n"
        "    rotation_degrees: 0\n    augmentation_factor: 1\n"
        "  isp:\n    white_balance:\n      r_gain: 5.0\n"  # > 2.0
    )
    with pytest.raises(ValueError, match="r_gain"):
        load_preprocessing_config(cfg)


def test_isp_black_level_wrong_length_raises(tmp_path):
    from src.config.loader import load_preprocessing_config
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        _PREP_CONFIG_YAML
        + "\nimage:\n  target_size: [64, 64]\n  color_mode: rgb\n"
        "  normalize: true\n  flatten: false\n  raw_input: true\n"
        "  augmentation:\n    enabled: false\n    horizontal_flip: false\n"
        "    rotation_degrees: 0\n    augmentation_factor: 1\n"
        "  isp:\n    black_level_correction:\n      black_level: [512, 512]\n"  # need 4
    )
    with pytest.raises(ValueError, match="black_level"):
        load_preprocessing_config(cfg)


def test_isp_explicit_black_level_roundtrips_as_tuple(tmp_path):
    """An explicit black_level list in YAML must be stored as a tuple of floats."""
    from src.config.loader import load_preprocessing_config
    cfg = tmp_path / "ok.yaml"
    cfg.write_text(
        _PREP_CONFIG_YAML
        + "\nimage:\n  target_size: [64, 64]\n  color_mode: rgb\n"
        "  normalize: true\n  flatten: false\n  raw_input: true\n"
        "  augmentation:\n    enabled: false\n    horizontal_flip: false\n"
        "    rotation_degrees: 0\n    augmentation_factor: 1\n"
        "  isp:\n    black_level_correction:\n      black_level: [512, 513, 514, 515]\n"
    )
    prep = load_preprocessing_config(cfg)
    bl = prep.image.isp.black_level_correction.black_level
    assert bl == (512.0, 513.0, 514.0, 515.0)
    assert isinstance(bl, tuple)


def test_isp_all_algorithms_accepted(tmp_path):
    """Every algorithm listed in VALID_*_ALGORITHMS must pass loader validation."""
    from src.config.loader import load_preprocessing_config
    from src.config.schema import (
        VALID_DEMOSAICING_ALGORITHMS,
        VALID_DENOISING_ALGORITHMS,
        VALID_SHARPENING_ALGORITHMS,
    )

    def _make_cfg(demosaicing="bilinear", denoising="gaussian", sharpening="unsharp_mask"):
        return (
            _PREP_CONFIG_YAML
            + "\nimage:\n  target_size: [64, 64]\n  color_mode: rgb\n"
            "  normalize: true\n  flatten: false\n  raw_input: true\n"
            "  augmentation:\n    enabled: false\n    horizontal_flip: false\n"
            "    rotation_degrees: 0\n    augmentation_factor: 1\n"
            f"  isp:\n"
            f"    demosaicing:\n      algorithm: {demosaicing}\n"
            f"    denoising:\n      algorithm: {denoising}\n      strength: 0.5\n"
            f"    sharpening:\n      algorithm: {sharpening}\n      radius: 1.0\n      amount: 1.0\n"
        )

    for algo in VALID_DEMOSAICING_ALGORITHMS:
        cfg = tmp_path / f"d_{algo}.yaml"
        cfg.write_text(_make_cfg(demosaicing=algo))
        load_preprocessing_config(cfg)  # must not raise

    for algo in VALID_DENOISING_ALGORITHMS:
        cfg = tmp_path / f"dn_{algo}.yaml"
        cfg.write_text(_make_cfg(denoising=algo))
        load_preprocessing_config(cfg)

    for algo in VALID_SHARPENING_ALGORITHMS:
        cfg = tmp_path / f"sh_{algo}.yaml"
        cfg.write_text(_make_cfg(sharpening=algo))
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

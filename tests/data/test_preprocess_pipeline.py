# =============================================================================
# tests/data/test_preprocess_pipeline.py — Full preprocessing pipeline tests
# =============================================================================
# Steps 15a-15r: pure-function unit tests + integration tests via
# run_preprocessing(). Covers feature-type resolution, pipeline construction,
# null fail-fast, output column naming, leakage-proof transform, idempotency,
# schema-dtype validation, and validate_on_skip behaviour.
# =============================================================================

import json
import joblib
import pytest
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from src.config.loader import (
    EncodingConfig,
    MissingValuesConfig,
    PreprocessingConfig,
    ScalingConfig,
)
from src.data.preprocess import (
    PIPELINE_VERSION,
    PREPROCESSED_SUBDIR,
    _build_pipeline,
    _check_nulls_fail_fast,
    _compute_preprocess_hash,
    _get_output_columns,
    _resolve_feature_types,
    run_preprocessing,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_version_dir(
    tmp_path: Path,
    features: list[str] | None = None,
    target: str = "label",
    task_type: str = "classification",
    schema: dict | None = None,
    constraints: dict | None = None,
    numeric_vals: dict | None = None,
    categorical_vals: dict | None = None,
    target_vals: list | None = None,
    train_n: int = 6,
    val_n: int = 3,
    test_n: int = 3,
    version_id: str = "v1",
    dataset_name: str = "ds",
) -> Path:
    """Build a minimal version directory usable by run_preprocessing()."""
    if features is None:
        features = ["num_a", "num_b"]
    if schema is None:
        schema = {f: "float" for f in features}
        schema[target] = "string" if task_type == "classification" else "float"
    if constraints is None:
        constraints = {"min_rows": 1, "max_null_fraction": 0.0}

    version_dir = tmp_path / dataset_name / version_id
    version_dir.mkdir(parents=True)

    metadata = {
        "name": dataset_name,
        "task_type": task_type,
        "features": features,
        "target": target,
        "schema": schema,
        "constraints": constraints,
    }
    with open(version_dir / "dataset.yaml", "w") as f:
        yaml.dump(metadata, f)

    def _make_df(n: int) -> pd.DataFrame:
        row: dict = {}
        for feat in features:
            if categorical_vals and feat in categorical_vals:
                vals = categorical_vals[feat]
                row[feat] = [vals[i % len(vals)] for i in range(n)]
            elif numeric_vals and feat in numeric_vals:
                vals = numeric_vals[feat]
                row[feat] = [vals[i % len(vals)] for i in range(n)]
            else:
                row[feat] = [float(i) for i in range(n)]
        if target_vals:
            row[target] = [target_vals[i % len(target_vals)] for i in range(n)]
        elif task_type == "classification":
            row[target] = ["A" if i % 2 == 0 else "B" for i in range(n)]
        else:
            row[target] = [float(i) for i in range(n)]
        return pd.DataFrame(row)

    train_df = _make_df(train_n)
    val_df = _make_df(val_n)
    test_df = _make_df(test_n)

    # Combined CSV at version root (required by validate_dataset)
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined.to_csv(version_dir / "data.csv", index=False)

    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        (version_dir / split).mkdir()
        df.to_csv(version_dir / split / "data.csv", index=False)

    return version_dir


def _write_prep_config(
    path: Path,
    encoding_enabled: bool = False,
    encoding_strategy: str = "onehot",
    handle_unknown: str = "ignore",
    min_frequency: int | None = None,
    scaling_enabled: bool = False,
    scaling_strategy: str = "standard",
    missing_policy: str = "passthrough",
    validate_on_skip: bool = False,
    validate_types: bool = False,
    numeric_features: list | None = None,
    categorical_features: list | None = None,
    min_rows: int = 1,
) -> Path:
    """Write a preprocessing YAML config to path and return it."""
    cfg: dict = {
        "fail_on_nulls": True,
        "min_rows": min_rows,
        "validate_types": validate_types,
        "validate_labels": False,
        "validate_on_skip": validate_on_skip,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "encoding": {
            "enabled": encoding_enabled,
            "strategy": encoding_strategy,
            "handle_unknown": handle_unknown,
            "min_frequency": min_frequency,
        },
        "scaling": {
            "enabled": scaling_enabled,
            "strategy": scaling_strategy,
        },
        "missing_values": {
            "policy": missing_policy,
            "numeric_strategy": "mean",
            "categorical_strategy": "most_frequent",
            "fill_value": None,
        },
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


# ---------------------------------------------------------------------------
# 15a-15e — _resolve_feature_types()
# ---------------------------------------------------------------------------

def test_15a_both_explicit_overrides_schema():
    """Both explicit lists → schema ignored, explicit lists used as-is."""
    prep = PreprocessingConfig(
        numeric_features=("a", "b"),
        categorical_features=("c",),
    )
    schema = {"a": "string", "b": "string", "c": "float"}  # deliberately wrong types
    df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": ["x"]})
    num, cat = _resolve_feature_types(["a", "b", "c"], "label", schema, df, prep, False)
    assert num == ["a", "b"]
    assert cat == ["c"]


def test_15b_auto_detect_from_schema():
    """No explicit overrides → float/int → numeric, string/bool → categorical."""
    prep = PreprocessingConfig()
    schema = {"num1": "float", "num2": "int", "cat1": "string", "cat2": "bool"}
    df = pd.DataFrame({
        "num1": [1.0], "num2": [1], "cat1": ["x"], "cat2": [True]
    })
    num, cat = _resolve_feature_types(
        ["num1", "num2", "cat1", "cat2"], "label", schema, df, prep, False
    )
    assert "num1" in num and "num2" in num
    assert "cat1" in cat and "cat2" in cat
    assert set(num) & set(cat) == set()


def test_15c_target_in_features_raises():
    """Target column appearing in features list must raise ValueError immediately."""
    prep = PreprocessingConfig()
    df = pd.DataFrame({"a": [1.0], "label": [0.0]})
    with pytest.raises(ValueError, match="target"):
        _resolve_feature_types(["a", "label"], "label", {}, df, prep, False)


def test_15d_dtype_mismatch_raises_when_validate_types_true():
    """Column declared 'int' but actual dtype is float → ValueError."""
    prep = PreprocessingConfig()
    schema = {"a": "int"}
    df = pd.DataFrame({"a": [1.5, 2.5]})  # float, not int
    with pytest.raises(ValueError, match="dtype"):
        _resolve_feature_types(["a"], "label", schema, df, prep, validate_types=True)


def test_15e_dtype_check_skipped_when_validate_types_false():
    """validate_types=False → dtype mismatch ignored, no error."""
    prep = PreprocessingConfig()
    schema = {"a": "int"}
    df = pd.DataFrame({"a": [1.5, 2.5]})  # float, declared as int — no error
    num, cat = _resolve_feature_types(["a"], "label", schema, df, prep, validate_types=False)
    assert "a" in num or "a" in cat  # assigned somewhere, no exception


# ---------------------------------------------------------------------------
# 15f-15h — _build_pipeline()
# ---------------------------------------------------------------------------

def _fit_pipeline(prep: PreprocessingConfig, numeric_cols, categorical_cols, df):
    """Helper: build and fit pipeline; return (pipeline, X_t)."""
    pipe = _build_pipeline(numeric_cols, categorical_cols, prep)
    X = df[numeric_cols + categorical_cols]
    X_t = pipe.fit_transform(X)
    return pipe, X_t


def test_15f_ohe_encoder_in_pipeline_when_encoding_enabled():
    """encoding.enabled=True, strategy='onehot' → OHE step inside cat sub-pipeline."""
    from sklearn.preprocessing import OneHotEncoder
    prep = PreprocessingConfig(encoding=EncodingConfig(enabled=True, strategy="onehot"))
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    pipe, _ = _fit_pipeline(prep, ["num"], ["cat"], df)
    ct = pipe.named_steps["preprocessor"]
    cat_pipe = ct.named_transformers_["cat"]
    assert hasattr(cat_pipe, "named_steps"), "Expected a Pipeline, not passthrough"
    assert isinstance(cat_pipe.named_steps["encoder"], OneHotEncoder)


def test_15g_scaler_in_pipeline_when_scaling_enabled():
    """scaling.enabled=True → scaler step inside numeric sub-pipeline."""
    from sklearn.preprocessing import StandardScaler
    prep = PreprocessingConfig(scaling=ScalingConfig(enabled=True, strategy="standard"))
    df = pd.DataFrame({"num1": [1.0, 2.0, 3.0], "num2": [4.0, 5.0, 6.0]})
    pipe, _ = _fit_pipeline(prep, ["num1", "num2"], [], df)
    ct = pipe.named_steps["preprocessor"]
    num_pipe = ct.named_transformers_["num"]
    assert hasattr(num_pipe, "named_steps"), "Expected a Pipeline, not passthrough"
    assert isinstance(num_pipe.named_steps["scaler"], StandardScaler)


def test_15h_passthrough_default_all_disabled():
    """All defaults disabled → numeric and categorical subs use 'passthrough'."""
    prep = PreprocessingConfig()  # all defaults: no encoding, no scaling, passthrough
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    pipe, X_t = _fit_pipeline(prep, ["num"], ["cat"], df)
    # Dense output, shape unchanged (1 numeric + 1 categorical = 2 cols)
    assert isinstance(X_t, np.ndarray)
    assert X_t.shape == (3, 2)


# ---------------------------------------------------------------------------
# 15i-15j — _check_nulls_fail_fast()
# ---------------------------------------------------------------------------

def test_15i_null_fail_fast_raises_on_nulls():
    """Null in any feature column → ValueError listing the column and count."""
    df = pd.DataFrame({"a": [1.0, float("nan"), 3.0], "b": [4.0, 5.0, 6.0]})
    with pytest.raises(ValueError, match="'a'"):
        _check_nulls_fail_fast(df, "train")


def test_15j_null_fail_fast_passes_on_clean_data():
    """No nulls → no exception."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    _check_nulls_fail_fast(df, "train")  # must not raise


# ---------------------------------------------------------------------------
# 15k-15l — _get_output_columns()
# ---------------------------------------------------------------------------

def test_15k_ohe_output_columns_expanded():
    """OHE → column names follow sklearn get_feature_names_out pattern."""
    from sklearn.preprocessing import OneHotEncoder
    prep = PreprocessingConfig(encoding=EncodingConfig(enabled=True, strategy="onehot"))
    df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["x", "y"]})
    pipe, X_t = _fit_pipeline(prep, ["num"], ["cat"], df)
    cols = _get_output_columns(pipe, ["num"], ["cat"], X_t)
    assert "num" in cols
    # OHE produces "cat_x" and "cat_y"
    assert any("cat_" in c for c in cols)
    assert len(cols) == X_t.shape[1]


def test_15l_passthrough_output_columns_original_names():
    """All-passthrough pipeline → output column names equal input names."""
    prep = PreprocessingConfig()
    df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["x", "y"]})
    pipe, X_t = _fit_pipeline(prep, ["num"], ["cat"], df)
    cols = _get_output_columns(pipe, ["num"], ["cat"], X_t)
    assert cols == ["num", "cat"]
    assert len(cols) == X_t.shape[1]


# ---------------------------------------------------------------------------
# 15m — Leakage-proof: pipeline fit on train only
# ---------------------------------------------------------------------------

def test_15m_pipeline_fit_on_train_only_unseen_categories_produce_zeros(tmp_path):
    """
    Leakage-proof contract: pipeline.fit_transform() called exclusively on train.
    val/test use pipeline.transform() only.

    Verification:
      - OHE categories_ learned from train only (no 'bird' from val)
      - Val row with unseen 'bird' → all-zero OHE output (handle_unknown='ignore')
    """
    version_dir = _make_version_dir(
        tmp_path,
        features=["num_feat", "cat_feat"],
        target="label",
        task_type="classification",
        schema={"num_feat": "float", "cat_feat": "string", "label": "string"},
        constraints={"min_rows": 1, "max_null_fraction": 0.0},
        categorical_vals={"cat_feat": ["cat", "dog"]},
        train_n=4,
        val_n=2,
    )

    # Overwrite val and test with unseen category 'bird'
    unseen_df = pd.DataFrame({
        "num_feat": [9.0, 10.0],
        "cat_feat": ["bird", "bird"],
        "label": ["A", "B"],
    })
    unseen_df.to_csv(version_dir / "val" / "data.csv", index=False)
    unseen_df.to_csv(version_dir / "test" / "data.csv", index=False)

    cfg = _write_prep_config(
        tmp_path / "prep.yaml",
        encoding_enabled=True,
        encoding_strategy="onehot",
        handle_unknown="ignore",
        numeric_features=["num_feat"],
        categorical_features=["cat_feat"],
    )

    run_preprocessing("ds", "v1", prep_config_path=cfg, processed_dir=tmp_path)

    preprocessed_dir = version_dir / PREPROCESSED_SUBDIR
    pipeline = joblib.load(preprocessed_dir / "pipeline.pkl")
    ct = pipeline.named_steps["preprocessor"]
    cat_pipe = ct.named_transformers_["cat"]
    ohe = cat_pipe.named_steps["encoder"]

    # OHE must only know train categories; 'bird' must NOT be in categories_
    train_cats = set(ohe.categories_[0])
    assert "cat" in train_cats, "OHE should have learned 'cat' from train"
    assert "dog" in train_cats, "OHE should have learned 'dog' from train"
    assert "bird" not in train_cats, (
        "'bird' is in OHE categories_ — pipeline was refit on val/test (leakage!)"
    )

    # Val row with unseen 'bird' → all-zero OHE output
    val_out = pd.read_csv(preprocessed_dir / "val.csv")
    ohe_cols = [c for c in val_out.columns if c.startswith("cat_feat_")]
    assert len(ohe_cols) > 0, "Expected OHE-expanded cat_feat columns in val.csv"
    for col in ohe_cols:
        assert val_out[col].iloc[0] == pytest.approx(0.0), (
            f"OHE col '{col}' is non-zero for unseen 'bird' — "
            "handle_unknown='ignore' must produce all zeros"
        )


# ---------------------------------------------------------------------------
# 15n-15o — Idempotency: config changes force rerun
# ---------------------------------------------------------------------------

def test_15n_encoding_change_forces_rerun(tmp_path):
    """Enabling encoding changes the hash → metadata.json must be overwritten."""
    _make_version_dir(tmp_path)
    cfg = _write_prep_config(tmp_path / "prep.yaml", encoding_enabled=False)
    run_preprocessing("ds", "v1", prep_config_path=cfg, processed_dir=tmp_path)

    meta_path = tmp_path / "ds" / "v1" / PREPROCESSED_SUBDIR / "metadata.json"
    mtime1 = meta_path.stat().st_mtime

    # Change config: enable encoding
    _write_prep_config(tmp_path / "prep.yaml", encoding_enabled=True)
    run_preprocessing("ds", "v1", prep_config_path=cfg, processed_dir=tmp_path)

    assert meta_path.stat().st_mtime != mtime1, (
        "metadata.json was NOT overwritten after encoding config changed — "
        "hash-based cache invalidation is broken"
    )


def test_15o_identical_config_skips(tmp_path):
    """Identical config on second run → metadata.json NOT overwritten (cache hit)."""
    _make_version_dir(tmp_path)
    cfg = _write_prep_config(tmp_path / "prep.yaml")
    run_preprocessing("ds", "v1", prep_config_path=cfg, processed_dir=tmp_path)

    meta_path = tmp_path / "ds" / "v1" / PREPROCESSED_SUBDIR / "metadata.json"
    mtime1 = meta_path.stat().st_mtime

    run_preprocessing("ds", "v1", prep_config_path=cfg, processed_dir=tmp_path)

    assert meta_path.stat().st_mtime == mtime1, (
        "metadata.json was overwritten on second run with identical config"
    )


# ---------------------------------------------------------------------------
# 15p-15q — Schema dtype validation
# ---------------------------------------------------------------------------

def test_15p_validate_types_raises_on_dtype_mismatch(tmp_path):
    """validate_types=True + actual dtype ≠ declared type → ValueError at runtime."""
    # Schema declares "num_a" as "int", but CSV will have float values
    version_dir = _make_version_dir(
        tmp_path,
        features=["num_a"],
        schema={"num_a": "int", "label": "string"},
        constraints={"min_rows": 1, "max_null_fraction": 0.0},
    )
    # Overwrite all splits with float data (violates "int" schema)
    float_df = pd.DataFrame({"num_a": [1.5, 2.5, 3.5], "label": ["A", "B", "A"]})
    float_df.to_csv(version_dir / "data.csv", index=False)
    for split in ["train", "val", "test"]:
        float_df.to_csv(version_dir / split / "data.csv", index=False)

    cfg = _write_prep_config(tmp_path / "prep.yaml", validate_types=True)
    with pytest.raises(ValueError, match="dtype"):
        run_preprocessing("ds", "v1", prep_config_path=cfg, processed_dir=tmp_path)


def test_15q_validate_types_false_does_not_block_run(tmp_path):
    """validate_types=False in prep config → _resolve_feature_types skips dtype check.
    A clean dataset (schema matches data) runs successfully regardless of the flag."""
    version_dir = _make_version_dir(
        tmp_path,
        features=["num_a"],
        schema={"num_a": "float", "label": "string"},  # schema matches actual float dtype
        constraints={"min_rows": 1, "max_null_fraction": 0.0},
    )

    cfg = _write_prep_config(tmp_path / "prep.yaml", validate_types=False)
    run_preprocessing("ds", "v1", prep_config_path=cfg, processed_dir=tmp_path)

    assert (version_dir / PREPROCESSED_SUBDIR / "metadata.json").exists()


# ---------------------------------------------------------------------------
# 15r — validate_on_skip behaviour
# ---------------------------------------------------------------------------

def test_15r_validate_on_skip_false_does_not_reread_splits(tmp_path):
    """
    validate_on_skip=False: on cache hit, val/test raw CSVs are NOT re-read.
    Corrupt val CSV after first run → second run must succeed (skip path only
    checks file existence, not content).
    """
    version_dir = _make_version_dir(
        tmp_path,
        constraints={"min_rows": 1, "max_null_fraction": 0.0},
    )
    cfg = _write_prep_config(tmp_path / "prep.yaml", validate_on_skip=False)
    run_preprocessing("ds", "v1", prep_config_path=cfg, processed_dir=tmp_path)

    # Introduce nulls in val raw split — would fail validate_split_data if re-read
    corrupt_df = pd.DataFrame({"num_a": [None, None, None], "num_b": [1.0, 2.0, 3.0], "label": ["A", "B", "A"]})
    corrupt_df.to_csv(version_dir / "val" / "data.csv", index=False)

    # Should succeed: validate_on_skip=False means val CSV is not re-read on cache hit
    run_preprocessing("ds", "v1", prep_config_path=cfg, processed_dir=tmp_path)


def test_15r_validate_on_skip_true_revalidates_splits(tmp_path):
    """
    validate_on_skip=True: on cache hit, val/test raw CSVs ARE re-read and
    validated. Corrupt val CSV after first run → second run must raise.
    """
    version_dir = _make_version_dir(
        tmp_path,
        constraints={"min_rows": 1, "max_null_fraction": 0.0},
    )
    cfg = _write_prep_config(tmp_path / "prep.yaml", validate_on_skip=True)
    run_preprocessing("ds", "v1", prep_config_path=cfg, processed_dir=tmp_path)

    # Introduce nulls (exceed max_null_fraction=0.0 → validation error)
    corrupt_df = pd.DataFrame({"num_a": [None, None, None], "num_b": [1.0, 2.0, 3.0], "label": ["A", "B", "A"]})
    corrupt_df.to_csv(version_dir / "val" / "data.csv", index=False)

    with pytest.raises(ValueError, match="null"):
        run_preprocessing("ds", "v1", prep_config_path=cfg, processed_dir=tmp_path)


# ---------------------------------------------------------------------------
# Integration: artefact completeness
# ---------------------------------------------------------------------------

def test_all_artefacts_written_after_first_run(tmp_path):
    """After a fresh run all artefacts must exist: train/val/test.csv, pipeline.pkl,
    feature_map.json, metadata.json."""
    version_dir = _make_version_dir(tmp_path)
    cfg = _write_prep_config(tmp_path / "prep.yaml")
    run_preprocessing("ds", "v1", prep_config_path=cfg, processed_dir=tmp_path)

    preprocessed_dir = version_dir / PREPROCESSED_SUBDIR
    for name in ["train.csv", "val.csv", "test.csv", "pipeline.pkl",
                 "feature_map.json", "metadata.json"]:
        assert (preprocessed_dir / name).exists(), f"{name} missing after run"


def test_feature_map_json_structure(tmp_path):
    """feature_map.json must have output_features, target, numeric_input, categorical_input."""
    version_dir = _make_version_dir(tmp_path)
    cfg = _write_prep_config(tmp_path / "prep.yaml")
    run_preprocessing("ds", "v1", prep_config_path=cfg, processed_dir=tmp_path)

    with open(version_dir / PREPROCESSED_SUBDIR / "feature_map.json") as f:
        fm = json.load(f)

    assert "output_features" in fm
    assert "target" in fm
    assert "numeric_input" in fm
    assert "categorical_input" in fm
    assert fm["target"] == "label"


def test_metadata_pipeline_version_recorded(tmp_path):
    """metadata.json must record the PIPELINE_VERSION constant."""
    version_dir = _make_version_dir(tmp_path)
    cfg = _write_prep_config(tmp_path / "prep.yaml")
    run_preprocessing("ds", "v1", prep_config_path=cfg, processed_dir=tmp_path)

    with open(version_dir / PREPROCESSED_SUBDIR / "metadata.json") as f:
        meta = json.load(f)

    assert meta.get("pipeline_version") == PIPELINE_VERSION


def test_target_is_last_column_in_preprocessed_csv(tmp_path):
    """Target column must be the last column in all preprocessed split CSVs."""
    version_dir = _make_version_dir(tmp_path, features=["num_a", "num_b"], target="label")
    cfg = _write_prep_config(tmp_path / "prep.yaml")
    run_preprocessing("ds", "v1", prep_config_path=cfg, processed_dir=tmp_path)

    preprocessed_dir = version_dir / PREPROCESSED_SUBDIR
    for split in ["train", "val", "test"]:
        df = pd.read_csv(preprocessed_dir / f"{split}.csv")
        assert df.columns[-1] == "label", (
            f"[{split}] target 'label' must be last column, got '{df.columns[-1]}'"
        )


def test_ohe_expands_columns_in_output_csv(tmp_path):
    """With OHE enabled, output train.csv must have more columns than raw input."""
    version_dir = _make_version_dir(
        tmp_path,
        features=["num_feat", "cat_feat"],
        schema={"num_feat": "float", "cat_feat": "string", "label": "string"},
        constraints={"min_rows": 1, "max_null_fraction": 0.0},
        categorical_vals={"cat_feat": ["x", "y", "z"]},
    )
    cfg = _write_prep_config(
        tmp_path / "prep.yaml",
        encoding_enabled=True,
        numeric_features=["num_feat"],
        categorical_features=["cat_feat"],
    )
    run_preprocessing("ds", "v1", prep_config_path=cfg, processed_dir=tmp_path)

    raw_cols = 3  # num_feat + cat_feat + label
    out_df = pd.read_csv(version_dir / PREPROCESSED_SUBDIR / "train.csv")
    # OHE for 3 categories → 3 new cols replacing 1 → total > raw
    assert len(out_df.columns) > raw_cols, (
        "OHE should expand categorical columns; output has same or fewer columns than raw"
    )

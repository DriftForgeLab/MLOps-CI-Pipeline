# =============================================================================
# src/data/preprocess.py — Preprocessing of versioned dataset splits
# =============================================================================
# Responsibility: Select the relevant columns from each split (train/val/test)
# and write the results to a preprocessed/ subdirectory inside the version dir.
#
# Design: Two-layer approach keeps concerns separated.
#   preprocess()        — pure function: no file I/O, no side effects.
#                         Easy to test in isolation.
#   run_preprocessing() — I/O boundary: reads splits, calls preprocess(),
#                         writes results.
#
# Idempotency: hash-based. A SHA-256 fingerprint of (sorted features + target)
# is stored in preprocessed/metadata.json. On re-run, if the hash matches the
# stored one, the step is skipped. If the config changed, it re-runs and
# overwrites — so callers always get outputs that match the current config.
#
# Column source: target and feature columns come from dataset.yaml, which is
# the single source of truth. No separate preprocessing config file is needed.
#
# Boundary contract: preprocess() applies BASE transforms only (column
# selection). Drift perturbations are NEVER applied here — they are applied
# separately to the already-preprocessed data. This keeps the preprocess
# output stable across all drift scenarios.
# =============================================================================

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import math
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src.common.io import atomic_write_csv, atomic_write_json, atomic_write_pickle
from src.config.loader import PreprocessingConfig, load_preprocessing_config
from src.data.validate import _is_dtype_compatible, validate_dataset, validate_split_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Bump PIPELINE_VERSION when _build_pipeline() output format changes in a
# breaking way (column order, encoder behaviour, etc.).
# Minor bump: new supported strategy added.
# Major bump: output format or column-order contract changed.
# History:
#   1.0.0 — initial full pipeline (encoding + scaling + impute)
PIPELINE_VERSION = "1.0.0"

# Sub-directory within a version dir where preprocessed artefacts are stored.
# Imported by training modules — single source of truth for this path segment.
PREPROCESSED_SUBDIR = "preprocessed"

_SCHEMA_TYPE_TO_CATEGORY: dict[str, str] = {
    "float": "numeric",
    "int": "numeric",
    "string": "categorical",
    "bool": "categorical",
}


# ---------------------------------------------------------------------------
# Step 3 — Feature-type resolution
# ---------------------------------------------------------------------------

def _resolve_feature_types(
    features: list[str],
    target: str,
    schema: dict[str, str],
    df_train: pd.DataFrame,
    prep_config: PreprocessingConfig,
    validate_types: bool,
) -> tuple[list[str], list[str]]:
    """Resolve which features are numeric and which are categorical.

    Priority:
      1. Explicit config overrides (prep_config.numeric_features /
         prep_config.categorical_features) — take precedence over schema.
      2. Auto-detection from dataset.yaml schema types.
      3. Unknown schema types default to numeric with a warning.

    When validate_types=True, actual pandas dtypes of df_train are checked
    against the declared schema type. Mismatches raise ValueError (fail-all).

    Invariants enforced (hard errors):
      - target must not appear in features.
      - Every feature must be assigned to exactly one list.
      - numeric_cols and categorical_cols must not overlap.

    Args:
        features:      Feature columns from dataset.yaml (original order preserved).
        target:        Target column name.
        schema:        Column name → declared type mapping from dataset.yaml.
        df_train:      Loaded training DataFrame (used for dtype validation only).
        prep_config:   Validated PreprocessingConfig.
        validate_types: When True, check actual dtype against schema.

    Returns:
        (numeric_cols, categorical_cols) — both in original feature-list order.

    Raises:
        ValueError: On any invariant violation or dtype mismatch.
    """
    # --- Invariant: target must never be a feature ---
    if target in features:
        raise ValueError(
            f"Target column '{target}' appears in the features list. "
            "The target must not be a feature — check dataset.yaml."
        )

    # --- Dtype validation against schema ---
    if validate_types:
        errors: list[str] = []
        for col in features:
            declared_type = schema.get(col)
            if declared_type is None:
                logger.warning(
                    "Column '%s' has no schema entry — skipping dtype check.", col
                )
                continue
            if col not in df_train.columns:
                errors.append(
                    f"Column '{col}' declared in features is missing from training data."
                )
                continue
            if not _is_dtype_compatible(df_train[col], declared_type):
                errors.append(
                    f"Column '{col}' declared as '{declared_type}' in schema "
                    f"but training data has dtype '{df_train[col].dtype}'. "
                    "Fix the data or schema before running preprocessing."
                )
        if errors:
            raise ValueError(
                "Feature dtype validation failed:\n  - " + "\n  - ".join(errors)
            )

    # --- Resolve numeric and categorical lists (original feature-list order) ---
    both_explicit = (
        prep_config.numeric_features is not None
        and prep_config.categorical_features is not None
    )
    num_explicit = prep_config.numeric_features is not None
    cat_explicit = prep_config.categorical_features is not None

    if both_explicit:
        numeric_cols = list(prep_config.numeric_features)
        categorical_cols = list(prep_config.categorical_features)
    elif num_explicit:
        
        numeric_cols = list(prep_config.numeric_features)
        explicit_num_set = set(numeric_cols)
        categorical_cols = [col for col in features if col not in explicit_num_set]
    elif cat_explicit:
        
        categorical_cols = list(prep_config.categorical_features)
        explicit_cat_set = set(categorical_cols)
        numeric_cols = [col for col in features if col not in explicit_cat_set]
    else:
        # Full auto-detect from schema
        numeric_cols = []
        categorical_cols = []
        for col in features:
            declared_type = schema.get(col)
            category = _SCHEMA_TYPE_TO_CATEGORY.get(declared_type or "")
            if category == "numeric":
                numeric_cols.append(col)
            elif category == "categorical":
                categorical_cols.append(col)
            else:
                logger.warning(
                    "Column '%s' has unknown schema type '%s'; treating as numeric. "
                    "Add an explicit override in preprocessing.yaml if this is wrong.",
                    col,
                    declared_type,
                )
                numeric_cols.append(col)

    # --- Final invariant checks ---
    resolution_errors: list[str] = []

    uncovered = set(features) - (set(numeric_cols) | set(categorical_cols))
    if uncovered:
        resolution_errors.append(
            f"Features not assigned to any type: {sorted(uncovered)}. "
            "Add them to numeric_features or categorical_features in preprocessing.yaml, "
            "or ensure they have a recognised schema type (float/int/string/bool)."
        )

    overlap = set(numeric_cols) & set(categorical_cols)
    if overlap:
        resolution_errors.append(
            f"Features in both numeric and categorical lists: {sorted(overlap)}. "
            "Each feature must appear in exactly one list."
        )

    if resolution_errors:
        raise ValueError(
            "Feature type resolution failed:\n  - " + "\n  - ".join(resolution_errors)
        )

    return numeric_cols, categorical_cols


# ---------------------------------------------------------------------------
# Step 4 — sklearn pipeline factory
# ---------------------------------------------------------------------------

def _build_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    prep_config: PreprocessingConfig,
) -> Pipeline:
    """Build an unfitted sklearn Pipeline from PreprocessingConfig.

    Output column order contract (enforced by ColumnTransformer declaration):
      [numeric output columns in original feature-list order]
      + [categorical expanded columns in deterministic order]

    ColumnTransformer is declared numeric FIRST, categorical SECOND.
    Output column names are derived via _get_output_columns() after fitting.

    sparse_output=False is mandatory on OneHotEncoder — dense output is a
    pipeline contract. A post-transform guard in run_preprocessing() enforces
    this for all transform types.

    Args:
        numeric_cols:     Numeric feature columns (original feature-list order).
        categorical_cols: Categorical feature columns (original feature-list order).
        prep_config:      Validated PreprocessingConfig.

    Returns:
        Unfitted sklearn Pipeline with a single "preprocessor" ColumnTransformer step.
    """
    enc_cfg = prep_config.encoding
    scl_cfg = prep_config.scaling
    mv_cfg = prep_config.missing_values

    # --- Numeric sub-pipeline ---
    numeric_steps: list[tuple[str, Any]] = []
    if mv_cfg.policy == "impute":
        fill_value = mv_cfg.fill_value if mv_cfg.numeric_strategy == "constant" else None
        numeric_steps.append((
            "imputer",
            SimpleImputer(strategy=mv_cfg.numeric_strategy, fill_value=fill_value),
        ))
    if scl_cfg.enabled:
        scaler = StandardScaler() if scl_cfg.strategy == "standard" else MinMaxScaler()
        numeric_steps.append(("scaler", scaler))

    # --- Categorical sub-pipeline ---
    categorical_steps: list[tuple[str, Any]] = []
    if mv_cfg.policy == "impute":
        cat_fill = mv_cfg.fill_value if mv_cfg.categorical_strategy == "constant" else None
        categorical_steps.append((
            "imputer",
            SimpleImputer(strategy=mv_cfg.categorical_strategy, fill_value=cat_fill),
        ))
    if enc_cfg.enabled:
        if enc_cfg.strategy == "onehot":
            ohe_kwargs: dict[str, Any] = {
                "handle_unknown": enc_cfg.handle_unknown,
                # sparse_output=False is mandatory — dense output is a pipeline contract.
                "sparse_output": False,
            }
            if enc_cfg.min_frequency is not None:
                ohe_kwargs["min_frequency"] = enc_cfg.min_frequency
            categorical_steps.append(("encoder", OneHotEncoder(**ohe_kwargs)))
        else:  # ordinal
            if enc_cfg.handle_unknown == "ignore":
                categorical_steps.append((
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ))
            else:
                categorical_steps.append(("encoder", OrdinalEncoder(handle_unknown="error")))

    # Use "passthrough" string when there are no steps for a sub-pipeline
    num_pipe: Pipeline | str = Pipeline(numeric_steps) if numeric_steps else "passthrough"
    cat_pipe: Pipeline | str = Pipeline(categorical_steps) if categorical_steps else "passthrough"

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))

    col_transformer = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocessor", col_transformer)])


# ---------------------------------------------------------------------------
# Step 5 — Null fail-fast check
# ---------------------------------------------------------------------------

def _check_nulls_fail_fast(df: pd.DataFrame, split_name: str) -> None:
    """Raise ValueError if any column in df contains NaN.

    Called when missing_values.policy == "fail". Checks X-only (target
    excluded before this call). Collects all null-containing columns before
    raising so the error message lists everything in one pass.

    Args:
        df:         Feature DataFrame (target column must not be present).
        split_name: Used in the error message to identify the offending split.

    Raises:
        ValueError: If any NaN found, with per-column null counts.
    """
    null_cols = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if df[col].isna().any()
    }
    if null_cols:
        details = ", ".join(f"'{col}': {cnt} null(s)" for col, cnt in null_cols.items())
        raise ValueError(
            f"Null values found in [{split_name}] X (missing_values.policy='fail'): "
            f"{details}. Fix the data or change missing_values.policy to "
            "'impute' or 'passthrough'."
        )


# ---------------------------------------------------------------------------
# Step 7 — Output column naming
# ---------------------------------------------------------------------------

def _get_output_columns(
    fitted_pipeline: Pipeline,
    numeric_cols: list[str],
    categorical_cols: list[str],
    X_transformed: np.ndarray,
) -> list[str]:
    """Derive output column names from a fitted pipeline in contract order.

    Output column order contract:
      [numeric output cols in original feature-list order]
      + [categorical expanded cols in deterministic order]

    Numeric: scalers and passthrough are order-preserving and do not rename.
      Output names = numeric_cols as-is.

    Categorical with OHE: get_feature_names_out(categorical_cols) gives
      deterministic expanded names (e.g. "species_setosa"). Do NOT re-sort —
      sklearn's OHE sorts .categories_ alphabetically at fit time, so the
      output is already deterministic.

    Categorical with ordinal/passthrough: output names = categorical_cols.

    An integrity assertion verifies len(output_cols) == X_transformed.shape[1]
    in every code path. A mismatch is always a bug in this function.

    Args:
        fitted_pipeline: Fitted sklearn Pipeline from _build_pipeline().
        numeric_cols:    Numeric columns in original feature-list order.
        categorical_cols: Categorical columns in original feature-list order.
        X_transformed:   The array returned by pipeline.fit_transform() or
                         .transform() — used only for the shape assertion.

    Returns:
        List of output column names, length == X_transformed.shape[1].

    Raises:
        ValueError: If derived column count does not match array width.
    """
    ct = fitted_pipeline.named_steps["preprocessor"]  # ColumnTransformer

    # ── Numeric output names ─────────────────────────────────────────────────
    # StandardScaler, MinMaxScaler, SimpleImputer, and passthrough all
    # preserve the original column order and do not rename.
    numeric_out: list[str] = list(numeric_cols)

    # ── Categorical output names ─────────────────────────────────────────────
    categorical_out: list[str]
    cat_transformer = ct.named_transformers_.get("cat")

    if cat_transformer is None or cat_transformer == "passthrough":
        # No categorical columns or no transforms applied — original names.
        categorical_out = list(categorical_cols)
    elif hasattr(cat_transformer, "named_steps"):
        # Pipeline — look for the encoder step.
        encoder = cat_transformer.named_steps.get("encoder")
        if encoder is None:
            # Imputer only, no encoder — original names.
            categorical_out = list(categorical_cols)
        elif isinstance(encoder, OneHotEncoder):
            # OHE: get_feature_names_out gives deterministic expanded names.
            # Do NOT re-sort — OHE sorts categories alphabetically at fit time;
            # the result of get_feature_names_out is already deterministic.
            categorical_out = list(encoder.get_feature_names_out(categorical_cols))
        else:
            # OrdinalEncoder or unknown encoder: output names = input names.
            categorical_out = list(categorical_cols)
    else:
        # Unexpected transformer structure — fall back to original names.
        categorical_out = list(categorical_cols)

    output_cols = numeric_out + categorical_out

    # ── Integrity assertion ──────────────────────────────────────────────────
    # This must hold in every code path. A mismatch is always a bug here,
    # never a data issue. Raise ValueError so it propagates to the caller.
    if len(output_cols) != X_transformed.shape[1]:
        raise ValueError(
            f"Column name count ({len(output_cols)}) != transformed array width "
            f"({X_transformed.shape[1]}). This is a bug in _get_output_columns — "
            f"numeric_out={numeric_out}, categorical_out={categorical_out}."
        )

    return output_cols


# ---------------------------------------------------------------------------
# Legacy column-selection function (kept for backward compatibility)
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    columns = features + [target]
    return df[columns].copy()

def _compute_preprocess_hash(
    features: list[str],
    target: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    prep_config: PreprocessingConfig,
    pipeline_version: str = PIPELINE_VERSION,
) -> str:
    """Deterministic fingerprint of the full preprocessing configuration.

    Covers the complete transform policy so that any change that would produce
    different output (different encoder, scaler, imputer, or resolved column
    lists) invalidates the cached artefacts and forces a re-run.

    IMPORTANT: columns are SORTED here for hash determinism only.
    The ColumnTransformer is constructed with the ORIGINAL feature-list order;
    do not use sorted lists for anything except this hash payload.

    Library versions (sklearn/pandas/numpy) are recorded in metadata.json for
    traceability but do NOT drive cache invalidation. Bump PIPELINE_VERSION
    explicitly if a dependency change would alter the transform output.

    Args:
        features:         All feature columns from dataset.yaml.
        target:           Target column name.
        numeric_cols:     Resolved numeric columns (before sorting for hash).
        categorical_cols: Resolved categorical columns (before sorting for hash).
        prep_config:      Validated PreprocessingConfig.
        pipeline_version: Semver string from PIPELINE_VERSION constant.

    Returns:
        "sha256:<hex>" fingerprint string.
    """
    # IMPORTANT: We sort here for hash determinism only. The ColumnTransformer
    # is constructed with the ORIGINAL feature-list order, not the sorted order.
    canonical = json.dumps(
        {
            "pipeline_version": pipeline_version,
            "features": sorted(features),
            "target": target,
            "numeric_cols": sorted(numeric_cols),
            "categorical_cols": sorted(categorical_cols),
            "encoding_enabled": prep_config.encoding.enabled,
            "encoding_strategy": prep_config.encoding.strategy,
            "handle_unknown": prep_config.encoding.handle_unknown,
            "min_frequency": prep_config.encoding.min_frequency,
            "scaling_enabled": prep_config.scaling.enabled,
            "scaling_strategy": prep_config.scaling.strategy,
            "missing_policy": prep_config.missing_values.policy,
            "numeric_impute_strategy": prep_config.missing_values.numeric_strategy,
            "categorical_impute_strategy": prep_config.missing_values.categorical_strategy,
        },
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _compute_dataset_signature(dataset_meta: dict) -> str:
    """
    Fingerprint of dataset contract inputs that *should* invalidate preprocessing outputs
    even if features/target stay constant.

    Includes schema + constraints + task_type + features/target.
    """
    canonical_obj = {
        "task_type": dataset_meta.get("task_type", "regression"),
        "features": sorted(dataset_meta.get("features", [])),
        "target": dataset_meta.get("target"),
        "schema": dataset_meta.get("schema", {}),
        "constraints": dataset_meta.get("constraints", {}),
    }
    canonical = json.dumps(canonical_obj, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()

def _load_existing_metadata(preprocessed_dir: Path) -> dict | None:
    """Load existing metadata from preprocessed/metadata.json if it exists and is valid.

    Args:
        preprocessed_dir (Path): The directory where preprocessed metadata.json is expected to be found.

    Returns:
        dict | None: The loaded metadata dictionary if it exists and is valid, otherwise None.
    """
    
    meta_path = preprocessed_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            logger.warning("metadata.json is not an object — will rebuild.")
            return None
        return meta
    except Exception as e:

        logger.warning("Failed to read metadata.json (%s) — will rebuild.", e)
        return None


def _compute_split_stats(
    df: pd.DataFrame,
    target: str,
    task_type: str,
) -> dict:
    """Compute observability stats for one preprocessed split.

    Used as the drift-analysis baseline — these numbers describe what
    'normal' data looks like coming out of preprocessing.
    """
    stats: dict = {
        "rows_out": len(df),
        "null_counts": {col: int(df[col].isna().sum()) for col in df.columns},
    }

    if task_type == "classification" and target in df.columns:

        vc = df[target].value_counts(dropna=False)
        stats["label_distribution"] = {str(k): int(v) for k, v in vc.items()}

    feature_cols = [c for c in df.columns if c != target]
    feature_stats: dict = {}
    categorical_stats: dict = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            mean = float(df[col].mean())
            std = float(df[col].std())
            mn = float(df[col].min())
            mx = float(df[col].max())
            feature_stats[col] = {
                "mean": round(mean, 6) if not (math.isnan(mean) or math.isinf(mean)) else None,
                "std":  round(std, 6) if not (math.isnan(std) or math.isinf(std)) else None,
                "min":  round(mn, 6) if not (math.isnan(mn) or math.isinf(mn)) else None,
                "max":  round(mx, 6) if not (math.isnan(mx) or math.isinf(mx)) else None,
            }
        else: 
            vc = df[col].astype("string").value_counts(dropna=False).head(10)
            categorical_stats[col] = {str(k): int(v) for k, v in vc.items()}
    if feature_stats:
        stats["feature_stats"] = feature_stats
    if categorical_stats:
        stats["categorical_top10"] = categorical_stats

    return stats

    
def _outputs_exist(preprocessed_dir: Path) -> bool:
    return (
        all((preprocessed_dir / f"{s}.csv").exists() for s in ("train", "val", "test"))
        and (preprocessed_dir / "metadata.json").exists()
        and (preprocessed_dir / "pipeline.pkl").exists()
        and (preprocessed_dir / "feature_map.json").exists()
    )

def run_preprocessing(
    dataset_name: str,
    version_id: str,
    prep_config_path: Path,
    processed_dir: Path = Path("data/processed"),
) -> None:
    """Run the preprocessing step for a given dataset version.

    Loads the preprocessing config, resolves feature types, builds and fits
    an sklearn Pipeline on the training split, transforms all splits without
    re-fitting (leakage-proof), and writes artefacts atomically.

    Idempotency: SHA-256 fingerprint of the full transform policy (prep config
    + resolved column lists + PIPELINE_VERSION + dataset contract). On re-run,
    if fingerprint matches stored metadata AND artefact files exist, the step is
    skipped. validate_on_skip controls whether raw splits are re-validated on skip.

    Args:
        dataset_name:     Dataset folder name under processed_dir.
        version_id:       Version folder under processed_dir/dataset_name.
        prep_config_path: Path to the preprocessing YAML config file.
        processed_dir:    Root directory for processed datasets.

    Raises:
        FileNotFoundError: If the versioned dataset directory does not exist.
        ValueError:        If dataset.yaml is malformed, data contract violated,
                           or the pipeline produces unexpected output.
    """
    # --- Load preprocessing config ---
    prep_config = load_preprocessing_config(prep_config_path)

    version_dir = processed_dir / dataset_name / version_id
    if not version_dir.exists():
        raise FileNotFoundError(
            f"No versioned dataset found at '{version_dir}'. "
            "Ensure the dataset has been ingested before running preprocessing."
        )

    validate_dataset(dataset_name, version_id, processed_dir=processed_dir)
    preprocessed_dir = version_dir / PREPROCESSED_SUBDIR

    # --- Load dataset contract ---
    yaml_path = version_dir / "dataset.yaml"
    with open(yaml_path) as f:
        metadata = yaml.safe_load(f)
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Invalid dataset.yaml at '{yaml_path}': expected a YAML mapping/object"
        )
    target: str = metadata["target"]
    features: list[str] = metadata["features"]
    task_type: str = metadata.get("task_type", "regression")
    schema: dict[str, str] = metadata.get("schema", {})

    # --- Load train split for dtype validation and pipeline fitting ---
    df_train = pd.read_csv(version_dir / "train" / "data.csv")
    validate_split_data(df_train, metadata, split_name="train")

    # --- Resolve feature types and compute fingerprints ---
    numeric_cols, categorical_cols = _resolve_feature_types(
        features, target, schema, df_train, prep_config, prep_config.validate_types
    )
    preprocess_hash = _compute_preprocess_hash(
        features, target, numeric_cols, categorical_cols, prep_config
    )
    dataset_signature = _compute_dataset_signature(metadata)

    # --- Idempotency check ---
    existing_meta = _load_existing_metadata(preprocessed_dir)
    if existing_meta:
        hash_ok = (
            existing_meta.get("preprocess_hash") == preprocess_hash
            and existing_meta.get("dataset_signature") == dataset_signature
            and _outputs_exist(preprocessed_dir)
        )
        if hash_ok:
            if prep_config.validate_on_skip:
                # Full validation on skip: re-read and validate remaining raw splits.
                # train was already validated above; only val/test remain.
                for split in ["val", "test"]:
                    df_s = pd.read_csv(version_dir / split / "data.csv")
                    validate_split_data(df_s, metadata, split_name=split)
            logger.info(
                "  Preprocessing up-to-date for version '%s' — skipping.", version_id
            )
            return

    if existing_meta:
        logger.info(
            "  Preprocess config or dataset contract changed — rerunning for version '%s'.",
            version_id,
        )

    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    # --- Null fail-fast on train X before fitting ---
    X_train = df_train[features]
    if prep_config.missing_values.policy == "fail":
        _check_nulls_fail_fast(X_train, "train")

    # --- Build and fit pipeline on train split only (leakage-proof) ---
    pipeline = _build_pipeline(numeric_cols, categorical_cols, prep_config)
    X_train_t: np.ndarray = pipeline.fit_transform(X_train)

    # Dense-output guard — a sparse return is a bug in _build_pipeline.
    if not isinstance(X_train_t, np.ndarray):
        raise ValueError(
            f"Pipeline returned {type(X_train_t).__name__} instead of numpy.ndarray "
            "after fit_transform on train. Ensure sparse_output=False is set on all "
            "encoders. This is a bug in _build_pipeline."
        )

    output_cols = _get_output_columns(pipeline, numeric_cols, categorical_cols, X_train_t)

    # --- Feature map artefact ---
    feature_map: dict = {
        "output_features": output_cols,
        "target": target,
        "numeric_input": numeric_cols,
        "categorical_input": categorical_cols,
    }

    # --- Write transformed splits ---
    split_stats: dict[str, dict] = {}
    null_counts_per_split: dict[str, dict] = {}

    # Train split (already transformed above — do not re-fit)
    y_train = df_train[target].reset_index(drop=True)
    train_out = pd.DataFrame(X_train_t, columns=output_cols)
    train_out[target] = y_train.values
    atomic_write_csv(train_out, preprocessed_dir / "train.csv")
    split_stats["train"] = _compute_split_stats(train_out, target, task_type)
    null_counts_per_split["train"] = {
        col: int(X_train[col].isna().sum()) for col in X_train.columns
    }

    # Val and test splits — transform-only, never fit
    for split in ["val", "test"]:
        df_split = pd.read_csv(version_dir / split / "data.csv")
        validate_split_data(df_split, metadata, split_name=split)

        X_split = df_split[features]
        if prep_config.missing_values.policy == "fail":
            _check_nulls_fail_fast(X_split, split)

        X_split_t: np.ndarray = pipeline.transform(X_split)
        if not isinstance(X_split_t, np.ndarray):
            raise ValueError(
                f"Pipeline returned {type(X_split_t).__name__} for split '{split}'. "
                "This is a bug in _build_pipeline — all transformers must return dense output."
            )

        y_split = df_split[target].reset_index(drop=True)
        split_out = pd.DataFrame(X_split_t, columns=output_cols)
        split_out[target] = y_split.values
        atomic_write_csv(split_out, preprocessed_dir / f"{split}.csv")
        split_stats[split] = _compute_split_stats(split_out, target, task_type)
        null_counts_per_split[split] = {
            col: int(df_split[col].isna().sum()) for col in X_split.columns
        }

    # --- Persist fitted pipeline artefact ---
    atomic_write_pickle(pipeline, preprocessed_dir / "pipeline.pkl")

    # --- Persist feature map ---
    atomic_write_json(preprocessed_dir / "feature_map.json", feature_map)

    # --- Write rich metadata ---
    transforms_active = (
        prep_config.encoding.enabled
        or prep_config.scaling.enabled
        or prep_config.missing_values.policy == "impute"
    )
    meta_payload: dict = {
        "dataset_version_id": version_id,
        "dataset_signature": dataset_signature,
        "preprocess_hash": preprocess_hash,
        "pipeline_version": PIPELINE_VERSION,
        "preprocessed_at": datetime.now(timezone.utc).isoformat(),
        "transform_description": (
            "full_pipeline" if transforms_active else "column_selection_only"
        ),
        "drift_safe": True,
        "output_features": output_cols,
        "target": target,
        "output_shape": {
            split: split_stats[split]["rows_out"] for split in ("train", "val", "test")
        },
        "null_counts_per_split": null_counts_per_split,
        "artefacts": {
            "pipeline": f"{PREPROCESSED_SUBDIR}/pipeline.pkl",
            "feature_map": f"{PREPROCESSED_SUBDIR}/feature_map.json",
        },
        "library_versions": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "splits": split_stats,
    }
    if prep_config.missing_values.policy == "passthrough":
        meta_payload["governance_note"] = (
            "missing_values.policy=passthrough: NaN values pass through silently. "
            "This is the legacy compatibility mode. "
            "Governance recommendation: switch to policy=fail for new pipelines."
        )
    atomic_write_json(preprocessed_dir / "metadata.json", meta_payload)

    logger.info(
        "  Preprocessing complete: %d features + target='%s' → %s/{train,val,test}.csv"
        " + pipeline.pkl + feature_map.json",
        len(features),
        target,
        PREPROCESSED_SUBDIR,
    )

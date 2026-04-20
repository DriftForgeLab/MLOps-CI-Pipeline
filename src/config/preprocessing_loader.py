"""Preprocessing config loader — validate, build, and return a PreprocessingConfig."""

import logging
from pathlib import Path

from src.config.schema import (
    VALID_ENCODING_STRATEGIES,
    VALID_HANDLE_UNKNOWN,
    VALID_SCALING_STRATEGIES,
    VALID_MISSING_POLICIES,
    VALID_NUMERIC_IMPUTE_STRATEGIES,
    VALID_CATEGORICAL_IMPUTE_STRATEGIES,
    VALID_IMAGE_COLOR_MODES,
    VALID_DEMOSAICING_ALGORITHMS,
    VALID_DENOISING_ALGORITHMS,
    VALID_SHARPENING_ALGORITHMS,
    _PREPROCESSING_TOP_LEVEL_KEYS,
    _IMAGE_KEYS,
    _IMAGE_AUGMENTATION_KEYS,
    _ISP_KEYS,
    _ISP_BLACK_LEVEL_KEYS,
    _ISP_DEMOSAICING_KEYS,
    _ISP_WHITE_BALANCE_KEYS,
    _ISP_COLOR_CORRECTION_KEYS,
    _ISP_DENOISING_KEYS,
    _ISP_SHARPENING_KEYS,
    _ISP_GAMMA_KEYS,
    _ENCODING_KEYS,
    _SCALING_KEYS,
    _MISSING_VALUES_KEYS,
    EncodingConfig,
    ScalingConfig,
    MissingValuesConfig,
    ImageAugmentationConfig,
    ISPBlackLevelConfig,
    ISPDemosaicingConfig,
    ISPWhiteBalanceConfig,
    ISPColorCorrectionConfig,
    ISPDenoisingConfig,
    ISPSharpeningConfig,
    ISPGammaConfig,
    ISPConfig,
    ImagePreprocessingConfig,
    PreprocessingConfig,
)
from src.config.validation import (
    _load_yaml,
    _validate_enum,
    _validate_bool,
    _validate_optional_section,
)

logger = logging.getLogger(__name__)


def _validate_feature_overrides(raw: dict, errors: list[str]) -> None:
    """Validate numeric_features / categorical_features lists and their overlap."""
    num_feats = raw.get("numeric_features")
    cat_feats = raw.get("categorical_features")
    for key, v in (("numeric_features", num_feats), ("categorical_features", cat_feats)):
        if v is not None:
            if not isinstance(v, list):
                errors.append(f"'{key}' must be null or a list of strings, got {type(v).__name__!r}")
            else:
                bad = [x for x in v if not isinstance(x, str) or not x.strip()]
                if bad:
                    errors.append(
                        f"'{key}' must contain non-empty strings only; bad entries: {bad!r}"
                    )
                elif len(v) != len(set(v)):
                    seen = [x for x in v if v.count(x) > 1]
                    errors.append(f"'{key}' contains duplicate entries: {sorted(set(seen))!r}")

    # Cross-field: warn if both explicit lists are provided and overlap
    if (
        isinstance(num_feats, list)
        and isinstance(cat_feats, list)
        and not errors  # only if both parsed cleanly
    ):
        overlap = set(num_feats) & set(cat_feats)
        if overlap:
            errors.append(
                f"'numeric_features' and 'categorical_features' overlap: {sorted(overlap)!r}. "
                "Each feature must appear in exactly one list."
            )


def _validate_isp_section(isp: dict, errors: list[str]) -> None:
    """Validate the image.isp sub-section with per-parameter bounds."""
    extra_isp = set(isp.keys()) - _ISP_KEYS
    if extra_isp:
        logger.warning("Unknown keys in 'image.isp' (possible typo): %s", ", ".join(sorted(extra_isp)))

    # black_level_correction
    bl = isp.get("black_level_correction")
    if bl is not None:
        if not isinstance(bl, dict):
            errors.append("'image.isp.black_level_correction' must be a mapping")
        else:
            extra = set(bl.keys()) - _ISP_BLACK_LEVEL_KEYS
            if extra:
                logger.warning("Unknown keys in 'image.isp.black_level_correction': %s", ", ".join(sorted(extra)))
            _validate_bool(bl, "enabled", errors, prefix="image.isp.black_level_correction.")
            if "black_level" in bl and bl["black_level"] is not None:
                bv = bl["black_level"]
                if not isinstance(bv, list) or len(bv) != 4 or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in bv):
                    errors.append(
                        f"'image.isp.black_level_correction.black_level' must be a list of 4 numbers or null, got {bv!r}"
                    )

    # demosaicing
    dm = isp.get("demosaicing")
    if dm is not None:
        if not isinstance(dm, dict):
            errors.append("'image.isp.demosaicing' must be a mapping")
        else:
            extra = set(dm.keys()) - _ISP_DEMOSAICING_KEYS
            if extra:
                logger.warning("Unknown keys in 'image.isp.demosaicing': %s", ", ".join(sorted(extra)))
            if "algorithm" in dm:
                _validate_enum(dm["algorithm"], VALID_DEMOSAICING_ALGORITHMS, "image.isp.demosaicing.algorithm", errors)

    # white_balance
    wb = isp.get("white_balance")
    if wb is not None:
        if not isinstance(wb, dict):
            errors.append("'image.isp.white_balance' must be a mapping")
        else:
            extra = set(wb.keys()) - _ISP_WHITE_BALANCE_KEYS
            if extra:
                logger.warning("Unknown keys in 'image.isp.white_balance': %s", ", ".join(sorted(extra)))
            for gain in ("r_gain", "g_gain", "b_gain"):
                if gain in wb and wb[gain] is not None:
                    v = wb[gain]
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        errors.append(f"'image.isp.white_balance.{gain}' must be a number or null, got {v!r}")
                    elif not (0.5 <= v <= 2.0):
                        errors.append(
                            f"'image.isp.white_balance.{gain}' = {v} is outside valid range [0.5, 2.0]"
                        )

    # color_correction
    cc = isp.get("color_correction")
    if cc is not None:
        if not isinstance(cc, dict):
            errors.append("'image.isp.color_correction' must be a mapping")
        else:
            extra = set(cc.keys()) - _ISP_COLOR_CORRECTION_KEYS
            if extra:
                logger.warning("Unknown keys in 'image.isp.color_correction': %s", ", ".join(sorted(extra)))
            _validate_bool(cc, "enabled", errors, prefix="image.isp.color_correction.")
            if "matrix" in cc and cc["matrix"] is not None:
                mv = cc["matrix"]
                if not isinstance(mv, list) or len(mv) != 9 or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in mv):
                    errors.append(
                        f"'image.isp.color_correction.matrix' must be a list of 9 numbers or null, got {mv!r}"
                    )

    # denoising
    dn = isp.get("denoising")
    if dn is not None:
        if not isinstance(dn, dict):
            errors.append("'image.isp.denoising' must be a mapping")
        else:
            extra = set(dn.keys()) - _ISP_DENOISING_KEYS
            if extra:
                logger.warning("Unknown keys in 'image.isp.denoising': %s", ", ".join(sorted(extra)))
            if "algorithm" in dn:
                _validate_enum(dn["algorithm"], VALID_DENOISING_ALGORITHMS, "image.isp.denoising.algorithm", errors)
            if "strength" in dn:
                v = dn["strength"]
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    errors.append(f"'image.isp.denoising.strength' must be a number, got {v!r}")
                elif not (0.0 <= v <= 1.0):
                    errors.append(f"'image.isp.denoising.strength' = {v} is outside valid range [0.0, 1.0]")

    # sharpening
    sh = isp.get("sharpening")
    if sh is not None:
        if not isinstance(sh, dict):
            errors.append("'image.isp.sharpening' must be a mapping")
        else:
            extra = set(sh.keys()) - _ISP_SHARPENING_KEYS
            if extra:
                logger.warning("Unknown keys in 'image.isp.sharpening': %s", ", ".join(sorted(extra)))
            if "algorithm" in sh:
                _validate_enum(sh["algorithm"], VALID_SHARPENING_ALGORITHMS, "image.isp.sharpening.algorithm", errors)
            if "radius" in sh:
                v = sh["radius"]
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    errors.append(f"'image.isp.sharpening.radius' must be a number, got {v!r}")
                elif not (0.1 <= v <= 5.0):
                    errors.append(f"'image.isp.sharpening.radius' = {v} is outside valid range [0.1, 5.0]")
            if "amount" in sh:
                v = sh["amount"]
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    errors.append(f"'image.isp.sharpening.amount' must be a number, got {v!r}")
                elif not (0.0 <= v <= 3.0):
                    errors.append(f"'image.isp.sharpening.amount' = {v} is outside valid range [0.0, 3.0]")

    # gamma_correction
    gc = isp.get("gamma_correction")
    if gc is not None:
        if not isinstance(gc, dict):
            errors.append("'image.isp.gamma_correction' must be a mapping")
        else:
            extra = set(gc.keys()) - _ISP_GAMMA_KEYS
            if extra:
                logger.warning("Unknown keys in 'image.isp.gamma_correction': %s", ", ".join(sorted(extra)))
            if "gamma" in gc:
                v = gc["gamma"]
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    errors.append(f"'image.isp.gamma_correction.gamma' must be a number, got {v!r}")
                elif not (1.0 <= v <= 2.8):
                    errors.append(f"'image.isp.gamma_correction.gamma' = {v} is outside valid range [1.0, 2.8]")

    _validate_bool(isp, "gpu_accelerated", errors, prefix="image.isp.")


def _validate_image_augmentation(aug: dict, errors: list[str]) -> None:
    """Validate the image.augmentation sub-section."""
    extra_aug = set(aug.keys()) - _IMAGE_AUGMENTATION_KEYS
    if extra_aug:
        logger.warning(
            "Unknown keys in 'image.augmentation' (possible typo): %s",
            ", ".join(sorted(extra_aug)),
        )
    for key in ("enabled", "horizontal_flip"):
        _validate_bool(aug, key, errors, prefix="image.augmentation.")
    if "rotation_degrees" in aug:
        rd = aug["rotation_degrees"]
        if isinstance(rd, bool) or not isinstance(rd, int) or rd < 0:
            errors.append(
                f"'image.augmentation.rotation_degrees' must be a non-negative integer, got {rd!r}"
            )
    if "augmentation_factor" in aug:
        af = aug["augmentation_factor"]
        if isinstance(af, bool) or not isinstance(af, int) or af < 1:
            errors.append(
                f"'image.augmentation.augmentation_factor' must be an integer >= 1, got {af!r}"
            )


def _validate_image_section(img: dict, errors: list[str]) -> None:
    """Validate the image preprocessing sub-section."""
    extra_img = set(img.keys()) - _IMAGE_KEYS
    if extra_img:
        logger.warning(
            "Unknown keys in 'image' (possible typo): %s",
            ", ".join(sorted(extra_img)),
        )
    if "target_size" in img:
        ts = img["target_size"]
        if (
            not isinstance(ts, list)
            or len(ts) != 2
            or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in ts)
        ):
            errors.append(
                f"'image.target_size' must be a list of two positive integers, got {ts!r}"
            )
    if "color_mode" in img:
        _validate_enum(img["color_mode"], VALID_IMAGE_COLOR_MODES, "image.color_mode", errors)
    for key in ("normalize", "flatten"):
        _validate_bool(img, key, errors, prefix="image.")

    aug = img.get("augmentation")
    if aug is not None:
        if not isinstance(aug, dict):
            errors.append("'image.augmentation' must be a mapping")
        else:
            _validate_image_augmentation(aug, errors)

    _validate_bool(img, "raw_input", errors, prefix="image.")

    isp = img.get("isp")
    raw_input_val = img.get("raw_input", False)
    if raw_input_val and isp is None:
        errors.append(
            "'image.raw_input' is true but 'image.isp' block is missing. "
            "Add an 'isp:' section to your preprocessing config."
        )
    if isp is not None:
        if not isinstance(isp, dict):
            errors.append("'image.isp' must be a mapping")
        else:
            _validate_isp_section(isp, errors)


def _validate_preprocessing(raw: dict) -> list[str]:
    """
    Validate a preprocessing config against the PreprocessingConfig schema.

    Collects ALL errors before raising (fail-all-at-once pattern).
    Emits logger.warning() for unknown keys (typo guard) and for
    cross-field invariant violations (e.g. min_frequency on ordinal encoder).
    Returns a list of error strings; empty list means config is valid.
    """
    errors: list[str] = []

    # --- Unknown top-level keys ---
    extra_top = set(raw.keys()) - _PREPROCESSING_TOP_LEVEL_KEYS
    if extra_top:
        logger.warning(
            "Unknown keys in preprocessing config (possible typo): %s",
            ", ".join(sorted(extra_top)),
        )

    # --- Top-level booleans ---
    for key in ("fail_on_nulls", "validate_types", "validate_labels", "validate_on_skip"):
        _validate_bool(raw, key, errors)

    # --- min_rows ---
    if "min_rows" in raw:
        v = raw["min_rows"]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            errors.append(f"'min_rows' must be a positive integer, got {v!r}")

    # --- Feature type overrides ---
    _validate_feature_overrides(raw, errors)

    # --- Encoding ---
    enc = _validate_optional_section(raw, "encoding", _ENCODING_KEYS, errors)
    if enc is not None:
        _validate_bool(enc, "enabled", errors, prefix="encoding.")
        if "strategy" in enc:
            _validate_enum(enc["strategy"], VALID_ENCODING_STRATEGIES, "encoding.strategy", errors)
        if "handle_unknown" in enc:
            _validate_enum(enc["handle_unknown"], VALID_HANDLE_UNKNOWN, "encoding.handle_unknown", errors)
        if "min_frequency" in enc and enc["min_frequency"] is not None:
            v = enc["min_frequency"]
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                errors.append(
                    f"'encoding.min_frequency' must be a positive integer or null, got {v!r}"
                )
            effective_strategy = enc.get("strategy", "onehot")
            if effective_strategy != "onehot":
                logger.warning(
                    "'encoding.min_frequency' is set but 'encoding.strategy' is %r. "
                    "min_frequency only applies to 'onehot' — it will be ignored.",
                    effective_strategy,
                )

    # --- Scaling ---
    scl = _validate_optional_section(raw, "scaling", _SCALING_KEYS, errors)
    if scl is not None:
        _validate_bool(scl, "enabled", errors, prefix="scaling.")
        if "strategy" in scl:
            _validate_enum(scl["strategy"], VALID_SCALING_STRATEGIES, "scaling.strategy", errors)

    # --- Missing values ---
    mv = _validate_optional_section(raw, "missing_values", _MISSING_VALUES_KEYS, errors)
    if mv is not None:
        if "policy" in mv:
            _validate_enum(mv["policy"], VALID_MISSING_POLICIES, "missing_values.policy", errors)
        if "numeric_strategy" in mv:
            _validate_enum(
                mv["numeric_strategy"],
                VALID_NUMERIC_IMPUTE_STRATEGIES,
                "missing_values.numeric_strategy",
                errors,
            )
        if "categorical_strategy" in mv:
            _validate_enum(
                mv["categorical_strategy"],
                VALID_CATEGORICAL_IMPUTE_STRATEGIES,
                "missing_values.categorical_strategy",
                errors,
            )
        _IMPUTE_DEFAULTS = {
            "numeric_strategy": "mean",
            "categorical_strategy": "most_frequent",
            "fill_value": None,
        }
        effective_policy = mv.get("policy", "passthrough")
        if effective_policy != "impute":
            for strat_key, default_val in _IMPUTE_DEFAULTS.items():
                if strat_key in mv and mv[strat_key] != default_val:
                    logger.warning(
                        "'missing_values.%s' is set to a non-default value (%r) but "
                        "'missing_values.policy' is %r. "
                        "Impute strategies are only applied when policy='impute' — "
                        "this value will be ignored.",
                        strat_key,
                        mv[strat_key],
                        effective_policy,
                    )

    # --- Image preprocessing ---
    img = raw.get("image")
    if img is not None:
        if not isinstance(img, dict):
            errors.append("'image' must be a mapping")
        else:
            _validate_image_section(img, errors)

    return errors


def _build_isp_config(raw: dict | None) -> ISPConfig | None:
    """Construct an ISPConfig from a validated raw dict. Returns None if raw is None."""
    if raw is None:
        return None

    bl_raw = raw.get("black_level_correction") or {}
    dm_raw = raw.get("demosaicing") or {}
    wb_raw = raw.get("white_balance") or {}
    cc_raw = raw.get("color_correction") or {}
    dn_raw = raw.get("denoising") or {}
    sh_raw = raw.get("sharpening") or {}
    gc_raw = raw.get("gamma_correction") or {}

    bl_value = bl_raw.get("black_level")
    cc_matrix = cc_raw.get("matrix")

    return ISPConfig(
        black_level_correction=ISPBlackLevelConfig(
            enabled=bl_raw.get("enabled", True),
            black_level=tuple(float(v) for v in bl_value) if bl_value is not None else None,
        ),
        demosaicing=ISPDemosaicingConfig(
            algorithm=dm_raw.get("algorithm", "bilinear"),
        ),
        white_balance=ISPWhiteBalanceConfig(
            r_gain=wb_raw.get("r_gain"),
            g_gain=wb_raw.get("g_gain"),
            b_gain=wb_raw.get("b_gain"),
        ),
        color_correction=ISPColorCorrectionConfig(
            enabled=cc_raw.get("enabled", True),
            matrix=tuple(float(v) for v in cc_matrix) if cc_matrix is not None else None,
        ),
        denoising=ISPDenoisingConfig(
            algorithm=dn_raw.get("algorithm", "gaussian"),
            strength=float(dn_raw.get("strength", 0.5)),
        ),
        sharpening=ISPSharpeningConfig(
            algorithm=sh_raw.get("algorithm", "unsharp_mask"),
            radius=float(sh_raw.get("radius", 1.0)),
            amount=float(sh_raw.get("amount", 1.0)),
        ),
        gamma_correction=ISPGammaConfig(
            gamma=float(gc_raw.get("gamma", 2.2)),
        ),
        gpu_accelerated=bool(raw.get("gpu_accelerated", False)),
    )


def _build_image_preprocessing_config(raw: dict | None) -> ImagePreprocessingConfig | None:
    if raw is None:
        return None
    aug_raw = raw.get("augmentation", {}) or {}
    ts = raw.get("target_size", [64, 64])
    return ImagePreprocessingConfig(
        target_size=tuple(ts),
        color_mode=raw.get("color_mode", "rgb"),
        normalize=raw.get("normalize", True),
        flatten=raw.get("flatten", False),
        augmentation=ImageAugmentationConfig(
            enabled=aug_raw.get("enabled", False),
            horizontal_flip=aug_raw.get("horizontal_flip", False),
            rotation_degrees=aug_raw.get("rotation_degrees", 0),
            augmentation_factor=aug_raw.get("augmentation_factor", 1),
        ),
        raw_input=raw.get("raw_input", False),
        isp=_build_isp_config(raw.get("isp")),
    )


def _build_preprocessing_config(raw: dict) -> PreprocessingConfig:
    """
    Construct a PreprocessingConfig from a validated raw dict.

    Assumes _validate_preprocessing has already passed. Applies defaults for
    any key absent from the YAML (matching field defaults in the dataclasses).
    """
    enc = raw.get("encoding", {}) or {}
    scl = raw.get("scaling", {}) or {}
    mv = raw.get("missing_values", {}) or {}

    numeric_raw = raw.get("numeric_features")
    categorical_raw = raw.get("categorical_features")

    return PreprocessingConfig(
        fail_on_nulls=raw.get("fail_on_nulls", True),
        min_rows=raw.get("min_rows", 10),
        validate_types=raw.get("validate_types", True),
        validate_labels=raw.get("validate_labels", True),
        validate_on_skip=raw.get("validate_on_skip", False),
        numeric_features=tuple(numeric_raw) if numeric_raw is not None else None,
        categorical_features=tuple(categorical_raw) if categorical_raw is not None else None,
        encoding=EncodingConfig(
            enabled=enc.get("enabled", False),
            strategy=enc.get("strategy", "onehot"),
            handle_unknown=enc.get("handle_unknown", "ignore"),
            min_frequency=enc.get("min_frequency", None),
        ),
        scaling=ScalingConfig(
            enabled=scl.get("enabled", False),
            strategy=scl.get("strategy", "standard"),
        ),
        missing_values=MissingValuesConfig(
            policy=mv.get("policy", "passthrough"),
            numeric_strategy=mv.get("numeric_strategy", "mean"),
            categorical_strategy=mv.get("categorical_strategy", "most_frequent"),
            fill_value=mv.get("fill_value", None),
        ),
        image=_build_image_preprocessing_config(raw.get("image")),
    )


def load_preprocessing_config(path: Path) -> PreprocessingConfig:
    """
    Load, validate, and return a PreprocessingConfig from a YAML file.

    Follows the same contract as load_config() and load_training_config().

    Raises:
        FileNotFoundError: Config file does not exist.
        ValueError: Config file is empty, malformed, or fails validation.
    """
    logger.debug("Loading preprocessing config from: %s", path)

    raw = _load_yaml(path)

    errors = _validate_preprocessing(raw)
    if errors:
        error_report = "\n - ".join(errors)
        raise ValueError(f"Preprocessing config validation failed:\n - {error_report}")

    config = _build_preprocessing_config(raw)

    logger.debug(
        "Preprocessing config loaded: encoding=%s, scaling=%s, missing_policy=%s",
        config.encoding.enabled,
        config.scaling.enabled,
        config.missing_values.policy,
    )
    return config

# =============================================================================
# src/config/validation.py — Shared validation primitives
# =============================================================================
# Reusable helpers for YAML loading and field validation.
# Consumed by loader.py. No project imports beyond stdlib + yaml.
# =============================================================================

from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    """
    Read a YAML file and return its contents as a dict.

    Raises FileNotFoundError if the path does not exist, and ValueError
    if the file is empty or does not contain a YAML mapping.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {path}")

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping, got {type(raw).__name__}: {path}"
        )

    return raw


def _validate_positive_int(
    value: object,
    field_path: str,
    errors: list[str],
    min_val: int = 1,
    allow_null: bool = False,
) -> None:
    """
    Validate that value is a non-bool integer >= min_val.
    If allow_null=True, None is accepted without error.
    """
    if value is None:
        if not allow_null:
            errors.append(
                f"'{field_path}' must be a positive integer, got None"
            )
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < min_val:
        if min_val > 1:
            desc = f"an integer >= {min_val}"
        elif allow_null:
            desc = "a positive integer or null"
        else:
            desc = "a positive integer"
        errors.append(f"'{field_path}' must be {desc}, got {value!r}")


def _validate_enum(
    value: object,
    valid_set: set[str],
    field_name: str,
    errors: list[str],
) -> None:
    """Validate that value is one of the allowed strings in valid_set."""
    if value not in valid_set:
        errors.append(
            f"Invalid {field_name} '{value}'. "
            f"Must be one of: {', '.join(sorted(valid_set))}"
        )


def _validate_section(
    raw: dict,
    section_name: str,
    required_keys: set[str],
    errors: list[str],
) -> dict | None:
    """
    Validate a nested dict section: type-check, required keys, warn on extras.
    Returns the section dict on success, None if the value is not a dict.
    """
    section = raw[section_name]
    if not isinstance(section, dict):
        errors.append(f"'{section_name}' must be a mapping")
        return None
    missing = required_keys - section.keys()
    if missing:
        errors.append(
            f"Missing required keys in '{section_name}': {', '.join(sorted(missing))}"
        )
    extra = section.keys() - required_keys
    if extra:
        logger.warning("Unknown keys in '%s': %s", section_name, ", ".join(sorted(extra)))
    return section


def _validate_bool(raw: dict, key: str, errors: list[str], prefix: str = "") -> None:
    """Validate that raw[key], if present, is a boolean."""
    if key in raw and not isinstance(raw[key], bool):
        path = f"{prefix}{key}" if prefix else key
        errors.append(f"'{path}' must be a boolean, got {type(raw[key]).__name__!r}")


def _validate_optional_section(
    raw: dict,
    key: str,
    valid_keys: set[str],
    errors: list[str],
) -> dict | None:
    """Validate an optional nested dict: type-check, warn on unknown keys.
    Returns the section dict if valid, None otherwise."""
    section = raw.get(key, {})
    if not isinstance(section, dict):
        errors.append(f"'{key}' must be a mapping")
        return None
    extra = set(section.keys()) - valid_keys
    if extra:
        logger.warning(
            "Unknown keys in '%s' (possible typo): %s",
            key, ", ".join(sorted(extra)),
        )
    return section

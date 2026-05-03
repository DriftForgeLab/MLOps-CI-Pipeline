# =============================================================================
# src/data/image_split.py — Stratified splitting for image datasets
# =============================================================================
# Splits an ImageFolder dataset into train/val/test by copying images into
# split subdirectories while preserving class proportions via stratification.
# =============================================================================

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

from src.common.io import atomic_write_json, atomic_write_text
from src.data.image_utils import scan_image_folder
from src.data.split import _validate_split_ratios

logger = logging.getLogger(__name__)

_TRAIN_PREFIX = "train_"
_TEST_PREFIX = "test_"


def _split_outputs_exist(version_dir: Path) -> bool:
    return all(
        (version_dir / s / "images").exists()
        and (version_dir / s / "metadata.json").exists()
        for s in ("train", "val", "test")
    )


def _label_dist(labels: list[str]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for label in labels:
        dist[label] = dist.get(label, 0) + 1
    return dist


def _partition_prefixed_holdout(
    entries: list[tuple[Path, str]],
    random_seed: int,
    val_ratio: float,
) -> tuple[list[Path], list[Path], list[Path], list[str], list[str], list[str]] | None:
    """Preserve an existing filename-encoded test split when present.

    CIFAR-10 import writes filenames like train_000001.png and test_000001.png.
    When every image follows that convention and both prefixes are present, keep
    all ``test_`` images in the test split and derive validation from ``train_``
    images only.
    """
    prefixed_entries: list[tuple[Path, str, str]] = []
    for path, label in entries:
        lower_name = path.name.lower()
        if lower_name.startswith(_TRAIN_PREFIX):
            prefixed_entries.append((path, label, "train"))
        elif lower_name.startswith(_TEST_PREFIX):
            prefixed_entries.append((path, label, "test"))
        else:
            return None

    train_entries = [(path, label) for path, label, split in prefixed_entries if split == "train"]
    test_entries = [(path, label) for path, label, split in prefixed_entries if split == "test"]
    if not train_entries or not test_entries:
        return None

    train_paths_all = [path for path, _ in train_entries]
    train_labels_all = [label for _, label in train_entries]
    test_paths = [path for path, _ in test_entries]
    test_labels = [label for _, label in test_entries]

    if val_ratio <= 0.0:
        return train_paths_all, [], test_paths, train_labels_all, [], test_labels

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths_all,
        train_labels_all,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=train_labels_all,
    )
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def split_image_dataset(
    dataset_name: str,
    version_id: str,
    random_seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    processed_dir: Path = Path("data/processed"),
) -> None:
    version_dir = processed_dir / dataset_name / version_id

    if _split_outputs_exist(version_dir):
        logger.info("  Image split outputs already exist for version '%s' — skipping.", version_id)
        return

    test_ratio = _validate_split_ratios(train_ratio, val_ratio)

    yaml_path = version_dir / "dataset.yaml"
    with open(yaml_path) as f:
        metadata = yaml.safe_load(f)

    image_props = metadata.get("image_properties", {}) or {}
    expected_formats = image_props.get("expected_formats")

    images_dir = version_dir / "images"
    entries = scan_image_folder(images_dir, expected_formats)
    if not entries:
        raise ValueError(f"No images found in {images_dir}")

    paths = [e[0] for e in entries]
    labels = [e[1] for e in entries]
    split_strategy = "stratified_resplit"
    preserved_original_test = False

    partitioned = _partition_prefixed_holdout(entries, random_seed, val_ratio)
    if partitioned is not None:
        (
            train_paths,
            val_paths,
            test_paths,
            train_labels,
            val_labels,
            test_labels,
        ) = partitioned
        split_strategy = "preserve_original_test"
        preserved_original_test = True
    else:
        # Stratified split — same pattern as tabular
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths, labels,
            test_size=(1.0 - train_ratio),
            random_state=random_seed,
            stratify=labels,
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=(test_ratio / (val_ratio + test_ratio)),
            random_state=random_seed,
            stratify=temp_labels,
        )

    # Copy images into split directories
    for split_name, split_paths, split_labels in [
        ("train", train_paths, train_labels),
        ("val", val_paths, val_labels),
        ("test", test_paths, test_labels),
    ]:
        split_dir = version_dir / split_name / "images"
        for img_path, label in zip(split_paths, split_labels):
            dest_dir = split_dir / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_dir / img_path.name)

    # Write per-split metadata
    now = datetime.now(timezone.utc).isoformat()

    total_rows = len(entries)
    actual_train_ratio = len(train_paths) / total_rows
    actual_val_ratio = len(val_paths) / total_rows
    actual_test_ratio = len(test_paths) / total_rows

    base_meta = {
        "dataset_name": dataset_name,
        "dataset_version_id": version_id,
        "created_at": now,
        "random_seed": random_seed,
        "train_ratio": round(actual_train_ratio, 4),
        "val_ratio": round(actual_val_ratio, 4),
        "test_ratio": round(actual_test_ratio, 4),
        "requested_train_ratio": train_ratio,
        "requested_val_ratio": val_ratio,
        "split_strategy": split_strategy,
        "preserved_original_test": preserved_original_test,
        "task_type": metadata.get("task_type", "image_classification_cnn"),
        "target": metadata.get("target", "label"),
    }

    for name, s_paths, s_labels in [
        ("train", train_paths, train_labels),
        ("val", val_paths, val_labels),
        ("test", test_paths, test_labels),
    ]:
        payload = dict(base_meta)
        payload.update({
            "split": name,
            "rows": len(s_paths),
            "label_distribution": _label_dist(s_labels),
        })
        split_dir = version_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(split_dir / "metadata.json", payload)

    # Update dataset.yaml with split block
    split_block = {
        "train": len(train_paths),
        "val": len(val_paths),
        "test": len(test_paths),
        "train_ratio": round(actual_train_ratio, 4),
        "val_ratio": round(actual_val_ratio, 4),
        "test_ratio": round(actual_test_ratio, 4),
        "requested_train_ratio": train_ratio,
        "requested_val_ratio": val_ratio,
        "random_seed": random_seed,
        "split_strategy": split_strategy,
        "preserved_original_test": preserved_original_test,
        "label_distribution": {
            "train": _label_dist(train_labels),
            "val": _label_dist(val_labels),
            "test": _label_dist(test_labels),
        },
    }
    metadata["split"] = split_block
    yaml_text = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
    atomic_write_text(yaml_path, yaml_text)

    print(f"  Image dataset split: {len(train_paths)} train / {len(val_paths)} val / {len(test_paths)} test")
    logger.info(
        "Split complete for %s/%s (seed=%s): train=%d val=%d test=%d",
        dataset_name, version_id, random_seed,
        len(train_paths), len(val_paths), len(test_paths),
    )

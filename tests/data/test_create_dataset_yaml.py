# =============================================================================
# tests/data/test_create_dataset_yaml.py
# =============================================================================

import pytest
import yaml
from pathlib import Path
from src.data.create_dataset_yaml import detect_and_generate, generate_for_dataset


# ---------------------------------------------------------------------------
# non_interactive mode: should raise immediately if any dataset.yaml is missing
# ---------------------------------------------------------------------------

def test_non_interactive_raises_when_dataset_yaml_missing(tmp_path):
    """In CI (non-interactive), a missing dataset.yaml must fail fast with RuntimeError."""
    ds_dir = tmp_path / "my_dataset"
    ds_dir.mkdir()
    (ds_dir / "data.csv").write_text("a,b\n1,2\n3,4\n")

    with pytest.raises(RuntimeError, match="non-interactive"):
        detect_and_generate(raw_dir=tmp_path, non_interactive=True)


def test_non_interactive_error_names_the_missing_dataset(tmp_path):
    """Error message must name the offending dataset so CI logs are actionable."""
    ds_dir = tmp_path / "missing_meta_dataset"
    ds_dir.mkdir()
    (ds_dir / "data.csv").write_text("x,y\n1,2\n")

    with pytest.raises(RuntimeError, match="missing_meta_dataset"):
        detect_and_generate(raw_dir=tmp_path, non_interactive=True)


def test_non_interactive_returns_empty_when_all_yamls_present(tmp_path):
    """In CI, if every dataset has a dataset.yaml, detect_and_generate returns [] silently."""
    ds_dir = tmp_path / "complete_dataset"
    ds_dir.mkdir()
    (ds_dir / "data.csv").write_text("a,b\n1,2\n")
    (ds_dir / "dataset.yaml").write_text("name: complete_dataset\n")

    result = detect_and_generate(raw_dir=tmp_path, non_interactive=True)

    assert result == []


def test_interactive_mode_unchanged_when_no_datasets_missing(tmp_path):
    """Default (interactive=False) still returns [] when nothing is missing."""
    ds_dir = tmp_path / "full_dataset"
    ds_dir.mkdir()
    (ds_dir / "data.csv").write_text("a,b\n1,2\n")
    (ds_dir / "dataset.yaml").write_text("name: full_dataset\n")

    result = detect_and_generate(raw_dir=tmp_path)

    assert result == []


# ---------------------------------------------------------------------------
# generate_for_dataset: constraints block auto-populated
# ---------------------------------------------------------------------------

def test_generate_writes_constraints_block(tmp_path, monkeypatch):
    """Generated dataset.yaml must contain a constraints block."""
    ds_dir = tmp_path / "flowers"
    ds_dir.mkdir()
    (ds_dir / "data.csv").write_text(
        "sepal_len,petal_len,species\n"
        "5.1,1.4,setosa\n4.9,1.4,setosa\n6.2,4.5,versicolor\n"
    )

    # Simulate user answering: target=species, type=C (classification)
    answers = iter(["species", "C", "test_source", "test description"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    generate_for_dataset(ds_dir)

    with open(ds_dir / "dataset.yaml") as f:
        meta = yaml.safe_load(f)

    assert "constraints" in meta, "Generated dataset.yaml must have a 'constraints' block"


def test_generate_constraints_includes_min_rows(tmp_path, monkeypatch):
    ds_dir = tmp_path / "flowers"
    ds_dir.mkdir()
    (ds_dir / "data.csv").write_text("x,y,label\n1,2,a\n3,4,b\n5,6,a\n")

    answers = iter(["label", "C", "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    generate_for_dataset(ds_dir)

    with open(ds_dir / "dataset.yaml") as f:
        meta = yaml.safe_load(f)

    assert "min_rows" in meta["constraints"]


def test_generate_classification_constraints_include_label_classes(tmp_path, monkeypatch):
    """For classification, constraints must auto-detect label_classes from data."""
    ds_dir = tmp_path / "flowers"
    ds_dir.mkdir()
    (ds_dir / "data.csv").write_text(
        "x,label\n1,setosa\n2,versicolor\n3,setosa\n4,virginica\n"
    )

    answers = iter(["label", "C", "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    generate_for_dataset(ds_dir)

    with open(ds_dir / "dataset.yaml") as f:
        meta = yaml.safe_load(f)

    label_classes = meta["constraints"].get("label_classes", [])
    assert set(label_classes) == {"setosa", "versicolor", "virginica"}


def test_generate_regression_constraints_have_no_label_classes(tmp_path, monkeypatch):
    """Regression tasks must not get label_classes in constraints."""
    ds_dir = tmp_path / "prices"
    ds_dir.mkdir()
    (ds_dir / "data.csv").write_text("sqft,rooms,price\n100,3,250000\n200,5,450000\n")

    answers = iter(["price", "R", "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    generate_for_dataset(ds_dir)

    with open(ds_dir / "dataset.yaml") as f:
        meta = yaml.safe_load(f)

    assert "label_classes" not in meta.get("constraints", {})

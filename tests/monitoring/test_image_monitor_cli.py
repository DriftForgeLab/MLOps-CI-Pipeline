"""Tests for src.monitoring.image_monitor_cli.

Covers:
- _resolve_method: all combinations of pipeline type and model availability
- _auto_resolve_model_pt: finds model when present, returns None when missing
- version_id mismatch warning in main()
- --input-dir one-step preprocessing + monitoring
- --batch-npz two-step monitoring
- mutual exclusion of --input-dir and --batch-npz
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.monitoring.image_monitor_cli import _resolve_method, _auto_resolve_model_pt


# =============================================================================
# _resolve_method
# =============================================================================

class TestResolveMethod:
    def test_raw_isp_no_explicit_method_returns_statistical(self):
        result = _resolve_method(
            explicit_method=None,
            is_raw_isp_pipeline=True,
            model_pt_available=False,
        )
        assert result == "statistical"

    def test_raw_isp_with_model_still_returns_statistical(self):
        result = _resolve_method(
            explicit_method=None,
            is_raw_isp_pipeline=True,
            model_pt_available=True,
        )
        assert result == "statistical"

    def test_standard_with_model_returns_embedding(self):
        result = _resolve_method(
            explicit_method=None,
            is_raw_isp_pipeline=False,
            model_pt_available=True,
        )
        assert result == "embedding"

    def test_standard_without_model_returns_statistical(self):
        result = _resolve_method(
            explicit_method=None,
            is_raw_isp_pipeline=False,
            model_pt_available=False,
        )
        assert result == "statistical"

    def test_explicit_statistical_respected_on_standard(self):
        result = _resolve_method(
            explicit_method="statistical",
            is_raw_isp_pipeline=False,
            model_pt_available=True,
        )
        assert result == "statistical"

    def test_explicit_embedding_respected_on_raw(self):
        result = _resolve_method(
            explicit_method="embedding",
            is_raw_isp_pipeline=True,
            model_pt_available=True,
        )
        assert result == "embedding"

    def test_standard_no_model_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.monitoring.image_monitor_cli"):
            _resolve_method(
                explicit_method=None,
                is_raw_isp_pipeline=False,
                model_pt_available=False,
            )
        assert any("falling back to method=statistical" in r.message for r in caplog.records)

    def test_explicit_statistical_on_standard_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.monitoring.image_monitor_cli"):
            _resolve_method(
                explicit_method="statistical",
                is_raw_isp_pipeline=False,
                model_pt_available=True,
            )
        assert any("semantic drift may go undetected" in r.message for r in caplog.records)

    def test_explicit_embedding_on_raw_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.monitoring.image_monitor_cli"):
            _resolve_method(
                explicit_method="embedding",
                is_raw_isp_pipeline=True,
                model_pt_available=True,
            )
        assert any("disables ISP scenario" in r.message for r in caplog.records)

    def test_raw_isp_auto_logs_info(self, caplog):
        with caplog.at_level(logging.INFO, logger="src.monitoring.image_monitor_cli"):
            _resolve_method(
                explicit_method=None,
                is_raw_isp_pipeline=True,
                model_pt_available=False,
            )
        assert any("Auto-selected method=statistical" in r.message for r in caplog.records)

    def test_standard_with_model_auto_logs_info(self, caplog):
        with caplog.at_level(logging.INFO, logger="src.monitoring.image_monitor_cli"):
            _resolve_method(
                explicit_method=None,
                is_raw_isp_pipeline=False,
                model_pt_available=True,
            )
        assert any("Auto-selected method=embedding" in r.message for r in caplog.records)


# =============================================================================
# _auto_resolve_model_pt
# =============================================================================

class TestAutoResolveModelPt:
    def test_finds_model_when_exists(self, tmp_path):
        version_id = "abc123"
        model_dir = tmp_path / version_id / "model"
        model_dir.mkdir(parents=True)
        pt_path = model_dir / "model.pt"
        pt_path.write_bytes(b"fake")

        result = _auto_resolve_model_pt(
            artifact_dir=tmp_path,
            version_id=version_id,
            is_raw_isp_pipeline=False,
        )
        assert result == pt_path

    def test_returns_none_when_model_missing(self, tmp_path):
        result = _auto_resolve_model_pt(
            artifact_dir=tmp_path,
            version_id="abc123",
            is_raw_isp_pipeline=False,
        )
        assert result is None

    def test_returns_none_for_raw_isp_pipeline(self, tmp_path):
        version_id = "abc123"
        model_dir = tmp_path / version_id / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.pt").write_bytes(b"fake")

        result = _auto_resolve_model_pt(
            artifact_dir=tmp_path,
            version_id=version_id,
            is_raw_isp_pipeline=True,
        )
        assert result is None

    def test_returns_none_when_version_id_is_none(self, tmp_path):
        result = _auto_resolve_model_pt(
            artifact_dir=tmp_path,
            version_id=None,
            is_raw_isp_pipeline=False,
        )
        assert result is None

    def test_logs_warning_when_model_missing(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="src.monitoring.image_monitor_cli"):
            _auto_resolve_model_pt(
                artifact_dir=tmp_path,
                version_id="abc123",
                is_raw_isp_pipeline=False,
            )
        assert any("not found at auto-resolved path" in r.message for r in caplog.records)

    def test_logs_info_when_model_found(self, tmp_path, caplog):
        version_id = "abc123"
        model_dir = tmp_path / version_id / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.pt").write_bytes(b"fake")

        with caplog.at_level(logging.INFO, logger="src.monitoring.image_monitor_cli"):
            _auto_resolve_model_pt(
                artifact_dir=tmp_path,
                version_id=version_id,
                is_raw_isp_pipeline=False,
            )
        assert any("Auto-resolved model" in r.message for r in caplog.records)


# =============================================================================
# Helpers for main() integration tests
# =============================================================================

def _make_images(n=20, h=8, w=8, c=3):
    return np.random.default_rng(0).random((n, h, w, c)).astype(np.float32)


def _write_batch_npz(path: Path, version_id: str | None = None):
    X = _make_images()
    path.parent.mkdir(parents=True, exist_ok=True)
    if version_id:
        np.savez(path, X=X, version_id=np.array(version_id))
    else:
        np.savez(path, X=X)


def _make_drift_result(severity="low"):
    return {
        "schema_version": "1.0.0",
        "drift_type": "image_statistical",
        "method": "statistical",
        "generated_at": "2024-01-01T00:00:00+00:00",
        "batch": {"size": 20},
        "reference": {"size": 100},
        "overall": {
            "drift_score": 0.05,
            "severity": severity,
            "dataset_drift_detected": severity != "low",
        },
        "scenario_interpretation": None,
    }


def _build_mock_config(tmp_path: Path, is_raw: bool = False):
    config = MagicMock()
    config.dataset = "testds"
    config.task_type = "image_classification_cnn"
    config.data.processed = str(tmp_path / "processed")
    config.data.drift_scenarios = str(tmp_path / "drift_scenarios")
    config.output_dir = str(tmp_path / "outputs")
    config.configs.drift = str(tmp_path / "drift.yaml")
    config.configs.preprocessing = str(tmp_path / "prep.yaml")
    # Prevent resolve_model_name from returning a MagicMock object, which would
    # be stringified as "MagicMock" and create real directories on disk.
    config.mlflow.registry_model_name = "test-model"
    return config


def _build_mock_prep_config(is_raw: bool = False):
    prep_config = MagicMock()
    if is_raw:
        prep_config.image.raw_input = True
        prep_config.image.isp = MagicMock()
    else:
        prep_config.image.raw_input = False
        prep_config.image.isp = None
    return prep_config


def _build_mock_drift_config():
    dc = MagicMock()
    dc.monitoring.min_batch_size = 5
    dc.monitoring.alert_severity = "high"
    return dc


# =============================================================================
# main() — version_id mismatch warning
# =============================================================================

class TestVersionIdMismatch:
    @patch("src.monitoring.image_monitor_cli.log_runtime_drift_to_mlflow")
    @patch("src.monitoring.image_monitor_cli.append_history_entry")
    @patch("src.monitoring.image_monitor_cli.atomic_write_json")
    @patch("src.monitoring.image_monitor_cli.monitor_image_batch")
    @patch("src.monitoring.image_drift_monitor.load_reference_images")
    @patch("src.monitoring.image_monitor_cli.resolve_latest_version")
    @patch("src.config.loader.load_preprocessing_config")
    @patch("src.monitoring.image_monitor_cli.load_drift_config")
    @patch("src.monitoring.image_monitor_cli.load_config")
    def test_version_mismatch_logs_warning(
        self,
        mock_load_config, mock_load_drift, mock_load_prep,
        mock_resolve, mock_load_ref, mock_monitor, mock_write,
        mock_history, mock_log_drift,
        tmp_path, caplog,
    ):
        batch_path = tmp_path / "batch.npz"
        _write_batch_npz(batch_path, version_id="v_old")

        mock_load_config.return_value = _build_mock_config(tmp_path)
        mock_load_drift.return_value = _build_mock_drift_config()
        mock_load_prep.return_value = _build_mock_prep_config(is_raw=False)
        mock_resolve.return_value = "v_new"
        mock_load_ref.return_value = _make_images(100)
        mock_monitor.return_value = _make_drift_result()

        import sys
        from src.monitoring.image_monitor_cli import main

        with caplog.at_level(logging.WARNING, logger="src.monitoring.image_monitor_cli"):
            with pytest.raises(SystemExit):
                sys.argv = [
                    "monitor-drift-image",
                    "--batch-npz", str(batch_path),
                    "--config", "dummy.yaml",
                    "--output-dir", str(tmp_path / "monitoring"),
                ]
                main()

        assert any("Version mismatch" in r.message for r in caplog.records)

    @patch("src.monitoring.image_monitor_cli.log_runtime_drift_to_mlflow")
    @patch("src.monitoring.image_monitor_cli.append_history_entry")
    @patch("src.monitoring.image_monitor_cli.atomic_write_json")
    @patch("src.monitoring.image_monitor_cli.monitor_image_batch")
    @patch("src.monitoring.image_drift_monitor.load_reference_images")
    @patch("src.monitoring.image_monitor_cli.resolve_latest_version")
    @patch("src.config.loader.load_preprocessing_config")
    @patch("src.monitoring.image_monitor_cli.load_drift_config")
    @patch("src.monitoring.image_monitor_cli.load_config")
    def test_matching_version_no_warning(
        self,
        mock_load_config, mock_load_drift, mock_load_prep,
        mock_resolve, mock_load_ref, mock_monitor, mock_write,
        mock_history, mock_log_drift,
        tmp_path, caplog,
    ):
        batch_path = tmp_path / "batch.npz"
        _write_batch_npz(batch_path, version_id="v_same")

        mock_load_config.return_value = _build_mock_config(tmp_path)
        mock_load_drift.return_value = _build_mock_drift_config()
        mock_load_prep.return_value = _build_mock_prep_config(is_raw=False)
        mock_resolve.return_value = "v_same"
        mock_load_ref.return_value = _make_images(100)
        mock_monitor.return_value = _make_drift_result()

        import sys
        from src.monitoring.image_monitor_cli import main

        with caplog.at_level(logging.WARNING, logger="src.monitoring.image_monitor_cli"):
            with pytest.raises(SystemExit):
                sys.argv = [
                    "monitor-drift-image",
                    "--batch-npz", str(batch_path),
                    "--config", "dummy.yaml",
                    "--output-dir", str(tmp_path / "monitoring"),
                ]
                main()

        assert not any("Version mismatch" in r.message for r in caplog.records)


# =============================================================================
# main() — method auto-selection via --batch-npz
# =============================================================================

class TestMethodAutoSelection:
    def _run_main(
        self,
        tmp_path,
        is_raw,
        model_exists,
        mock_load_config,
        mock_load_drift,
        mock_load_prep,
        mock_resolve,
        mock_load_ref,
        mock_monitor,
        mock_write,
    ):
        batch_path = tmp_path / "batch.npz"
        _write_batch_npz(batch_path, version_id="v1")

        mock_load_config.return_value = _build_mock_config(tmp_path)
        mock_load_drift.return_value = _build_mock_drift_config()
        mock_load_prep.return_value = _build_mock_prep_config(is_raw=is_raw)
        mock_resolve.return_value = "v1"
        mock_load_ref.return_value = _make_images(100)
        mock_monitor.return_value = _make_drift_result()

        if model_exists:
            model_dir = tmp_path / "v1" / "model"
            model_dir.mkdir(parents=True)
            (model_dir / "model.pt").write_bytes(b"fake")

        import sys
        from src.monitoring.image_monitor_cli import main

        with patch("src.monitoring.image_monitor_cli.append_history_entry"), \
             patch("src.monitoring.image_monitor_cli.log_runtime_drift_to_mlflow"), \
             patch("torch.load", return_value=MagicMock()):
            with pytest.raises(SystemExit):
                sys.argv = [
                    "monitor-drift-image",
                    "--batch-npz", str(batch_path),
                    "--config", "dummy.yaml",
                    "--artifact-dir", str(tmp_path),
                    "--output-dir", str(tmp_path / "monitoring"),
                ]
                main()

        return mock_monitor.call_args

    @patch("src.monitoring.image_monitor_cli.atomic_write_json")
    @patch("src.monitoring.image_monitor_cli.monitor_image_batch")
    @patch("src.monitoring.image_drift_monitor.load_reference_images")
    @patch("src.monitoring.image_monitor_cli.resolve_latest_version")
    @patch("src.config.loader.load_preprocessing_config")
    @patch("src.monitoring.image_monitor_cli.load_drift_config")
    @patch("src.monitoring.image_monitor_cli.load_config")
    def test_raw_pipeline_uses_statistical(
        self, mock_load_config, mock_load_drift, mock_load_prep,
        mock_resolve, mock_load_ref, mock_monitor, mock_write, tmp_path,
    ):
        call_args = self._run_main(
            tmp_path, is_raw=True, model_exists=False,
            mock_load_config=mock_load_config, mock_load_drift=mock_load_drift,
            mock_load_prep=mock_load_prep, mock_resolve=mock_resolve,
            mock_load_ref=mock_load_ref, mock_monitor=mock_monitor, mock_write=mock_write,
        )
        assert call_args.kwargs["method"] == "statistical"

    @patch("src.monitoring.image_monitor_cli.atomic_write_json")
    @patch("src.monitoring.image_monitor_cli.monitor_image_batch")
    @patch("src.monitoring.image_drift_monitor.load_reference_images")
    @patch("src.monitoring.image_monitor_cli.resolve_latest_version")
    @patch("src.config.loader.load_preprocessing_config")
    @patch("src.monitoring.image_monitor_cli.load_drift_config")
    @patch("src.monitoring.image_monitor_cli.load_config")
    def test_standard_pipeline_with_model_uses_embedding(
        self, mock_load_config, mock_load_drift, mock_load_prep,
        mock_resolve, mock_load_ref, mock_monitor, mock_write, tmp_path,
    ):
        call_args = self._run_main(
            tmp_path, is_raw=False, model_exists=True,
            mock_load_config=mock_load_config, mock_load_drift=mock_load_drift,
            mock_load_prep=mock_load_prep, mock_resolve=mock_resolve,
            mock_load_ref=mock_load_ref, mock_monitor=mock_monitor, mock_write=mock_write,
        )
        assert call_args.kwargs["method"] == "embedding"

    @patch("src.monitoring.image_monitor_cli.atomic_write_json")
    @patch("src.monitoring.image_monitor_cli.monitor_image_batch")
    @patch("src.monitoring.image_drift_monitor.load_reference_images")
    @patch("src.monitoring.image_monitor_cli.resolve_latest_version")
    @patch("src.config.loader.load_preprocessing_config")
    @patch("src.monitoring.image_monitor_cli.load_drift_config")
    @patch("src.monitoring.image_monitor_cli.load_config")
    def test_standard_pipeline_no_model_uses_statistical(
        self, mock_load_config, mock_load_drift, mock_load_prep,
        mock_resolve, mock_load_ref, mock_monitor, mock_write, tmp_path,
    ):
        call_args = self._run_main(
            tmp_path, is_raw=False, model_exists=False,
            mock_load_config=mock_load_config, mock_load_drift=mock_load_drift,
            mock_load_prep=mock_load_prep, mock_resolve=mock_resolve,
            mock_load_ref=mock_load_ref, mock_monitor=mock_monitor, mock_write=mock_write,
        )
        assert call_args.kwargs["method"] == "statistical"


# =============================================================================
# main() — --input-dir one-step workflow
# =============================================================================

class TestInputDirWorkflow:
    @patch("src.monitoring.image_monitor_cli.log_runtime_drift_to_mlflow")
    @patch("src.monitoring.image_monitor_cli.append_history_entry")
    @patch("src.monitoring.image_monitor_cli.atomic_write_json")
    @patch("src.monitoring.image_monitor_cli.atomic_write_npz")
    @patch("src.monitoring.image_monitor_cli.monitor_image_batch")
    @patch("src.monitoring.image_drift_monitor.load_reference_images")
    @patch("src.monitoring.image_monitor_cli.resolve_latest_version")
    @patch("src.config.loader.load_preprocessing_config")
    @patch("src.monitoring.image_monitor_cli.load_drift_config")
    @patch("src.monitoring.image_monitor_cli.load_config")
    def test_input_dir_calls_prepare_image_batch(
        self,
        mock_load_config, mock_load_drift, mock_load_prep,
        mock_resolve, mock_load_ref, mock_monitor,
        mock_write_npz, mock_write_json,
        mock_history, mock_log_drift,
        tmp_path, monkeypatch,
    ):
        """--input-dir mode should preprocess images and run monitoring."""
        input_dir = tmp_path / "incoming"
        input_dir.mkdir()
        # Create a dummy image file so the directory is non-empty
        (input_dir / "img.jpg").write_bytes(b"fake-image")

        mock_load_config.return_value = _build_mock_config(tmp_path)
        mock_load_drift.return_value = _build_mock_drift_config()
        mock_load_prep.return_value = _build_mock_prep_config(is_raw=False)
        mock_resolve.return_value = "v1"
        mock_load_ref.return_value = _make_images(100)
        mock_monitor.return_value = _make_drift_result()

        X_fake = _make_images(10)

        # Patch prepare_image_batch and load_training_stats inside the CLI module
        with patch("src.data.prepare_batch.prepare_image_batch", return_value=X_fake) as mock_prep, \
             patch("src.data.prepare_batch.load_training_stats", return_value=(None, None, None)):

            # atomic_write_npz is patched so no real file is written;
            # we need batch loading to still work — patch np.load for the batch
            npz_mock = MagicMock()
            npz_mock.__contains__ = lambda self, key: key == "X"
            npz_mock.__getitem__ = lambda self, key: X_fake if key == "X" else None

            import sys
            from src.monitoring.image_monitor_cli import main

            with patch("src.monitoring.image_monitor_cli.np.load", return_value=npz_mock):
                with pytest.raises(SystemExit):
                    sys.argv = [
                        "monitor-drift-image",
                        "--input-dir", str(input_dir),
                        "--config", "dummy.yaml",
                        "--output-dir", str(tmp_path / "monitoring"),
                    ]
                    main()

            mock_prep.assert_called_once()
            mock_write_npz.assert_called_once()

    def test_input_dir_and_batch_npz_are_mutually_exclusive(self, tmp_path):
        """Providing both --input-dir and --batch-npz should fail at argparse."""
        import sys
        from src.monitoring.image_monitor_cli import _parse_args

        with pytest.raises(SystemExit):
            sys.argv = [
                "monitor-drift-image",
                "--input-dir", str(tmp_path),
                "--batch-npz", "some.npz",
                "--config", "dummy.yaml",
            ]
            _parse_args()

    def test_neither_input_dir_nor_batch_npz_fails(self):
        """Providing neither --input-dir nor --batch-npz should fail at argparse."""
        import sys
        from src.monitoring.image_monitor_cli import _parse_args

        with pytest.raises(SystemExit):
            sys.argv = [
                "monitor-drift-image",
                "--config", "dummy.yaml",
            ]
            _parse_args()

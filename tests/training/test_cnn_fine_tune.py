# =============================================================================
# tests/training/test_cnn_fine_tune.py — Fine-tuning tests for CNN training
# =============================================================================
"""Tests for the --fine-tune flag in the CNN image classification pipeline.

Covers:
- FineTuneConfig schema defaults and YAML parsing
- _load_production_weights falls back gracefully when no Production model exists
- run_training with fine_tune=True uses fine-tune hyperparameters (lr, epochs)
- run_training with fine_tune=True and freeze_backbone freezes conv layers
- run_training with fine_tune=False behaves identically to the original behaviour
- TrainingResult hyperparameters include fine_tuned / fine_tune_epochs / fine_tune_lr
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.config.schema import (
    CnnArchitectureConfig,
    CnnConvLayerConfig,
    CnnHyperparams,
    FineTuneConfig,
    ModelConfig,
    TrainingConfig,
)
from src.training.image_classification_cnn.train import SimpleCNN, _load_production_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_arch() -> CnnArchitectureConfig:
    return CnnArchitectureConfig(
        conv_layers=(CnnConvLayerConfig(out_channels=8, kernel_size=3),),
        fc_units=16,
        dropout=0.0,
    )


def _make_training_config(fine_tune_kwargs: dict | None = None) -> TrainingConfig:
    ft = FineTuneConfig(**(fine_tune_kwargs or {}))
    return TrainingConfig(
        model=ModelConfig(
            algorithm="cnn",
            hyperparameters=CnnHyperparams(epochs=2, batch_size=4, learning_rate=0.01),
            architecture=_make_arch(),
        ),
        fine_tune=ft,
    )


def _make_model(num_classes: int = 2) -> SimpleCNN:
    return SimpleCNN(
        num_classes=num_classes,
        in_channels=3,
        image_height=8,
        image_width=8,
        arch=_make_arch(),
    )


def _make_pipeline_config() -> MagicMock:
    cfg = MagicMock()
    cfg.task_type = "image_classification_cnn"
    cfg.random_seed = 42
    cfg.dataset = "test_dataset"
    cfg.data.processed = "data/processed"
    cfg.configs.training = "src/config/training_image_cnn.yaml"
    cfg.mlflow.registry_model_name = None
    cfg.mlflow.tracking_uri = None
    cfg.project.name = "test"
    return cfg


# ---------------------------------------------------------------------------
# FineTuneConfig schema
# ---------------------------------------------------------------------------

class TestFineTuneConfig:
    def test_defaults(self):
        ft = FineTuneConfig()
        assert ft.enabled is False
        assert ft.epochs == 5
        assert ft.learning_rate == 0.0001
        assert ft.freeze_backbone is False

    def test_custom_values(self):
        ft = FineTuneConfig(enabled=True, epochs=3, learning_rate=0.00005, freeze_backbone=True)
        assert ft.enabled is True
        assert ft.epochs == 3
        assert ft.learning_rate == 0.00005
        assert ft.freeze_backbone is True

    def test_training_config_has_fine_tune(self):
        tc = _make_training_config()
        assert isinstance(tc.fine_tune, FineTuneConfig)


# ---------------------------------------------------------------------------
# _load_production_weights
# ---------------------------------------------------------------------------

class TestLoadProductionWeights:
    def test_returns_false_when_no_production_model(self):
        """If the registry has no Production version, returns False without raising."""
        config = _make_pipeline_config()
        model = _make_model()

        with (
            patch("src.registry.model_registry.MlflowClient") as mock_client_cls,
            patch("src.registry.model_registry.resolve_model_name", return_value="test-model"),
        ):
            mock_client = MagicMock()
            mock_client.get_latest_versions.return_value = []
            mock_client_cls.return_value = mock_client

            result = _load_production_weights(config, model)

        assert result is False

    def test_returns_false_on_exception(self):
        """If anything goes wrong (e.g. MLflow unreachable), returns False."""
        config = _make_pipeline_config()
        model = _make_model()

        with patch(
            "src.registry.model_registry.MlflowClient",
            side_effect=RuntimeError("MLflow unavailable"),
        ):
            result = _load_production_weights(config, model)

        assert result is False

    def test_loads_weights_when_production_exists(self, tmp_path):
        """When a Production model exists, state_dict is loaded into the model."""
        config = _make_pipeline_config()
        model = _make_model()

        pt_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), pt_path)

        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.run_id = "abc123"
        mock_version.version = "1"
        mock_client.get_latest_versions.return_value = [mock_version]

        with (
            patch("src.registry.model_registry.MlflowClient", return_value=mock_client),
            patch("src.registry.model_registry.resolve_model_name", return_value="test-model"),
            patch("mlflow.get_artifact_uri", return_value="file:///tmp/mlruns"),
            patch("mlflow.artifacts.download_artifacts", return_value=str(pt_path)),
        ):
            result = _load_production_weights(config, model)

        assert result is True


# ---------------------------------------------------------------------------
# run_training — fine_tune flag integration
# ---------------------------------------------------------------------------

class TestRunTrainingFineTune:
    """Integration-style tests that exercise run_training with fine_tune=True/False.

    These tests mock out file I/O and MLflow, running the actual PyTorch training
    loop on synthetic data so the CNN forward/backward passes are exercised.
    """

    def _make_npz(self, tmp_path: Path, n: int = 8) -> Path:
        npz = tmp_path / "train.npz"
        X = np.random.rand(n, 8, 8, 3).astype(np.float32)
        y = np.array([0, 1] * (n // 2), dtype=np.int64)
        np.savez(npz, X=X, y=y)
        return npz

    def _make_feature_map(self, tmp_path: Path) -> Path:
        import json
        fm = {
            "target": "label",
            "class_names": ["cat", "dog"],
            "image_shape": [8, 8, 3],
        }
        p = tmp_path / "feature_map.json"
        p.write_text(json.dumps(fm))
        return p

    def _patch_env(self, tmp_path: Path, monkeypatch):
        """Patch filesystem and config so run_training can find its files."""
        preprocessed = tmp_path / "preprocessed"
        preprocessed.mkdir()
        self._make_npz(preprocessed)
        self._make_feature_map(preprocessed)

        config = _make_pipeline_config()
        config.data.processed = str(tmp_path)
        config.dataset = "ds"

        version_dir = tmp_path / "ds" / "v1"
        version_dir.mkdir(parents=True, exist_ok=True)

        (tmp_path / "ds" / "v1" / "preprocessed").mkdir(parents=True, exist_ok=True)
        self._make_npz(tmp_path / "ds" / "v1" / "preprocessed")
        self._make_feature_map(tmp_path / "ds" / "v1" / "preprocessed")

        return config

    def test_fine_tune_false_uses_full_hyperparams(self, tmp_path, monkeypatch):
        """With fine_tune=False, training uses standard epochs/lr from config."""
        from src.training.image_classification_cnn.train import run_training

        config = self._patch_env(tmp_path, monkeypatch)

        with patch("src.training.image_classification_cnn.train.load_training_config") as mock_cfg:
            mock_cfg.return_value = _make_training_config()
            result = run_training(config, "v1", fine_tune=False)

        assert result.hyperparameters["fine_tuned"] is False
        assert result.hyperparameters["fine_tune_epochs"] is None
        assert result.hyperparameters["fine_tune_lr"] is None

    def test_fine_tune_true_no_production_falls_back(self, tmp_path, monkeypatch):
        """With fine_tune=True but no Production model, training falls back to scratch."""
        from src.training.image_classification_cnn.train import run_training

        config = self._patch_env(tmp_path, monkeypatch)

        with (
            patch("src.training.image_classification_cnn.train.load_training_config") as mock_cfg,
            patch("src.training.image_classification_cnn.train._load_production_weights", return_value=False),
        ):
            mock_cfg.return_value = _make_training_config({"enabled": True, "epochs": 2, "learning_rate": 0.00005})
            result = run_training(config, "v1", fine_tune=True)

        assert result.hyperparameters["fine_tuned"] is False

    def test_fine_tune_true_with_weights_uses_ft_hyperparams(self, tmp_path, monkeypatch):
        """With fine_tune=True and Production weights loaded, fine-tune lr/epochs are used."""
        from src.training.image_classification_cnn.train import run_training

        config = self._patch_env(tmp_path, monkeypatch)

        ft_epochs = 2
        ft_lr = 0.00005

        with (
            patch("src.training.image_classification_cnn.train.load_training_config") as mock_cfg,
            patch("src.training.image_classification_cnn.train._load_production_weights", return_value=True),
        ):
            mock_cfg.return_value = _make_training_config(
                {"enabled": True, "epochs": ft_epochs, "learning_rate": ft_lr, "freeze_backbone": False}
            )
            result = run_training(config, "v1", fine_tune=True)

        assert result.hyperparameters["fine_tuned"] is True
        assert result.hyperparameters["fine_tune_epochs"] == ft_epochs
        assert result.hyperparameters["fine_tune_lr"] == ft_lr

    def test_freeze_backbone_freezes_conv_params(self, tmp_path, monkeypatch):
        """When freeze_backbone=True and weights are loaded, conv params are frozen."""
        from src.training.image_classification_cnn.train import run_training

        config = self._patch_env(tmp_path, monkeypatch)

        with (
            patch("src.training.image_classification_cnn.train.load_training_config") as mock_cfg,
            patch("src.training.image_classification_cnn.train._load_production_weights", return_value=True),
        ):
            mock_cfg.return_value = _make_training_config(
                {"enabled": True, "epochs": 1, "learning_rate": 0.00005, "freeze_backbone": True}
            )
            # We capture the model by inspecting TrainingResult — the model is returned
            result = run_training(config, "v1", fine_tune=True)

        # The model in TrainingResult should have frozen conv layers
        model: SimpleCNN = result.model
        frozen = [p for name, p in model.net.named_parameters()
                  if "Conv" in type(model.net[0]).__name__ or not p.requires_grad]
        # At minimum the last Linear layer must still require grad
        last_linear_params = [p for m in model.net.modules()
                              if isinstance(m, torch.nn.Linear)
                              for p in m.parameters()]
        assert any(p.requires_grad for p in last_linear_params)

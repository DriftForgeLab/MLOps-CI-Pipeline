# =============================================================================
# src/training/image_classification_cnn/train.py — CNN image classification
# =============================================================================
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.config.loader import PipelineConfig
from src.data.preprocess import PREPROCESSED_SUBDIR
from src.training import TrainingResult

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """Simple configurable CNN for image classification."""

    def __init__(self, num_classes: int, in_channels: int, image_size: int, arch_config: dict):
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = in_channels
        current_size = image_size

        for conv in arch_config["conv_layers"]:
            out_ch = conv["out_channels"]
            k = conv["kernel_size"]
            layers += [
                nn.Conv2d(current_channels, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            current_channels = out_ch
            current_size = current_size // 2

        flat_size = current_channels * current_size * current_size
        layers += [
            nn.Flatten(),
            nn.Linear(flat_size, arch_config["fc_units"]),
            nn.ReLU(),
            nn.Dropout(arch_config["dropout"]),
            nn.Linear(arch_config["fc_units"], num_classes),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """sklearn-compatible predict interface."""
        self.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32)
            output = self.net(tensor)
            return output.argmax(dim=1).numpy()


def run_training(config: PipelineConfig, version_id: str) -> TrainingResult:
    """
    Train a CNN on preprocessed image data (NPZ with 4D arrays).

    Expects preprocessing with flatten=false, producing arrays of shape
    (N, H, W, C) which are transposed to (N, C, H, W) for PyTorch.

    Args:
        config:     Validated PipelineConfig
        version_id: Dataset version ID

    Returns:
        TrainingResult with fitted CNN model and training metadata.
    """
    import yaml as _yaml
    training_cfg_path = Path(config.configs.training)
    with open(training_cfg_path) as f:
        training_cfg = _yaml.safe_load(f)

    arch = training_cfg["model"]["architecture"]
    hp = training_cfg["model"]["hyperparameters"]
    epochs = hp["epochs"]
    batch_size = hp["batch_size"]
    lr = hp["learning_rate"]

    version_dir = Path(config.data.processed) / config.dataset / version_id
    preprocessed_dir = version_dir / PREPROCESSED_SUBDIR

    # Load feature map
    feature_map_path = preprocessed_dir / "feature_map.json"
    if not feature_map_path.exists():
        raise FileNotFoundError(f"feature_map.json not found at '{feature_map_path}'.")
    with open(feature_map_path) as f:
        feature_map = json.load(f)

    num_classes = len(feature_map["class_names"])
    image_shape = feature_map["image_shape"]  # [H, W, C]
    image_size = image_shape[0]
    in_channels = image_shape[2] if len(image_shape) == 3 else 1

    # Load NPZ data
    train_npz_path = preprocessed_dir / "train.npz"
    if not train_npz_path.exists():
        raise FileNotFoundError(f"Preprocessed training data not found: {train_npz_path}")

    data = np.load(train_npz_path)
    X_np = data["X"]  # (N, H, W, C)
    y_np = data["y"]  # (N,)

    # Transpose to (N, C, H, W) for PyTorch
    if X_np.ndim == 4:
        X_np = X_np.transpose(0, 3, 1, 2)
    elif X_np.ndim == 3:
        X_np = X_np[:, np.newaxis, :, :]

    torch.manual_seed(config.random_seed)

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleCNN(num_classes, in_channels, image_size, arch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        logger.info("  Epoch %d/%d — loss: %.4f", epoch + 1, epochs, avg_loss)

    logger.info(
        "  CNN training complete: %d classes, %d samples",
        num_classes, len(X_tensor),
    )

    return TrainingResult(
        model=model,
        algorithm="cnn",
        hyperparameters={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "conv_layers": arch["conv_layers"],
            "fc_units": arch["fc_units"],
            "dropout": arch["dropout"],
        },
        dataset_version_id=version_id,
        random_seed=config.random_seed,
        trained_at=datetime.now(timezone.utc).isoformat(),
        train_rows=len(X_tensor),
    )
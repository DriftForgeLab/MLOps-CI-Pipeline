# =============================================================================
# src/training/image_classification_cnn/train.py — CNN image classification
# =============================================================================
# Responsibility: Train a PyTorch CNN on preprocessed image data (NPZ with 4D
# arrays) and return the fitted model with training metadata.
#
# Uses validated CnnArchitectureConfig and CnnHyperparams from the config
# system — no raw YAML parsing. Follows the same TrainingResult contract as
# all other training modules.
# =============================================================================
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.config.loader import PipelineConfig, load_training_config
from src.config.schema import CnnArchitectureConfig, CnnHyperparams
from src.data.preprocess import PREPROCESSED_SUBDIR
from src.training import TrainingResult

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """Simple configurable CNN for image classification.

    Supports non-square images by tracking height and width independently
    through pooling layers.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        image_height: int,
        image_width: int,
        arch: CnnArchitectureConfig,
    ):
        super().__init__()
        self.in_channels = in_channels
        layers: list[nn.Module] = []
        current_channels = in_channels
        current_h = image_height
        current_w = image_width

        for conv in arch.conv_layers:
            layers += [
                nn.Conv2d(current_channels, conv.out_channels, kernel_size=conv.kernel_size, padding=conv.kernel_size // 2),
                nn.BatchNorm2d(conv.out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            current_channels = conv.out_channels
            current_h = current_h // 2
            current_w = current_w // 2

        flat_size = current_channels * current_h * current_w
        layers += [
            nn.Flatten(),
            nn.Linear(flat_size, arch.fc_units),
            nn.ReLU(),
            nn.Dropout(arch.dropout),
            nn.Linear(arch.fc_units, num_classes),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """sklearn-compatible predict interface. Accepts NHWC or NCHW input."""
        self.eval()
        with torch.no_grad():
            if X.ndim == 4 and X.shape[1] != self.in_channels:
                X = X.transpose(0, 3, 1, 2)
            tensor = torch.tensor(X, dtype=torch.float32)
            output = self.net(tensor)
            return output.argmax(dim=1).numpy()


def _load_production_weights(config: PipelineConfig, model: "SimpleCNN") -> bool:
    """Load Production model weights into model in-place.

    Fetches the latest Production version from the MLflow Model Registry,
    downloads model.pt, and loads the state_dict. Returns True on success,
    False if no Production model exists or loading fails.

    Args:
        config: PipelineConfig (used to resolve registry model name and tracking URI).
        model:  SimpleCNN instance to load weights into. Must match the
                architecture of the registered model.
    """
    try:
        from src.registry.model_registry import get_mlflow_client, resolve_model_name
        import mlflow
        client = get_mlflow_client(config)
        model_name = resolve_model_name(config)
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            logger.warning(
                "  Fine-tune: no Production model found for '%s' — training from scratch.", model_name
            )
            return False
        prod_version = versions[0]
        run_id = prod_version.run_id
        artifact_uri = mlflow.get_artifact_uri()  # base; we use download below
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model/model.pt",
        )
        # weights_only=False because model.pt is saved as a full model object
        # (torch.save(model)) by metadata.py — not as a state_dict.
        # The artifact comes from our own MLflow Registry, so the source is trusted.
        loaded = torch.load(local_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict):
            model.load_state_dict(loaded)
        else:
            model.load_state_dict(loaded.state_dict())
        logger.info(
            "  Fine-tune: loaded Production weights from '%s' version %s (run_id=%s)",
            model_name, prod_version.version, run_id,
        )
        return True
    except Exception as e:
        logger.warning(
            "  Fine-tune: failed to load Production weights (%s) — training from scratch.", e
        )
        return False


def run_training(config: PipelineConfig, version_id: str, fine_tune: bool = False) -> TrainingResult:
    """Train a CNN on preprocessed image data (NPZ with 4D arrays).

    Expects preprocessing with flatten=false, producing arrays of shape
    (N, H, W, C) which are transposed to (N, C, H, W) for PyTorch.

    When ``fine_tune=True``, the existing Production model weights are loaded
    before training begins. Training then continues with the fine-tune
    hyperparameters (fewer epochs, lower learning rate) defined in
    ``training_image_cnn.yaml`` under the ``fine_tune:`` block. If no
    Production model exists, a warning is logged and training proceeds from
    scratch with the standard hyperparameters.

    Args:
        config:     Validated PipelineConfig
        version_id: Dataset version ID
        fine_tune:  If True, load Production weights and use fine-tune hyperparams.

    Returns:
        TrainingResult with fitted CNN model and training metadata.
    """
    training_config = load_training_config(Path(config.configs.training))
    arch: CnnArchitectureConfig = training_config.model.architecture
    hp: CnnHyperparams = training_config.model.hyperparameters

    version_dir = Path(config.data.processed) / config.dataset / version_id
    preprocessed_dir = version_dir / PREPROCESSED_SUBDIR

    # --- Load feature contract from feature_map.json ---
    feature_map_path = preprocessed_dir / "feature_map.json"
    if not feature_map_path.exists():
        raise FileNotFoundError(
            f"feature_map.json not found at '{feature_map_path}'. "
            "Run the preprocessing stage before training."
        )
    with open(feature_map_path) as f:
        feature_map = json.load(f)

    target: str = feature_map["target"]

    # --- Load dataset.yaml to confirm target (belt-and-suspenders) ---
    yaml_path = version_dir / "dataset.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        yaml_target = meta.get("target")
        if yaml_target and yaml_target != target:
            raise ValueError(
                f"Target mismatch: feature_map.json says '{target}' "
                f"but dataset.yaml says '{yaml_target}'. "
                "Re-run preprocessing to regenerate feature_map.json."
            )

    num_classes = len(feature_map["class_names"])
    image_shape = feature_map["image_shape"]  # [H, W, C]
    image_height = image_shape[0]
    image_width = image_shape[1]
    in_channels = image_shape[2] if len(image_shape) == 3 else 1

    # --- Load NPZ data ---
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    shuffle_gen = torch.Generator().manual_seed(config.random_seed)
    loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, generator=shuffle_gen)

    model = SimpleCNN(num_classes, in_channels, image_height, image_width, arch)

    # --- Fine-tuning: load Production weights if requested ---
    ft_cfg = training_config.fine_tune
    weights_loaded = False
    if fine_tune:
        weights_loaded = _load_production_weights(config, model)
        if weights_loaded and ft_cfg.freeze_backbone:
            # Freeze all layers except the final Linear classifier
            for name, param in model.net.named_parameters():
                param.requires_grad = False
            # Unfreeze only the last Linear layer
            last_linear = [m for m in model.net.modules() if isinstance(m, nn.Linear)][-1]
            for param in last_linear.parameters():
                param.requires_grad = True
            logger.info("  Fine-tune: backbone frozen — training classifier head only.")

    # Choose hyperparams: fine-tune overrides when weights loaded successfully
    if fine_tune and weights_loaded:
        epochs = ft_cfg.epochs
        learning_rate = ft_cfg.learning_rate
        logger.info(
            "  Fine-tune mode: epochs=%d  lr=%g  freeze_backbone=%s",
            epochs, learning_rate, ft_cfg.freeze_backbone,
        )
    else:
        epochs = hp.epochs
        learning_rate = hp.learning_rate

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
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

    hyperparameters = dict(vars(hp))
    hyperparameters["fine_tuned"] = fine_tune and weights_loaded
    hyperparameters["fine_tune_epochs"] = epochs if (fine_tune and weights_loaded) else None
    hyperparameters["fine_tune_lr"] = learning_rate if (fine_tune and weights_loaded) else None
    hyperparameters["conv_layers"] = [
        {"out_channels": c.out_channels, "kernel_size": c.kernel_size}
        for c in arch.conv_layers
    ]
    hyperparameters["fc_units"] = arch.fc_units
    hyperparameters["dropout"] = arch.dropout

    return TrainingResult(
        model=model,
        algorithm=training_config.model.algorithm,
        hyperparameters=hyperparameters,
        dataset_version_id=version_id,
        random_seed=config.random_seed,
        trained_at=datetime.now(timezone.utc).isoformat(),
        train_rows=len(X_tensor),
    )

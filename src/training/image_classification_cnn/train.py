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

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset, TensorDataset, WeightedRandomSampler

from src.common.device import resolve_device
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

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """sklearn-compatible predict interface. Accepts NHWC or NCHW input."""
        self.eval()
        with torch.no_grad():
            if X.ndim == 4 and X.shape[1] != self.in_channels:
                if isinstance(X, torch.Tensor):
                    X = X.permute(0, 3, 1, 2)
                else:
                    X = X.transpose(0, 3, 1, 2)
            device = next(self.parameters()).device
            tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
            output = self.net(tensor)
            return output.argmax(dim=1).cpu().numpy()


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
        # Note: artefacts are CPU-only by contract (metadata.py forces .cpu()
        # before save), so map_location="cpu" stays correct regardless of the
        # active training device. The model is moved to the active device
        # later in run_training().
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


def _load_image_split_npz(
    preprocessed_dir: Path, split_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load one preprocessed image split and convert it to NCHW for PyTorch.

    Returns ``(X_nchw, y, is_drifted)`` where ``is_drifted`` is the optional
    boolean mask written into ``train.npz`` by image preprocessing when any
    training images carried the ``_drifted`` filename suffix. Returns
    ``None`` for ``is_drifted`` if the array is absent (older NPZ files, or
    val/test splits which never carry the mask).
    """
    npz_path = preprocessed_dir / f"{split_name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Preprocessed {split_name} data not found: {npz_path}")

    data = np.load(npz_path)
    X_np = data["X"]
    y_np = data["y"]
    if X_np.ndim == 4:
        X_np = X_np.transpose(0, 3, 1, 2)
    elif X_np.ndim == 3:
        X_np = X_np[:, np.newaxis, :, :]
    is_drifted_np: np.ndarray | None = None
    if "is_drifted" in data.files:
        is_drifted_np = data["is_drifted"].astype(np.bool_)
    return X_np, y_np, is_drifted_np


def _build_drift_val_indices(
    is_drifted: np.ndarray,
    y: np.ndarray,
    drift_val_ratio: float,
    seed: int,
) -> np.ndarray:
    """Pick a stratified slice of drifted training samples for drift_val.

    Stratifies by class so each class contributes the same fraction.
    Returns an int64 array of indices into the training arrays.
    """
    rng = np.random.RandomState(seed)
    drifted_idx = np.flatnonzero(is_drifted)
    if drifted_idx.size == 0:
        return np.empty((0,), dtype=np.int64)
    chosen: list[int] = []
    for cls in np.unique(y[drifted_idx]):
        cls_idx = drifted_idx[y[drifted_idx] == cls]
        n_take = max(1, int(round(len(cls_idx) * drift_val_ratio)))
        n_take = min(n_take, len(cls_idx) - 1) if len(cls_idx) > 1 else 0
        if n_take <= 0:
            continue
        picks = rng.choice(cls_idx, size=n_take, replace=False)
        chosen.extend(int(i) for i in picks)
    return np.array(sorted(chosen), dtype=np.int64)


def _build_oversample_weights(
    is_drifted: np.ndarray,
    target_drift_ratio: float,
) -> np.ndarray:
    """Compute per-sample weights so a WeightedRandomSampler yields drifted
    samples at the requested expected fraction.

    With ``D`` drifted and ``C`` clean samples and target fraction ``r``:
        w_drift / w_clean = (r / (1 - r)) * (C / D)
    Setting ``w_clean = 1.0`` gives the formula below. Weights are not
    normalised — ``WeightedRandomSampler`` handles that.
    """
    n_drifted = int(is_drifted.sum())
    n_clean = int(len(is_drifted) - n_drifted)
    if n_drifted == 0 or n_clean == 0:
        return np.ones(len(is_drifted), dtype=np.float64)
    w_drift = (target_drift_ratio / (1.0 - target_drift_ratio)) * (n_clean / n_drifted)
    weights = np.where(is_drifted, w_drift, 1.0).astype(np.float64)
    return weights


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute classification accuracy on a validation loader."""
    was_training = model.training
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            correct += int((preds == y_batch).sum().item())
            total += int(y_batch.numel())

    if was_training:
        model.train()

    return (correct / total) if total > 0 else 0.0


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
    X_np, y_np, is_drifted_np = _load_image_split_npz(preprocessed_dir, "train")
    X_val_np, y_val_np, _ = _load_image_split_npz(preprocessed_dir, "val")

    device = resolve_device()
    logger.info("  CNN training device: %s", device)

    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
        # cudnn flags are only meaningful on CUDA. They are no-ops elsewhere
        # but setting them under non-CUDA backends has caused warnings in
        # past torch versions, so guard them too.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_np, dtype=torch.long)

    full_train_dataset = TensorDataset(X_tensor, y_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    shuffle_gen = torch.Generator().manual_seed(config.random_seed)

    # --- Fine-tuning: load Production weights if requested ---
    ft_cfg = training_config.fine_tune
    weights_loaded = False
    model = SimpleCNN(num_classes, in_channels, image_height, image_width, arch).to(device)
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

    # --- Drift-aware training arrangement (fine-tune only) ---
    # When fine-tuning with a drifted training set we can:
    #   1) carve out a stratified drift_val slice and use it for best-epoch
    #      selection (more honest than the clean val split), and
    #   2) oversample the remaining drifted samples so they make up a target
    #      fraction of each batch (combats the 4%-minority dilution problem).
    use_drift_val = (
        fine_tune
        and weights_loaded
        and ft_cfg.use_drift_val_for_best_epoch
        and is_drifted_np is not None
        and bool(is_drifted_np.any())
    )
    train_indices = np.arange(len(full_train_dataset), dtype=np.int64)
    drift_val_loader: DataLoader | None = None
    if use_drift_val:
        drift_val_idx = _build_drift_val_indices(
            is_drifted_np, y_np, ft_cfg.drift_val_ratio, config.random_seed
        )
        if drift_val_idx.size > 0:
            train_indices = np.setdiff1d(train_indices, drift_val_idx, assume_unique=False)
            drift_val_subset = Subset(full_train_dataset, drift_val_idx.tolist())
            drift_val_loader = DataLoader(
                drift_val_subset,
                batch_size=hp.batch_size,
                shuffle=False,
                pin_memory=(device.type == "cuda"),
                num_workers=0,
            )
            logger.info(
                "  Fine-tune: held out %d drifted sample(s) as drift_val for best-epoch selection.",
                int(drift_val_idx.size),
            )
        else:
            use_drift_val = False
            logger.warning(
                "  Fine-tune: use_drift_val_for_best_epoch=True but no drifted samples could be held out — "
                "falling back to standard val for best-epoch selection."
            )

    train_dataset = Subset(full_train_dataset, train_indices.tolist())

    sampler: WeightedRandomSampler | None = None
    use_oversample = (
        fine_tune
        and weights_loaded
        and ft_cfg.oversample_drift_ratio is not None
        and is_drifted_np is not None
        and bool(is_drifted_np.any())
    )
    if use_oversample:
        is_drifted_train = is_drifted_np[train_indices]
        if is_drifted_train.any() and (~is_drifted_train).any():
            weights = _build_oversample_weights(is_drifted_train, ft_cfg.oversample_drift_ratio)
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=True,
                generator=torch.Generator().manual_seed(config.random_seed),
            )
            logger.info(
                "  Fine-tune: oversampling drift to ~%.0f%% per batch "
                "(%d drifted / %d clean in training pool).",
                ft_cfg.oversample_drift_ratio * 100,
                int(is_drifted_train.sum()),
                int((~is_drifted_train).sum()),
            )
        else:
            logger.info(
                "  Fine-tune: oversample_drift_ratio set but training pool has only one class "
                "of drift status — uniform shuffling will be used."
            )

    # pin_memory only helps for CUDA host->device transfer. DirectML and CPU
    # do not benefit, and DirectML may warn. num_workers=0 because the dataset
    # already lives in memory (and on the active device) — workers would only
    # add IPC overhead.
    if sampler is not None:
        loader = DataLoader(
            train_dataset,
            batch_size=hp.batch_size,
            sampler=sampler,
            pin_memory=(device.type == "cuda"),
            num_workers=0,
        )
    else:
        loader = DataLoader(
            train_dataset,
            batch_size=hp.batch_size,
            shuffle=True,
            generator=shuffle_gen,
            pin_memory=(device.type == "cuda"),
            num_workers=0,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

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
    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    selection_split = "drift_val" if drift_val_loader is not None else "val"
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        val_accuracy = _evaluate_model(model, val_loader, device)
        drift_val_accuracy: float | None = None
        if drift_val_loader is not None:
            drift_val_accuracy = _evaluate_model(model, drift_val_loader, device)
        score = drift_val_accuracy if drift_val_accuracy is not None else val_accuracy
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        if drift_val_accuracy is not None:
            logger.info(
                "  Epoch %d/%d — loss: %.4f — val_accuracy: %.4f — drift_val_accuracy: %.4f",
                epoch + 1, epochs, avg_loss, val_accuracy, drift_val_accuracy,
            )
        else:
            logger.info(
                "  Epoch %d/%d — loss: %.4f — val_accuracy: %.4f",
                epoch + 1, epochs, avg_loss, val_accuracy,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info(
        "  CNN training complete: %d classes, %d samples — best %s_accuracy=%.4f at epoch %d",
        num_classes,
        len(train_dataset),
        selection_split,
        best_score,
        best_epoch,
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
    hyperparameters["best_val_accuracy"] = round(best_score, 4) if best_epoch else None
    hyperparameters["best_val_split"] = selection_split if best_epoch else None
    hyperparameters["best_epoch"] = best_epoch if best_epoch else None
    hyperparameters["oversample_drift_ratio"] = (
        ft_cfg.oversample_drift_ratio if (fine_tune and weights_loaded and sampler is not None) else None
    )
    hyperparameters["use_drift_val_for_best_epoch"] = (
        bool(use_drift_val) if (fine_tune and weights_loaded) else None
    )

    return TrainingResult(
        model=model,
        algorithm=training_config.model.algorithm,
        hyperparameters=hyperparameters,
        dataset_version_id=version_id,
        random_seed=config.random_seed,
        trained_at=datetime.now(timezone.utc).isoformat(),
        train_rows=len(train_dataset),
    )

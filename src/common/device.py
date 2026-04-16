"""Single torch device resolver for the project.

All code paths that move tensors or models to a device should call
``resolve_device()`` rather than constructing ``torch.device(...)`` directly,
so the choice can be controlled by the ``MODEL_DEVICE`` environment variable
and so the ``torch_directml`` import stays isolated to this module.

MODEL_DEVICE values:
    auto      (default) — try CUDA, then DirectML, then CPU.
    cpu                  — force CPU.
    cuda                 — force CUDA; fall back to CPU if unavailable.
    directml             — force DirectML; fall back to CPU if torch_directml
                           is not installed.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def resolve_device() -> torch.device:
    pref = os.getenv("MODEL_DEVICE", "auto").strip().lower()

    if pref == "cpu":
        return torch.device("cpu")

    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("MODEL_DEVICE=cuda but CUDA unavailable; falling back to CPU.")
        return torch.device("cpu")

    if pref == "directml":
        return _try_directml(required=True)

    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        dml = _try_directml(required=False)
        if dml.type != "cpu":
            return dml
        return torch.device("cpu")

    logger.warning("Unknown MODEL_DEVICE=%r; falling back to CPU.", pref)
    return torch.device("cpu")


def _try_directml(*, required: bool) -> torch.device:
    try:
        import torch_directml  # type: ignore
    except ImportError:
        if required:
            logger.warning(
                "MODEL_DEVICE=directml but torch_directml is not installed; "
                "falling back to CPU. Install with: pip install torch-directml"
            )
        return torch.device("cpu")
    dev = torch_directml.device()
    logger.info("Using DirectML device: %s", dev)
    return dev

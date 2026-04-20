"""Single torch device resolver for the project.

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

_cached_device: torch.device | None = None


def resolve_device() -> torch.device:
    """Resolve the target device, caching the result for subsequent calls.

    The device is determined once (based on MODEL_DEVICE env var and hardware
    availability) and reused for the lifetime of the process. This avoids
    repeated probe/log overhead when called per-image in ISP pipelines.
    """
    global _cached_device
    if _cached_device is not None:
        return _cached_device
    _cached_device = _resolve_device_uncached()
    return _cached_device


def _resolve_device_uncached() -> torch.device:
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

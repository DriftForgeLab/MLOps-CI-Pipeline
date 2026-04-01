# =============================================================================
# src/data/isp_pipeline.py — Image Signal Processing (ISP) pipeline
# =============================================================================
# Responsibility: Process raw Bayer images through a configurable 7-step ISP
# pipeline. Pure NumPy. No side effects. Used both by the CNN preprocessing
# pipeline (15.3) and by the drift scenario simulator (16.1).
#
# Steps (executed in order):
#   1. Black level correction  — subtract per-channel sensor offset
#   2. Demosaicing             — Bayer mosaic → RGB (bilinear/Malvar/Menon)
#   3. White balance           — per-channel gain (R/G/B)
#   4. Color correction        — 3×3 matrix transform
#   5. Denoising               — gaussian or median filter on luminance
#   6. Sharpening              — unsharp mask or convolution filter on luminance
#   7. Gamma correction        — power-law tone mapping
#
# Camera parameters (black_level, white_balance, color_matrix) are read from
# DNG metadata by default. Config values override DNG values per-parameter,
# enabling controlled drift scenarios in ID 16.
#
# Usage:
#   camera_params = read_camera_params(dng_path)
#   rgb = run_isp(raw_array, isp_config, camera_params)
# =============================================================================

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.config.schema import ISPConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Public API
# =============================================================================

def read_camera_params(dng_path: Path) -> dict:
    """Read camera-specific ISP parameters from a DNG file via rawpy.

    Extracts the three parameter sets needed by the ISP pipeline:
    - black_level:   [R, G1, G2, B] sensor black level offsets
    - white_balance: [r_gain, g_gain, b_gain] normalized so g_gain = 1.0
    - color_matrix:  9 floats (3×3, row-major) for colour correction

    Args:
        dng_path: Path to the DNG file.

    Returns:
        dict with keys "black_level", "white_balance", "color_matrix".

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError:       If rawpy is not installed.
        ValueError:        If black_level or white_balance metadata is missing
                           or incomplete. Set the affected parameters explicitly
                           in the isp: config block instead.
    """
    try:
        import rawpy
    except ImportError:
        raise ImportError(
            "rawpy is required to read DNG files. Install it with: pip install rawpy"
        )

    if not dng_path.exists():
        raise FileNotFoundError(f"DNG file not found: {dng_path}")

    with rawpy.imread(str(dng_path)) as raw:
        # --- Black level (4 values: R, G1, G2, B) ---
        bl = raw.black_level_per_channel
        if bl is None or len(bl) < 4:
            raise ValueError(
                f"'{dng_path.name}' has incomplete black_level_per_channel metadata "
                f"(got {bl!r}). Set 'image.isp.black_level_correction.black_level' "
                "explicitly in config."
            )
        black_level = [float(v) for v in bl[:4]]

        # --- White balance ([R, G, B], some cameras return [R, G, B, G]) ---
        wb = raw.camera_whitebalance
        if wb is None or len(wb) < 3:
            raise ValueError(
                f"'{dng_path.name}' has incomplete camera_whitebalance metadata "
                f"(got {wb!r}). Set 'image.isp.white_balance.[r/g/b]_gain' "
                "explicitly in config."
            )
        r_raw, g_raw, b_raw = float(wb[0]), float(wb[1]), float(wb[2])
        # Normalize so g_gain = 1.0; all gains relative to green channel
        if g_raw > 0:
            white_balance = [r_raw / g_raw, 1.0, b_raw / g_raw]
        else:
            logger.warning(
                "'%s' has g_gain = 0 in camera_whitebalance — using identity [1, 1, 1].",
                dng_path.name,
            )
            white_balance = [1.0, 1.0, 1.0]

        # --- Color correction matrix (3×3) ---
        cm = raw.color_matrix
        if cm is None or cm.shape[0] < 3 or cm.shape[1] < 3:
            logger.warning(
                "'%s' has no valid color_matrix — falling back to identity matrix.",
                dng_path.name,
            )
            color_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        else:
            color_matrix = [float(v) for v in cm[:3, :3].flatten()]

    return {
        "black_level": black_level,
        "white_balance": white_balance,
        "color_matrix": color_matrix,
    }


def run_isp(
    raw: np.ndarray,
    isp_config: ISPConfig,
    camera_params: dict | None = None,
) -> np.ndarray:
    """Process a raw Bayer image through the full 7-step ISP pipeline.

    Config values override DNG camera params per-parameter, so a partial
    override like setting only r_gain still reads b_gain from DNG metadata.

    Args:
        raw:           2D Bayer array (H, W) from rawpy.raw_image_visible,
                       dtype float32 or uint16.
        isp_config:    Validated ISPConfig from preprocessing config.
        camera_params: Dict from read_camera_params(). If None, all parameters
                       must be set explicitly in isp_config.

    Returns:
        RGB image array of shape (H, W, 3), dtype float64, values in [0.0, 1.0].

    Raises:
        ValueError: If a required parameter is absent from both config and
                    camera_params.
    """
    img = raw.astype(np.float64)
    params = camera_params or {}

    # --- Step 1: Black level correction ---
    if isp_config.black_level_correction.enabled:
        bl = (
            list(isp_config.black_level_correction.black_level)
            if isp_config.black_level_correction.black_level is not None
            else params.get("black_level")
        )
        if bl is None:
            raise ValueError(
                "Black level correction is enabled but no black_level found in config "
                "or DNG metadata. Set 'image.isp.black_level_correction.black_level' "
                "or ensure the DNG file has black_level_per_channel metadata."
            )
        img = _black_level_correction(img, bl)

    # Normalize Bayer mosaic to [0, 1] before demosaicing
    img_max = img.max()
    if img_max > 0:
        img = img / img_max

    # --- Step 2: Demosaicing (Bayer → RGB) ---
    img = _demosaicing(img, isp_config.demosaicing.algorithm)

    # --- Step 3: White balance ---
    wb_cfg = isp_config.white_balance
    dng_wb = params.get("white_balance", [1.0, 1.0, 1.0])
    r_gain = wb_cfg.r_gain if wb_cfg.r_gain is not None else dng_wb[0]
    g_gain = wb_cfg.g_gain if wb_cfg.g_gain is not None else dng_wb[1]
    b_gain = wb_cfg.b_gain if wb_cfg.b_gain is not None else dng_wb[2]
    img = _white_balance(img, [r_gain, g_gain, b_gain])

    # --- Step 4: Color correction ---
    if isp_config.color_correction.enabled:
        matrix = (
            list(isp_config.color_correction.matrix)
            if isp_config.color_correction.matrix is not None
            else params.get("color_matrix")
        )
        if matrix is None:
            logger.warning(
                "Color correction enabled but no matrix found in config or DNG — "
                "using identity matrix."
            )
            matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        img = _color_correction(img, matrix)

    # --- Step 5: Denoising ---
    if isp_config.denoising.algorithm != "none":
        img = _denoising(img, isp_config.denoising.algorithm, isp_config.denoising.strength)

    # --- Step 6: Sharpening ---
    if isp_config.sharpening.algorithm != "none":
        img = _sharpening(
            img,
            isp_config.sharpening.algorithm,
            isp_config.sharpening.radius,
            isp_config.sharpening.amount,
        )

    # --- Step 7: Gamma correction ---
    img = np.clip(img, 0.0, 1.0)
    img = _gamma_correction(img, isp_config.gamma_correction.gamma)

    return img


# =============================================================================
# Private step implementations
# =============================================================================

def _black_level_correction(raw: np.ndarray, black_level: list[float]) -> np.ndarray:
    """Subtract per-channel black level from the raw Bayer mosaic.

    Bayer pattern assumed: RGGB (standard for most cameras).
    Clips to zero to prevent negative values.
    """
    img = raw.copy()
    img[0::2, 0::2] -= black_level[0]   # R
    img[0::2, 1::2] -= black_level[1]   # G1
    img[1::2, 0::2] -= black_level[2]   # G2
    img[1::2, 1::2] -= black_level[3]   # B
    return np.clip(img, 0.0, None)


def _demosaicing(bayer: np.ndarray, algorithm: str) -> np.ndarray:
    """Demosaic Bayer pattern to RGB using the specified algorithm.

    Algorithms:
      bilinear  — fast, slight colour fringing at edges
      malvar2004 — Malvar 2004, better quality, moderate speed
      menon2007 — Menon 2007, highest quality, slowest
    """
    try:
        from colour_demosaicing import (
            demosaicing_CFA_Bayer_bilinear,
            demosaicing_CFA_Bayer_Malvar2004,
            demosaicing_CFA_Bayer_Menon2007,
        )
    except ImportError:
        raise ImportError(
            "colour-demosaicing is required for Bayer demosaicing. "
            "Install it with: pip install colour-demosaicing"
        )

    if algorithm == "bilinear":
        return demosaicing_CFA_Bayer_bilinear(bayer)
    elif algorithm == "malvar2004":
        return demosaicing_CFA_Bayer_Malvar2004(bayer)
    elif algorithm == "menon2007":
        return demosaicing_CFA_Bayer_Menon2007(bayer)
    else:
        raise ValueError(
            f"Unknown demosaicing algorithm: '{algorithm}'. "
            "Must be one of: bilinear, malvar2004, menon2007"
        )


def _white_balance(rgb: np.ndarray, gains: list[float]) -> np.ndarray:
    """Apply per-channel multiplicative white balance gains."""
    result = rgb.copy()
    result[..., 0] *= gains[0]  # R
    result[..., 1] *= gains[1]  # G
    result[..., 2] *= gains[2]  # B
    return result


def _color_correction(rgb: np.ndarray, matrix: list[float]) -> np.ndarray:
    """Apply a 3×3 color correction matrix (row-major, 9 floats)."""
    ccm = np.array(matrix, dtype=np.float64).reshape(3, 3)
    # einsum: for each pixel (h,w), output[c] = sum_k(input[k] * ccm[c,k])
    return np.einsum("hwk,ck->hwc", rgb, ccm)


def _denoising(rgb: np.ndarray, algorithm: str, strength: float) -> np.ndarray:
    """Apply denoising to the luminance (Y) channel in YUV space.

    Operating on luminance only preserves colour fidelity.
    strength: sigma for gaussian; converted to odd kernel size for median.
    """
    try:
        from skimage.color import rgb2yuv, yuv2rgb
        from scipy import ndimage
    except ImportError:
        raise ImportError(
            "scikit-image and scipy are required for denoising. "
            "Install with: pip install scikit-image scipy"
        )

    yuv = rgb2yuv(np.clip(rgb, 0.0, 1.0))

    if algorithm == "gaussian":
        yuv[..., 0] = ndimage.gaussian_filter(yuv[..., 0], sigma=strength)
    elif algorithm == "median":
        # Map strength [0, 1] → odd kernel size [3, 21]
        kernel_size = max(3, int(strength * 20) | 1)
        yuv[..., 0] = ndimage.median_filter(yuv[..., 0], size=kernel_size)
    else:
        raise ValueError(
            f"Unknown denoising algorithm: '{algorithm}'. Must be: gaussian, median, none"
        )

    return yuv2rgb(yuv)


def _sharpening(rgb: np.ndarray, algorithm: str, radius: float, amount: float) -> np.ndarray:
    """Apply sharpening to the luminance (Y) channel in YUV space.

    Operating on luminance only avoids colour artefacts.
    """
    try:
        from skimage.color import rgb2yuv, yuv2rgb
    except ImportError:
        raise ImportError(
            "scikit-image is required for sharpening. "
            "Install with: pip install scikit-image"
        )

    yuv = rgb2yuv(np.clip(rgb, 0.0, 1.0))

    if algorithm == "unsharp_mask":
        from skimage.filters import unsharp_mask as _unsharp_mask
        yuv[..., 0] = _unsharp_mask(
            yuv[..., 0], radius=radius, amount=amount, preserve_range=True
        )
    elif algorithm == "sharpen_filter":
        from scipy.signal import convolve2d
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
        yuv[..., 0] = convolve2d(
            yuv[..., 0], kernel, mode="same", boundary="fill", fillvalue=0
        )
    else:
        raise ValueError(
            f"Unknown sharpening algorithm: '{algorithm}'. "
            "Must be: unsharp_mask, sharpen_filter, none"
        )

    return yuv2rgb(yuv)


def _gamma_correction(rgb: np.ndarray, gamma: float) -> np.ndarray:
    """Apply power-law gamma correction: out = in^(1/gamma)."""
    inv_gamma = 1.0 / gamma
    return np.power(np.clip(rgb, 0.0, 1.0), inv_gamma)

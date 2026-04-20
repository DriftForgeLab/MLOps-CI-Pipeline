# =============================================================================
# src/data/isp_pipeline.py — Image Signal Processing (ISP) pipeline
# =============================================================================
# Responsibility: Process raw Bayer images through a configurable 7-step ISP
# pipeline. Used both by the CNN preprocessing
# pipeline and by the drift scenario simulator.
#
# Pipeline steps (executed in order):
#   1. Black level correction  — subtract per-channel sensor offset [R, G1, G2, B]
#   2. Demosaicing             — Bayer mosaic → RGB (bilinear / Malvar2004 / Menon2007)
#   3. White balance           — per-channel multiplicative gain (R / G / B)
#   4. Color correction        — 3×3 linear transform in RGB space
#   5. Denoising               — Gaussian or median filter applied to luminance (YUV)
#   6. Sharpening              — Unsharp mask or convolution filter on luminance (YUV)
#   7. Gamma correction        — Power-law tone mapping (out = in^(1/gamma))
#
# Parameter sources
# -----------------
# Steps 1, 3, and 4 read their parameters from DNG file metadata by default.
# Any parameter can be overridden by setting it explicitly in the ISP config
# block. Partial overrides are supported: for example, setting only r_gain
# leaves g_gain and b_gain sourced from the DNG. This override mechanism is
# used by the drift scenario simulator to produce controlled, reproducible
# drift variants from the same raw images.
#
# Quick-start usage
# -----------------
#   from pathlib import Path
#   import rawpy
#   from src.data.isp_pipeline import read_camera_params, run_isp
#   from src.config.loader import load_preprocessing_config
#
#   prep = load_preprocessing_config(Path("src/config/preprocessing_raw_image.yaml"))
#   dng_path = Path("data/raw/my_dataset/train/images/cls/frame_001.dng")
#
#   with rawpy.imread(str(dng_path)) as raw:
#       bayer = raw.raw_image_visible.copy()
#
#   camera_params = read_camera_params(dng_path)   # reads DNG metadata
#   rgb = run_isp(bayer, prep.image.isp, camera_params)  # float64 [0, 1], shape (H, W, 3)
#
# Config structure (src/config/preprocessing_raw_image.yaml, image.isp block)
# --------------------------------------------------------------------------
#   isp:
#     black_level_correction:
#       enabled: true
#       black_level: null          # [R, G1, G2, B] offsets; null → read from DNG
#     demosaicing:
#       algorithm: "bilinear"      # bilinear | malvar2004 | menon2007
#     white_balance:
#       r_gain: null               # float ≥ 0; null → read from DNG (g normalized to 1.0)
#       g_gain: null
#       b_gain: null
#     color_correction:
#       enabled: true
#       matrix: null               # 9 floats, row-major 3×3; null → read from DNG
#     denoising:
#       algorithm: "gaussian"      # gaussian | median | none
#       strength: 0.5              # [0.0, 1.0]; see _denoising() for exact mapping
#     sharpening:
#       algorithm: "unsharp_mask"  # unsharp_mask | sharpen_filter | none
#       radius: 1.0                # [0.1, 5.0] — blur radius for unsharp mask
#       amount: 1.0                # [0.0, 3.0] — sharpening intensity
#     gamma_correction:
#       gamma: 2.2                 # [1.0, 2.8] — standard sRGB is 2.2
#
# Dependencies
# ------------
#   rawpy             — read DNG metadata (read_camera_params only)
#   colour-demosaicing — Bayer → RGB algorithms
#   scikit-image      — YUV conversion, unsharp mask, resize
#   scipy             — Gaussian/median filter, convolution
# =============================================================================

from __future__ import annotations

import logging
import time
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
    profile: bool = False,
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
        profile:       If True, log per-stage wall-clock timings at INFO level.
                       Use this to establish baseline measurements before GPU work.

    Returns:
        RGB image array of shape (H, W, 3), dtype float64, values in [0.0, 1.0].

    Raises:
        ValueError: If a required parameter is absent from both config and
                    camera_params.
    """
    img = raw.astype(np.float64)
    params = camera_params or {}

    _gpu_device = None
    if isp_config.gpu_accelerated:
        from src.common.device import resolve_device
        _gpu_device = resolve_device()
        if _gpu_device.type == "cpu":
            _gpu_device = None  # no GPU available — fall back to CPU paths

    if profile:
        _t0 = _t = time.perf_counter()

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

    if profile:
        _now = time.perf_counter()
        logger.info("ISP step 1 black_level:  %.1f ms", (_now - _t) * 1000)
        _t = _now

    # Normalize Bayer mosaic to [0, 1] before demosaicing
    img_max = img.max()
    if img_max > 0:
        img = img / img_max

    # --- Step 2: Demosaicing (Bayer → RGB) ---
    # GPU path only supports bilinear; malvar2004/menon2007 always run on CPU.
    if _gpu_device is not None and isp_config.demosaicing.algorithm == "bilinear":
        from src.data.isp_gpu import gpu_demosaic_bilinear
        img = gpu_demosaic_bilinear(img, _gpu_device)
    else:
        img = _demosaicing(img, isp_config.demosaicing.algorithm)

    if profile:
        _now = time.perf_counter()
        logger.info("ISP step 2 demosaicing:  %.1f ms", (_now - _t) * 1000)
        _t = _now

    # --- Step 3: White balance ---
    wb_cfg = isp_config.white_balance
    dng_wb = params.get("white_balance", [1.0, 1.0, 1.0])
    r_gain = wb_cfg.r_gain if wb_cfg.r_gain is not None else dng_wb[0]
    g_gain = wb_cfg.g_gain if wb_cfg.g_gain is not None else dng_wb[1]
    b_gain = wb_cfg.b_gain if wb_cfg.b_gain is not None else dng_wb[2]
    img = _white_balance(img, [r_gain, g_gain, b_gain])

    if profile:
        _now = time.perf_counter()
        logger.info("ISP step 3 white_balance: %.1f ms", (_now - _t) * 1000)
        _t = _now

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

    if profile:
        _now = time.perf_counter()
        logger.info("ISP step 4 color_correction: %.1f ms", (_now - _t) * 1000)
        _t = _now

    # --- Steps 5 + 6: Denoising + Sharpening ---
    # When gpu_accelerated=True and both steps are active, run them fused on GPU
    # (single YUV roundtrip). Otherwise fall back to separate CPU steps.
    _denoise_on = isp_config.denoising.algorithm != "none"
    _sharpen_on = isp_config.sharpening.algorithm != "none"

    _use_gpu_fused = (
        _gpu_device is not None
        and _denoise_on
        and _sharpen_on
    )
    _ran_gpu_fused = False
    if _use_gpu_fused:
        from src.data.isp_gpu import gpu_denoise_sharpen_fused
        img = gpu_denoise_sharpen_fused(
            img,
            isp_config.denoising.algorithm,
            isp_config.denoising.strength,
            isp_config.sharpening.algorithm,
            isp_config.sharpening.radius,
            isp_config.sharpening.amount,
            _gpu_device,
        )
        _ran_gpu_fused = True
    else:
        if _gpu_device is None and isp_config.gpu_accelerated and _denoise_on and _sharpen_on:
            logger.debug("gpu_accelerated=True but no GPU found; denoising+sharpening on CPU.")
        if _denoise_on:
            img = _denoising(img, isp_config.denoising.algorithm, isp_config.denoising.strength)
        if _sharpen_on:
            img = _sharpening(
                img,
                isp_config.sharpening.algorithm,
                isp_config.sharpening.radius,
                isp_config.sharpening.amount,
            )

    if profile:
        _now = time.perf_counter()
        logger.info(
            "ISP step 5+6 denoise+sharpen: %.1f ms%s",
            (_now - _t) * 1000,
            " [GPU fused]" if _ran_gpu_fused else "",
        )
        _t = _now

    # --- Step 7: Gamma correction ---
    img = np.clip(img, 0.0, 1.0)
    img = _gamma_correction(img, isp_config.gamma_correction.gamma)

    if profile:
        _now = time.perf_counter()
        logger.info("ISP step 7 gamma:        %.1f ms", (_now - _t) * 1000)
        logger.info("ISP total:               %.1f ms", (_now - _t0) * 1000)

    return img


# =============================================================================
# Private step implementations
# =============================================================================

def _black_level_correction(raw: np.ndarray, black_level: list[float]) -> np.ndarray:
    """Subtract per-channel black level from the raw Bayer mosaic.

    Raw sensor ADC values include a fixed electronic offset (the "black level")
    that must be subtracted before any other processing. Without this step,
    shadows appear lifted and colour balance is wrong.

    Assumes RGGB Bayer pattern (standard for most cameras):
      black_level[0] → R   pixels at even rows, even cols
      black_level[1] → G1  pixels at even rows, odd  cols
      black_level[2] → G2  pixels at odd  rows, even cols
      black_level[3] → B   pixels at odd  rows, odd  cols

    Typical sensor values range from 512 to 4096 (12-bit) or 1024 to 16384
    (14-bit). DNG metadata provides the exact value for each camera model.
    Result is clipped to [0, ∞) to prevent negative values from underexposed
    pixels.

    Args:
        raw:         2D Bayer array (H, W), float64, pre-normalization values.
        black_level: [R, G1, G2, B] offsets. Length must be 4.

    Returns:
        Black-level-corrected Bayer array, same shape and dtype as input,
        clipped to [0, ∞).
    """
    img = raw.copy()
    img[0::2, 0::2] -= black_level[0]   # R
    img[0::2, 1::2] -= black_level[1]   # G1
    img[1::2, 0::2] -= black_level[2]   # G2
    img[1::2, 1::2] -= black_level[3]   # B
    return np.clip(img, 0.0, None)


def _demosaicing(bayer: np.ndarray, algorithm: str) -> np.ndarray:
    """Demosaic Bayer pattern to full-colour RGB.

    Raw sensors capture only one colour channel per pixel in a mosaic pattern
    (RGGB). Demosaicing interpolates the missing two channels at each pixel
    from its neighbours, producing a full (H, W, 3) RGB array.

    Algorithms (quality / speed trade-off):
      bilinear   — simple bilinear interpolation; fast, slight colour fringing
                   at high-contrast edges. Good default for ML preprocessing.
      malvar2004 — Malvar–He–Cutler 2004; better edge reconstruction at
                   moderate cost. Recommended when edge fidelity matters.
      menon2007  — Menon–Andriani–Calvagno 2007; highest quality, slowest.
                   Use when ground-truth colour accuracy is critical.

    Args:
        bayer:     2D Bayer array (H, W), float64, values in [0, 1] after
                   black-level correction and normalization.
        algorithm: One of "bilinear", "malvar2004", "menon2007".

    Returns:
        RGB array (H, W, 3), float64, values in [0, 1].

    Raises:
        ValueError: If algorithm is not one of the supported options.
        ImportError: If colour-demosaicing is not installed.
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
    """Apply per-channel multiplicative white balance gains.

    Corrects for the colour cast introduced by the camera's illuminant (light
    source). Each channel is multiplied by its gain independently. The green
    channel gain (gains[1]) is normalized to 1.0 by convention; red and blue
    gains are expressed relative to green.

    Typical gain ranges (relative to green = 1.0):
      r_gain: 1.2 – 2.5  (red channel, often brightest under tungsten/daylight)
      g_gain: 1.0         (reference, always 1.0 after normalization)
      b_gain: 0.8 – 2.0  (blue channel, high under incandescent, low outdoors)

    Drift simulation uses gains outside these ranges (config parameter_bounds)
    to model systematic WB shifts, e.g. r_gain=0.5 (extreme cool) or 2.5
    (extreme warm).

    Note: values above 1.0 can exceed the [0, 1] range. The clip in step 7
    (gamma correction) handles this — no intermediate clip is applied here to
    avoid quantization artefacts.

    Args:
        rgb:   (H, W, 3) float64 array in [0, 1] from demosaicing.
        gains: [r_gain, g_gain, b_gain], all floats ≥ 0.

    Returns:
        White-balanced RGB array, same shape and dtype, values possibly > 1.0.
    """
    result = rgb.copy()
    result[..., 0] *= gains[0]  # R
    result[..., 1] *= gains[1]  # G
    result[..., 2] *= gains[2]  # B
    return result


def _color_correction(rgb: np.ndarray, matrix: list[float]) -> np.ndarray:
    """Apply a 3×3 color correction matrix (CCM) to the RGB image.

    The CCM maps from the camera's native colour space (sensor response) to a
    standard output colour space (e.g. sRGB). Each output channel is a linear
    combination of all three input channels:

        out_R = m[0]*in_R + m[1]*in_G + m[2]*in_B
        out_G = m[3]*in_R + m[4]*in_G + m[5]*in_B
        out_B = m[6]*in_R + m[7]*in_G + m[8]*in_B

    The matrix is provided as 9 floats in row-major order (left-to-right, top-
    to-bottom). A camera-specific matrix is stored in DNG metadata and used by
    default; setting `matrix: null` in config reads it from there.

    Matrix conventions:
      - Identity matrix [1,0,0, 0,1,0, 0,0,1] applies no correction.
      - Most camera matrices have positive diagonal entries near 1.0 and small
        off-diagonal cross-talk corrections (typically ±0.3).
      - Rows should sum to approximately 1.0 to preserve overall brightness.

    Args:
        rgb:    (H, W, 3) float64 array from white balance step.
        matrix: 9 floats, row-major 3×3. Length must be exactly 9.

    Returns:
        Colour-corrected RGB array, same shape and dtype. Values may exceed
        [0, 1] and will be clipped during gamma correction (step 7).
    """
    ccm = np.array(matrix, dtype=np.float64).reshape(3, 3)
    # einsum: for each pixel (h,w), output[c] = sum_k(input[k] * ccm[c,k])
    return np.einsum("hwk,ck->hwc", rgb, ccm)


def _denoising(rgb: np.ndarray, algorithm: str, strength: float) -> np.ndarray:
    """Apply denoising to the luminance (Y) channel in YUV space.

    Denoising operates exclusively on the luminance (Y) channel after
    converting from RGB to YUV. This preserves colour fidelity — smoothing
    the U/V channels would desaturate colours. After filtering, the image is
    converted back to RGB.

    Algorithms:
      gaussian  — Gaussian blur; strength is used directly as the sigma value.
                  Smooth, isotropic noise reduction; slightly blurs edges.
                  Best for high-frequency sensor noise (high ISO).
                  strength → sigma: strength=0.5 → sigma=0.5 (default, mild).
      median    — Median filter; strength maps to an odd kernel size:
                  kernel_size = max(3, int(strength × 20) | 1).
                  Preserves edges better than Gaussian; slower for large kernels.
                  strength=0.5 → kernel=11; strength=1.0 → kernel=21.
      none      — Denoising step is skipped entirely.

    Parameter:
      strength: float in [0.0, 1.0], default 0.5.
        Gaussian:  0.0 → sigma=0.0 (no effect); 0.5 → sigma=0.5; 1.0 → sigma=1.0
        Median:    0.0 → kernel=3 (minimal);    0.5 → kernel=11; 1.0 → kernel=21

    For drift simulation: setting strength=0.0 models minimal denoising
    (noisy sensor output); strength=1.0 models aggressive smoothing (loss
    of fine texture).

    Args:
        rgb:       (H, W, 3) float64 array in [0, 1] from color correction.
        algorithm: "gaussian", "median", or "none".
        strength:  Filter intensity, [0.0, 1.0].

    Returns:
        Denoised RGB array, same shape and dtype.

    Raises:
        ValueError: If algorithm is not one of the supported options.
        ImportError: If scikit-image or scipy is not installed.
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

    Sharpening operates on the luminance (Y) channel after converting to YUV,
    for the same reason as denoising: colour channels are left untouched to
    avoid saturation artefacts.

    Algorithms:
      unsharp_mask   — Subtracts a blurred copy of the image from itself,
                       then adds the difference back scaled by `amount`:
                         out = original + amount × (original − blur(radius))
                       Provides smooth, controllable edge enhancement.
                       radius: [0.1, 5.0] — Gaussian blur radius (in pixels).
                         Small radius (0.5–1.5) sharpens fine detail.
                         Large radius (3.0–5.0) sharpens coarser structures.
                       amount: [0.0, 3.0] — sharpening intensity.
                         0.0 → no sharpening; 1.0 → standard; >2.0 → artefacts likely.
      sharpen_filter — Convolves with a fixed 3×3 kernel:
                         [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
                       Simple, fast, but can introduce ringing at strong edges.
                       radius and amount parameters are ignored for this algorithm.
      none           — Sharpening step is skipped entirely.

    Args:
        rgb:       (H, W, 3) float64 array in [0, 1] from denoising step.
        algorithm: "unsharp_mask", "sharpen_filter", or "none".
        radius:    Blur radius for unsharp_mask, in pixels. Default 1.0.
                   Ignored by sharpen_filter.
        amount:    Sharpening intensity for unsharp_mask. Default 1.0.
                   Ignored by sharpen_filter.

    Returns:
        Sharpened RGB array, same shape and dtype.

    Raises:
        ValueError: If algorithm is not one of the supported options.
        ImportError: If scikit-image or scipy is not installed.
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
    """Apply power-law gamma correction: out = clip(in, 0, 1)^(1/gamma).

    Gamma correction maps the linear light values produced by the sensor into
    a perceptual brightness scale. Human vision is approximately logarithmic,
    so applying a power curve before display makes the image look natural.

    The standard sRGB gamma is 2.2. Lower values (closer to 1.0) produce a
    brighter, more linear output; higher values (closer to 2.8) produce darker,
    more compressed shadows.

    Gamma values for drift simulation (config parameter_bounds: [1.0, 2.8]):
      gamma=1.0  — no tone mapping (linear, very bright)
      gamma=1.8  — legacy Mac/printing standard
      gamma=2.2  — sRGB standard (default, recommended)
      gamma=2.8  — darker output, compressed highlights

    Input is clipped to [0, 1] before the power function to avoid NaN from
    negative values and to ensure all out-of-gamut values are handled cleanly.

    Args:
        rgb:   (H, W, 3) float64 array, may contain values outside [0, 1]
               from earlier pipeline steps. Clipped internally.
        gamma: Gamma exponent. Default 2.2. Range: [1.0, 2.8].

    Returns:
        Gamma-corrected RGB array, float64, values in [0.0, 1.0].
    """
    inv_gamma = 1.0 / gamma
    return np.power(np.clip(rgb, 0.0, 1.0), inv_gamma)

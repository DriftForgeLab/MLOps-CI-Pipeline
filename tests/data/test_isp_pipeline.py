# =============================================================================
# tests/data/test_isp_pipeline.py — ISP pipeline unit tests
# =============================================================================
# Covers:
#   - read_camera_params: missing / incomplete / unusual DNG metadata
#   - _demosaicing: all three algorithms produce same shape and dtype
#   - run_isp: output shape, dtype, value range, and partial overrides
#   - run_isp with camera_params=None (black_level required from config)
# =============================================================================

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config.schema import (
    ISPBlackLevelConfig,
    ISPColorCorrectionConfig,
    ISPConfig,
    ISPDemosaicingConfig,
    ISPDenoisingConfig,
    ISPGammaConfig,
    ISPSharpeningConfig,
    ISPWhiteBalanceConfig,
)
from src.data.isp_pipeline import (
    _demosaicing,
    read_camera_params,
    run_isp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def _mock_rawpy(mock_raw_obj):
    """Inject a fake rawpy module into sys.modules for the duration of the block.

    read_camera_params imports rawpy inside its body, so we must patch
    sys.modules rather than a module-level attribute.
    """
    mock_rawpy = MagicMock()
    # rawpy.imread(path).__enter__ should return the raw object
    mock_rawpy.imread.return_value.__enter__ = lambda s: mock_raw_obj
    mock_rawpy.imread.return_value.__exit__ = MagicMock(return_value=False)

    original = sys.modules.get("rawpy")
    sys.modules["rawpy"] = mock_rawpy
    try:
        yield mock_rawpy
    finally:
        if original is None:
            sys.modules.pop("rawpy", None)
        else:
            sys.modules["rawpy"] = original


def _make_raw_obj(
    *,
    black_level=None,
    camera_whitebalance=None,
    color_matrix=None,
) -> MagicMock:
    """Build a minimal rawpy RawPy mock with the given metadata attributes."""
    obj = MagicMock()
    obj.black_level_per_channel = black_level
    obj.camera_whitebalance = camera_whitebalance
    obj.color_matrix = color_matrix
    return obj


def _make_isp_config(
    *,
    black_level: tuple[float, ...] | None = (512.0, 512.0, 512.0, 512.0),
    demosaicing_algorithm: str = "bilinear",
    r_gain: float | None = 1.2,
    g_gain: float | None = 1.0,
    b_gain: float | None = 1.5,
    color_correction_enabled: bool = False,
    denoising_algorithm: str = "none",
    sharpening_algorithm: str = "none",
    gamma: float = 2.2,
) -> ISPConfig:
    """Minimal ISPConfig with sensible defaults for tests."""
    return ISPConfig(
        black_level_correction=ISPBlackLevelConfig(enabled=True, black_level=black_level),
        demosaicing=ISPDemosaicingConfig(algorithm=demosaicing_algorithm),
        white_balance=ISPWhiteBalanceConfig(r_gain=r_gain, g_gain=g_gain, b_gain=b_gain),
        color_correction=ISPColorCorrectionConfig(enabled=color_correction_enabled),
        denoising=ISPDenoisingConfig(algorithm=denoising_algorithm, strength=0.5),
        sharpening=ISPSharpeningConfig(algorithm=sharpening_algorithm, radius=1.0, amount=1.0),
        gamma_correction=ISPGammaConfig(gamma=gamma),
    )


def _make_bayer(h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    """Synthetic Bayer mosaic (H, W) float32 with values in [512, 4095]."""
    rng = np.random.RandomState(seed)
    return rng.uniform(512, 4095, size=(h, w)).astype(np.float32)


# ---------------------------------------------------------------------------
# read_camera_params: DNG metadata edge cases
# ---------------------------------------------------------------------------

class TestReadCameraParams:
    """Fallback and error behaviour for missing/unusual DNG metadata."""

    def _read(self, tmp_path: Path, raw_obj: MagicMock) -> dict:
        dng = tmp_path / "test.DNG"
        dng.write_bytes(b"fake")
        with _mock_rawpy(raw_obj):
            return read_camera_params(dng)

    # -- black_level --

    def test_black_level_none_raises(self, tmp_path):
        raw = _make_raw_obj(
            black_level=None,
            camera_whitebalance=[2048.0, 1024.0, 1536.0],
            color_matrix=np.eye(3, 4).astype(np.float32),
        )
        with pytest.raises(ValueError, match="black_level_per_channel"):
            self._read(tmp_path, raw)

    def test_black_level_too_short_raises(self, tmp_path):
        raw = _make_raw_obj(
            black_level=[512.0, 512.0],  # only 2 of the required 4 values
            camera_whitebalance=[2048.0, 1024.0, 1536.0],
            color_matrix=np.eye(3, 4).astype(np.float32),
        )
        with pytest.raises(ValueError, match="black_level_per_channel"):
            self._read(tmp_path, raw)

    def test_black_level_valid_roundtrips(self, tmp_path):
        raw = _make_raw_obj(
            black_level=[512.0, 513.0, 514.0, 515.0],
            camera_whitebalance=[2048.0, 1024.0, 1536.0],
            color_matrix=np.eye(3, 4).astype(np.float32),
        )
        params = self._read(tmp_path, raw)
        assert params["black_level"] == [512.0, 513.0, 514.0, 515.0]

    # -- white_balance --

    def test_white_balance_none_raises(self, tmp_path):
        raw = _make_raw_obj(
            black_level=[512.0, 512.0, 512.0, 512.0],
            camera_whitebalance=None,
            color_matrix=np.eye(3, 4).astype(np.float32),
        )
        with pytest.raises(ValueError, match="camera_whitebalance"):
            self._read(tmp_path, raw)

    def test_white_balance_too_short_raises(self, tmp_path):
        raw = _make_raw_obj(
            black_level=[512.0, 512.0, 512.0, 512.0],
            camera_whitebalance=[2048.0, 1024.0],  # only 2 values, need >= 3
            color_matrix=np.eye(3, 4).astype(np.float32),
        )
        with pytest.raises(ValueError, match="camera_whitebalance"):
            self._read(tmp_path, raw)

    def test_white_balance_normalized_g_equals_one(self, tmp_path):
        raw = _make_raw_obj(
            black_level=[512.0, 512.0, 512.0, 512.0],
            camera_whitebalance=[2048.0, 1024.0, 1536.0],
            color_matrix=np.eye(3, 4).astype(np.float32),
        )
        params = self._read(tmp_path, raw)
        r, g, b = params["white_balance"]
        assert g == pytest.approx(1.0)
        assert r == pytest.approx(2048.0 / 1024.0)
        assert b == pytest.approx(1536.0 / 1024.0)

    def test_white_balance_g_zero_falls_back_to_identity(self, tmp_path):
        """g_gain=0 in DNG metadata is physically invalid; pipeline uses [1,1,1]."""
        raw = _make_raw_obj(
            black_level=[512.0, 512.0, 512.0, 512.0],
            camera_whitebalance=[2048.0, 0.0, 1536.0],
            color_matrix=np.eye(3, 4).astype(np.float32),
        )
        params = self._read(tmp_path, raw)
        assert params["white_balance"] == [1.0, 1.0, 1.0]

    def test_white_balance_four_element_uses_first_three(self, tmp_path):
        """Some cameras return [R, G, B, G2]; only the first three are used."""
        raw = _make_raw_obj(
            black_level=[512.0, 512.0, 512.0, 512.0],
            camera_whitebalance=[2048.0, 1024.0, 1536.0, 1024.0],
            color_matrix=np.eye(3, 4).astype(np.float32),
        )
        params = self._read(tmp_path, raw)
        assert len(params["white_balance"]) == 3

    # -- color_matrix --

    def test_color_matrix_none_falls_back_to_identity(self, tmp_path):
        raw = _make_raw_obj(
            black_level=[512.0, 512.0, 512.0, 512.0],
            camera_whitebalance=[2048.0, 1024.0, 1536.0],
            color_matrix=None,
        )
        params = self._read(tmp_path, raw)
        assert params["color_matrix"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    def test_color_matrix_too_small_falls_back_to_identity(self, tmp_path):
        raw = _make_raw_obj(
            black_level=[512.0, 512.0, 512.0, 512.0],
            camera_whitebalance=[2048.0, 1024.0, 1536.0],
            color_matrix=np.eye(2).astype(np.float32),
        )
        params = self._read(tmp_path, raw)
        assert params["color_matrix"] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    def test_color_matrix_3x4_extracts_3x3_block(self, tmp_path):
        """rawpy sometimes returns a (3, 4) matrix; only the 3×3 block is used."""
        cm = np.array([
            [1.1, 0.1, -0.1, 0.0],
            [-0.2, 1.2, 0.0, 0.0],
            [0.0, -0.1, 1.1, 0.0],
        ], dtype=np.float32)
        raw = _make_raw_obj(
            black_level=[512.0, 512.0, 512.0, 512.0],
            camera_whitebalance=[2048.0, 1024.0, 1536.0],
            color_matrix=cm,
        )
        params = self._read(tmp_path, raw)
        expected = [float(v) for v in cm[:3, :3].flatten()]
        assert params["color_matrix"] == pytest.approx(expected)

    def test_file_not_found_raises(self, tmp_path):
        missing = tmp_path / "no_such.DNG"
        raw = _make_raw_obj(
            black_level=[512.0, 512.0, 512.0, 512.0],
            camera_whitebalance=[2048.0, 1024.0, 1536.0],
            color_matrix=np.eye(3, 4).astype(np.float32),
        )
        with _mock_rawpy(raw):
            with pytest.raises(FileNotFoundError):
                read_camera_params(missing)


# ---------------------------------------------------------------------------
# _demosaicing: shape, dtype, and channel consistency across algorithms
# ---------------------------------------------------------------------------

class TestDemosaicing:
    """All three demosaicing algorithms must produce (H, W, 3) float64."""

    ALGORITHMS = ["bilinear", "malvar2004", "menon2007"]

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_output_shape(self, algorithm):
        bayer = _make_bayer(32, 32).astype(np.float64) / 4095.0
        result = _demosaicing(bayer, algorithm)
        assert result.shape == (32, 32, 3), (
            f"{algorithm}: expected (32, 32, 3), got {result.shape}"
        )

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_output_dtype_is_float(self, algorithm):
        bayer = _make_bayer(32, 32).astype(np.float64) / 4095.0
        result = _demosaicing(bayer, algorithm)
        assert np.issubdtype(result.dtype, np.floating), (
            f"{algorithm}: expected float dtype, got {result.dtype}"
        )

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_non_square_input_preserves_spatial_dims(self, algorithm):
        bayer = _make_bayer(24, 32).astype(np.float64) / 4095.0
        result = _demosaicing(bayer, algorithm)
        assert result.shape == (24, 32, 3)

    def test_all_algorithms_same_channel_count(self):
        bayer = _make_bayer(32, 32).astype(np.float64) / 4095.0
        channel_counts = {alg: _demosaicing(bayer, alg).shape[2] for alg in self.ALGORITHMS}
        assert all(c == 3 for c in channel_counts.values()), channel_counts

    def test_unknown_algorithm_raises(self):
        bayer = _make_bayer(16, 16).astype(np.float64) / 4095.0
        with pytest.raises(ValueError, match="Unknown demosaicing algorithm"):
            _demosaicing(bayer, "unknown_algo")


# ---------------------------------------------------------------------------
# run_isp: output contract (shape, dtype, value range)
# ---------------------------------------------------------------------------

class TestRunISP:
    """run_isp must return (H, W, 3) float64 in [0.0, 1.0]."""

    def test_output_shape(self):
        result = run_isp(_make_bayer(32, 32), _make_isp_config())
        assert result.shape == (32, 32, 3)

    def test_output_dtype(self):
        result = run_isp(_make_bayer(32, 32), _make_isp_config())
        assert result.dtype == np.float64

    def test_output_values_in_0_1(self):
        result = run_isp(_make_bayer(32, 32), _make_isp_config())
        assert result.min() >= 0.0, f"min = {result.min()}"
        assert result.max() <= 1.0, f"max = {result.max()}"

    def test_camera_params_none_black_level_from_config(self):
        """camera_params=None is valid when black_level is set in the config."""
        cfg = _make_isp_config(black_level=(512.0, 512.0, 512.0, 512.0))
        result = run_isp(_make_bayer(32, 32), cfg, camera_params=None)
        assert result.shape == (32, 32, 3)

    def test_camera_params_none_no_black_level_raises(self):
        """camera_params=None + no config black_level must raise a clear error."""
        cfg = _make_isp_config(black_level=None)
        with pytest.raises(ValueError, match="Black level correction"):
            run_isp(_make_bayer(32, 32), cfg, camera_params=None)

    def test_config_white_balance_overrides_camera_params(self):
        """White balance in config must override values from DNG metadata."""
        camera_params = {
            "black_level": [512.0, 512.0, 512.0, 512.0],
            "white_balance": [1.0, 1.0, 1.0],
            "color_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
        cfg_override = _make_isp_config(r_gain=2.0, g_gain=1.0, b_gain=1.0)
        cfg_passthrough = _make_isp_config(r_gain=None, g_gain=None, b_gain=None)

        bayer = _make_bayer(32, 32)
        result_override = run_isp(bayer, cfg_override, camera_params=camera_params)
        result_passthrough = run_isp(bayer, cfg_passthrough, camera_params=camera_params)

        # r_gain=2.0 vs r_gain=1.0: red channel must differ
        assert not np.allclose(result_override[..., 0], result_passthrough[..., 0])

    def test_black_level_correction_disabled_changes_output(self):
        """Disabling black level correction must produce different results."""
        camera_params = {
            "black_level": [1000.0, 1000.0, 1000.0, 1000.0],
            "white_balance": [1.5, 1.0, 1.2],
            "color_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
        cfg_on = ISPConfig(
            black_level_correction=ISPBlackLevelConfig(enabled=True),
            demosaicing=ISPDemosaicingConfig(algorithm="bilinear"),
            white_balance=ISPWhiteBalanceConfig(),
            color_correction=ISPColorCorrectionConfig(enabled=False),
            denoising=ISPDenoisingConfig(algorithm="none"),
            sharpening=ISPSharpeningConfig(algorithm="none"),
            gamma_correction=ISPGammaConfig(gamma=2.2),
        )
        cfg_off = ISPConfig(
            black_level_correction=ISPBlackLevelConfig(enabled=False),
            demosaicing=ISPDemosaicingConfig(algorithm="bilinear"),
            white_balance=ISPWhiteBalanceConfig(),
            color_correction=ISPColorCorrectionConfig(enabled=False),
            denoising=ISPDenoisingConfig(algorithm="none"),
            sharpening=ISPSharpeningConfig(algorithm="none"),
            gamma_correction=ISPGammaConfig(gamma=2.2),
        )
        bayer = _make_bayer(32, 32)
        result_on = run_isp(bayer, cfg_on, camera_params=camera_params)
        result_off = run_isp(bayer, cfg_off, camera_params=camera_params)
        assert not np.allclose(result_on, result_off)

    def test_missing_white_balance_in_camera_params_uses_identity(self):
        """If camera_params has no 'white_balance' key, ISP uses identity [1,1,1]."""
        camera_params = {
            "black_level": [512.0, 512.0, 512.0, 512.0],
            # intentionally no 'white_balance' key
            "color_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
        cfg = _make_isp_config(r_gain=None, g_gain=None, b_gain=None)
        result = run_isp(_make_bayer(32, 32), cfg, camera_params=camera_params)
        assert result.shape == (32, 32, 3)

    @pytest.mark.parametrize("algorithm", ["bilinear", "malvar2004", "menon2007"])
    def test_all_demosaicing_algorithms_complete_pipeline(self, algorithm):
        """Each demosaicing algorithm must complete the full pipeline without error."""
        cfg = _make_isp_config(demosaicing_algorithm=algorithm)
        result = run_isp(_make_bayer(32, 32), cfg)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.float64
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_gamma_higher_value_produces_brighter_output(self):
        """gamma=2.8 applies x^(1/2.8) which lifts midtones above gamma=1.0."""
        bayer = _make_bayer(32, 32)
        result_flat = run_isp(bayer, _make_isp_config(gamma=1.0))
        result_lifted = run_isp(bayer, _make_isp_config(gamma=2.8))
        assert result_lifted.mean() > result_flat.mean()

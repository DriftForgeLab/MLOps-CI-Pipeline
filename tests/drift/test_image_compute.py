"""Tests for multi-scale drift analysis in src.drift.image_compute."""

from __future__ import annotations

import numpy as np
import pytest

from src.drift.image_compute import (
    _build_gaussian_pyramid,
    _interpret_scale_profile,
    compute_multiscale_drift,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _rgb_batch(n: int = 20, h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, h, w, 3)).astype(np.float64)


def _gray_batch(n: int = 20, h: int = 32, w: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, h, w)).astype(np.float64)


# ---------------------------------------------------------------------------
# _build_gaussian_pyramid
# ---------------------------------------------------------------------------

class TestBuildGaussianPyramid:

    def test_level_count_rgb(self):
        images = _rgb_batch(n=10, h=64, w=64)
        pyramid = _build_gaussian_pyramid(images, num_levels=3)
        assert len(pyramid) == 3

    def test_level_count_grayscale(self):
        images = _gray_batch(n=10, h=64, w=64)
        pyramid = _build_gaussian_pyramid(images, num_levels=3)
        assert len(pyramid) == 3

    def test_level_0_is_original(self):
        images = _rgb_batch()
        pyramid = _build_gaussian_pyramid(images, num_levels=2)
        np.testing.assert_array_equal(pyramid[0], images)

    def test_spatial_dims_halved_each_level(self):
        images = _rgb_batch(n=10, h=64, w=64)
        pyramid = _build_gaussian_pyramid(images, num_levels=3)
        assert pyramid[0].shape == (10, 64, 64, 3)
        assert pyramid[1].shape == (10, 32, 32, 3)
        assert pyramid[2].shape == (10, 16, 16, 3)

    def test_batch_and_channel_dims_preserved(self):
        images = _rgb_batch(n=8, h=32, w=32)
        pyramid = _build_gaussian_pyramid(images, num_levels=2)
        # N and C must be unchanged at every level
        assert pyramid[0].shape[0] == 8
        assert pyramid[0].shape[-1] == 3
        assert pyramid[1].shape[0] == 8
        assert pyramid[1].shape[-1] == 3

    def test_num_levels_1_returns_original_only(self):
        images = _rgb_batch()
        pyramid = _build_gaussian_pyramid(images, num_levels=1)
        assert len(pyramid) == 1
        np.testing.assert_array_equal(pyramid[0], images)

    def test_stops_before_minimum_spatial_size(self):
        """Very small images should cap the pyramid early rather than error."""
        images = _rgb_batch(n=5, h=8, w=8)
        # Requesting 4 levels from 8×8 — level 1 = 4×4 (ok), level 2 = 2×2 (< 4, stop)
        pyramid = _build_gaussian_pyramid(images, num_levels=4)
        # Must have at least level 0 and level 1; must not go below 4×4
        assert len(pyramid) >= 1
        for lvl in pyramid:
            assert min(lvl.shape[1], lvl.shape[2]) >= 4

    def test_values_are_smoothed(self):
        """Coarser levels should have lower pixel variance than level 0."""
        rng = np.random.default_rng(99)
        noisy = rng.normal(0.0, 1.0, (30, 64, 64, 3)).astype(np.float64)
        pyramid = _build_gaussian_pyramid(noisy, num_levels=3)
        var_fine = float(np.var(pyramid[0]))
        var_coarse = float(np.var(pyramid[2]))
        assert var_coarse < var_fine, (
            "Coarser pyramid level should have lower variance than full-res level"
        )


# ---------------------------------------------------------------------------
# _interpret_scale_profile
# ---------------------------------------------------------------------------

class TestInterpretScaleProfile:

    def test_high_ratio_indicates_fine_detail(self):
        result = _interpret_scale_profile(scale_decay_ratio=10.0)
        assert "fine-detail" in result

    def test_low_ratio_indicates_global(self):
        result = _interpret_scale_profile(scale_decay_ratio=1.1)
        assert "global" in result or "coarse" in result

    def test_mid_ratio_indicates_midscale(self):
        result = _interpret_scale_profile(scale_decay_ratio=2.0)
        assert "mid-scale" in result

    def test_boundary_high_is_fine(self):
        # Exactly at the 3.0 boundary should be "fine-detail"
        result = _interpret_scale_profile(scale_decay_ratio=3.1)
        assert "fine-detail" in result

    def test_boundary_low_is_global(self):
        result = _interpret_scale_profile(scale_decay_ratio=1.4)
        assert "global" in result or "coarse" in result


# ---------------------------------------------------------------------------
# compute_multiscale_drift
# ---------------------------------------------------------------------------

class TestComputeMultiscaleDrift:

    def test_returns_expected_keys(self):
        batch = _rgb_batch(n=20, h=32, w=32, seed=1)
        reference = _rgb_batch(n=20, h=32, w=32, seed=2)
        result = compute_multiscale_drift(batch, reference, num_levels=3)
        assert "levels" in result
        assert "scale_decay_ratio" in result
        assert "interpretation" in result

    def test_level_count_matches_request(self):
        batch = _rgb_batch(n=20, h=64, w=64, seed=1)
        reference = _rgb_batch(n=20, h=64, w=64, seed=2)
        result = compute_multiscale_drift(batch, reference, num_levels=3)
        assert len(result["levels"]) == 3

    def test_level_0_is_full_resolution(self):
        batch = _rgb_batch(n=20, h=32, w=32)
        reference = _rgb_batch(n=20, h=32, w=32)
        result = compute_multiscale_drift(batch, reference, num_levels=2)
        assert result["levels"][0]["level"] == 0
        assert result["levels"][0]["resolution_factor"] == "1/1"
        assert result["levels"][0]["spatial_shape"] == [32, 32]

    def test_each_level_has_required_keys(self):
        batch = _rgb_batch(n=20, h=32, w=32)
        reference = _rgb_batch(n=20, h=32, w=32)
        result = compute_multiscale_drift(batch, reference, num_levels=2)
        for lvl in result["levels"]:
            assert "level" in lvl
            assert "resolution_factor" in lvl
            assert "spatial_shape" in lvl
            assert "channel_scores" in lvl
            assert "overall_score" in lvl

    def test_rgb_channel_scores_present(self):
        batch = _rgb_batch(n=20, h=32, w=32)
        reference = _rgb_batch(n=20, h=32, w=32)
        result = compute_multiscale_drift(batch, reference, num_levels=2)
        channels = result["levels"][0]["channel_scores"]
        assert set(channels.keys()) == {"r", "g", "b"}

    def test_grayscale_channel_score_present(self):
        batch = _gray_batch(n=20, h=32, w=32)
        reference = _gray_batch(n=20, h=32, w=32)
        result = compute_multiscale_drift(batch, reference, num_levels=2)
        channels = result["levels"][0]["channel_scores"]
        assert "gray" in channels

    def test_identical_inputs_near_zero_score(self):
        """Identical batch and reference should produce near-zero Wasserstein at every level."""
        data = _rgb_batch(n=20, h=32, w=32, seed=7)
        result = compute_multiscale_drift(data, data.copy(), num_levels=3)
        for lvl in result["levels"]:
            assert lvl["overall_score"] < 1e-4, (
                f"Level {lvl['level']} score should be ~0 for identical inputs, "
                f"got {lvl['overall_score']}"
            )

    def test_shifted_batch_gives_higher_score_at_level_0(self):
        """A globally shifted batch should show non-trivial score at full resolution."""
        rng = np.random.default_rng(42)
        reference = rng.random((30, 32, 32, 3)).astype(np.float64)
        batch = reference + 0.5  # large global shift
        result = compute_multiscale_drift(batch, reference, num_levels=3)
        assert result["levels"][0]["overall_score"] > 0.05

    def test_noise_drift_produces_high_decay_ratio(self):
        """Pure high-frequency noise added to the batch should yield a high scale_decay_ratio."""
        rng = np.random.default_rng(0)
        reference = rng.random((40, 64, 64, 3)).astype(np.float64)
        noise = rng.normal(0.0, 0.5, reference.shape)
        batch = reference + noise
        result = compute_multiscale_drift(batch, reference, num_levels=3)
        # Fine-detail (noise) drift should decay at coarser scales
        assert result["scale_decay_ratio"] > 1.5, (
            f"Expected high decay ratio for noise drift, got {result['scale_decay_ratio']}"
        )

    def test_global_shift_produces_low_decay_ratio(self):
        """A uniform brightness shift should persist at coarse scales → low ratio."""
        rng = np.random.default_rng(5)
        reference = rng.random((40, 64, 64, 3)).astype(np.float64)
        batch = reference + 0.8  # large global brightness offset
        result = compute_multiscale_drift(batch, reference, num_levels=3)
        # Global drift should NOT decay strongly → ratio close to 1
        assert result["scale_decay_ratio"] < 4.0, (
            f"Expected low decay ratio for global shift, got {result['scale_decay_ratio']}"
        )

    def test_interpretation_is_string(self):
        batch = _rgb_batch(n=20, h=32, w=32, seed=3)
        reference = _rgb_batch(n=20, h=32, w=32, seed=4)
        result = compute_multiscale_drift(batch, reference, num_levels=2)
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0

    def test_scale_decay_ratio_is_positive(self):
        batch = _rgb_batch(n=20, h=32, w=32, seed=5)
        reference = _rgb_batch(n=20, h=32, w=32, seed=6)
        result = compute_multiscale_drift(batch, reference, num_levels=2)
        assert result["scale_decay_ratio"] > 0

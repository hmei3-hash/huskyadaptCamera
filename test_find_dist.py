"""
test_find_dist.py — Unit tests for find_dist.py core logic.

Tests cover:
  - calibrate_scale
  - sample_keypoint_depths
  - person_distance
  - find_label_anchor

Heavy ML models (YOLO, MiDaS) are NOT imported here; only pure numpy
functions are exercised so the suite runs without GPU or model weights.
"""

import numpy as np
import pytest

from find_dist import (
    calibrate_scale,
    sample_keypoint_depths,
    person_distance,
    find_label_anchor,
)


# ── helpers ──────────────────────────────────────────────────────

def flat_depth(shape=(480, 640), value=100.0):
    return np.full(shape, value, dtype=np.float32)


def kp(x, y, conf):
    return np.array([x, y, conf], dtype=np.float32)


# ── calibrate_scale ───────────────────────────────────────────────

class TestCalibrateScale:
    def test_uniform_depth_equals_ref_times_raw(self):
        depth = flat_depth(value=100.0)
        scale = calibrate_scale(depth, ref_dist=1.0)
        assert scale == pytest.approx(100.0, rel=1e-4)

    def test_ref_dist_multiplies_scale(self):
        depth = flat_depth(value=50.0)
        assert calibrate_scale(depth, ref_dist=2.0) == pytest.approx(100.0, rel=1e-4)

    def test_zero_depth_no_divide_by_zero(self):
        depth = flat_depth(value=0.0)
        scale = calibrate_scale(depth, ref_dist=1.0)
        # floor is 1e-6
        assert scale == pytest.approx(1e-6, rel=1e-2)

    def test_non_square_frame(self):
        depth = flat_depth(shape=(240, 320), value=80.0)
        scale = calibrate_scale(depth, ref_dist=0.5)
        assert scale == pytest.approx(40.0, rel=1e-3)

    def test_nonuniform_uses_center(self):
        depth = np.zeros((100, 100), dtype=np.float32)
        # Centre 30×30 region = 200, rest = 0
        depth[35:65, 35:65] = 200.0
        scale = calibrate_scale(depth, ref_dist=1.0)
        assert scale == pytest.approx(200.0, rel=0.05)


# ── sample_keypoint_depths ────────────────────────────────────────

class TestSampleKeypointDepths:
    def test_visible_keypoint_gives_correct_dist(self):
        depth = flat_depth(value=100.0)
        scale = 100.0  # dist = scale / raw = 100/100 = 1 m
        samples = sample_keypoint_depths(depth, scale, [kp(100, 200, 0.9)])
        assert len(samples) == 1
        xi, yi, d = samples[0]
        assert xi == 100 and yi == 200
        assert d == pytest.approx(1.0, rel=1e-4)

    def test_low_confidence_skipped(self):
        depth = flat_depth(value=100.0)
        samples = sample_keypoint_depths(depth, 100.0, [kp(100, 200, 0.3)])
        assert samples == []

    def test_exactly_at_threshold_included(self):
        depth = flat_depth(value=100.0)
        samples = sample_keypoint_depths(depth, 100.0, [kp(100, 200, 0.5)])
        assert len(samples) == 1

    def test_out_of_bounds_x_skipped(self):
        depth = flat_depth(shape=(480, 640), value=100.0)
        samples = sample_keypoint_depths(depth, 100.0, [kp(700, 200, 0.9)])
        assert samples == []

    def test_out_of_bounds_y_skipped(self):
        depth = flat_depth(shape=(480, 640), value=100.0)
        samples = sample_keypoint_depths(depth, 100.0, [kp(100, 500, 0.9)])
        assert samples == []

    def test_last_valid_pixel_included(self):
        depth = flat_depth(shape=(480, 640), value=100.0)
        samples = sample_keypoint_depths(depth, 100.0, [kp(639, 479, 0.9)])
        assert len(samples) == 1

    def test_origin_pixel_included(self):
        depth = flat_depth(shape=(480, 640), value=100.0)
        samples = sample_keypoint_depths(depth, 100.0, [kp(0, 0, 0.9)])
        assert len(samples) == 1

    def test_multiple_keypoints_mixed_visibility(self):
        depth = flat_depth(value=100.0)
        kps = [kp(10, 10, 0.9), kp(50, 50, 0.8), kp(0, 0, 0.1)]
        samples = sample_keypoint_depths(depth, 100.0, kps)
        assert len(samples) == 2

    def test_zero_raw_depth_clamped_not_inf(self):
        depth = np.zeros((480, 640), dtype=np.float32)
        samples = sample_keypoint_depths(depth, 100.0, [kp(100, 100, 0.9)])
        assert len(samples) == 1
        # clamp floor is 1e-3, so dist = 100/1e-3 = 1e5
        assert samples[0][2] == pytest.approx(1e5, rel=0.01)

    def test_custom_conf_threshold(self):
        depth = flat_depth(value=100.0)
        kps = [kp(10, 10, 0.6)]
        assert len(sample_keypoint_depths(depth, 100.0, kps, conf_thresh=0.7)) == 0
        assert len(sample_keypoint_depths(depth, 100.0, kps, conf_thresh=0.5)) == 1

    def test_varying_depth_values(self):
        depth = np.zeros((480, 640), dtype=np.float32)
        depth[100, 100] = 200.0
        depth[200, 200] = 50.0
        kps = [kp(100, 100, 0.9), kp(200, 200, 0.9)]
        samples = sample_keypoint_depths(depth, 100.0, kps)
        assert len(samples) == 2
        dists = {(s[0], s[1]): s[2] for s in samples}
        assert dists[(100, 100)] == pytest.approx(0.5, rel=1e-4)   # 100/200
        assert dists[(200, 200)] == pytest.approx(2.0, rel=1e-4)   # 100/50


# ── person_distance ───────────────────────────────────────────────

class TestPersonDistance:
    def test_single_sample(self):
        assert person_distance([(0, 0, 2.5)]) == pytest.approx(2.5)

    def test_median_three_values(self):
        samples = [(0, 0, 1.0), (0, 0, 3.0), (0, 0, 5.0)]
        assert person_distance(samples) == pytest.approx(3.0)

    def test_median_even_count(self):
        samples = [(0, 0, 1.0), (0, 0, 3.0)]
        assert person_distance(samples) == pytest.approx(2.0)

    def test_empty_returns_none(self):
        assert person_distance([]) is None

    def test_outlier_robustness(self):
        # Median should not be pulled to the outlier
        samples = [(0, 0, 1.0), (0, 0, 1.1), (0, 0, 1.2), (0, 0, 100.0)]
        assert person_distance(samples) == pytest.approx(1.15, rel=0.01)

    def test_all_equal(self):
        samples = [(i, i, 3.0) for i in range(10)]
        assert person_distance(samples) == pytest.approx(3.0)


# ── find_label_anchor ─────────────────────────────────────────────

class TestFindLabelAnchor:
    def test_returns_first_visible(self):
        assert find_label_anchor([kp(10, 20, 0.9)]) == (10, 20)

    def test_skips_low_confidence(self):
        kps = [kp(5, 5, 0.3), kp(10, 20, 0.9)]
        assert find_label_anchor(kps) == (10, 20)

    def test_all_low_conf_returns_none(self):
        kps = [kp(5, 5, 0.1), kp(10, 10, 0.2)]
        assert find_label_anchor(kps) is None

    def test_empty_returns_none(self):
        assert find_label_anchor([]) is None

    def test_returns_first_not_highest_conf(self):
        # Should return the FIRST visible, not the highest-confidence one
        kps = [kp(10, 20, 0.6), kp(30, 40, 0.95)]
        assert find_label_anchor(kps) == (10, 20)

    def test_exactly_at_threshold_included(self):
        assert find_label_anchor([kp(7, 8, 0.5)]) == (7, 8)

    def test_custom_threshold(self):
        kps = [kp(10, 20, 0.6)]
        assert find_label_anchor(kps, conf_thresh=0.7) is None
        assert find_label_anchor(kps, conf_thresh=0.5) == (10, 20)

    def test_coordinates_are_integers(self):
        result = find_label_anchor([kp(10.7, 20.3, 0.9)])
        assert result == (10, 20)
        assert isinstance(result[0], int) and isinstance(result[1], int)

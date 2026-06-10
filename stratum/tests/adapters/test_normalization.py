import os
os.environ.setdefault("SKRUB_RUST", "1")
import numpy as np
import pytest

from stratum import _rust_backend as rb

pytestmark = pytest.mark.skipif(not rb.HAVE_RUST, reason="Rust backend not built")

EPS = 1e-5


# ---- Helpers ----

def _f32(arr):
    return np.asarray(arr, dtype=np.float32)


# ---- normalize_l2 ----

class TestNormalizeL2:
    def test_basic_row_unit_norm(self):
        data = _f32([[3, 4], [1, 0], [5, 12]])
        out = rb.normalize_l2(data)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=EPS)

    def test_zero_row_unchanged(self):
        data = _f32([[0, 0, 0], [1, 0, 0]])
        out = rb.normalize_l2(data)
        np.testing.assert_allclose(out[0], [0, 0, 0], atol=EPS)
        np.testing.assert_allclose(out[1], [1, 0, 0], atol=EPS)

    def test_negative_values(self):
        data = _f32([[-3, 4]])
        out = rb.normalize_l2(data)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0], atol=EPS)
        assert out[0, 0] < 0  # sign preserved

    def test_does_not_modify_input(self):
        data = _f32([[3, 4]])
        original = data.copy()
        rb.normalize_l2(data)
        np.testing.assert_array_equal(data, original)

    def test_single_row(self):
        data = _f32([[0, 2, 0]])
        out = rb.normalize_l2(data)
        np.testing.assert_allclose(out, [[0, 1, 0]], atol=EPS)

    def test_output_dtype_is_float32(self):
        data = _f32([[1, 2]])
        out = rb.normalize_l2(data)
        assert out.dtype == np.float32


# ---- normalize_l1 ----

class TestNormalizeL1:
    def test_basic_row_unit_l1_norm(self):
        data = _f32([[3, 1, 2], [0, 4, 0]])
        out = rb.normalize_l1(data)
        l1_norms = np.abs(out).sum(axis=1)
        np.testing.assert_allclose(l1_norms, 1.0, atol=EPS)

    def test_negative_values_absolute_sum(self):
        data = _f32([[-2, 2]])
        out = rb.normalize_l1(data)
        np.testing.assert_allclose(out, [[-0.5, 0.5]], atol=EPS)

    def test_zero_row_unchanged(self):
        data = _f32([[0, 0]])
        out = rb.normalize_l1(data)
        np.testing.assert_allclose(out[0], [0, 0], atol=EPS)

    def test_known_values(self):
        data = _f32([[6, 3, 1]])
        out = rb.normalize_l1(data)
        np.testing.assert_allclose(out, [[0.6, 0.3, 0.1]], atol=EPS)


# ---- normalize_max ----

class TestNormalizeMax:
    def test_max_abs_becomes_one(self):
        data = _f32([[2, -6, 3], [10, 1, 0]])
        out = rb.normalize_max(data)
        max_abs = np.abs(out).max(axis=1)
        np.testing.assert_allclose(max_abs, 1.0, atol=EPS)

    def test_negative_max(self):
        data = _f32([[1, -4]])
        out = rb.normalize_max(data)
        np.testing.assert_allclose(np.abs(out).max(), 1.0, atol=EPS)
        assert out[0, 1] == pytest.approx(-1.0, abs=EPS)

    def test_zero_row_unchanged(self):
        data = _f32([[0, 0]])
        out = rb.normalize_max(data)
        np.testing.assert_allclose(out[0], [0, 0], atol=EPS)

    def test_known_values(self):
        data = _f32([[2, -6, 3]])
        out = rb.normalize_max(data)
        np.testing.assert_allclose(out, [[2/6, -1.0, 3/6]], atol=EPS)


# ---- min_max_fit / min_max_transform ----

class TestMinMaxScaler:
    def test_output_in_unit_interval(self):
        data = _f32([[0, 10], [5, 20], [10, 30]])
        _, out = rb.min_max_fit(data)
        assert out.min() >= -EPS
        assert out.max() <= 1.0 + EPS

    def test_min_maps_to_zero_max_maps_to_one(self):
        data = _f32([[0, 10], [5, 20], [10, 30]])
        _, out = rb.min_max_fit(data)
        np.testing.assert_allclose(out[:, 0], [0.0, 0.5, 1.0], atol=EPS)
        np.testing.assert_allclose(out[:, 1], [0.0, 0.5, 1.0], atol=EPS)

    def test_constant_column_outputs_zero(self):
        data = _f32([[5, 1], [5, 3], [5, 5]])
        _, out = rb.min_max_fit(data)
        np.testing.assert_allclose(out[:, 0], [0, 0, 0], atol=EPS)

    def test_transform_matches_fit(self):
        data = _f32([[0, 0], [4, 8], [8, 16]])
        model_id, fit_out = rb.min_max_fit(data)
        transform_out = rb.min_max_transform(model_id, data)
        np.testing.assert_allclose(fit_out, transform_out, atol=EPS)

    def test_transform_new_data(self):
        train = _f32([[0, 0], [10, 100]])
        model_id, _ = rb.min_max_fit(train)
        # Point at midpoint of range
        test = _f32([[5, 50]])
        out = rb.min_max_transform(model_id, test)
        np.testing.assert_allclose(out, [[0.5, 0.5]], atol=EPS)

    def test_transform_ncols_mismatch_raises(self):
        data = _f32([[0, 0], [1, 1]])
        model_id, _ = rb.min_max_fit(data)
        bad = _f32([[0, 0, 0]])
        with pytest.raises(Exception):
            rb.min_max_transform(model_id, bad)

    def test_invalid_model_id_raises(self):
        bad_id = 999_999_999
        data = _f32([[1, 2]])
        with pytest.raises(Exception):
            rb.min_max_transform(bad_id, data)

    def test_output_dtype_is_float32(self):
        data = _f32([[0, 1], [1, 2]])
        _, out = rb.min_max_fit(data)
        assert out.dtype == np.float32


# ---- standard_scaler_fit / standard_scaler_transform ----

class TestStandardScaler:
    def test_zero_mean_after_fit(self):
        data = _f32([[1, 100], [2, 200], [3, 300]])
        _, out = rb.standard_scaler_fit(data)
        col_means = out.mean(axis=0)
        np.testing.assert_allclose(col_means, [0, 0], atol=EPS)

    def test_unit_population_std_after_fit(self):
        data = _f32([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        _, out = rb.standard_scaler_fit(data)
        col_vars = ((out - out.mean(axis=0)) ** 2).mean(axis=0)
        np.testing.assert_allclose(col_vars, [1.0, 1.0], atol=EPS)

    def test_constant_column_outputs_zero(self):
        data = _f32([[7, 1], [7, 2], [7, 3]])
        _, out = rb.standard_scaler_fit(data)
        np.testing.assert_allclose(out[:, 0], [0, 0, 0], atol=EPS)

    def test_transform_matches_fit(self):
        data = _f32([[1, 10], [2, 20], [3, 30]])
        model_id, fit_out = rb.standard_scaler_fit(data)
        transform_out = rb.standard_scaler_transform(model_id, data)
        np.testing.assert_allclose(fit_out, transform_out, atol=EPS)

    def test_transform_new_data_uses_train_stats(self):
        # Train on [0, 2] -> mean=1, std=1 (population)
        train = _f32([[0, 0], [2, 0]])
        model_id, _ = rb.standard_scaler_fit(train)
        # Test point 3 -> (3 - 1) / 1 = 2.0
        test = _f32([[3, 0]])
        out = rb.standard_scaler_transform(model_id, test)
        np.testing.assert_allclose(out[0, 0], 2.0, atol=EPS)

    def test_transform_ncols_mismatch_raises(self):
        data = _f32([[0, 0], [1, 1]])
        model_id, _ = rb.standard_scaler_fit(data)
        bad = _f32([[0]])
        with pytest.raises(Exception):
            rb.standard_scaler_transform(model_id, bad)

    def test_invalid_model_id_raises(self):
        bad_id = 999_999_999
        data = _f32([[1, 2]])
        with pytest.raises(Exception):
            rb.standard_scaler_transform(bad_id, data)

    def test_output_dtype_is_float32(self):
        data = _f32([[0, 1], [1, 2]])
        _, out = rb.standard_scaler_fit(data)
        assert out.dtype == np.float32

    def test_single_row_constant_treated_as_zero_variance(self):
        data = _f32([[5, 3]])
        _, out = rb.standard_scaler_fit(data)
        # Single row: mean = value, variance = 0 -> output = 0
        np.testing.assert_allclose(out, [[0, 0]], atol=EPS)

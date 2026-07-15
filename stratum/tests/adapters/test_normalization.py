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


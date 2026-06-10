"""
Normalization example using the Rust backend.

Demonstrates three normalization strategies:
  - Row-wise normalization (L2, L1, max): scale each sample independently
  - MinMaxScaler:  fit per-column min/max on training data, map to [0, 1]
  - StandardScaler: fit per-column mean/std on training data, produce z-scores

Run after building the extension:
    cd _rust && maturin develop --release && cd ..
    SKRUB_RUST=1 python examples/normalization_example.py
"""
import os
os.environ.setdefault("SKRUB_RUST", "1")

import numpy as np
from stratum import _rust_backend as rb

if not rb.HAVE_RUST:
    raise SystemExit(
        "Rust backend not available. Build it first:\n"
        "  cd _rust && maturin develop --release"
    )


def section(title: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


rng = np.random.default_rng(42)

# ── Row-wise normalizations ───────────────────────────────────────────────────

section("L2 normalization (each row becomes unit vector)")

embeddings = rng.standard_normal((5, 4)).astype(np.float32)
print("Input:\n", embeddings.round(3))

l2_out = rb.normalize_l2(embeddings)
print("Output:\n", l2_out.round(4))
print("Row L2 norms:", np.linalg.norm(l2_out, axis=1).round(6))  # all ~1.0


section("L1 normalization (each row sums to 1 in absolute value)")

counts = rng.integers(0, 10, size=(4, 6)).astype(np.float32)
print("Input (pseudo-counts):\n", counts)

l1_out = rb.normalize_l1(counts)
print("Output:\n", l1_out.round(4))
print("Row L1 norms:", np.abs(l1_out).sum(axis=1).round(6))  # all ~1.0


section("Max normalization (each row divided by its absolute max)")

data = np.array([[2.0, -6.0, 3.0], [10.0, 1.0, 5.0]], dtype=np.float32)
print("Input:\n", data)

max_out = rb.normalize_max(data)
print("Output:\n", max_out.round(4))
print("Row max abs:", np.abs(max_out).max(axis=1))  # all 1.0


# ── MinMaxScaler ──────────────────────────────────────────────────────────────

section("MinMaxScaler — fit on train, transform train and test")

train = np.array([
    [10.0,  0.0,  1.5],
    [20.0, 50.0,  3.0],
    [30.0, 100.0, 4.5],
], dtype=np.float32)

test = np.array([
    [15.0, 25.0, 2.25],
    [35.0, 110.0, 5.0],   # outside train range — clipping NOT applied
], dtype=np.float32)

model_id, train_scaled = rb.min_max_fit(train)
print("Train scaled (should be in [0,1]):\n", train_scaled.round(4))

test_scaled = rb.min_max_transform(model_id, test)
print("Test scaled:\n", test_scaled.round(4))

# Verify train range
print(f"Train min={train_scaled.min():.4f}, max={train_scaled.max():.4f}")


# ── StandardScaler ────────────────────────────────────────────────────────────

section("StandardScaler — fit on train, transform train and test")

train = rng.standard_normal((100, 4)).astype(np.float32)
# Shift columns to give non-trivial means/stds
train[:, 0] += 10
train[:, 1] *= 5
train[:, 2] -= 3
train[:, 3] *= 0.1

model_id, train_z = rb.standard_scaler_fit(train)
print("Train z-scores — per-column stats (should be ~mean=0, std=1):")
print("  means:", train_z.mean(axis=0).round(4))
print("  stds: ", train_z.std(axis=0).round(4))

test = rng.standard_normal((10, 4)).astype(np.float32)
test[:, 0] += 10
test[:, 1] *= 5
test[:, 2] -= 3
test[:, 3] *= 0.1

test_z = rb.standard_scaler_transform(model_id, test)
print("\nTest z-scores (first 3 rows):\n", test_z[:3].round(4))


# ── Edge cases ────────────────────────────────────────────────────────────────

section("Edge cases")

# All-zero row: normalization leaves it untouched
zero_row = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
out = rb.normalize_l2(zero_row)
assert np.all(out[0] == 0), "Zero row should stay zero"
print("Zero row handled correctly (stays zero).")

# Constant column in MinMax: output is 0
const_col = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]], dtype=np.float32)
_, out = rb.min_max_fit(const_col)
assert np.all(out[:, 0] == 0), "Constant column should map to 0"
print("Constant MinMax column handled correctly (maps to 0).")

# Constant column in StandardScaler: output is 0
_, out = rb.standard_scaler_fit(const_col)
assert np.all(out[:, 0] == 0), "Zero-variance column should map to 0"
print("Zero-variance StandardScaler column handled correctly (maps to 0).")

print("\nAll checks passed.")

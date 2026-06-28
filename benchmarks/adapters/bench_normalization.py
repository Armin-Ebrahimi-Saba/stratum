"""
Benchmark: Rust normalization kernels vs. sklearn equivalents.

Kernels compared
----------------
  normalize_l2    sklearn.preprocessing.normalize(X, norm='l2')
  normalize_l1    sklearn.preprocessing.normalize(X, norm='l1')
  normalize_max   sklearn.preprocessing.normalize(X, norm='max')
  min_max fit     MinMaxScaler().fit_transform(X)
  min_max transform MinMaxScaler().transform(X)    (pre-fitted)
  std_scaler fit  StandardScaler().fit_transform(X)
  std_scaler transform StandardScaler().transform(X)  (pre-fitted)

All inputs are float32. Results are printed as a table.

Usage
-----
    cd <repo_root>
    uv run python benchmarks/adapters/bench_normalization.py

    # Larger sizes:
    uv run python benchmarks/adapters/bench_normalization.py --large
"""
import os
os.environ["SKRUB_RUST"] = "1"

import argparse
import gc
import sys
import time

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize

from stratum import _rust_backend as rb

# ── guard ────────────────────────────────────────────────────────────────────
if not rb.HAVE_RUST:
    sys.exit(
        "Rust backend not built. Run:\n"
        "  cd _rust && maturin develop --release"
    )

# ── helpers ───────────────────────────────────────────────────────────────────

def make_data(n_rows: int, n_cols: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Mix of positive and negative values; some near-zero rows
    data = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
    # Sprinkle all-zero rows (~1%) to stress edge-case handling
    zero_idx = rng.integers(0, n_rows, size=max(1, n_rows // 100))
    data[zero_idx] = 0.0
    return data


def timeit(fn, n_reps: int = 5) -> float:
    """Return median wall-clock time over n_reps repetitions (seconds)."""
    times = []
    for _ in range(n_reps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def warmup(fn, n: int = 3):
    for _ in range(n):
        fn()
    gc.collect()


def timeit_inplace(make_buf, fn, n_reps: int = 5) -> float:
    """Like timeit, but rebuilds the (mutated) buffer before each rep so the
    copy cost stays outside the timed region."""
    for _ in range(3):
        fn(make_buf())
    gc.collect()
    times = []
    for _ in range(n_reps):
        buf = make_buf()
        gc.collect()
        t0 = time.perf_counter()
        fn(buf)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ── per-kernel benchmark ──────────────────────────────────────────────────────

def bench_normalize(data: np.ndarray, n_reps: int) -> dict:
    results = {}

    for norm in ("l2", "l1", "max"):
        rust_fn = {"l2": rb.normalize_l2, "l1": rb.normalize_l1, "max": rb.normalize_max}[norm]

        warmup(lambda: normalize(data, norm=norm, copy=True))
        sklearn_t = timeit(lambda: normalize(data, norm=norm, copy=True), n_reps)

        warmup(lambda: rust_fn(data))
        rust_t = timeit(lambda: rust_fn(data), n_reps)

        results[f"normalize_{norm}"] = (sklearn_t, rust_t)

    return results


def bench_normalize_inplace(data: np.ndarray, n_reps: int) -> dict:
    results = {}

    for norm in ("l2", "l1", "max"):
        rust_fn = {
            "l2": rb.normalize_l2_inplace,
            "l1": rb.normalize_l1_inplace,
            "max": rb.normalize_max_inplace,
        }[norm]

        sklearn_t = timeit_inplace(
            lambda: data.copy(),
            lambda buf: normalize(buf, norm=norm, copy=False),
            n_reps,
        )
        rust_t = timeit_inplace(lambda: data.copy(), rust_fn, n_reps)

        results[f"normalize_{norm}_inplace"] = (sklearn_t, rust_t)

    return results


def bench_min_max(data: np.ndarray, n_reps: int) -> dict:
    results = {}

    # fit (fit_transform, sklearn has no separate fit that returns output)
    sk = MinMaxScaler()
    warmup(lambda: MinMaxScaler().fit_transform(data))
    sklearn_fit_t = timeit(lambda: MinMaxScaler().fit_transform(data), n_reps)

    warmup(lambda: rb.min_max_fit(data))
    rust_fit_t = timeit(lambda: rb.min_max_fit(data), n_reps)

    results["min_max_fit"] = (sklearn_fit_t, rust_fit_t)

    # transform (re-use already fitted model)
    sk = MinMaxScaler().fit(data)
    model_id, _ = rb.min_max_fit(data)

    warmup(lambda: sk.transform(data))
    sklearn_tr_t = timeit(lambda: sk.transform(data), n_reps)

    warmup(lambda: rb.min_max_transform(model_id, data))
    rust_tr_t = timeit(lambda: rb.min_max_transform(model_id, data), n_reps)

    results["min_max_transform"] = (sklearn_tr_t, rust_tr_t)

    # transform in-place (overwrite the caller's buffer). Both sides mutate, so
    # each rep gets a fresh, un-transformed input via timeit_inplace; the rebuild
    # cost stays outside the timed region.
    sk_ip = MinMaxScaler(copy=False).fit(data)
    sklearn_tr_ip_t = timeit_inplace(
        lambda: data.copy(), lambda buf: sk_ip.transform(buf), n_reps
    )
    rust_tr_ip_t = timeit_inplace(
        lambda: data.copy(), lambda buf: rb.min_max_transform_inplace(model_id, buf), n_reps
    )

    results["min_max_transform_inplace"] = (sklearn_tr_ip_t, rust_tr_ip_t)

    return results


def bench_standard_scaler(data: np.ndarray, n_reps: int) -> dict:
    results = {}

    # fit
    warmup(lambda: StandardScaler().fit_transform(data))
    sklearn_fit_t = timeit(lambda: StandardScaler().fit_transform(data), n_reps)

    warmup(lambda: rb.standard_scaler_fit(data))
    rust_fit_t = timeit(lambda: rb.standard_scaler_fit(data), n_reps)

    results["standard_scaler_fit"] = (sklearn_fit_t, rust_fit_t)

    # transform
    sk = StandardScaler().fit(data)
    model_id, _ = rb.standard_scaler_fit(data)

    warmup(lambda: sk.transform(data))
    sklearn_tr_t = timeit(lambda: sk.transform(data), n_reps)

    warmup(lambda: rb.standard_scaler_transform(model_id, data))
    rust_tr_t = timeit(lambda: rb.standard_scaler_transform(model_id, data), n_reps)

    results["standard_scaler_transform"] = (sklearn_tr_t, rust_tr_t)

    # transform in-place (overwrite the caller's buffer). Fresh input each rep
    # via timeit_inplace since both sides mutate their argument.
    sk_ip = StandardScaler(copy=False).fit(data)
    sklearn_tr_ip_t = timeit_inplace(
        lambda: data.copy(), lambda buf: sk_ip.transform(buf), n_reps
    )
    rust_tr_ip_t = timeit_inplace(
        lambda: data.copy(), lambda buf: rb.standard_scaler_transform_inplace(model_id, buf), n_reps
    )

    results["standard_scaler_transform_inplace"] = (sklearn_tr_ip_t, rust_tr_ip_t)

    return results


# ── printing ──────────────────────────────────────────────────────────────────

COL_W = 22

def _ms(t: float) -> str:
    return f"{t * 1000:.1f} ms"


def print_header():
    header = (
        f"{'kernel':<{COL_W}}"
        f"{'sklearn':>{COL_W}}"
        f"{'rust':>{COL_W}}"
        f"{'speedup':>{COL_W}}"
    )
    print(header)
    print("-" * len(header))


def print_row(name: str, sklearn_t: float, rust_t: float):
    speedup = sklearn_t / rust_t if rust_t > 0 else float("inf")
    marker = " ✓" if speedup >= 1.0 else " ✗"
    print(
        f"{name:<{COL_W}}"
        f"{_ms(sklearn_t):>{COL_W}}"
        f"{_ms(rust_t):>{COL_W}}"
        f"{speedup:>{COL_W-2}.2f}x{marker}"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def run_suite(sizes: list[tuple[int, int]], n_reps: int):
    for n_rows, n_cols in sizes:
        mb = n_rows * n_cols * 4 / 1e6
        print(f"\n{'='*88}")
        print(f"  Shape: ({n_rows:,} × {n_cols})   ~{mb:.0f} MB float32   reps={n_reps}")
        print(f"{'='*88}")

        data = make_data(n_rows, n_cols)
        print_header()

        for name, (sk, ru) in bench_normalize(data, n_reps).items():
            print_row(name, sk, ru)
        for name, (sk, ru) in bench_normalize_inplace(data, n_reps).items():
            print_row(name, sk, ru)
        for name, (sk, ru) in bench_min_max(data, n_reps).items():
            print_row(name, sk, ru)
        for name, (sk, ru) in bench_standard_scaler(data, n_reps).items():
            print_row(name, sk, ru)

        del data
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Bench Rust normalization vs sklearn")
    parser.add_argument("--large", action="store_true", help="Add very large sizes")
    parser.add_argument("--reps", type=int, default=5, help="Repetitions per kernel (default 5)")
    args = parser.parse_args()

    sizes = [
        (10_000,   100),
        (100_000,  100),
        (100_000, 1_000),
        (1_000_000, 50),
    ]
    if args.large:
        sizes += [
            (1_000_000, 100),
            (1_000_000, 500),
        ]

    print(f"sklearn {__import__('sklearn').__version__}  |  numpy {np.__version__}  |  reps={args.reps}")
    run_suite(sizes, args.reps)


if __name__ == "__main__":
    main()

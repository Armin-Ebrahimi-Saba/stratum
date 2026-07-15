"""
Benchmark: Rust normalization kernels vs. sklearn equivalents.

Metrics: sklearn time, rust time, speedup, and a correctness check comparing the
Rust output against sklearn's element-wise.

Kernels compared
----------------
  normalize_l2    sklearn.preprocessing.normalize(X, norm='l2')
  normalize_l1    sklearn.preprocessing.normalize(X, norm='l1')
  normalize_max   sklearn.preprocessing.normalize(X, norm='max')
  (each also has an in-place variant)

All inputs are float32. Results are printed as a table.

Correctness
-----------
Each kernel's Rust output is compared to sklearn's with
np.allclose(rtol=1e-3, atol=1e-4); the table shows PASS/FAIL and the max
absolute error. Pass --no-verify to skip the check.

Usage
-----
    cd <repo_root>
    uv run python benchmarks/adapters/bench_normalization.py

    # Larger sizes:
    uv run python benchmarks/adapters/bench_normalization.py --large

    # Skip the correctness check:
    uv run python benchmarks/adapters/bench_normalization.py --no-verify
"""
import os
os.environ["SKRUB_RUST"] = "1"

import argparse
import gc
import sys
import time

import numpy as np
from sklearn.preprocessing import normalize

from stratum import _rust_backend as rb

# ── guard ────────────────────────────────────────────────────────────────────
if not rb.HAVE_RUST:
    sys.exit(
        "Rust backend not built. Run:\n"
        "  cd _rust && maturin develop --release"
    )

# ── correctness tolerances ────────────────────────────────────────────────────
VERIFY_RTOL = 1e-3
VERIFY_ATOL = 1e-4

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


def verify_arrays(rust_out, sklearn_out) -> dict:
    """Compare a Rust kernel's output against sklearn's element-wise."""
    ru = np.asarray(rust_out, dtype=np.float64)
    sk = np.asarray(sklearn_out, dtype=np.float64)
    diff = np.abs(ru - sk)
    return {
        "passed":  bool(np.allclose(ru, sk, rtol=VERIFY_RTOL, atol=VERIFY_ATOL)),
        "max_abs": float(diff.max()),
        "max_rel": float((diff / (np.abs(sk) + 1e-6)).max()),
    }


# ── per-kernel benchmark ──────────────────────────────────────────────────────

def bench_normalize(data: np.ndarray, n_reps: int, verify: bool) -> dict:
    results = {}

    for norm in ("l2", "l1", "max"):
        rust_fn = {"l2": rb.normalize_l2, "l1": rb.normalize_l1, "max": rb.normalize_max}[norm]

        warmup(lambda: normalize(data, norm=norm, copy=True))
        sklearn_t = timeit(lambda: normalize(data, norm=norm, copy=True), n_reps)

        warmup(lambda: rust_fn(data))
        rust_t = timeit(lambda: rust_fn(data), n_reps)

        v = verify_arrays(rust_fn(data), normalize(data, norm=norm, copy=True)) if verify else None
        results[f"normalize_{norm}"] = (sklearn_t, rust_t, v)

    return results


def bench_normalize_inplace(data: np.ndarray, n_reps: int, verify: bool) -> dict:
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

        v = None
        if verify:
            buf = data.copy()
            rust_fn(buf)
            v = verify_arrays(buf, normalize(data, norm=norm, copy=True))
        results[f"normalize_{norm}_inplace"] = (sklearn_t, rust_t, v)

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
        f"{'correctness':>{COL_W}}"
    )
    print(header)
    print("-" * len(header))


def print_row(name: str, sklearn_t: float, rust_t: float, verify_info: dict | None) -> bool:
    """Print a result row; returns True if a correctness check FAILED."""
    speedup = sklearn_t / rust_t if rust_t > 0 else float("inf")
    marker = " ✓" if speedup >= 1.0 else " ✗"

    failed = False
    if verify_info is None:
        corr = "— skipped"
    else:
        failed = not verify_info["passed"]
        tag = "PASS" if verify_info["passed"] else "FAIL"
        corr = f"{tag}  (max_abs={verify_info['max_abs']:.1e})"

    print(
        f"{name:<{COL_W}}"
        f"{_ms(sklearn_t):>{COL_W}}"
        f"{_ms(rust_t):>{COL_W}}"
        f"{speedup:>{COL_W-4}.2f}x{marker:>2}"
        f"{corr:>{COL_W}}"
    )
    return failed


# ── main ──────────────────────────────────────────────────────────────────────

def run_suite(sizes: list[tuple[int, int]], n_reps: int, verify: bool) -> bool:
    """Run every kernel across every size. Returns True if any check failed."""
    any_fail = False
    for n_rows, n_cols in sizes:
        mb = n_rows * n_cols * 4 / 1e6
        print(f"\n{'='*110}")
        print(f"  Shape: ({n_rows:,} × {n_cols})   ~{mb:.0f} MB float32   reps={n_reps}"
              f"   verify={'on' if verify else 'off'}")
        print(f"{'='*110}")

        data = make_data(n_rows, n_cols)
        print_header()

        for group in (bench_normalize, bench_normalize_inplace):
            for name, (sk, ru, v) in group(data, n_reps, verify).items():
                any_fail |= print_row(name, sk, ru, v)

        del data
        gc.collect()
    return any_fail


def main():
    parser = argparse.ArgumentParser(description="Bench Rust normalization vs sklearn")
    parser.add_argument("--large", action="store_true", help="Add very large sizes")
    parser.add_argument("--reps", type=int, default=5, help="Repetitions per kernel (default 5)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the rust-vs-sklearn correctness check")
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

    verify = not args.no_verify
    print(f"sklearn {__import__('sklearn').__version__}  |  numpy {np.__version__}  "
          f"|  reps={args.reps}  |  verify={'on' if verify else 'off'}")
    any_fail = run_suite(sizes, args.reps, verify)

    if verify:
        print()
        if any_fail:
            print("RESULT: one or more kernels FAILED the correctness check ✗")
            sys.exit(1)
        print("RESULT: all kernels passed the correctness check ✓")


if __name__ == "__main__":
    main()

"""
Benchmark: Rust ElasticNet vs sklearn ElasticNet.

Metrics: fit time, predict time, coefficient MSE vs sklearn.

Usage
-----
    cd <repo_root>
    uv run python benchmarks/adapters/bench_elastic_net.py
    uv run python benchmarks/adapters/bench_elastic_net.py --reps 10
"""
import os
os.environ["SKRUB_RUST"] = "1"

import argparse
import gc
import sys
import time

import numpy as np
from sklearn.linear_model import ElasticNet

from stratum import _rust_backend as rb

if not rb.HAVE_RUST:
    sys.exit("Rust backend not built. Run: cd _rust && maturin develop --release")

# ── helpers ───────────────────────────────────────────────────────────────────

def make_data(n_rows, n_cols, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
    true_coef = rng.standard_normal(n_cols).astype(np.float32)
    y = (X @ true_coef + 0.1 * rng.standard_normal(n_rows)).astype(np.float32)
    return X, y


def timeit(fn, n_reps=5):
    times = []
    for _ in range(n_reps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def warmup(fn, n=2):
    for _ in range(n):
        fn()
    gc.collect()


# ── benchmark ─────────────────────────────────────────────────────────────────

COL_W = 22

def _ms(t): return f"{t * 1000:.1f} ms"

def print_header():
    h = (f"{'kernel':<{COL_W}}"
         f"{'sklearn':>{COL_W}}"
         f"{'rust':>{COL_W}}"
         f"{'speedup':>{COL_W}}")
    print(h)
    print("-" * len(h))

def print_row(name, sk_t, ru_t, extra=""):
    sp = sk_t / ru_t if ru_t > 0 else float("inf")
    mark = " ✓" if sp >= 1.0 else " ✗"
    print(f"{name:<{COL_W}}{_ms(sk_t):>{COL_W}}{_ms(ru_t):>{COL_W}}"
          f"{sp:>{COL_W - 2}.2f}x{mark}{extra}")


def bench_one(X, y, alpha, l1_ratio, max_iter, tol, n_reps):
    # ── sklearn ──
    sk = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter,
                    tol=tol, fit_intercept=True)
    Xf64, yf64 = X.astype(np.float64), y.astype(np.float64)

    warmup(lambda: ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter,
                               tol=tol).fit(Xf64, yf64))
    sk_fit_t = timeit(lambda: ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                          max_iter=max_iter, tol=tol).fit(Xf64, yf64), n_reps)
    sk.fit(Xf64, yf64)
    warmup(lambda: sk.predict(Xf64))
    sk_pred_t = timeit(lambda: sk.predict(Xf64), n_reps)

    # ── rust ──
    warmup(lambda: rb.elastic_net_fit(X, y, alpha, l1_ratio, max_iter, tol, True))
    ru_fit_t = timeit(lambda: rb.elastic_net_fit(X, y, alpha, l1_ratio, max_iter, tol, True), n_reps)

    mid, coef_ru, intercept_ru, n_iter_ru = rb.elastic_net_fit(
        X, y, alpha, l1_ratio, max_iter, tol, True)
    warmup(lambda: rb.elastic_net_predict(mid, X))
    ru_pred_t = timeit(lambda: rb.elastic_net_predict(mid, X), n_reps)

    # ── coefficient agreement ──
    coef_sk = sk.coef_.astype(np.float32)
    coef_mse = float(np.mean((coef_ru - coef_sk) ** 2))

    return sk_fit_t, ru_fit_t, sk_pred_t, ru_pred_t, coef_mse, n_iter_ru


def run_suite(sizes, alpha, l1_ratio, n_reps):
    for n_rows, n_cols in sizes:
        mb = n_rows * n_cols * 4 / 1e6
        max_iter = 1000
        tol = 1e-4
        print(f"\n{'='*88}")
        print(f"  ({n_rows:,} × {n_cols})  ~{mb:.0f} MB  alpha={alpha}  l1_ratio={l1_ratio}  "
              f"max_iter={max_iter}  reps={n_reps}")
        print(f"{'='*88}")

        X, y = make_data(n_rows, n_cols)
        sk_fit_t, ru_fit_t, sk_pred_t, ru_pred_t, coef_mse, n_iter = \
            bench_one(X, y, alpha, l1_ratio, max_iter, tol, n_reps)

        print_header()
        print_row("elastic_net fit", sk_fit_t, ru_fit_t,
                  f"  (rust n_iter={n_iter})")
        print_row("elastic_net predict", sk_pred_t, ru_pred_t,
                  f"  coef_mse={coef_mse:.2e}")
        del X, y
        gc.collect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps",     type=int,   default=5)
    p.add_argument("--alpha",    type=float, default=0.1)
    p.add_argument("--l1-ratio", type=float, default=0.5)
    p.add_argument("--large",    action="store_true")
    args = p.parse_args()

    sizes = [
        (10_000,   50),
        (50_000,   50),
        (100_000,  50),
        (100_000, 100),
    ]
    if args.large:
        sizes += [(500_000, 50), (1_000_000, 20)]

    print(f"sklearn {__import__('sklearn').__version__}  |  numpy {np.__version__}  "
          f"|  reps={args.reps}")
    run_suite(sizes, args.alpha, args.l1_ratio, args.reps)


if __name__ == "__main__":
    main()

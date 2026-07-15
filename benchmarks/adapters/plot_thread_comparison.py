#!/usr/bin/env python3
"""
Runtime vs thread count for four representative input sizes.

One panel per size tier. Each panel plots HEAD (block-parallel) runtime in ms
against thread count, for fit and predict, with the sklearn baseline shown as a
dashed reference line. The label on top of each bar is the speedup relative to
sklearn (sklearn_time / rust_time).

Produces two PNGs:
  elastic_net_thread_comparison_small_large.png    — Small + Large
  elastic_net_thread_comparison_xlarge_xxlarge.png — X-Large + XX-Large

Reads benchmarks/results/elastic_net_full_cache.json.

Usage:
    uv run python benchmarks/adapters/plot_thread_comparison.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
RESULTS    = REPO_ROOT / "benchmarks" / "results"
CACHE_FILE = RESULTS / "elastic_net_full_cache.json"
HEAD       = "10f16bd"

THREADS      = [1, 2, 4, 8]
FIT_COLOR    = "#2255CC"
PRED_COLOR   = "#FF9800"
SKLEARN_FIT  = "#CC2222"
SKLEARN_PRED = "#9C27B0"


def size_label(r: int, c: int) -> str:
    if r >= 1_000_000:
        return f"{r // 1_000_000}M×{c}"
    return f"{r // 1000}k×{c}"


def draw_panel(ax, cache: dict, tier: str, r: int, c: int) -> None:
    x = np.arange(len(THREADS))
    bar_w = 0.38

    fit  = [cache.get(f"{HEAD}__t{t}__{r}__{c}", {}).get("fit")     for t in THREADS]
    pred = [cache.get(f"{HEAD}__t{t}__{r}__{c}", {}).get("predict") for t in THREADS]
    fit_ms  = [v * 1000 if v else np.nan for v in fit]
    pred_ms = [v * 1000 if v else np.nan for v in pred]

    sk = cache.get(f"sklearn__{r}__{c}", {})
    sk_fit_ms  = sk.get("fit")     and sk["fit"]     * 1000
    sk_pred_ms = sk.get("predict") and sk["predict"] * 1000

    ax.bar(x - bar_w/2, fit_ms,  bar_w, color=FIT_COLOR,  label="fit (rust)",     zorder=3)
    ax.bar(x + bar_w/2, pred_ms, bar_w, color=PRED_COLOR, label="predict (rust)", zorder=3)

    # speedup-over-sklearn labels on top of each bar
    for xi, (f, p) in enumerate(zip(fit_ms, pred_ms)):
        if f and not np.isnan(f) and sk_fit_ms:
            ax.text(x[xi] - bar_w/2, f * 1.03, f"{sk_fit_ms/f:.1f}×",
                    ha="center", va="bottom", fontsize=8, color=FIT_COLOR, fontweight="bold")
        if p and not np.isnan(p) and sk_pred_ms:
            ax.text(x[xi] + bar_w/2, p * 1.03, f"{sk_pred_ms/p:.1f}×",
                    ha="center", va="bottom", fontsize=8, color=PRED_COLOR, fontweight="bold")

    # sklearn reference lines
    if sk_fit_ms:
        ax.axhline(sk_fit_ms, color=SKLEARN_FIT, ls="--", lw=1.4,
                   label="sklearn fit", zorder=2)
    if sk_pred_ms:
        ax.axhline(sk_pred_ms, color=SKLEARN_PRED, ls=":", lw=1.4,
                   label="sklearn predict", zorder=2)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}T" for t in THREADS])
    ax.set_xlabel("Thread count")
    ax.set_ylabel("Median time (ms)")
    mb = r * c * 4 / 1e6
    ax.set_title(f"{tier}: {size_label(r, c)}  ({mb:.0f} MB)",
                 fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:.2f}"))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="best")


def make_figure(cache: dict, panels: list, out_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(bottom=0.16, wspace=0.28)
    fig.suptitle("Runtime vs thread count  (labels = speedup over sklearn)",
                 fontsize=14, fontweight="bold", y=1.0)
    for ax, (tier, (r, c)) in zip(axes, panels):
        draw_panel(ax, cache, tier, r, c)
    out = RESULTS / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


def main() -> None:
    cache = json.loads(CACHE_FILE.read_text())

    make_figure(cache,
                [("Small", (100_000, 50)), ("Large", (500_000, 50))],
                "elastic_net_thread_comparison_small_large.png")

    make_figure(cache,
                [("X-Large", (2_000_000, 50)), ("XX-Large", (10_000_000, 50))],
                "elastic_net_thread_comparison_xlarge_xxlarge.png")

    print("Done.")


if __name__ == "__main__":
    main()

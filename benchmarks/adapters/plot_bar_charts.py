#!/usr/bin/env python3
"""
Generate two bar-chart PNGs comparing sklearn vs Rust (HEAD, 8 threads).

  elastic_net_bars_small_large.png   — Small + Large tiers
  elastic_net_bars_xlarge_xxlarge.png — X-Large + XX-Large tiers

Both files are written to benchmarks/results/.
Data is read from benchmarks/results/elastic_net_full_cache.json.

Usage:
    uv run python benchmarks/adapters/plot_bar_charts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
RESULTS    = REPO_ROOT / "benchmarks" / "results"
CACHE_FILE = RESULTS / "elastic_net_full_cache.json"
HEAD       = "10f16bd"

SKLEARN_COLOR = "#CC2222"
RUST_COLOR    = "#2255CC"
BAR_W         = 0.2
GAP           = 0.05


def load_tier(cache: dict, sizes: list[tuple[int, int]]) -> dict:
    """Pull sklearn and Rust (HEAD, 8T) times for a list of sizes."""
    labels, sk_fit, ru_fit, sk_pred, ru_pred = [], [], [], [], []
    for r, c in sizes:
        sk = cache.get(f"sklearn__{r}__{c}", {})
        ru = cache.get(f"{HEAD}__t8__{r}__{c}", {})
        if not sk or not ru:
            print(f"  warning: missing cache entry for {r}×{c} — skipped")
            continue
        k = r // 1000
        labels.append(f"{k}k×{c}" if k < 1000 else f"{k//1000}M×{c}")
        sk_fit.append(sk["fit"]     * 1000)
        ru_fit.append(ru["fit"]     * 1000)
        sk_pred.append(sk["predict"] * 1000)
        ru_pred.append(ru["predict"] * 1000)
    return dict(labels=labels,
                sk_fit=np.array(sk_fit),   ru_fit=np.array(ru_fit),
                sk_pred=np.array(sk_pred), ru_pred=np.array(ru_pred))


def draw_tier(ax: plt.Axes, data: dict, title: str) -> None:
    """Draw one panel (one size tier) on ax."""
    n = len(data["labels"])
    x = np.arange(n, dtype=float)

    sk_fit_pos  = x - BAR_W - GAP / 2
    ru_fit_pos  = x         - GAP / 2
    sk_pred_pos = x + GAP / 2
    ru_pred_pos = x + GAP / 2 + BAR_W

    ax.bar(sk_fit_pos,  data["sk_fit"],  BAR_W, color=SKLEARN_COLOR, alpha=1.0, zorder=3)
    ax.bar(ru_fit_pos,  data["ru_fit"],  BAR_W, color=RUST_COLOR,    alpha=1.0, zorder=3)
    ax.bar(sk_pred_pos, data["sk_pred"], BAR_W, color=SKLEARN_COLOR, alpha=0.5, zorder=3, hatch="//")
    ax.bar(ru_pred_pos, data["ru_pred"], BAR_W, color=RUST_COLOR,    alpha=0.5, zorder=3, hatch="//")

    # speedup labels above each Rust bar
    for xi in range(n):
        sp_fit  = data["sk_fit"][xi]  / data["ru_fit"][xi]
        sp_pred = data["sk_pred"][xi] / data["ru_pred"][xi]
        ax.text(ru_fit_pos[xi],  data["ru_fit"][xi]  * 1.35, f"{sp_fit:.1f}×",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1a3d99")
        ax.text(ru_pred_pos[xi], data["ru_pred"][xi] * 1.35, f"{sp_pred:.1f}×",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#1a3d99", alpha=0.85)

    tick_pos = (sk_fit_pos + ru_pred_pos) / 2
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(data["labels"], fontsize=10)
    ax.set_xlabel("Input size (rows × cols)", fontsize=10)
    ax.set_ylabel("Median time (ms)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:.2f}"))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.5, n - 0.5)


def make_legend(fig: plt.Figure) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SKLEARN_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=RUST_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=SKLEARN_COLOR, alpha=0.5, hatch="//"),
        plt.Rectangle((0, 0), 1, 1, color=RUST_COLOR,    alpha=0.5, hatch="//"),
    ]
    fig.legend(handles,
               ["sklearn — fit", "rust — fit (8T)",
                "sklearn — predict", "rust — predict (8T)"],
               loc="lower center", ncol=4, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, 0.01))


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path.relative_to(REPO_ROOT)}")
    plt.close(fig)


def main() -> None:
    cache = json.loads(CACHE_FILE.read_text())
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Small + Large ───────────────────────────────────────────────
    small = load_tier(cache, [(10_000,50),(50_000,50),(100_000,50),(100_000,100)])
    large = load_tier(cache, [(500_000,50),(1_000_000,20)])

    n_small = len(small["labels"])
    n_large = len(large["labels"])
    fig1, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6 * (n_small + n_large) / 3, 5.5),
        gridspec_kw={"width_ratios": [n_small, n_large], "wspace": 0.4},
    )
    fig1.subplots_adjust(bottom=0.2)
    draw_tier(ax1, small, "Small  (10k – 100k rows)")
    draw_tier(ax2, large, "Large  (500k – 1M rows)")
    make_legend(fig1)
    save(fig1, RESULTS / "elastic_net_bars_small_large.png")

    # ── Figure 2: X-Large + XX-Large ─────────────────────────────────────────
    xlarge  = load_tier(cache, [(2_000_000,50),(5_000_000,20)])
    xxlarge = load_tier(cache, [(10_000_000,50)])

    n_xl  = len(xlarge["labels"])
    n_xxl = len(xxlarge["labels"])
    fig2, (ax3, ax4) = plt.subplots(
        1, 2, figsize=(6 * (n_xl + n_xxl) / 2, 5.5),
        gridspec_kw={"width_ratios": [n_xl, n_xxl], "wspace": 0.4},
    )
    fig2.subplots_adjust(bottom=0.2)
    draw_tier(ax3, xlarge,  "X-Large  (2M – 5M rows)")
    draw_tier(ax4, xxlarge, "XX-Large  (10M rows)")
    make_legend(fig2)
    save(fig2, RESULTS / "elastic_net_bars_xlarge_xxlarge.png")

    print("Done.")


if __name__ == "__main__":
    main()

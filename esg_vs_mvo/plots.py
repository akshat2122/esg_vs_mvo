"""
Chart generation. Every figure is sized for LinkedIn-post readability
(1200x675 px at 100 dpi = 12x6.75 in), labeled, and carries a source line.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as cfg
from .metrics import drawdown_series

SOURCE_LINE = "Data: Yahoo Finance via yfinance. ESG: Sustainalytics via yfinance " \
              "(placeholder overrides used where unavailable — see README)."
FIGSIZE = (12, 6.75)
DPI = 100


def _finish(fig, path: Path, title: str):
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, SOURCE_LINE, fontsize=8, color="gray")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_cumulative_returns(returns_out_sample: pd.DataFrame, path: Path):
    growth = 100 * (1 + returns_out_sample).cumprod()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for col in growth.columns:
        ax.plot(growth.index, growth[col], label=col, linewidth=2)
    ax.set_ylabel("Growth of €100")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    _finish(fig, path, "Out-of-Sample Performance: ESG-Screened vs. Standard MVO Portfolio")


def plot_drawdowns(returns_out_sample: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for col in returns_out_sample.columns:
        dd = drawdown_series(returns_out_sample[col]) * 100
        ax.plot(dd.index, dd.values, label=col, linewidth=2)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    _finish(fig, path, "Out-of-Sample Drawdowns")


def plot_weights(weights_a: pd.Series, weights_b: pd.Series, sector_map: dict, path: Path):
    tickers = sorted(set(weights_a.index) | set(weights_b.index))
    df = pd.DataFrame({
        "Portfolio A": weights_a.reindex(tickers).fillna(0),
        "Portfolio B": weights_b.reindex(tickers).fillna(0),
    })
    df = df[(df["Portfolio A"] > 1e-4) | (df["Portfolio B"] > 1e-4)]
    df["sector"] = [sector_map.get(t, "") for t in df.index]
    df = df.sort_values(["sector", "Portfolio A"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = range(len(df))
    width = 0.4
    ax.bar([i - width / 2 for i in x], df["Portfolio A"] * 100, width, label="Portfolio A (MVO)")
    ax.bar([i + width / 2 for i in x], df["Portfolio B"] * 100, width, label="Portfolio B (ESG-screened)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df.index, rotation=90, fontsize=8)
    ax.set_ylabel("Weight (%)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, axis="y")
    _finish(fig, path, "Portfolio Weights by Name: A vs. B")


def plot_sector_exposure(weights_a: pd.Series, weights_b: pd.Series, sector_map: dict,
                          universe_tickers: list[str], market_caps: pd.Series | None, path: Path):
    def sector_weights(weights: pd.Series) -> pd.Series:
        s = pd.Series(0.0, index=sorted(set(sector_map.values())))
        for t, w in weights.items():
            if t in sector_map:
                s[sector_map[t]] += w
        return s

    ew = pd.Series(1.0 / len(universe_tickers), index=universe_tickers)
    ew_sector = sector_weights(ew)

    if market_caps is not None and market_caps.notna().sum() > 0:
        cw = market_caps.reindex(universe_tickers).fillna(0)
        cw = cw / cw.sum()
    else:
        cw = ew
    cw_sector = sector_weights(cw)

    df = pd.DataFrame({
        "Portfolio A": sector_weights(weights_a),
        "Portfolio B": sector_weights(weights_b),
        "Equal-Weight": ew_sector,
        "Cap-Weighted": cw_sector,
    }) * 100

    fig, ax = plt.subplots(figsize=FIGSIZE)
    df.plot(kind="bar", ax=ax)
    ax.set_ylabel("Sector weight (%)")
    ax.set_xlabel("")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _finish(fig, path, "Sector Exposure: Portfolios vs. Benchmarks")

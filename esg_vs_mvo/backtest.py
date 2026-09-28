"""
Split years 1-4 (in-sample / estimation) vs. year 5 (out-of-sample / test),
build the two optimized portfolios and the two benchmarks, and produce
daily return series for all four lines over both windows.

No look-ahead: the optimizer in `optimize.py` only ever sees the in-sample
return matrix. Out-of-sample weights are frozen and simply applied
(buy-and-hold) to the year-5 returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg
from .optimize import annualize_returns, covariance_matrix, max_sharpe_portfolio


@dataclass
class BacktestResult:
    weights_a: pd.Series
    weights_b: pd.Series
    returns_in_sample: pd.DataFrame   # columns: A, B, EW, CW
    returns_out_sample: pd.DataFrame  # columns: A, B, EW, CW
    b_universe: list[str]


def split_dates(prices: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return (start, split, end) where [start, split) is in-sample and
    [split, end] is out-of-sample, based on calendar years back from the
    last available date."""
    end = prices.index.max()
    split = end - pd.DateOffset(years=cfg.OUT_SAMPLE_YEARS)
    start = end - pd.DateOffset(years=cfg.LOOKBACK_YEARS)
    return start, split, end


def portfolio_daily_returns(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    tickers = [t for t in weights.index if t in returns.columns]
    w = weights[tickers]
    return (returns[tickers] * w.values).sum(axis=1)


def equal_weight_returns(returns: pd.DataFrame) -> pd.Series:
    n = returns.shape[1]
    return returns.mean(axis=1)  # equivalent to equal weights summed


def cap_weighted_benchmark_returns(bench_prices: pd.Series | None,
                                    fallback_returns: pd.DataFrame,
                                    market_caps: pd.Series | None) -> tuple[pd.Series, bool]:
    """
    Prefer a real index series (^SBF120 / ^FCHI). If unavailable, build a
    cap-weighted return series from the universe's own (current) market
    caps. Returns (series, used_fallback_flag).
    """
    if bench_prices is not None and bench_prices.notna().sum() > 0:
        r = bench_prices.pct_change().dropna()
        return r, False

    if market_caps is not None and market_caps.notna().sum() > 0:
        w = market_caps.reindex(fallback_returns.columns).fillna(0)
        w = w / w.sum()
        r = (fallback_returns * w.values).sum(axis=1)
        return r, True

    # last resort: equal weight, flagged
    return equal_weight_returns(fallback_returns), True


def run_backtest(
    prices: pd.DataFrame,
    esg_df: pd.DataFrame,
    sector_map: dict[str, str],
    bench_prices: pd.Series | None,
    market_caps: pd.Series | None,
) -> BacktestResult:
    start, split, end = split_dates(prices)

    all_returns = prices.pct_change().dropna(how="all")
    in_sample_returns = all_returns.loc[start:split].dropna(axis=0, how="any")
    out_sample_returns = all_returns.loc[split:end].dropna(axis=0, how="any")

    # --- Portfolio A: full universe, no ESG input --------------------------
    mu_a = annualize_returns(in_sample_returns)
    cov_a = covariance_matrix(in_sample_returns)
    weights_a = max_sharpe_portfolio(mu_a, cov_a, sector_map)

    # --- Portfolio B: drop bottom quartile by ESG (highest risk score) -----
    scored = esg_df.dropna(subset=["esg_score"]).copy()
    n_drop = int(np.floor(len(scored) * cfg.ESG_BOTTOM_QUARTILE))
    # sort descending by risk score -> highest risk (worst) first -> drop them
    worst = scored.sort_values("esg_score", ascending=False).head(n_drop)["ticker"].tolist()
    b_universe = [t for t in in_sample_returns.columns if t not in worst]

    in_sample_b = in_sample_returns[b_universe]
    mu_b = annualize_returns(in_sample_b)
    cov_b = covariance_matrix(in_sample_b)
    sector_map_b = {t: sector_map[t] for t in b_universe}
    weights_b = max_sharpe_portfolio(mu_b, cov_b, sector_map_b)

    # --- Benchmarks ----------------------------------------------------
    ew_returns_in = equal_weight_returns(in_sample_returns)
    ew_returns_out = equal_weight_returns(out_sample_returns)

    bench_in = bench_prices.reindex(in_sample_returns.index) if bench_prices is not None else None
    bench_out = bench_prices.reindex(out_sample_returns.index) if bench_prices is not None else None
    cw_returns_in, _ = cap_weighted_benchmark_returns(bench_in, in_sample_returns, market_caps)
    cw_returns_out, used_fallback = cap_weighted_benchmark_returns(bench_out, out_sample_returns, market_caps)

    # --- Assemble the 4 lines, in-sample and out-of-sample ------------------
    a_in = portfolio_daily_returns(weights_a, in_sample_returns)
    a_out = portfolio_daily_returns(weights_a, out_sample_returns)
    b_in = portfolio_daily_returns(weights_b, in_sample_returns[b_universe])
    b_out = portfolio_daily_returns(weights_b, out_sample_returns[b_universe])

    returns_in_sample = pd.DataFrame({
        "Portfolio A (MVO)": a_in, "Portfolio B (ESG-screened MVO)": b_in,
        "Equal-Weight": ew_returns_in, "Cap-Weighted Benchmark": cw_returns_in,
    })
    returns_out_sample = pd.DataFrame({
        "Portfolio A (MVO)": a_out, "Portfolio B (ESG-screened MVO)": b_out,
        "Equal-Weight": ew_returns_out, "Cap-Weighted Benchmark": cw_returns_out,
    }).dropna()

    return BacktestResult(
        weights_a=weights_a, weights_b=weights_b,
        returns_in_sample=returns_in_sample, returns_out_sample=returns_out_sample,
        b_universe=b_universe,
    )

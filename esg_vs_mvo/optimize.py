"""
Covariance estimation and constrained max-Sharpe mean-variance optimization.

Both portfolios (A: full universe, B: ESG-screened universe) go through the
exact same optimizer and the exact same per-name / per-sector constraints —
the only difference is which tickers are in the input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from . import config as cfg


def annualize_returns(daily_returns: pd.DataFrame) -> pd.Series:
    return daily_returns.mean() * 252


def annualize_vol(daily_returns: pd.DataFrame) -> pd.Series:
    return daily_returns.std() * np.sqrt(252)


def covariance_matrix(daily_returns: pd.DataFrame, method: str = cfg.COV_METHOD) -> pd.DataFrame:
    """Annualized covariance matrix, with optional Ledoit-Wolf shrinkage."""
    sample_cov = daily_returns.cov() * 252
    if method == "sample":
        return sample_cov

    lw = LedoitWolf().fit(daily_returns.values)
    shrunk = pd.DataFrame(lw.covariance_ * 252, index=daily_returns.columns,
                           columns=daily_returns.columns)
    return shrunk


def max_sharpe_portfolio(
    mu: pd.Series,
    cov: pd.DataFrame,
    sector_map: dict[str, str],
    risk_free_rate: float = cfg.RISK_FREE_RATE,
    max_weight_per_name: float = cfg.MAX_WEIGHT_PER_NAME,
    max_weight_per_sector: float = cfg.MAX_WEIGHT_PER_SECTOR,
) -> pd.Series:
    """
    Long-only, fully-invested max-Sharpe optimization with per-name and
    per-sector caps, via PyPortfolioOpt's EfficientFrontier.
    """
    from pypfopt import EfficientFrontier
    from pypfopt.exceptions import OptimizationError

    tickers = list(mu.index)
    sector_per_ticker = {t: sector_map[t] for t in tickers}
    sector_upper = {s: max_weight_per_sector for s in set(sector_per_ticker.values())}
    sector_lower = {s: 0.0 for s in set(sector_per_ticker.values())}

    ef = EfficientFrontier(mu, cov, weight_bounds=(0, max_weight_per_name))
    ef.add_sector_constraints(sector_per_ticker, sector_lower, sector_upper)

    try:
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    except OptimizationError:
        # Infrequent numerical edge case (e.g. all-negative excess returns):
        # fall back to min-volatility so the pipeline still produces a result.
        ef = EfficientFrontier(mu, cov, weight_bounds=(0, max_weight_per_name))
        ef.add_sector_constraints(sector_per_ticker, sector_lower, sector_upper)
        ef.min_volatility()

    weights = pd.Series(ef.clean_weights())
    return weights

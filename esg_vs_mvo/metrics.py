"""
Performance statistics computed from a daily return series (or an equity
curve). Kept dependency-free (numpy/pandas only) and unit-tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def cumulative_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    n = len(returns)
    if n == 0:
        return float("nan")
    total = (1 + returns).prod()
    years = n / periods_per_year
    return float(total ** (1 / years) - 1) if years > 0 else float("nan")


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = cfg.RISK_FREE_RATE,
                  periods_per_year: int = 252) -> float:
    excess = returns - risk_free_rate / periods_per_year
    vol = returns.std(ddof=1) * np.sqrt(periods_per_year)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float(excess.mean() * periods_per_year / vol)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = cfg.RISK_FREE_RATE,
                   periods_per_year: int = 252) -> float:
    excess = returns - risk_free_rate / periods_per_year
    downside = returns[returns < 0]
    dvol = downside.std(ddof=1) * np.sqrt(periods_per_year) if len(downside) > 1 else np.nan
    if not dvol or np.isnan(dvol) or dvol == 0:
        return float("nan")
    return float(excess.mean() * periods_per_year / dvol)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    mdd = max_drawdown(returns)
    ar = annualized_return(returns, periods_per_year)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(ar / abs(mdd))


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    return equity / running_max - 1


def summarize(returns: pd.Series, risk_free_rate: float = cfg.RISK_FREE_RATE) -> dict:
    return {
        "cumulative_return": cumulative_return(returns),
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns),
    }

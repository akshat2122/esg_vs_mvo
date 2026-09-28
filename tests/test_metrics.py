import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esg_vs_mvo.metrics import (
    annualized_return, annualized_volatility, cumulative_return,
    max_drawdown, sharpe_ratio, calmar_ratio, drawdown_series,
)


def test_cumulative_return_simple():
    r = pd.Series([0.1, 0.1])  # +10% then +10% = 21%
    assert cumulative_return(r) == pytest.approx(0.21, abs=1e-9)


def test_cumulative_return_zero():
    r = pd.Series([0.0, 0.0, 0.0])
    assert cumulative_return(r) == pytest.approx(0.0)


def test_annualized_return_matches_daily_compounding():
    # constant daily return of x over 252 days should annualize to (1+x)^252 - 1
    x = 0.0005
    r = pd.Series([x] * 252)
    expected = (1 + x) ** 252 - 1
    assert annualized_return(r) == pytest.approx(expected, rel=1e-6)


def test_annualized_volatility_scales_by_sqrt_252():
    np.random.seed(0)
    r = pd.Series(np.random.normal(0, 0.01, 1000))
    assert annualized_volatility(r) == pytest.approx(r.std() * np.sqrt(252), rel=1e-9)


def test_max_drawdown_known_path():
    # Prices: 100 -> 110 -> 88 -> 99  => drawdown from 110 to 88 = -20%
    prices = pd.Series([100, 110, 88, 99])
    returns = prices.pct_change().dropna()
    assert max_drawdown(returns) == pytest.approx(-0.2, abs=1e-9)


def test_max_drawdown_monotonic_up_is_zero():
    prices = pd.Series([100, 101, 102, 103])
    returns = prices.pct_change().dropna()
    assert max_drawdown(returns) == pytest.approx(0.0, abs=1e-9)


def test_drawdown_series_never_positive():
    np.random.seed(1)
    r = pd.Series(np.random.normal(0.0002, 0.01, 500))
    dd = drawdown_series(r)
    assert (dd <= 1e-12).all()


def test_sharpe_ratio_zero_vol_is_nan():
    r = pd.Series([0.0, 0.0, 0.0])
    assert np.isnan(sharpe_ratio(r))


def test_sharpe_ratio_positive_for_strong_positive_returns():
    r = pd.Series([0.001] * 252)
    assert sharpe_ratio(r, risk_free_rate=0.0) > 0


def test_calmar_ratio_nan_when_no_drawdown():
    r = pd.Series([0.0] * 10)
    assert np.isnan(calmar_ratio(r)) or calmar_ratio(r) == 0

"""
Entry point: runs the full pipeline end to end.

    python -m esg_vs_mvo.main [--refresh]

--refresh forces a fresh price download instead of using the parquet cache.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from . import config as cfg
from .backtest import run_backtest
from .data import download_prices, get_esg_scores, screen_universe
from .metrics import summarize
from .plots import plot_cumulative_returns, plot_drawdowns, plot_sector_exposure, plot_weights


def _fetch_market_caps(tickers: list[str]) -> pd.Series:
    caps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            caps[t] = info.get("market_cap") or info.get("marketCap")
        except Exception:
            caps[t] = np.nan
    return pd.Series(caps, dtype="float64")


def _fetch_benchmark_series() -> pd.Series | None:
    end = cfg.END_DATE
    start = end.replace(year=end.year - cfg.LOOKBACK_YEARS)
    for symbol in cfg.SBF120_TICKER_CANDIDATES:
        try:
            hist = yf.download(symbol, start=start.isoformat(), end=end.isoformat(),
                                auto_adjust=True, progress=False)
            if hist is None or hist.empty:
                print(f"[data] benchmark {symbol} returned no data; trying next candidate")
                continue
            close = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 0]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if close.notna().sum() > 100:
                print(f"[data] using {symbol} as the cap-weighted benchmark index")
                return close
        except Exception as e:
            print(f"[data] could not fetch benchmark {symbol}: {e}")
    print("[data] no benchmark index series available; will build a cap-weighted "
          "benchmark from universe market caps instead (see README limitations).")
    return None


def main():
    parser = argparse.ArgumentParser(description="ESG-tilted vs. standard MVO portfolio comparison")
    parser.add_argument("--refresh", action="store_true", help="force re-download of price data")
    args = parser.parse_args()

    np.random.seed(cfg.RANDOM_SEED)
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(cfg.DATA_DIR).mkdir(parents=True, exist_ok=True)

    log_lines = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    log(f"=== ESG vs. MVO backtest run: {datetime.now().isoformat()} ===")
    log(f"Universe size (candidates): {len(cfg.UNIVERSE)}")
    log(f"Lookback: {cfg.LOOKBACK_YEARS}y | In-sample: {cfg.IN_SAMPLE_YEARS}y | "
        f"Out-of-sample: {cfg.OUT_SAMPLE_YEARS}y")
    log(f"Risk-free rate: {cfg.RISK_FREE_RATE:.2%} | Cov method: {cfg.COV_METHOD}")

    # 1. Prices + screen
    prices = download_prices(refresh=args.refresh)
    universe_df = screen_universe(prices)
    included = universe_df[universe_df["included"] == "Y"]["ticker"].tolist()
    log(f"Eligible names after screening: {len(included)}/{len(cfg.UNIVERSE)}")
    prices = prices[included].dropna(how="any")
    sector_map = {t: cfg.UNIVERSE[t]["sector"] for t in included}

    # 2. ESG
    esg_df = get_esg_scores(included)

    # 3. Benchmarks
    bench_series = _fetch_benchmark_series()
    market_caps = _fetch_market_caps(included)
    if market_caps.notna().sum() == 0:
        log("[data] WARNING: could not fetch market caps for any name; cap-weighted "
            "benchmark will fall back further (see cap_weighted_benchmark_returns).")

    # 4. Backtest (in-sample optimization, out-of-sample test)
    result = run_backtest(prices, esg_df, sector_map, bench_series, market_caps)

    dropped_from_b = sorted(set(included) - set(result.b_universe))
    log(f"Portfolio B universe: {len(result.b_universe)}/{len(included)} names "
        f"(dropped bottom-quartile ESG risk + any still-missing scores): {dropped_from_b}")

    # 5. Metrics
    rows = []
    for period_name, returns in [("In-Sample", result.returns_in_sample),
                                  ("Out-of-Sample", result.returns_out_sample)]:
        for col in returns.columns:
            stats = summarize(returns[col])
            stats.update({"period": period_name, "line": col})
            rows.append(stats)
    table = pd.DataFrame(rows)[["period", "line", "cumulative_return", "annualized_return",
                                 "annualized_volatility", "sharpe_ratio", "sortino_ratio",
                                 "max_drawdown", "calmar_ratio"]]
    table.to_csv(Path(cfg.OUTPUT_DIR) / "performance_table.csv", index=False)

    pct_cols = ["cumulative_return", "annualized_return", "annualized_volatility", "max_drawdown"]
    fmt_table = table.copy()
    for c in pct_cols:
        fmt_table[c] = (fmt_table[c] * 100).round(2).astype(str) + "%"
    for c in ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]:
        fmt_table[c] = fmt_table[c].round(2)
    with open(Path(cfg.OUTPUT_DIR) / "performance_table.md", "w") as f:
        f.write(fmt_table.to_markdown(index=False))

    log("\n=== Out-of-sample performance ===")
    log(fmt_table[fmt_table["period"] == "Out-of-Sample"].to_string(index=False))

    # 6. Weights
    result.weights_a[result.weights_a > 1e-4].to_frame("weight").to_csv(
        Path(cfg.OUTPUT_DIR) / "weights_a.csv")
    result.weights_b[result.weights_b > 1e-4].to_frame("weight").to_csv(
        Path(cfg.OUTPUT_DIR) / "weights_b.csv")
    weights_combined = pd.DataFrame({
        "weight_a": result.weights_a, "weight_b": result.weights_b.reindex(result.weights_a.index).fillna(0),
    })
    weights_combined.to_csv(Path(cfg.OUTPUT_DIR) / "weights.csv")

    # 7. Plots
    plot_cumulative_returns(result.returns_out_sample, Path(cfg.OUTPUT_DIR) / "cumulative_returns.png")
    plot_drawdowns(result.returns_out_sample, Path(cfg.OUTPUT_DIR) / "drawdowns.png")
    plot_weights(result.weights_a, result.weights_b, sector_map, Path(cfg.OUTPUT_DIR) / "weights.png")
    plot_sector_exposure(result.weights_a, result.weights_b, sector_map, included, market_caps,
                          Path(cfg.OUTPUT_DIR) / "sector_exposure.png")

    log(f"\nDate window: {prices.index.min().date()} -> {prices.index.max().date()}")
    log("All outputs written to ./outputs/")

    with open(Path(cfg.OUTPUT_DIR) / "run_log.txt", "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    sys.exit(main())

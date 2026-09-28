"""
Data layer: price download + caching, universe eligibility screening, and
ESG score retrieval with an explicit, logged fallback path.

Nothing here silently drops a name. Every exclusion is written to
outputs/universe.csv or outputs/esg_coverage.csv with a reason.
"""
from __future__ import annotations

import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from . import config as cfg

warnings.filterwarnings("ignore", category=FutureWarning)


def _date_window() -> tuple[date, date]:
    end = cfg.END_DATE
    start = end - timedelta(days=int(365.25 * cfg.LOOKBACK_YEARS))
    return start, end


def download_prices(refresh: bool = False) -> pd.DataFrame:
    """Download 5y daily adjusted close for the universe, with parquet caching."""
    cache_path = Path(cfg.DATA_DIR) / "prices.parquet"
    if cache_path.exists() and not refresh:
        prices = pd.read_parquet(cache_path)
        print(f"[data] loaded cached prices from {cache_path} ({prices.shape[0]} rows, "
              f"{prices.shape[1]} tickers)")
        return prices

    start, end = _date_window()
    tickers = list(cfg.UNIVERSE.keys())
    print(f"[data] downloading {len(tickers)} tickers from {start} to {end} ...")
    raw = yf.download(
        tickers, start=start.isoformat(), end=end.isoformat(),
        auto_adjust=True, progress=False, group_by="ticker", threads=True,
    )

    prices = pd.DataFrame(index=raw.index)
    for t in tickers:
        try:
            col = raw[t]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
            prices[t] = col
        except KeyError:
            prices[t] = np.nan

    prices = prices.sort_index()
    Path(cfg.DATA_DIR).mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cache_path)
    print(f"[data] cached prices -> {cache_path}")
    return prices


def screen_universe(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the 3-year-history / <5%-missing-days screen. Returns a DataFrame
    (also written to outputs/universe.csv) with one row per candidate ticker
    and an explicit inclusion decision + reason.
    """
    rows = []
    min_obs = int(252 * cfg.MIN_HISTORY_YEARS)
    total_days = prices.shape[0]

    for t, meta in cfg.UNIVERSE.items():
        series = prices[t] if t in prices.columns else pd.Series(dtype=float)
        first_valid = series.first_valid_index()
        n_obs = series.notna().sum()
        missing_frac = 1 - (n_obs / total_days) if total_days else 1.0

        included, reason = True, ""
        if first_valid is None or n_obs == 0:
            included, reason = False, "no price data returned by yfinance"
        elif n_obs < min_obs:
            included, reason = False, (
                f"only {n_obs} obs (~{n_obs/252:.1f}y), need >= {cfg.MIN_HISTORY_YEARS}y"
            )
        elif missing_frac > cfg.MAX_MISSING_DAY_FRACTION:
            included, reason = False, f"{missing_frac:.1%} missing trading days (> 5% cap)"

        rows.append({
            "ticker": t,
            "name": meta["name"],
            "sector": meta["sector"],
            "first_date": first_valid.date().isoformat() if first_valid is not None else "",
            "n_obs": int(n_obs),
            "included": "Y" if included else "N",
            "exclusion_reason": reason,
        })

    df = pd.DataFrame(rows)
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(cfg.OUTPUT_DIR) / "universe.csv", index=False)

    excluded = df[df["included"] == "N"]
    if not excluded.empty:
        print(f"[data] excluded {len(excluded)} names from the universe:")
        for _, r in excluded.iterrows():
            print(f"    - {r['ticker']} ({r['name']}): {r['exclusion_reason']}")
    return df


def fetch_esg_scores(tickers: list[str]) -> pd.DataFrame:
    """
    Attempt to pull the Sustainalytics ESG Risk Score (lower = better) for
    each ticker via yfinance. Every ticker is logged with a status, whether
    or not a score was found — nothing is dropped silently here.
    """
    rows = []
    for t in tickers:
        status, score, reason = "missing", np.nan, ""
        try:
            tk = yf.Ticker(t)
            sust = tk.sustainability
            if sust is not None and not sust.empty and cfg.ESG_FIELD in sust.index:
                score = float(sust.loc[cfg.ESG_FIELD].iloc[0])
                status = "ok"
            else:
                reason = "yfinance sustainability field empty/unavailable for this ticker"
        except Exception as e:  # yfinance raises a mix of exception types
            reason = f"yfinance error: {e}"
        rows.append({
            "ticker": t, "sector": cfg.UNIVERSE[t]["sector"],
            "esg_score": score, "status": status, "reason": reason,
        })
    return pd.DataFrame(rows)


def _load_manual_override() -> pd.DataFrame:
    path = Path(cfg.MANUAL_ESG_OVERRIDE_PATH)
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "esg_score", "source", "as_of_date"])
    return pd.read_csv(path)


def get_esg_scores(tickers: list[str]) -> pd.DataFrame:
    """
    Full ESG resolution pipeline:
      1. Try yfinance for every ticker.
      2. Log full coverage to outputs/esg_coverage.csv.
      3. If live coverage >= threshold: use live scores, drop the few
         missing names from the ESG-aware portfolio only (with a warning).
      4. If live coverage < threshold: fall back to data/esg_manual.csv.
         If that file doesn't cover enough names either, stop with a clear
         message rather than silently proceeding on partial data.
    Returns a DataFrame: ticker, sector, esg_score, source, status.
    """
    live = fetch_esg_scores(tickers)
    live_coverage = (live["status"] == "ok").mean()
    print(f"[esg] live yfinance ESG coverage: {live_coverage:.0%} "
          f"({(live['status'] == 'ok').sum()}/{len(live)} names)")

    manual = _load_manual_override()
    manual_map = dict(zip(manual.get("ticker", []), manual.get("esg_score", [])))
    manual_source_map = dict(zip(manual.get("ticker", []), manual.get("source", [])))

    resolved = live.copy()
    resolved["source"] = np.where(resolved["status"] == "ok", "yfinance/Sustainalytics", "")

    if live_coverage < cfg.ESG_COVERAGE_MIN_FRACTION:
        print(f"[esg] live coverage ({live_coverage:.0%}) is below the "
              f"{cfg.ESG_COVERAGE_MIN_FRACTION:.0%} threshold. Falling back to "
              f"manual override file: {cfg.MANUAL_ESG_OVERRIDE_PATH}")
        for i, row in resolved.iterrows():
            if row["status"] != "ok" and row["ticker"] in manual_map:
                resolved.at[i, "esg_score"] = manual_map[row["ticker"]]
                resolved.at[i, "status"] = "manual_override"
                resolved.at[i, "source"] = manual_source_map.get(row["ticker"], "manual override")
                resolved.at[i, "reason"] = "filled from data/esg_manual.csv"

    combined_coverage = resolved["esg_score"].notna().mean()
    resolved.to_csv(Path(cfg.OUTPUT_DIR) / "esg_coverage.csv", index=False)

    if combined_coverage < cfg.ESG_COVERAGE_MIN_FRACTION:
        missing = resolved[resolved["esg_score"].isna()]["ticker"].tolist()
        print(
            "\n[esg] STOPPING: ESG coverage after live pull + manual override is only "
            f"{combined_coverage:.0%}, below the required {cfg.ESG_COVERAGE_MIN_FRACTION:.0%}.\n"
            f"Missing scores for: {', '.join(missing)}\n"
            f"Add rows (ticker, esg_score, source, as_of_date) for these names to "
            f"{cfg.MANUAL_ESG_OVERRIDE_PATH} and re-run.\n"
        )
        sys.exit(1)

    still_missing = resolved[resolved["esg_score"].isna()]
    if not still_missing.empty:
        print(f"[esg] WARNING: {len(still_missing)} names still have no ESG score and will be "
              f"kept in Portfolio A but dropped from Portfolio B's eligible set:")
        for _, r in still_missing.iterrows():
            print(f"    - {r['ticker']}")

    return resolved

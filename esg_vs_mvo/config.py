"""
Static configuration for the ESG vs. MVO project: the investable universe,
sector map, date window, optimizer constraints and the risk-free rate.

Everything that is a *choice* (which stocks, which constraints, which rate)
lives here so the rest of the pipeline stays free of magic numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# ---------------------------------------------------------------------------
# Universe: SBF 120 constituents, Yahoo Finance tickers (Euronext Paris, ".PA")
# Sector labels are a simplified GICS-style mapping maintained by hand because
# yfinance's own "sector" field is frequently missing or inconsistent for
# European tickers. This is deliberately coarse (8 buckets) — good enough to
# enforce diversification, not a substitute for a licensed classification.
# ---------------------------------------------------------------------------
UNIVERSE: dict[str, dict[str, str]] = {
    # ticker:      name,                          sector
    "MC.PA":  {"name": "LVMH",                    "sector": "Consumer Discretionary"},
    "RMS.PA": {"name": "Hermes International",     "sector": "Consumer Discretionary"},
    "KER.PA": {"name": "Kering",                   "sector": "Consumer Discretionary"},
    "RNO.PA": {"name": "Renault",                  "sector": "Consumer Discretionary"},

    "OR.PA":  {"name": "L'Oreal",                  "sector": "Consumer Staples"},
    "BN.PA":  {"name": "Danone",                   "sector": "Consumer Staples"},
    "RI.PA":  {"name": "Pernod Ricard",             "sector": "Consumer Staples"},

    "BNP.PA": {"name": "BNP Paribas",               "sector": "Financials"},
    "GLE.PA": {"name": "Societe Generale",          "sector": "Financials"},
    "ACA.PA": {"name": "Credit Agricole",           "sector": "Financials"},
    "CS.PA":  {"name": "AXA",                       "sector": "Financials"},
    "AMUN.PA":{"name": "Amundi",                    "sector": "Financials"},

    "AIR.PA": {"name": "Airbus",                    "sector": "Industrials"},
    "SAF.PA": {"name": "Safran",                    "sector": "Industrials"},
    "SU.PA":  {"name": "Schneider Electric",        "sector": "Industrials"},
    "LR.PA":  {"name": "Legrand",                   "sector": "Industrials"},
    "VIE.PA": {"name": "Veolia Environnement",      "sector": "Industrials"},

    "SGO.PA": {"name": "Saint-Gobain",              "sector": "Materials"},
    "AI.PA":  {"name": "Air Liquide",               "sector": "Materials"},
    "ML.PA":  {"name": "Michelin",                  "sector": "Materials"},

    "SAN.PA": {"name": "Sanofi",                    "sector": "Health Care"},
    "EL.PA":  {"name": "EssilorLuxottica",          "sector": "Health Care"},
    "IPN.PA": {"name": "Ipsen",                     "sector": "Health Care"},

    "CAP.PA": {"name": "Capgemini",                 "sector": "Information Technology"},
    "DSY.PA": {"name": "Dassault Systemes",         "sector": "Information Technology"},
    "STMPA.PA": {"name": "STMicroelectronics",      "sector": "Information Technology"},

    "PUB.PA": {"name": "Publicis Groupe",           "sector": "Communication Services"},
    "ORA.PA": {"name": "Orange",                    "sector": "Communication Services"},
}

# Cap-weighted benchmark index (yfinance symbol). If this series is not
# available at run time, data.py falls back to building a cap-weighted
# benchmark from the universe's own market caps (see README limitations).
BENCHMARK_INDEX_TICKER = "^FCHI"  # CAC 40 used as a liquid proxy if SBF120 series is unavailable
SBF120_TICKER_CANDIDATES = ["^SBF120", "^FCHI"]

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
END_DATE = date.today()
LOOKBACK_YEARS = 5
# In-sample = years 1-4 (estimation window), out-of-sample = year 5 (test window)
IN_SAMPLE_YEARS = 4
OUT_SAMPLE_YEARS = 1

MIN_HISTORY_YEARS = 3          # minimum listed history required to be eligible
MAX_MISSING_DAY_FRACTION = 0.05  # exclude if >5% of trading days are missing

# ---------------------------------------------------------------------------
# ESG
# ---------------------------------------------------------------------------
# yfinance exposes Sustainalytics' "ESG Risk Score" (totalEsg): LOWER = LESS
# ESG RISK = BETTER. This is a risk score, not a "goodness" score.
ESG_FIELD = "totalEsg"
ESG_COVERAGE_MIN_FRACTION = 0.70   # below this, the pipeline halts (see data.py)
ESG_BOTTOM_QUARTILE = 0.25         # fraction of highest-risk names excluded from Portfolio B
MANUAL_ESG_OVERRIDE_PATH = "data/esg_manual.csv"

# ---------------------------------------------------------------------------
# Optimizer constraints (both portfolios)
# ---------------------------------------------------------------------------
MAX_WEIGHT_PER_NAME = 0.20
MAX_WEIGHT_PER_SECTOR = 0.35
RISK_FREE_RATE = 0.03  # annualized, fixed & documented (approx. average €STR over the window)

# Covariance shrinkage
COV_METHOD = "ledoit_wolf"  # "ledoit_wolf" or "sample"

RANDOM_SEED = 42

OUTPUT_DIR = "outputs"
DATA_DIR = "data"


@dataclass(frozen=True)
class RunConfig:
    universe: dict = field(default_factory=lambda: UNIVERSE)
    end_date: date = END_DATE
    lookback_years: int = LOOKBACK_YEARS
    in_sample_years: int = IN_SAMPLE_YEARS
    out_sample_years: int = OUT_SAMPLE_YEARS
    min_history_years: int = MIN_HISTORY_YEARS
    max_missing_day_fraction: float = MAX_MISSING_DAY_FRACTION
    esg_coverage_min_fraction: float = ESG_COVERAGE_MIN_FRACTION
    esg_bottom_quartile: float = ESG_BOTTOM_QUARTILE
    max_weight_per_name: float = MAX_WEIGHT_PER_NAME
    max_weight_per_sector: float = MAX_WEIGHT_PER_SECTOR
    risk_free_rate: float = RISK_FREE_RATE
    cov_method: str = COV_METHOD
    random_seed: int = RANDOM_SEED

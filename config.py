"""
Global configuration for the stock analyzer.

Centralizing tunables here lets you adjust behavior without touching engine code.
Every weight, threshold, and lookback should be reviewed for your strategy and
re-validated with backtests before being trusted.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
CHARTS_DIR: Path = PROJECT_ROOT / "charts"
for _d in (DATA_DIR, REPORTS_DIR, CHARTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------
# Yahoo Finance ticker conventions:
#   US:    AAPL, MSFT
#   Saudi: 2222.SR (Aramco), 1120.SR (Al Rajhi Bank)
MARKET_SUFFIX: Dict[str, str] = {
    "US": "",
    "SAUDI": ".SR",
    "GLOBAL": "",
}

BENCHMARKS: Dict[str, str] = {
    "US": "^GSPC",        # S&P 500
    "US_TECH": "^IXIC",   # Nasdaq Composite
    "US_DOW": "^DJI",
    "SAUDI": "^TASI.SR",  # TASI — yfinance coverage is incomplete; falls back if missing
    "GLOBAL": "^GSPC",
}

DEFAULT_BENCHMARK_PER_MARKET: Dict[str, str] = {
    "US": "^GSPC",
    "SAUDI": "^TASI.SR",
    "GLOBAL": "^GSPC",
}

# Curated screener universes. Replace with full S&P 500 / TASI lists as needed.
DEFAULT_UNIVERSE: Dict[str, List[str]] = {
    "US": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO",
        "JPM", "V", "MA", "UNH", "JNJ", "PG", "HD", "XOM", "CVX", "WMT",
        "KO", "PEP", "COST", "MRK", "LLY", "ABBV", "ORCL", "CRM", "ADBE",
        "NFLX", "AMD", "INTC", "QCOM", "CSCO", "TXN", "IBM", "GE", "BA",
        "CAT", "DE", "GS", "MS", "BAC", "WFC", "T", "VZ", "DIS", "MCD",
    ],
    "SAUDI": [
        # Major Tadawul names (subject to yfinance availability)
        "2222.SR",  # Saudi Aramco
        "1120.SR",  # Al Rajhi Bank
        "2010.SR",  # SABIC
        "7010.SR",  # STC
        "1180.SR",  # SNB
        "2350.SR",  # Saudi Kayan
        "4030.SR",  # Bahri
        "4002.SR",  # Mouwasat
        "1211.SR",  # Maaden
        "2380.SR",  # Petro Rabigh
    ],
}


# ---------------------------------------------------------------------------
# Technical analysis parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IndicatorConfig:
    sma_periods: tuple = (20, 50, 100, 200)
    ema_periods: tuple = (20, 50, 100, 200)
    rsi_period: int = 14
    rsi_healthy_low: float = 45.0
    rsi_healthy_high: float = 70.0
    rsi_overbought: float = 75.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    swing_lookback: int = 20
    volume_avg_period: int = 20
    volume_breakout_multiple: float = 1.5

INDICATORS = IndicatorConfig()


# ---------------------------------------------------------------------------
# Scoring weights — total = 100
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoringWeights:
    technical_trend: int = 25
    momentum: int = 15
    volume: int = 15
    fundamental: int = 25
    valuation: int = 10
    risk_reward: int = 10

    def total(self) -> int:
        return (self.technical_trend + self.momentum + self.volume
                + self.fundamental + self.valuation + self.risk_reward)

WEIGHTS = ScoringWeights()
assert WEIGHTS.total() == 100, "Scoring weights must sum to 100"

# Classification thresholds on the 0–100 score
CLASSIFICATION_THRESHOLDS: Dict[str, int] = {
    "Strong Candidate":   75,
    "Watchlist Candidate": 60,
    "Neutral":             45,
    "Weak":                30,
    # below 30 => "Avoid"
}


# ---------------------------------------------------------------------------
# Risk management defaults
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskConfig:
    atr_stop_multiplier: float = 1.5      # SL = entry - 1.5 * ATR (long)
    min_risk_reward: float = 2.0          # Reject setups below 1:2 R:R
    max_account_risk_pct: float = 1.0     # Max % of account per trade
    pullback_ema_period: int = 20
    breakout_lookback: int = 20

RISK = RiskConfig()


# ---------------------------------------------------------------------------
# Screener filters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScreenerConfig:
    min_avg_volume: int = 200_000
    min_price: float = 2.0
    max_price: float = 10_000.0
    require_above_50_sma: bool = True
    require_above_200_sma: bool = True
    rsi_min: float = 45.0
    rsi_max: float = 70.0
    min_rel_strength: float = 0.0   # vs benchmark, % over lookback
    rs_lookback_days: int = 63      # ~3 months

SCREENER = ScreenerConfig()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
CACHE_TTL_MINUTES: int = 30
CACHE_DIR: Path = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


DISCLAIMER: str = (
    "DISCLAIMER: This tool is decision-support software for educational and "
    "research purposes. It is not financial advice, not a recommendation to "
    "buy or sell any security, and not a guarantee of future performance. "
    "Markets carry risk of substantial loss. Always perform independent "
    "research and consult a licensed advisor before making investment decisions."
)

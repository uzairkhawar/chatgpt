"""
Technical analysis engine — pure functions over OHLCV DataFrames.

All functions accept a DataFrame with columns: Open, High, Low, Close, Volume
(DatetimeIndex). They return either Series (most indicators) or dicts of
diagnostic values (interpretation helpers).

Design choice: implement indicators directly in NumPy/pandas instead of pulling
in TA-Lib. This avoids a heavy native dependency and keeps the math auditable.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from src.utils import logger

C = config.INDICATORS


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Attach SMA_20/50/100/200 and EMA_20/50/100/200 columns."""
    out = df.copy()
    close = out["Close"]
    for p in C.sma_periods:
        out[f"SMA_{p}"] = sma(close, p)
    for p in C.ema_periods:
        out[f"EMA_{p}"] = ema(close, p)
    return out


def trend_direction(df: pd.DataFrame) -> str:
    """Classify trend by stacking of price vs the major SMAs."""
    last = df.iloc[-1]
    px = last["Close"]
    sma50 = last.get("SMA_50")
    sma200 = last.get("SMA_200")
    if pd.isna(sma50) or pd.isna(sma200):
        return "Insufficient data"
    if px > sma50 > sma200:
        return "Uptrend"
    if px < sma50 < sma200:
        return "Downtrend"
    if sma50 > sma200 and px < sma50:
        return "Uptrend (pullback)"
    if sma50 < sma200 and px > sma50:
        return "Downtrend (rally)"
    return "Sideways / mixed"


def golden_or_death_cross(df: pd.DataFrame, lookback: int = 60) -> Optional[str]:
    """Detect a recent SMA50/SMA200 crossover."""
    if "SMA_50" not in df or "SMA_200" not in df:
        return None
    s50 = df["SMA_50"].dropna()
    s200 = df["SMA_200"].dropna()
    common = s50.index.intersection(s200.index)
    if len(common) < lookback:
        return None
    a = s50.loc[common].iloc[-lookback:]
    b = s200.loc[common].iloc[-lookback:]
    diff = a - b
    sign = np.sign(diff)
    crosses = sign.diff().fillna(0)
    if (crosses > 0).any():
        return "Golden cross (recent)"
    if (crosses < 0).any():
        return "Death cross (recent)"
    return None


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    # Wilder smoothing
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    f = ema(close, fast)
    s = ema(close, slow)
    line = f - s
    sig = ema(line, signal)
    hist = line - sig
    return pd.DataFrame({"MACD": line, "MACD_Signal": sig, "MACD_Hist": hist})


def stochastic_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    r = rsi(close, period)
    lo = r.rolling(period).min()
    hi = r.rolling(period).max()
    return ((r - lo) / (hi - lo).replace(0, np.nan)).fillna(0.5) * 100


def rate_of_change(close: pd.Series, period: int = 12) -> pd.Series:
    return (close / close.shift(period) - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------
def obv(df: pd.DataFrame) -> pd.Series:
    """On-balance volume."""
    direction = np.sign(df["Close"].diff().fillna(0))
    return (direction * df["Volume"]).cumsum()


def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    mfv = mfm.fillna(0) * df["Volume"]
    return mfv.cumsum()


def volume_diagnostics(df: pd.DataFrame) -> Dict:
    avg = df["Volume"].rolling(C.volume_avg_period).mean().iloc[-1]
    last = df["Volume"].iloc[-1]
    ratio = last / avg if avg and not pd.isna(avg) else np.nan
    breakout = bool(ratio is not np.nan and ratio >= C.volume_breakout_multiple)
    # OBV trend
    o = obv(df)
    obv_slope = o.diff(20).iloc[-1]
    return {
        "last_volume": float(last) if not pd.isna(last) else None,
        "avg_volume_20": float(avg) if not pd.isna(avg) else None,
        "volume_ratio": float(ratio) if not pd.isna(ratio) else None,
        "volume_breakout": breakout,
        "obv_trend": "rising" if obv_slope and obv_slope > 0 else ("falling" if obv_slope and obv_slope < 0 else "flat"),
    }


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev).abs(),
                    (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger_bands(close: pd.Series, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return pd.DataFrame({
        "BB_Mid": mid,
        "BB_Upper": mid + k * std,
        "BB_Lower": mid - k * std,
    })


def historical_volatility(close: pd.Series, period: int = 30) -> pd.Series:
    """Annualized realized volatility from daily log returns."""
    rets = np.log(close / close.shift(1))
    return rets.rolling(period).std() * np.sqrt(252) * 100.0


# ---------------------------------------------------------------------------
# Support / resistance via swing pivots
# ---------------------------------------------------------------------------
def swing_pivots(df: pd.DataFrame, lookback: int = 20) -> Tuple[List[float], List[float]]:
    """Identify swing highs (resistance) and swing lows (support)."""
    highs, lows = [], []
    h, l = df["High"], df["Low"]
    n = len(df)
    for i in range(lookback, n - lookback):
        window_h = h.iloc[i - lookback : i + lookback + 1]
        window_l = l.iloc[i - lookback : i + lookback + 1]
        if h.iloc[i] == window_h.max():
            highs.append(float(h.iloc[i]))
        if l.iloc[i] == window_l.min():
            lows.append(float(l.iloc[i]))
    return highs, lows


def nearest_levels(price: float, highs: List[float], lows: List[float]) -> Dict:
    """Find the nearest support below price and nearest resistance above price."""
    res_above = sorted([h for h in highs if h > price])
    sup_below = sorted([l for l in lows if l < price], reverse=True)
    return {
        "nearest_resistance": res_above[0] if res_above else None,
        "next_resistance": res_above[1] if len(res_above) > 1 else None,
        "third_resistance": res_above[2] if len(res_above) > 2 else None,
        "nearest_support": sup_below[0] if sup_below else None,
        "next_support": sup_below[1] if len(sup_below) > 1 else None,
    }


# ---------------------------------------------------------------------------
# Price-action / structure
# ---------------------------------------------------------------------------
def market_structure(df: pd.DataFrame, lookback: int = 20) -> str:
    """Classify by sequence of recent swing highs/lows."""
    highs, lows = swing_pivots(df.tail(200), lookback=max(5, lookback // 2))
    if len(highs) < 2 or len(lows) < 2:
        return "Insufficient structure"
    hh = highs[-1] > highs[-2]
    hl = lows[-1] > lows[-2]
    lh = highs[-1] < highs[-2]
    ll = lows[-1] < lows[-2]
    if hh and hl:
        return "Higher highs / higher lows (uptrend structure)"
    if lh and ll:
        return "Lower highs / lower lows (downtrend structure)"
    return "Mixed / consolidation"


# ---------------------------------------------------------------------------
# Master enricher
# ---------------------------------------------------------------------------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all indicator columns to a copy of df."""
    out = add_moving_averages(df)
    out["RSI"] = rsi(out["Close"], C.rsi_period)
    macd_df = macd(out["Close"], C.macd_fast, C.macd_slow, C.macd_signal)
    out = pd.concat([out, macd_df], axis=1)
    out["StochRSI"] = stochastic_rsi(out["Close"], C.stoch_rsi_period)
    out["ATR"] = atr(out, C.atr_period)
    bb = bollinger_bands(out["Close"], C.bb_period, C.bb_std)
    out = pd.concat([out, bb], axis=1)
    out["HV"] = historical_volatility(out["Close"], 30)
    out["OBV"] = obv(out)
    out["AD"] = accumulation_distribution(out)
    return out


def snapshot(df_enriched: pd.DataFrame) -> Dict:
    """Latest-bar snapshot for the report."""
    if df_enriched is None or df_enriched.empty:
        return {}
    last = df_enriched.iloc[-1]
    px = float(last["Close"])
    s = {
        "price": px,
        "sma_20": float(last.get("SMA_20", np.nan)),
        "sma_50": float(last.get("SMA_50", np.nan)),
        "sma_100": float(last.get("SMA_100", np.nan)),
        "sma_200": float(last.get("SMA_200", np.nan)),
        "ema_20": float(last.get("EMA_20", np.nan)),
        "rsi": float(last.get("RSI", np.nan)),
        "macd": float(last.get("MACD", np.nan)),
        "macd_signal": float(last.get("MACD_Signal", np.nan)),
        "macd_hist": float(last.get("MACD_Hist", np.nan)),
        "stoch_rsi": float(last.get("StochRSI", np.nan)),
        "atr": float(last.get("ATR", np.nan)),
        "bb_upper": float(last.get("BB_Upper", np.nan)),
        "bb_lower": float(last.get("BB_Lower", np.nan)),
        "hv": float(last.get("HV", np.nan)),
        "trend": trend_direction(df_enriched),
        "structure": market_structure(df_enriched),
        "cross": golden_or_death_cross(df_enriched),
    }
    s.update({"volume": volume_diagnostics(df_enriched)})
    highs, lows = swing_pivots(df_enriched, C.swing_lookback)
    s["levels"] = nearest_levels(px, highs, lows)
    return s


def relative_strength(stock_close: pd.Series, bench_close: pd.Series, lookback: int = 63) -> Optional[float]:
    """Stock return minus benchmark return over the lookback window (in %)."""
    try:
        s = stock_close.dropna().iloc[-lookback:]
        b = bench_close.dropna().iloc[-lookback:]
        if len(s) < 5 or len(b) < 5:
            return None
        sr = (s.iloc[-1] / s.iloc[0] - 1) * 100
        br = (b.iloc[-1] / b.iloc[0] - 1) * 100
        return float(sr - br)
    except Exception as e:
        logger.warning("relative_strength failed: %s", e)
        return None

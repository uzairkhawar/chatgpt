"""
Risk management & trade-setup geometry.

Given enriched OHLCV + a snapshot, compute entry zones, stop loss, take-profit
levels, and risk/reward. Entry styles supported:

    - 'breakout'    : entry slightly above recent N-day high
    - 'pullback'    : entry near a key moving average (default EMA 20)
    - 'conservative': entry on confirmed retest of breakout level
    - 'aggressive'  : entry at current price

All numbers are *suggestions* — the user must confirm with their own
discretion and broker constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from src.utils import logger, round_or_none

R = config.RISK


# ---------------------------------------------------------------------------
# Setup data class
# ---------------------------------------------------------------------------
@dataclass
class TradeSetup:
    style: str                  # breakout / pullback / conservative / aggressive
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: Optional[float]
    risk_per_share: float
    reward_per_share_tp1: float
    risk_reward_tp1: float
    risk_reward_tp2: float
    rationale: str
    invalidation: str
    acceptable: bool            # passes minimum R:R?

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _structure_stop(df: pd.DataFrame, lookback: int = 20) -> Optional[float]:
    """Lowest low of the recent window — a structural stop for long setups."""
    if df is None or df.empty:
        return None
    return float(df["Low"].iloc[-lookback:].min())


def _atr_stop(price: float, atr_value: float, mult: float = R.atr_stop_multiplier) -> float:
    return float(price - mult * atr_value)


# ---------------------------------------------------------------------------
# Entry/SL/TP construction
# ---------------------------------------------------------------------------
def build_setups(df_enriched: pd.DataFrame, snap: Dict) -> List[TradeSetup]:
    """Construct a list of plausible long setups from the enriched data."""
    if df_enriched is None or df_enriched.empty:
        return []

    price = float(snap["price"])
    atr_v = float(snap.get("atr") or np.nan)
    levels = snap.get("levels", {}) or {}
    nearest_res = levels.get("nearest_resistance")
    next_res = levels.get("next_resistance")
    third_res = levels.get("third_resistance")
    nearest_sup = levels.get("nearest_support")

    # Recent N-day high for breakout entry
    n = R.breakout_lookback
    recent_high = float(df_enriched["High"].iloc[-n:].max())
    ema20 = float(snap.get("ema_20") or df_enriched["EMA_20"].iloc[-1])
    sma50 = float(snap.get("sma_50") or df_enriched["SMA_50"].iloc[-1])
    structure_stop = _structure_stop(df_enriched, lookback=n) or (price * 0.92)

    setups: List[TradeSetup] = []

    def _make_setup(style: str, entry: float, sl: float, rationale: str,
                    invalidation: str) -> Optional[TradeSetup]:
        if entry <= 0 or sl <= 0 or sl >= entry:
            return None
        # Targets: prefer real swing highs; fallback to measured-move
        risk = entry - sl
        tp1 = nearest_res if (nearest_res and nearest_res > entry) else entry + 2 * risk
        tp2 = next_res if (next_res and next_res > tp1) else entry + 3 * risk
        tp3 = third_res if (third_res and third_res > tp2) else entry + 5 * risk
        rr1 = (tp1 - entry) / risk if risk > 0 else 0
        rr2 = (tp2 - entry) / risk if risk > 0 else 0
        return TradeSetup(
            style=style,
            entry=round(entry, 2),
            stop_loss=round(sl, 2),
            tp1=round(tp1, 2),
            tp2=round(tp2, 2),
            tp3=round(tp3, 2) if tp3 else None,
            risk_per_share=round(risk, 2),
            reward_per_share_tp1=round(tp1 - entry, 2),
            risk_reward_tp1=round(rr1, 2),
            risk_reward_tp2=round(rr2, 2),
            rationale=rationale,
            invalidation=invalidation,
            acceptable=rr1 >= R.min_risk_reward,
        )

    # --- Breakout: entry slightly above recent N-day high
    if recent_high > price:
        bo_entry = recent_high * 1.005
        sl = max(structure_stop, _atr_stop(bo_entry, atr_v) if not np.isnan(atr_v) else 0)
        s = _make_setup(
            "breakout", bo_entry, sl,
            rationale=f"Breakout entry above {n}-day high ({recent_high:.2f}). "
                      "Confirms trend continuation; favoured when volume expands on the break.",
            invalidation=f"Daily close back below {recent_high:.2f} or below {round(sl, 2)}.",
        )
        if s: setups.append(s)

    # --- Pullback: entry near EMA20 if uptrend
    if not np.isnan(ema20) and price > ema20 * 0.97:
        pb_entry = ema20
        # Stop below recent swing low or 1.5x ATR below entry
        sl_atr = _atr_stop(pb_entry, atr_v) if not np.isnan(atr_v) else pb_entry * 0.95
        sl = min(sl_atr, structure_stop)
        s = _make_setup(
            "pullback", pb_entry, sl,
            rationale="Pullback entry near 20-EMA — common re-entry level in healthy uptrends. "
                      "Wait for a bullish reversal candle to confirm.",
            invalidation=f"Daily close below {round(sl, 2)} or break of recent swing low.",
        )
        if s: setups.append(s)

    # --- Conservative: wait for retest of nearest_resistance after breakout
    if nearest_res and nearest_res > price:
        cons_entry = nearest_res * 1.002
        sl = max(structure_stop,
                 _atr_stop(cons_entry, atr_v) if not np.isnan(atr_v) else cons_entry * 0.95)
        s = _make_setup(
            "conservative", cons_entry, sl,
            rationale="Wait for breakout above nearest resistance and successful retest as support — "
                      "lowest probability of fakeouts.",
            invalidation=f"Failed retest, daily close below {round(sl, 2)}.",
        )
        if s: setups.append(s)

    # --- Aggressive: enter at current price (only if uptrend & RSI OK)
    rsi_v = snap.get("rsi")
    trend = snap.get("trend", "")
    if "Uptrend" in (trend or "") and rsi_v and 45 <= rsi_v <= 70:
        agg_entry = price
        sl_atr = _atr_stop(agg_entry, atr_v) if not np.isnan(atr_v) else agg_entry * 0.95
        sl = max(structure_stop, sl_atr)
        s = _make_setup(
            "aggressive", agg_entry, sl,
            rationale="Market entry — trend is up and momentum healthy. Higher risk of "
                      "buying near short-term highs; size smaller.",
            invalidation=f"Close below {round(sl, 2)} (structure or 1.5*ATR stop).",
        )
        if s: setups.append(s)

    return setups


def best_setup(setups: List[TradeSetup]) -> Optional[TradeSetup]:
    """Pick the best acceptable setup by R:R, prefer pullback over breakout when tied."""
    candidates = [s for s in setups if s.acceptable]
    if not candidates:
        return max(setups, key=lambda s: s.risk_reward_tp1) if setups else None
    style_rank = {"pullback": 4, "breakout": 3, "conservative": 2, "aggressive": 1}
    candidates.sort(key=lambda s: (s.risk_reward_tp1, style_rank.get(s.style, 0)), reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------
def position_size(account_equity: float, setup: TradeSetup,
                  risk_pct: float = R.max_account_risk_pct) -> Dict:
    """Compute share count using fixed-fractional risk."""
    risk_dollars = account_equity * (risk_pct / 100.0)
    shares = int(risk_dollars // max(setup.risk_per_share, 0.01))
    notional = shares * setup.entry
    return {
        "account_equity": account_equity,
        "risk_pct": risk_pct,
        "risk_dollars": round(risk_dollars, 2),
        "risk_per_share": setup.risk_per_share,
        "shares": shares,
        "notional": round(notional, 2),
        "exposure_pct": round((notional / account_equity) * 100, 2) if account_equity else None,
    }


def risk_summary(setups: List[TradeSetup]) -> Dict:
    """Aggregate stats for the report."""
    if not setups:
        return {"acceptable_count": 0, "best": None}
    best = best_setup(setups)
    return {
        "count": len(setups),
        "acceptable_count": sum(1 for s in setups if s.acceptable),
        "best": best.to_dict() if best else None,
    }

"""
Transparent 0–100 scoring engine.

Total = 100, broken down per config.WEIGHTS:
    technical_trend  25
    momentum         15
    volume           15
    fundamental      25
    valuation        10
    risk_reward      10

Each component returns (score_0_to_max, list_of_reasons). Reasons are surfaced
verbatim in the report so users can audit *why* a stock got its score.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np

import config
from src.fundamentals import fundamentals_available
from src.utils import logger

W = config.WEIGHTS


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------
def score_trend(snap: Dict) -> Tuple[float, List[str]]:
    """25 pts. Stacking of price vs SMAs + trend classification + cross."""
    max_pts = W.technical_trend
    pts = 0.0
    reasons: List[str] = []
    px = snap.get("price")
    s50 = snap.get("sma_50")
    s200 = snap.get("sma_200")
    ema20 = snap.get("ema_20")

    if px and s200 and not np.isnan(s200):
        if px > s200:
            pts += 6; reasons.append("Price above 200-SMA (long-term uptrend).")
        else:
            reasons.append("Price below 200-SMA (long-term downtrend).")
    if px and s50 and not np.isnan(s50):
        if px > s50:
            pts += 5; reasons.append("Price above 50-SMA.")
        else:
            reasons.append("Price below 50-SMA.")
    if s50 and s200 and not np.isnan(s50) and not np.isnan(s200):
        if s50 > s200:
            pts += 4; reasons.append("50-SMA above 200-SMA (bullish stack).")
        else:
            reasons.append("50-SMA below 200-SMA (bearish stack).")
    if px and ema20 and not np.isnan(ema20):
        if px > ema20:
            pts += 3; reasons.append("Price above 20-EMA (short-term momentum up).")

    trend = (snap.get("trend") or "").lower()
    if "uptrend" in trend and "pullback" not in trend:
        pts += 4; reasons.append(f"Trend classification: {snap.get('trend')}.")
    elif "uptrend (pullback)" in trend:
        pts += 2; reasons.append(f"Trend classification: {snap.get('trend')}.")
    elif "sideways" in trend:
        pts += 1; reasons.append(f"Trend classification: {snap.get('trend')}.")
    else:
        reasons.append(f"Trend classification: {snap.get('trend')}.")

    cross = snap.get("cross")
    if cross and "Golden" in cross:
        pts += 3; reasons.append("Recent golden cross (50/200 SMA).")
    elif cross and "Death" in cross:
        pts -= 3; reasons.append("Recent death cross (50/200 SMA) — bearish flag.")

    pts = max(0.0, min(max_pts, pts))
    return pts, reasons


def score_momentum(snap: Dict) -> Tuple[float, List[str]]:
    """15 pts. RSI healthy zone + MACD posture + StochRSI."""
    max_pts = W.momentum
    pts = 0.0
    reasons: List[str] = []
    rsi = snap.get("rsi")
    if rsi is not None and not np.isnan(rsi):
        if 45 <= rsi <= 70:
            pts += 6; reasons.append(f"RSI {rsi:.1f} in healthy bullish zone (45–70).")
        elif 40 <= rsi < 45 or 70 < rsi <= 75:
            pts += 3; reasons.append(f"RSI {rsi:.1f} borderline.")
        elif rsi > 75:
            pts += 1; reasons.append(f"RSI {rsi:.1f} overbought — risk of pullback.")
        elif rsi < 30:
            pts += 1; reasons.append(f"RSI {rsi:.1f} oversold — possible mean reversion.")
        else:
            reasons.append(f"RSI {rsi:.1f} weak.")

    macd_v = snap.get("macd"); macd_s = snap.get("macd_signal"); macd_h = snap.get("macd_hist")
    if all(v is not None and not np.isnan(v) for v in (macd_v, macd_s, macd_h)):
        if macd_v > macd_s and macd_h > 0:
            pts += 5; reasons.append("MACD above signal with positive histogram (bullish momentum).")
        elif macd_v > macd_s:
            pts += 3; reasons.append("MACD above signal (momentum turning up).")
        else:
            reasons.append("MACD below signal (momentum down).")

    sr = snap.get("stoch_rsi")
    if sr is not None and not np.isnan(sr):
        if 20 <= sr <= 80:
            pts += 2; reasons.append(f"StochRSI {sr:.0f} (neutral-to-trending).")
        elif sr > 80:
            pts += 1; reasons.append(f"StochRSI {sr:.0f} overbought.")
        elif sr < 20:
            pts += 1; reasons.append(f"StochRSI {sr:.0f} oversold.")

    # Bonus for relative strength when provided through snap
    rs = snap.get("relative_strength_pct")
    if rs is not None:
        if rs > 5:
            pts += 2; reasons.append(f"Outperforming benchmark by {rs:.1f}% (3M).")
        elif rs < -5:
            reasons.append(f"Underperforming benchmark by {rs:.1f}% (3M).")

    pts = max(0.0, min(max_pts, pts))
    return pts, reasons


def score_volume(snap: Dict) -> Tuple[float, List[str]]:
    """15 pts. Volume vs average + breakout flag + OBV trend."""
    max_pts = W.volume
    pts = 0.0
    reasons: List[str] = []
    v = snap.get("volume", {}) or {}
    ratio = v.get("volume_ratio")
    if ratio is not None:
        if ratio >= 1.5:
            pts += 6; reasons.append(f"Volume {ratio:.2f}x 20-day avg (breakout-quality).")
        elif ratio >= 1.0:
            pts += 4; reasons.append(f"Volume {ratio:.2f}x 20-day avg (above average).")
        elif ratio >= 0.7:
            pts += 2; reasons.append(f"Volume {ratio:.2f}x 20-day avg (slightly soft).")
        else:
            reasons.append(f"Volume {ratio:.2f}x 20-day avg (weak).")
    if v.get("volume_breakout"):
        pts += 3; reasons.append("Confirmed volume breakout day.")
    obv = v.get("obv_trend")
    if obv == "rising":
        pts += 4; reasons.append("OBV rising — accumulation pattern.")
    elif obv == "falling":
        reasons.append("OBV falling — distribution pattern.")
    else:
        pts += 2; reasons.append("OBV flat.")

    pts = max(0.0, min(max_pts, pts))
    return pts, reasons


def score_fundamental(f: Optional[Dict]) -> Tuple[float, List[str]]:
    """25 pts. Growth, margins, balance sheet quality, cash flow."""
    max_pts = W.fundamental
    if not f or not fundamentals_available(f):
        return 0.0, ["Fundamentals unavailable for this ticker — score skipped."]
    pts = 0.0
    reasons: List[str] = []

    rg = f.get("revenue_growth_yoy_pct")
    if rg is not None:
        if rg > 15:    pts += 5; reasons.append(f"Strong revenue growth YoY ({rg:.1f}%).")
        elif rg > 5:   pts += 3; reasons.append(f"Healthy revenue growth YoY ({rg:.1f}%).")
        elif rg > 0:   pts += 1; reasons.append(f"Modest revenue growth YoY ({rg:.1f}%).")
        else:          reasons.append(f"Revenue declining YoY ({rg:.1f}%).")
    nig = f.get("net_income_growth_yoy_pct")
    if nig is not None:
        if nig > 15:   pts += 4; reasons.append(f"Strong net income growth YoY ({nig:.1f}%).")
        elif nig > 0:  pts += 2; reasons.append(f"Positive net income growth YoY ({nig:.1f}%).")
        else:          reasons.append(f"Net income contracting YoY ({nig:.1f}%).")

    gm = f.get("gross_margin_pct")
    if gm is not None:
        if gm > 50:    pts += 3; reasons.append(f"High gross margin ({gm:.1f}%).")
        elif gm > 30:  pts += 2; reasons.append(f"Solid gross margin ({gm:.1f}%).")
        elif gm > 15:  pts += 1; reasons.append(f"Modest gross margin ({gm:.1f}%).")
    nm = f.get("net_margin_pct")
    if nm is not None:
        if nm > 15:    pts += 2; reasons.append(f"Strong net margin ({nm:.1f}%).")
        elif nm > 5:   pts += 1; reasons.append(f"Acceptable net margin ({nm:.1f}%).")
        elif nm < 0:   reasons.append(f"Negative net margin ({nm:.1f}%).")

    roe = f.get("roe")
    if roe is not None:
        if roe > 15:   pts += 3; reasons.append(f"ROE {roe:.1f}% — efficient capital use.")
        elif roe > 8:  pts += 2; reasons.append(f"ROE {roe:.1f}% — adequate.")
        elif roe < 0:  reasons.append(f"ROE {roe:.1f}% — unprofitable.")

    de = f.get("debt_to_equity")
    if de is not None:
        if de < 50:    pts += 3; reasons.append(f"D/E {de:.0f} — conservative leverage.")
        elif de < 100: pts += 2; reasons.append(f"D/E {de:.0f} — moderate leverage.")
        elif de < 200: pts += 1; reasons.append(f"D/E {de:.0f} — elevated leverage.")
        else:          reasons.append(f"D/E {de:.0f} — high leverage risk.")

    fcf = f.get("free_cash_flow")
    if fcf is not None:
        if fcf > 0:    pts += 3; reasons.append("Positive free cash flow.")
        else:          reasons.append("Negative free cash flow.")

    cr = f.get("current_ratio")
    if cr is not None:
        if cr >= 1.5:  pts += 2; reasons.append(f"Current ratio {cr:.2f} — strong liquidity.")
        elif cr >= 1:  pts += 1; reasons.append(f"Current ratio {cr:.2f}.")
        else:          reasons.append(f"Current ratio {cr:.2f} — liquidity risk.")

    pts = max(0.0, min(max_pts, pts))
    return pts, reasons


def score_valuation(f: Optional[Dict]) -> Tuple[float, List[str]]:
    """10 pts. P/E, P/S, P/B, PEG sanity (lower is better, with caveats)."""
    max_pts = W.valuation
    if not f or not fundamentals_available(f):
        return 0.0, ["Valuation skipped — no fundamentals."]
    pts = 0.0
    reasons: List[str] = []
    pe = f.get("pe")
    if pe is not None and pe > 0:
        if pe < 15:    pts += 3; reasons.append(f"P/E {pe:.1f} — value-leaning.")
        elif pe < 25:  pts += 2; reasons.append(f"P/E {pe:.1f} — reasonable.")
        elif pe < 40:  pts += 1; reasons.append(f"P/E {pe:.1f} — premium.")
        else:          reasons.append(f"P/E {pe:.1f} — expensive.")
    elif pe is not None and pe <= 0:
        reasons.append("Negative earnings — P/E not meaningful.")

    ps = f.get("ps")
    if ps is not None and ps > 0:
        if ps < 2:     pts += 2; reasons.append(f"P/S {ps:.2f} — low.")
        elif ps < 5:   pts += 1; reasons.append(f"P/S {ps:.2f} — moderate.")
        else:          reasons.append(f"P/S {ps:.2f} — high.")

    pb = f.get("pb")
    if pb is not None and pb > 0:
        if pb < 1.5:   pts += 2; reasons.append(f"P/B {pb:.2f} — near book value.")
        elif pb < 4:   pts += 1; reasons.append(f"P/B {pb:.2f} — moderate.")
        else:          reasons.append(f"P/B {pb:.2f} — high.")

    peg = f.get("peg")
    if peg is not None and peg > 0:
        if peg < 1:    pts += 3; reasons.append(f"PEG {peg:.2f} — growth at reasonable price.")
        elif peg < 2:  pts += 1; reasons.append(f"PEG {peg:.2f} — fair.")
        else:          reasons.append(f"PEG {peg:.2f} — overpaying for growth.")

    pts = max(0.0, min(max_pts, pts))
    return pts, reasons


def score_risk_reward(best_setup: Optional[Dict]) -> Tuple[float, List[str]]:
    """10 pts. Quality of best available trade setup."""
    max_pts = W.risk_reward
    if not best_setup:
        return 0.0, ["No actionable trade setup identified."]
    rr1 = best_setup.get("risk_reward_tp1") or 0
    rr2 = best_setup.get("risk_reward_tp2") or 0
    pts = 0.0
    reasons: List[str] = []
    if rr1 >= 3:   pts += 6; reasons.append(f"Excellent R:R to TP1 ({rr1:.2f}).")
    elif rr1 >= 2: pts += 5; reasons.append(f"Good R:R to TP1 ({rr1:.2f}).")
    elif rr1 >= 1.5: pts += 3; reasons.append(f"Marginal R:R to TP1 ({rr1:.2f}).")
    else:          reasons.append(f"Poor R:R to TP1 ({rr1:.2f}).")
    if rr2 >= 3:   pts += 4; reasons.append(f"Strong R:R to TP2 ({rr2:.2f}).")
    elif rr2 >= 2: pts += 2; reasons.append(f"Reasonable R:R to TP2 ({rr2:.2f}).")

    pts = max(0.0, min(max_pts, pts))
    return pts, reasons


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def classify(score: float) -> str:
    th = config.CLASSIFICATION_THRESHOLDS
    if score >= th["Strong Candidate"]:    return "Strong Candidate"
    if score >= th["Watchlist Candidate"]: return "Watchlist Candidate"
    if score >= th["Neutral"]:             return "Neutral"
    if score >= th["Weak"]:                return "Weak"
    return "Avoid"


def confidence_from_components(components: Dict[str, float]) -> str:
    """Confidence reflects breadth of signal agreement, not raw score."""
    # Count how many components scored >= 60% of their max
    max_map = {
        "trend": W.technical_trend, "momentum": W.momentum, "volume": W.volume,
        "fundamental": W.fundamental, "valuation": W.valuation, "risk_reward": W.risk_reward,
    }
    ratios = [components[k] / max_map[k] for k in components if max_map.get(k)]
    strong = sum(1 for r in ratios if r >= 0.6)
    if strong >= 5:  return "High"
    if strong >= 3:  return "Medium"
    return "Low"


def score_stock(snap: Dict, fundamentals: Optional[Dict],
                best_setup: Optional[Dict]) -> Dict:
    t_pts, t_r  = score_trend(snap)
    m_pts, m_r  = score_momentum(snap)
    v_pts, v_r  = score_volume(snap)
    f_pts, f_r  = score_fundamental(fundamentals)
    val_pts, val_r = score_valuation(fundamentals)
    rr_pts, rr_r = score_risk_reward(best_setup)

    components = {
        "trend": t_pts, "momentum": m_pts, "volume": v_pts,
        "fundamental": f_pts, "valuation": val_pts, "risk_reward": rr_pts,
    }
    total = sum(components.values())
    return {
        "components": components,
        "max_components": {
            "trend": W.technical_trend, "momentum": W.momentum, "volume": W.volume,
            "fundamental": W.fundamental, "valuation": W.valuation, "risk_reward": W.risk_reward,
        },
        "total": round(total, 1),
        "classification": classify(total),
        "confidence": confidence_from_components(components),
        "reasons": {
            "trend": t_r, "momentum": m_r, "volume": v_r,
            "fundamental": f_r, "valuation": val_r, "risk_reward": rr_r,
        },
    }

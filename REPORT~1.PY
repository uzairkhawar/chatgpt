"""
Report generator — builds a complete analysis dict and renders Markdown / HTML.

The dict is the canonical output. Markdown is for export and the in-app text
section. Excel/PDF export is a thin layer over the dict.
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import config
from src.data_loader import load_stock, DataProvider, DEFAULT_PROVIDER
from src.fundamentals import extract_fundamentals, fundamentals_available
from src.indicators import enrich, snapshot, relative_strength
from src.risk_management import build_setups, best_setup, position_size
from src.scoring import score_stock
from src.utils import logger, is_market_data_valid, round_or_none


# ---------------------------------------------------------------------------
# Scenario engine
# ---------------------------------------------------------------------------
def build_scenarios(snap: Dict, score: Dict, setup: Optional[Dict]) -> Dict:
    """Bullish / base / bearish scenarios from levels + score."""
    px = snap.get("price")
    levels = snap.get("levels", {}) or {}
    res1 = levels.get("nearest_resistance")
    res2 = levels.get("next_resistance")
    sup1 = levels.get("nearest_support")
    cls = score["classification"]

    bullish = (
        f"If price closes decisively above {res1:.2f} on above-average volume, "
        f"the next target is {res2:.2f}. Trend, momentum and structure align."
        if res1 else "If price reclaims recent highs on rising volume, the trend continues."
    )
    base = (
        f"With price at {px:.2f}, base case is range-bound action between "
        f"{sup1:.2f} and {res1:.2f}." if sup1 and res1
        else "Base case is consolidation around current levels."
    )
    bearish = (
        f"A daily close below {sup1:.2f} would invalidate the bullish setup and "
        f"open the door to retests of lower swing lows."
        if sup1 else "Loss of the lower Bollinger Band on volume would shift bias bearish."
    )
    if setup:
        invalidation = setup.get("invalidation")
        confirmation = (f"Daily close above {res1:.2f}." if res1 else
                        "Higher high on rising volume.")
    else:
        invalidation = (f"Close below {sup1:.2f}." if sup1 else "Loss of 200-SMA.")
        confirmation = (f"Close above {res1:.2f}." if res1 else "Reclaim of 50-SMA.")

    horizon_map = {
        "Strong Candidate": "1–3 months",
        "Watchlist Candidate": "1–2 months",
        "Neutral": "Wait for confirmation",
        "Weak": "Avoid until structure improves",
        "Avoid": "Not a buy candidate",
    }

    return {
        "bullish": bullish, "base": base, "bearish": bearish,
        "confirmation": confirmation, "invalidation": invalidation,
        "expected_horizon": horizon_map.get(cls, "Conditional"),
    }


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------
def analyze_stock(
    ticker: str, market: str = "US", horizon: str = "1Y",
    account_equity: Optional[float] = None,
    provider: Optional[DataProvider] = None,
) -> Dict:
    """Run the full pipeline and return a comprehensive result dict."""
    provider = provider or DEFAULT_PROVIDER
    bundle = load_stock(ticker, market, horizon, provider)
    if not is_market_data_valid(bundle["ohlcv"], min_rows=200):
        return {"ok": False, "error": f"Insufficient data for {ticker}.",
                "ticker": bundle["ticker"]}

    df_e = enrich(bundle["ohlcv"])
    snap = snapshot(df_e)

    # Relative strength vs benchmark
    bench_df = bundle.get("benchmark_ohlcv")
    rs = None
    if isinstance(bench_df, pd.DataFrame) and not bench_df.empty:
        rs = relative_strength(df_e["Close"], bench_df["Close"], 63)
    snap["relative_strength_pct"] = rs

    f = extract_fundamentals(bundle["info"], bundle["financials"])

    setups = build_setups(df_e, snap)
    best = best_setup(setups)
    best_d = best.to_dict() if best else None

    sc = score_stock(snap, f, best_d)
    scenarios = build_scenarios(snap, sc, best_d)

    pos_size = None
    if account_equity and best:
        pos_size = position_size(account_equity, best)

    # Suitable profile
    profile_map = {
        "Strong Candidate":   "Suitable for swing traders and long-term investors with risk control.",
        "Watchlist Candidate":"Watchlist — wait for confirmation. Suitable for patient investors.",
        "Neutral":            "Not actionable. Suitable for observation only.",
        "Weak":               "Not suitable for new long positions.",
        "Avoid":              "Avoid for long entries; consider only on full structure repair.",
    }

    market_condition = _market_condition(bench_df) if isinstance(bench_df, pd.DataFrame) else "Unknown"

    return {
        "ok": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ticker": bundle["ticker"],
        "market": bundle["market"],
        "name": f.get("name"),
        "sector": f.get("sector"),
        "industry": f.get("industry"),
        "currency": f.get("currency"),
        "snapshot": snap,
        "fundamentals": f,
        "fundamentals_available": fundamentals_available(f),
        "setups": [s.to_dict() for s in setups],
        "best_setup": best_d,
        "score": sc,
        "scenarios": scenarios,
        "position_sizing": pos_size,
        "market_condition": market_condition,
        "benchmark_ticker": bundle["benchmark_ticker"],
        "suitable_profile": profile_map.get(sc["classification"], ""),
        "df_enriched": df_e,  # for charting; not included in markdown
    }


def _market_condition(bench_df: pd.DataFrame) -> str:
    """Quick read of broad-market posture: above/below 50/200 SMAs."""
    if bench_df is None or bench_df.empty or len(bench_df) < 200:
        return "Unknown"
    close = bench_df["Close"]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    last = close.iloc[-1]
    if last > sma50 > sma200:
        return "Bullish (broad uptrend)"
    if last < sma50 < sma200:
        return "Bearish (broad downtrend)"
    if last > sma200:
        return "Mixed (above 200-SMA, below 50-SMA — caution)"
    return "Weak (below 200-SMA)"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def _fmt(v, digits=2, suffix=""):
    try:
        if v is None: return "—"
        if isinstance(v, float):
            if v != v: return "—"
            return f"{v:,.{digits}f}{suffix}"
        return f"{v}{suffix}"
    except Exception:
        return str(v)


def to_markdown(result: Dict) -> str:
    """Render the analysis as a Markdown report."""
    if not result.get("ok"):
        return f"# Analysis failed\n\n{result.get('error', 'Unknown error')}\n"

    snap = result["snapshot"]
    f = result["fundamentals"] or {}
    sc = result["score"]
    setups = result["setups"]
    best = result["best_setup"]
    sc_components = sc["components"]
    sc_max = sc["max_components"]
    scen = result["scenarios"]

    # Header
    lines: List[str] = []
    name = result.get("name") or result["ticker"]
    lines += [
        f"# Stock Analysis Report — {result['ticker']} ({name})",
        f"_Generated: {result['generated_at']} · Market: {result['market']} · Benchmark: {result['benchmark_ticker']}_",
        "",
        f"**Sector:** {result.get('sector') or '—'} · **Industry:** {result.get('industry') or '—'} · **Currency:** {result.get('currency') or '—'}",
        "",
        "## Verdict",
        f"- **Classification:** {sc['classification']}",
        f"- **Score:** {sc['total']} / 100",
        f"- **Confidence:** {sc['confidence']}",
        f"- **Market condition:** {result['market_condition']}",
        f"- **Suitable for:** {result['suitable_profile']}",
        "",
    ]

    # Scorecard
    lines += [
        "## Scorecard",
        "| Component | Score | Max |",
        "|---|---:|---:|",
    ]
    for k in ("trend", "momentum", "volume", "fundamental", "valuation", "risk_reward"):
        lines.append(f"| {k.replace('_',' ').title()} | {sc_components[k]:.1f} | {sc_max[k]} |")
    lines.append(f"| **Total** | **{sc['total']}** | **100** |")
    lines.append("")

    # Reasons
    lines.append("### Why this score (per component)")
    for k, label in [("trend","Trend"),("momentum","Momentum"),("volume","Volume"),
                     ("fundamental","Fundamentals"),("valuation","Valuation"),("risk_reward","Risk/Reward")]:
        lines.append(f"**{label}:**")
        for r in sc["reasons"].get(k, []) or ["—"]:
            lines.append(f"- {r}")
        lines.append("")

    # Snapshot
    vol = snap.get("volume", {}) or {}
    lvls = snap.get("levels", {}) or {}
    lines += [
        "## Technical Snapshot",
        f"- Price: **{_fmt(snap.get('price'))}** · Trend: **{snap.get('trend')}** · Structure: {snap.get('structure')}",
        f"- 20-SMA: {_fmt(snap.get('sma_20'))} · 50-SMA: {_fmt(snap.get('sma_50'))} · 200-SMA: {_fmt(snap.get('sma_200'))}",
        f"- RSI(14): {_fmt(snap.get('rsi'),1)} · MACD: {_fmt(snap.get('macd'),3)} (signal {_fmt(snap.get('macd_signal'),3)}, hist {_fmt(snap.get('macd_hist'),3)})",
        f"- ATR(14): {_fmt(snap.get('atr'))} · Hist Vol(30, ann.): {_fmt(snap.get('hv'),1,'%')}",
        f"- Volume vs avg(20): {_fmt(vol.get('volume_ratio'),2,'x')} · OBV trend: {vol.get('obv_trend')}",
        f"- Cross signal: {snap.get('cross') or 'none'}",
        f"- Nearest support: {_fmt(lvls.get('nearest_support'))} · Nearest resistance: {_fmt(lvls.get('nearest_resistance'))} · Next R: {_fmt(lvls.get('next_resistance'))}",
        f"- Relative strength vs benchmark (3M): {_fmt(snap.get('relative_strength_pct'),1,'%')}",
        "",
    ]

    # Setups table
    lines += [
        "## Trade Setups",
        "| Style | Entry | Stop | TP1 | TP2 | TP3 | R:R TP1 | Acceptable |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for s in setups:
        lines.append(
            f"| {s['style']} | {_fmt(s['entry'])} | {_fmt(s['stop_loss'])} | "
            f"{_fmt(s['tp1'])} | {_fmt(s['tp2'])} | {_fmt(s.get('tp3'))} | "
            f"{_fmt(s['risk_reward_tp1'])} | {'Yes' if s['acceptable'] else 'No'} |"
        )
    lines.append("")

    if best:
        lines += [
            "### Recommended Setup",
            f"- **Style:** {best['style']}",
            f"- **Entry:** {_fmt(best['entry'])} · **Stop:** {_fmt(best['stop_loss'])}",
            f"- **TP1:** {_fmt(best['tp1'])} · **TP2:** {_fmt(best['tp2'])} · **TP3:** {_fmt(best.get('tp3'))}",
            f"- **Risk per share:** {_fmt(best['risk_per_share'])} · **R:R TP1:** {_fmt(best['risk_reward_tp1'])} · **R:R TP2:** {_fmt(best['risk_reward_tp2'])}",
            f"- **Rationale:** {best['rationale']}",
            f"- **Invalidation:** {best['invalidation']}",
            "",
        ]

    if result.get("position_sizing"):
        ps = result["position_sizing"]
        lines += [
            "### Position Sizing",
            f"- Account equity: {_fmt(ps['account_equity'])} · Risk %: {ps['risk_pct']}%",
            f"- Risk $: {_fmt(ps['risk_dollars'])} · Risk per share: {_fmt(ps['risk_per_share'])}",
            f"- Suggested shares: **{ps['shares']}** · Notional: {_fmt(ps['notional'])} ({_fmt(ps['exposure_pct'],1,'%')} of equity)",
            "",
        ]

    # Fundamentals
    if result.get("fundamentals_available"):
        lines += [
            "## Fundamentals",
            "| Metric | Value |",
            "|---|---:|",
            f"| Market cap | {_fmt(f.get('market_cap'),0)} |",
            f"| Revenue | {_fmt(f.get('revenue'),0)} |",
            f"| Revenue growth YoY | {_fmt(f.get('revenue_growth_yoy_pct'),1,'%')} |",
            f"| Net income | {_fmt(f.get('net_income'),0)} |",
            f"| Net income growth YoY | {_fmt(f.get('net_income_growth_yoy_pct'),1,'%')} |",
            f"| Gross margin | {_fmt(f.get('gross_margin_pct'),1,'%')} |",
            f"| Operating margin | {_fmt(f.get('operating_margin_pct'),1,'%')} |",
            f"| Net margin | {_fmt(f.get('net_margin_pct'),1,'%')} |",
            f"| ROE | {_fmt(f.get('roe'),1,'%')} |",
            f"| ROA | {_fmt(f.get('roa'),1,'%')} |",
            f"| P/E | {_fmt(f.get('pe'),1)} |",
            f"| Forward P/E | {_fmt(f.get('forward_pe'),1)} |",
            f"| P/S | {_fmt(f.get('ps'),2)} |",
            f"| P/B | {_fmt(f.get('pb'),2)} |",
            f"| PEG | {_fmt(f.get('peg'),2)} |",
            f"| Debt/Equity | {_fmt(f.get('debt_to_equity'),0)} |",
            f"| Current ratio | {_fmt(f.get('current_ratio'),2)} |",
            f"| Free cash flow | {_fmt(f.get('free_cash_flow'),0)} |",
            f"| Dividend yield | {_fmt(f.get('dividend_yield_pct'),2,'%')} |",
            "",
        ]
    else:
        lines += ["## Fundamentals", "_Not enough fundamental data was available from the provider._", ""]

    # Scenarios
    lines += [
        "## Scenarios",
        f"- **Bullish:** {scen['bullish']}",
        f"- **Base case:** {scen['base']}",
        f"- **Bearish:** {scen['bearish']}",
        f"- **Confirmation:** {scen['confirmation']}",
        f"- **Invalidation:** {scen['invalidation']}",
        f"- **Expected horizon:** {scen['expected_horizon']}",
        "",
        "## Disclaimer",
        config.DISCLAIMER,
        "",
    ]
    return "\n".join(lines)


def save_markdown(result: Dict, output_dir: Optional[Path] = None) -> Path:
    output_dir = output_dir or config.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result['ticker'].replace('.', '_')}_report.md"
    path.write_text(to_markdown(result), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------
def to_excel_bytes(result: Dict) -> bytes:
    """Multi-sheet Excel export for the analysis."""
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        # Summary
        sc = result["score"]
        summary = {
            "Field": ["Ticker", "Name", "Market", "Sector", "Industry", "Currency",
                      "Classification", "Score", "Confidence", "Market Condition",
                      "Suitable Profile", "Generated"],
            "Value": [
                result["ticker"], result.get("name") or "", result["market"],
                result.get("sector") or "", result.get("industry") or "",
                result.get("currency") or "", sc["classification"], sc["total"],
                sc["confidence"], result["market_condition"],
                result["suitable_profile"], result["generated_at"],
            ],
        }
        pd.DataFrame(summary).to_excel(xl, sheet_name="Summary", index=False)

        # Scorecard
        comp = sc["components"]; mx = sc["max_components"]
        rows = [{"Component": k, "Score": comp[k], "Max": mx[k]} for k in comp]
        rows.append({"Component": "Total", "Score": sc["total"], "Max": 100})
        pd.DataFrame(rows).to_excel(xl, sheet_name="Scorecard", index=False)

        # Setups
        if result["setups"]:
            pd.DataFrame(result["setups"]).to_excel(xl, sheet_name="Setups", index=False)

        # Fundamentals
        if result.get("fundamentals"):
            f = result["fundamentals"]
            pd.DataFrame(
                [{"Metric": k, "Value": v} for k, v in f.items()]
            ).to_excel(xl, sheet_name="Fundamentals", index=False)

        # Reasons
        rows = []
        for k, items in sc["reasons"].items():
            for r in items:
                rows.append({"Component": k, "Reason": r})
        pd.DataFrame(rows).to_excel(xl, sheet_name="Reasons", index=False)

        # Last 250 bars of OHLCV+indicators (truncate to keep file small)
        df = result.get("df_enriched")
        if df is not None and not df.empty:
            df.tail(250).to_excel(xl, sheet_name="Indicators")

    out.seek(0)
    return out.read()

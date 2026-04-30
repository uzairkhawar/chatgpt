"""
Stock screener — scans a universe of tickers against technical, fundamental
and risk filters, then ranks candidates by overall score.

Built to be embarrassingly parallel. v1 runs sequentially with caching;
v2 should swap in a thread/process pool.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

import config
from src.data_loader import DEFAULT_PROVIDER, DataProvider
from src.fundamentals import extract_fundamentals
from src.indicators import (
    enrich, snapshot, relative_strength,
)
from src.risk_management import build_setups, best_setup
from src.scoring import score_stock
from src.utils import logger, normalize_ticker, is_market_data_valid

S = config.SCREENER


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def passes_technical(snap: Dict, df_enriched: pd.DataFrame) -> Dict:
    reasons: List[str] = []
    px = snap.get("price")
    s50 = snap.get("sma_50")
    s200 = snap.get("sma_200")
    rsi = snap.get("rsi")
    avg_v = (snap.get("volume") or {}).get("avg_volume_20")

    ok = True
    if avg_v is not None and avg_v < S.min_avg_volume:
        ok = False; reasons.append(f"Avg volume {avg_v:,.0f} below min {S.min_avg_volume:,}")
    if px is not None and (px < S.min_price or px > S.max_price):
        ok = False; reasons.append(f"Price {px} outside [{S.min_price}, {S.max_price}]")
    if S.require_above_50_sma and s50 is not None and px is not None and px < s50:
        ok = False; reasons.append("Price below 50-SMA")
    if S.require_above_200_sma and s200 is not None and px is not None and px < s200:
        ok = False; reasons.append("Price below 200-SMA")
    if rsi is not None and not (S.rsi_min <= rsi <= S.rsi_max):
        ok = False; reasons.append(f"RSI {rsi:.1f} outside [{S.rsi_min},{S.rsi_max}]")
    return {"passes": ok, "reasons_failed": reasons}


def passes_relative_strength(stock_close: pd.Series,
                             bench_close: Optional[pd.Series],
                             min_rs: float = S.min_rel_strength) -> Dict:
    if bench_close is None:
        return {"passes": True, "rs": None, "note": "no benchmark"}
    rs = relative_strength(stock_close, bench_close, S.rs_lookback_days)
    if rs is None:
        return {"passes": True, "rs": None, "note": "RS unavailable"}
    return {"passes": rs >= min_rs, "rs": rs}


# ---------------------------------------------------------------------------
# Screener entry point
# ---------------------------------------------------------------------------
def screen(
    tickers: Optional[List[str]] = None,
    market: str = "US",
    provider: Optional[DataProvider] = None,
    require_fundamentals: bool = False,
    benchmark_ticker: Optional[str] = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Returns a DataFrame ranked by total score.
    Columns: ticker, score, classification, confidence, price, rsi, trend,
             rel_strength, setup_style, rr_tp1, sector, market_cap,
             pass_technical, fail_reasons.
    """
    provider = provider or DEFAULT_PROVIDER
    market_u = market.upper()
    tickers = tickers or config.DEFAULT_UNIVERSE.get(market_u, [])
    bench_sym = benchmark_ticker or config.DEFAULT_BENCHMARK_PER_MARKET.get(market_u, "^GSPC")
    bench_df = provider.fetch_ohlcv(bench_sym, period="1y", interval="1d")
    bench_close = bench_df["Close"] if isinstance(bench_df, pd.DataFrame) and not bench_df.empty else None

    rows: List[Dict] = []
    total = len(tickers)
    for i, raw_t in enumerate(tickers, 1):
        t = normalize_ticker(raw_t, market_u)
        if progress_callback:
            try: progress_callback(i, total, t)
            except Exception: pass

        df = provider.fetch_ohlcv(t, period="2y", interval="1d")
        if not is_market_data_valid(df, min_rows=200):
            rows.append({"ticker": t, "score": None, "classification": "Skipped",
                         "fail_reasons": "insufficient data"})
            continue

        try:
            df_e = enrich(df)
            snap = snapshot(df_e)
            # add RS into snapshot for momentum bonus
            rs_info = passes_relative_strength(df_e["Close"], bench_close)
            snap["relative_strength_pct"] = rs_info.get("rs")

            tech = passes_technical(snap, df_e)
            info = provider.fetch_info(t) if require_fundamentals else {}
            fin = provider.fetch_financials(t) if require_fundamentals else {}
            f = extract_fundamentals(info, fin) if require_fundamentals else None

            setups = build_setups(df_e, snap)
            best = best_setup(setups)
            best_d = best.to_dict() if best else None
            sc = score_stock(snap, f, best_d)

            rows.append({
                "ticker": t,
                "name": (f or {}).get("name") or info.get("longName") or "",
                "sector": (f or {}).get("sector") or info.get("sector"),
                "market_cap": (f or {}).get("market_cap") or info.get("marketCap"),
                "price": snap.get("price"),
                "rsi": snap.get("rsi"),
                "trend": snap.get("trend"),
                "rel_strength_pct": rs_info.get("rs"),
                "score": sc["total"],
                "classification": sc["classification"],
                "confidence": sc["confidence"],
                "setup_style": best.style if best else None,
                "rr_tp1": best.risk_reward_tp1 if best else None,
                "pass_technical": tech["passes"],
                "fail_reasons": "; ".join(tech["reasons_failed"]) if tech["reasons_failed"] else "",
            })
        except Exception as e:
            logger.error("Screen error for %s: %s", t, e)
            rows.append({"ticker": t, "score": None, "classification": "Error",
                         "fail_reasons": str(e)})

    df_out = pd.DataFrame(rows)
    if "score" in df_out.columns:
        df_out = df_out.sort_values(by=["score"], ascending=False, na_position="last")
    return df_out.reset_index(drop=True)

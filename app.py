"""
Streamlit dashboard for the local stock analyzer.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `import config` and `from src.X import ...` when run via `streamlit run app.py`
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

import config
from src.charting import plot_full_analysis
from src.report_generator import (
    analyze_stock, to_markdown, save_markdown, to_excel_bytes,
)
from src.screener import screen


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Analyzer — Decision Support",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.title("Stock Analyzer — Local Decision-Support Dashboard")
st.caption("Evidence-based stock analysis. Decision-support only. Not financial advice.")
with st.expander("Important disclaimer", expanded=False):
    st.warning(config.DISCLAIMER)


# ---------------------------------------------------------------------------
# Sidebar — global controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    market = st.selectbox("Market", ["US", "SAUDI", "GLOBAL"], index=0)
    horizon = st.selectbox("Timeframe", ["1M", "3M", "6M", "1Y", "3Y", "5Y"], index=3)
    strategy = st.selectbox(
        "Strategy style",
        ["Swing Trade", "Long-Term Investment", "Breakout", "Pullback"],
        index=0,
    )
    risk_profile = st.selectbox(
        "Risk profile", ["Conservative", "Balanced", "Aggressive"], index=1,
    )
    account_equity = st.number_input(
        "Account equity (for sizing)", min_value=0.0, value=10_000.0, step=500.0,
    )
    st.markdown("---")
    st.caption("v1 · yfinance + custom indicators · transparent scoring")


tab_analyze, tab_screen, tab_methodology = st.tabs(
    ["Analyze", "Screener", "Methodology"]
)


# ---------------------------------------------------------------------------
# Analyze tab
# ---------------------------------------------------------------------------
with tab_analyze:
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_in = st.text_input(
            "Ticker (e.g. AAPL, MSFT, 2222 for Saudi Aramco)",
            value="AAPL",
        ).strip()
    with col2:
        run = st.button("Analyze", use_container_width=True, type="primary")

    if run and ticker_in:
        with st.spinner(f"Analyzing {ticker_in}..."):
            result = analyze_stock(
                ticker_in, market=market, horizon=horizon,
                account_equity=account_equity if account_equity > 0 else None,
            )

        if not result.get("ok"):
            st.error(result.get("error", "Analysis failed."))
        else:
            sc = result["score"]
            best = result["best_setup"]

            # ---- Top metrics ----
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Score", f"{sc['total']}/100", sc["classification"])
            m2.metric("Confidence", sc["confidence"])
            m3.metric("Trend", result["snapshot"].get("trend") or "—")
            m4.metric("RSI(14)", f"{result['snapshot'].get('rsi'):.1f}"
                      if result['snapshot'].get('rsi') is not None else "—")
            m5.metric("Market", result["market_condition"])

            # ---- Verdict box ----
            cls = sc["classification"]
            color = {"Strong Candidate": "success", "Watchlist Candidate": "info",
                     "Neutral": "warning", "Weak": "warning", "Avoid": "error"}.get(cls, "info")
            getattr(st, color)(
                f"**{cls}** · Score {sc['total']}/100 · Confidence {sc['confidence']}\n\n"
                f"_{result['suitable_profile']}_"
            )

            # ---- Chart ----
            st.subheader("Chart")
            try:
                chart_path = plot_full_analysis(
                    result["df_enriched"], result["snapshot"], best,
                    result["ticker"],
                )
                st.image(str(chart_path), use_column_width=True)
            except Exception as e:
                st.warning(f"Chart could not be rendered: {e}")

            # ---- Scorecard ----
            st.subheader("Scorecard")
            comp = sc["components"]; mx = sc["max_components"]
            df_score = pd.DataFrame([
                {"Component": k.replace("_", " ").title(),
                 "Score": comp[k], "Max": mx[k],
                 "% of Max": round(comp[k] / mx[k] * 100, 1) if mx[k] else 0}
                for k in comp
            ])
            st.dataframe(df_score, use_container_width=True, hide_index=True)

            with st.expander("Why this score (component reasons)", expanded=False):
                for k in ("trend","momentum","volume","fundamental","valuation","risk_reward"):
                    st.markdown(f"**{k.replace('_',' ').title()}:**")
                    for r in sc["reasons"].get(k, []):
                        st.markdown(f"- {r}")

            # ---- Setups ----
            st.subheader("Trade Setups")
            if result["setups"]:
                df_setups = pd.DataFrame(result["setups"])
                st.dataframe(df_setups, use_container_width=True, hide_index=True)
            else:
                st.info("No actionable long setups in current price action.")

            if best:
                st.markdown("**Recommended setup**")
                cols = st.columns(4)
                cols[0].metric("Style", best["style"])
                cols[1].metric("Entry", f"{best['entry']:.2f}")
                cols[2].metric("Stop", f"{best['stop_loss']:.2f}")
                cols[3].metric("R:R TP1", f"{best['risk_reward_tp1']:.2f}")
                cols2 = st.columns(3)
                cols2[0].metric("TP1", f"{best['tp1']:.2f}")
                cols2[1].metric("TP2", f"{best['tp2']:.2f}")
                cols2[2].metric("TP3", f"{best.get('tp3'):.2f}" if best.get("tp3") else "—")
                st.write(f"**Rationale:** {best['rationale']}")
                st.write(f"**Invalidation:** {best['invalidation']}")

                if result.get("position_sizing"):
                    ps = result["position_sizing"]
                    st.markdown("**Position sizing**")
                    pcols = st.columns(4)
                    pcols[0].metric("Shares", f"{ps['shares']}")
                    pcols[1].metric("Notional", f"{ps['notional']:.2f}")
                    pcols[2].metric("Risk $", f"{ps['risk_dollars']:.2f}")
                    pcols[3].metric("Exposure", f"{ps['exposure_pct']:.1f}%"
                                    if ps.get("exposure_pct") else "—")

            # ---- Fundamentals ----
            st.subheader("Fundamentals")
            if result.get("fundamentals_available"):
                f = result["fundamentals"]
                df_f = pd.DataFrame(
                    [{"Metric": k, "Value": v} for k, v in f.items()]
                )
                st.dataframe(df_f, use_container_width=True, hide_index=True)
            else:
                st.info("Fundamentals not available for this ticker from the provider.")

            # ---- Scenarios ----
            st.subheader("Scenarios")
            scen = result["scenarios"]
            st.markdown(f"- **Bullish:** {scen['bullish']}")
            st.markdown(f"- **Base case:** {scen['base']}")
            st.markdown(f"- **Bearish:** {scen['bearish']}")
            st.markdown(f"- **Confirmation:** {scen['confirmation']}")
            st.markdown(f"- **Invalidation:** {scen['invalidation']}")
            st.markdown(f"- **Expected horizon:** {scen['expected_horizon']}")

            # ---- Exports ----
            st.subheader("Exports")
            md = to_markdown(result)
            colA, colB, colC = st.columns(3)
            with colA:
                st.download_button(
                    "Download Markdown report", data=md,
                    file_name=f"{result['ticker']}_report.md", mime="text/markdown",
                )
            with colB:
                st.download_button(
                    "Download Excel workbook", data=to_excel_bytes(result),
                    file_name=f"{result['ticker']}_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with colC:
                if st.button("Save report to disk", use_container_width=True):
                    p = save_markdown(result)
                    st.success(f"Saved to {p}")

            with st.expander("Full Markdown report", expanded=False):
                st.markdown(md)


# ---------------------------------------------------------------------------
# Screener tab
# ---------------------------------------------------------------------------
with tab_screen:
    st.subheader("Stock Screener")
    universe_default = ", ".join(config.DEFAULT_UNIVERSE.get(market, []))
    universe_in = st.text_area(
        "Tickers (comma-separated). Defaults to a curated universe per market.",
        value=universe_default, height=100,
    )
    require_fund = st.checkbox(
        "Include fundamentals (slower, hits API more)", value=False,
    )
    if st.button("Run screener", type="primary"):
        tickers = [t.strip() for t in universe_in.split(",") if t.strip()]
        bar = st.progress(0.0, text="Starting...")
        def cb(i, total, t):
            bar.progress(i / total, text=f"{i}/{total} — {t}")
        with st.spinner("Scanning universe..."):
            df = screen(
                tickers=tickers, market=market,
                require_fundamentals=require_fund, progress_callback=cb,
            )
        bar.empty()
        st.success(f"Scanned {len(df)} tickers.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV", data=df.to_csv(index=False),
            file_name=f"screener_{market}.csv", mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Methodology tab
# ---------------------------------------------------------------------------
with tab_methodology:
    st.subheader("Methodology — How the score is computed")
    st.markdown(f"""
**Total = 100 points, broken down per `config.WEIGHTS`:**

- **Trend** ({config.WEIGHTS.technical_trend} pts): price vs 20/50/200 SMA stacking, 50/200 cross, classification.
- **Momentum** ({config.WEIGHTS.momentum} pts): RSI healthy zone, MACD posture, StochRSI, relative strength bonus.
- **Volume** ({config.WEIGHTS.volume} pts): volume vs 20-day average, breakout day, OBV trend.
- **Fundamentals** ({config.WEIGHTS.fundamental} pts): revenue and net income growth, margins, ROE, leverage, free cash flow, liquidity.
- **Valuation** ({config.WEIGHTS.valuation} pts): P/E, P/S, P/B, PEG.
- **Risk/Reward** ({config.WEIGHTS.risk_reward} pts): quality of best available trade setup (R:R to TP1, TP2).

**Classification thresholds:**
- ≥ 75 → Strong Candidate
- ≥ 60 → Watchlist Candidate
- ≥ 45 → Neutral
- ≥ 30 → Weak
- < 30 → Avoid

**Confidence** is independent of raw score: it counts how many of the six components scored ≥ 60% of max. Five+ → High; three–four → Medium; otherwise Low. This penalizes one-dimensional setups (e.g. great price action but ugly fundamentals).

**Trade-setup geometry** is derived from swing-pivot S/R combined with ATR(14) and 20-day structural lows. Every setup is scored on R:R to TP1 and rejected from "acceptable" status if R:R < {config.RISK.min_risk_reward}.

**Limitations:**
- yfinance fundamentals are best-effort; many Tadawul tickers will not have ratios.
- TASI index symbol coverage on yfinance is partial — falls back to S&P 500 if missing.
- All indicators are computed on daily candles regardless of UI horizon (daily is more reliable for trend/MA/RSI signals than intraday).
- This tool does not predict prices. Scenarios are conditional descriptions of what *would* confirm or invalidate the current setup.
""")


st.markdown("---")
st.caption(config.DISCLAIMER)

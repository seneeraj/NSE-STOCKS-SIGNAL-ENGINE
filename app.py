import re
import numpy as np
import pandas as pd
import streamlit as st

from src.universe import load_universe
from src.market_data import download_prices
from src.fundamentals import collect_fundamentals
from src.scoring import calculate_scores
from src.sector_rotation import build_sector_scores
from src.market_regime import fetch_market_regime
from src.valuation import add_valuation_to_ranking
from src.scenario_valuation import add_scenario_valuation
from src.risk_engine import add_risk_engine
from src.backtesting import (
    load_snapshots,
    save_scan_snapshot,
    backtest_snapshots,
    score_bucket_analysis,
    download_backtest_prices,
)


def explain_risk_flag(flag):
    explanations = {
        "HIGH DEBT/EQUITY": "Leverage is elevated. A weaker-than-expected recovery could put pressure on cash flow and solvency.",
        "WEAK CFO/PAT": "Operating cash flow is weak relative to reported profit. Check earnings quality and working-capital movements.",
        "BEAR DOWNSIDE >30%": "The modeled Bear scenario is more than 30% below the current price. This is a valuation/downside-protection warning, not a forecast.",
        "TURNAROUND NOT YET CONFIRMED": "Fundamental improvement has not yet reached the confirmation stage. Treat this as an early/watch opportunity.",
        "Cyclical earnings require normalization": "Current earnings may be influenced by the business cycle. Avoid treating peak/trough earnings as permanently sustainable.",
        "Extreme growth rate; likely base-effect distortion": "The reported growth rate may be inflated by an unusually weak comparison period. Validate the underlying trend.",
        "Very high P/E; earnings denominator may be temporarily depressed": "The P/E is unusually high. Investigate whether current earnings are temporarily depressed before concluding the stock is expensive.",
        "Very high EV/EBITDA; investigate earnings/capital-cycle normalization": "EV/EBITDA is unusually high. Check whether EBITDA is temporarily depressed or the capital cycle is distorting the multiple.",
        "Primary valuation input unavailable": "A key valuation input is unavailable. Avoid treating the valuation score as fully reliable until the missing data is resolved.",
    }
    return explanations.get(flag, "Review the underlying company data before making an investment decision.")

def render_risk_flags(flags):
    if flags is None or (isinstance(flags, float) and pd.isna(flags)):
        flags = "NONE"
    flags = str(flags).strip()
    if not flags or flags.upper() == "NONE":
        st.success("No major automated risk flag detected.")
        return

    items = [x.strip() for x in re.split(r"\s*\|\s*|\s*;\s*", flags) if x.strip()]
    for flag in items:
        severe = any(k in flag for k in [
            "HIGH DEBT", "WEAK CFO", "Very high P/E", "Very high EV/EBITDA"
        ])
        icon = "🔴" if severe else "🟠"
        with st.expander(f"{icon} {flag}", expanded=False):
            st.write(explain_risk_flag(flag))

st.set_page_config(
    page_title="India Early Turnaround Engine",
    page_icon="📈",
    layout="wide",
)

st.title("📈 NSE Stocks Signal Engine")
st.caption(
    "V1.10 — Foundation + Turnaround + Sector Rotation + Valuation + Risk Engine + Market Regime + Backtesting"
)

with st.sidebar:
    st.header("Scan Settings")
    universe_size = st.selectbox(
        "Universe", ["Starter 50", "Custom CSV"], index=0
    )
    period = st.selectbox(
        "Price history", ["1y", "2y", "5y"], index=1
    )
    run = st.button("🔄 Run Scan", type="primary")

st.info(
    "The engine looks for improving businesses before the market fully recognizes the change. "
    "Scores are screening signals, not intrinsic value estimates or guaranteed-return predictions."
)

if universe_size == "Custom CSV":
    st.write("Place `data/universe.csv` with columns: symbol, company, sector.")

# Persistent state is essential because Streamlit reruns the script whenever
# a widget changes.
if "scores" not in st.session_state:
    st.session_state.scores = None
    st.session_state.sectors = None
    st.session_state.scan_universe = None
    st.session_state.scan_period = None
    st.session_state.market_regime = None

if run:
    universe = load_universe(custom=(universe_size == "Custom CSV"))
    st.write(f"Universe loaded: **{len(universe)} companies**")

    with st.spinner("Downloading market data..."):
        prices = download_prices(universe["ticker"].tolist(), period=period)

    with st.spinner("Collecting multi-period financial data..."):
        fundamentals = collect_fundamentals(universe["ticker"].tolist())

    scores = calculate_scores(universe, fundamentals, prices)
    sectors = build_sector_scores(scores, prices)
    scores = add_valuation_to_ranking(scores, fundamentals)

    # Scenario valuation needs the complete latest fundamentals row in addition
    # to the ranked score fields. Merge only the raw inputs required by the
    # scenario engine, avoiding duplicate score columns.
    scenario_inputs = [
        "ticker", "revenue_growth_pct", "net_income_growth_pct",
        "net_income", "equity", "ebitda", "net_debt", "fcf",
        "roce_proxy", "market_cap", "current_price",
    ]
    scenario_inputs = [c for c in scenario_inputs if c in fundamentals.columns]
    fund_for_scenario = fundamentals[scenario_inputs].copy()
    if "ticker" in fund_for_scenario.columns:
        fund_for_scenario = fund_for_scenario.drop_duplicates("ticker")
    scores = scores.merge(
        fund_for_scenario,
        on="ticker",
        how="left",
        suffixes=("", "_fund"),
    )

    # Prefer the values already calculated by the ranking pipeline when
    # available, otherwise use the raw fundamentals copy.
    for col in [
        "revenue_growth_pct", "net_income_growth_pct", "net_income",
        "equity", "ebitda", "net_debt", "fcf", "roce_proxy",
        "market_cap", "current_price",
    ]:
        fund_col = f"{col}_fund"
        if fund_col in scores.columns:
            if col not in scores.columns:
                scores[col] = scores[fund_col]
            else:
                scores[col] = scores[col].combine_first(scores[fund_col])
            scores.drop(columns=[fund_col], inplace=True)

    scores = add_scenario_valuation(scores)

    if not sectors.empty:
        scores = scores.merge(
            sectors[["sector", "sector_recovery_score", "sector_stage"]],
            on="sector",
            how="left",
        )
    else:
        scores["sector_recovery_score"] = 50.0
        scores["sector_stage"] = "INSUFFICIENT DATA"

    # Market breadth is calculated from the same scanned stocks.
    breadth = (
        float((scores["momentum_60d_pct"] > 0).mean() * 100)
        if "momentum_60d_pct" in scores.columns
        else None
    )

    with st.spinner("Calculating market regime..."):
        regime = fetch_market_regime(breadth_pct=breadth)

    # Stock-level opportunity remains the dominant component.
    scores["opportunity_score_pre_regime"] = (
        0.55 * scores["turnaround_score"]
        + 0.15 * scores["foundation_score"]
        + 0.15 * scores["sector_recovery_score"]
        + 0.15 * scores["valuation_disconnect"]
    )

    regime_score = (
        float(regime["market_regime_score"])
        if pd.notna(regime["market_regime_score"])
        else 50.0
    )
    scores["market_regime_score"] = regime_score
    scores["opportunity_score"] = (
        0.92 * scores["opportunity_score_pre_regime"]
        + 0.08 * regime_score
    ).clip(0, 100)

    # V1.10: risk and decision layer sits after opportunity scoring.
    scores = add_risk_engine(scores)

    scores = scores.sort_values(
        ["opportunity_score", "turnaround_score"],
        ascending=False,
    ).reset_index(drop=True)

    # Persist the complete scan so dropdowns and navigation can rerun safely.
    st.session_state.scores = scores.copy()
    st.session_state.sectors = sectors.copy()
    st.session_state.scan_universe = universe.copy()
    st.session_state.scan_period = period
    st.session_state.market_regime = regime

    # Save a point-in-time ranking for future walk-forward testing.
    try:
        save_scan_snapshot(scores)
    except Exception:
        pass

if st.session_state.scores is not None:
    scores = st.session_state.scores
    sectors = st.session_state.sectors
    universe = st.session_state.scan_universe
    period = st.session_state.scan_period
    market_regime = st.session_state.market_regime

    section = st.radio(
        "Section",
        [
            "🏆 Opportunities",
            "🛡️ Risk & Decision",
            "🔄 Sector Rotation",
            "💰 Valuation",
            "🌐 Market Regime",
            "🔎 Stock Diagnostics",
            "📈 Backtesting",
            "📊 Data",
        ],
        horizontal=True,
        key="main_section",
        label_visibility="collapsed",
    )

    if section == "🏆 Opportunities":
        st.subheader("Early Opportunity Ranking")
        st.caption(
            "Turnaround evidence remains the primary driver. Market regime contributes only 8%."
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Market Regime", market_regime.get("market_regime", "UNKNOWN"))
        m2.metric(
            "Regime Score",
            f'{market_regime["market_regime_score"]:.0f}/100'
            if pd.notna(market_regime.get("market_regime_score"))
            else "N/A",
        )
        m3.metric("Deployment Bias", market_regime.get("deployment_bias", "N/A"))

        display_cols = [
            "symbol", "company", "sector", "opportunity_score",
            "turnaround_score", "foundation_score",
            "sector_recovery_score", "valuation_disconnect",
            "sector_stage", "turnaround_stage", "data_completeness_pct", "investment_decision",
        ]
        st.dataframe(
            scores[display_cols].style.format({
                "opportunity_score": "{:.1f}",
                "turnaround_score": "{:.1f}",
                "foundation_score": "{:.1f}",
                "sector_recovery_score": "{:.1f}",
                "valuation_disconnect": "{:.1f}",
                "data_completeness_pct": "{:.0f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download opportunity ranking CSV",
            scores.to_csv(index=False).encode("utf-8"),
            "opportunity_ranking.csv",
            "text/csv",
        )

    if section == "🛡️ Risk & Decision":
        st.subheader("Risk Engine & Investment Decision Matrix")
        st.caption("V1.10 adds an explicit research conclusion, next confirmation step, thesis invalidation condition, and research priority. It is a screening framework, not a guaranteed-return forecast.")
        risk_cols = [
            "symbol", "company", "sector", "opportunity_score", "risk_score",
            "survival_score", "scenario_risk_reward_score", "turnaround_stage",
            "valuation_data_status", "investment_decision", "position_bias",
            "research_priority", "risk_flags",
        ]
        st.dataframe(scores[risk_cols].style.format({
            "opportunity_score": "{:.1f}", "risk_score": "{:.1f}",
            "survival_score": "{:.1f}", "scenario_risk_reward_score": "{:.1f}",
        }), use_container_width=True, hide_index=True)

        st.markdown("### 🔬 Decision Research Card")
        decision_symbol = st.selectbox(
            "Company", scores["symbol"].tolist(), key="decision_company"
        )
        drow = scores[scores["symbol"] == decision_symbol].iloc[0]

        d1, d2, d3 = st.columns(3)
        d1.metric("Decision", drow.get("investment_decision", "N/A"))
        d2.metric("Research Priority", drow.get("research_priority", "N/A"))
        d3.metric("Position Bias", drow.get("position_bias", "N/A"))

        st.markdown("**Why this decision?**")
        st.info(drow.get("decision_rationale", "N/A"))

        st.markdown("**What should confirm the thesis next?**")
        st.write(drow.get("next_confirmation", "N/A"))

        st.markdown("**What would invalidate the thesis?**")
        st.warning(drow.get("thesis_invalidation", "N/A"))

        st.markdown("**Risk Flags**")
        render_risk_flags(drow.get("risk_flags", "NONE"))

        st.markdown("### Decision Logic")
        st.write("**ATTRACTIVE RISK/REWARD** → strong opportunity + adequate risk score + EARLY/CONFIRMING turnaround; still requires staged validation.")
        st.write("**PROMISING — WATCH / CONFIRM** → promising evidence, but confirmation or risk/reward is not strong enough yet.")
        st.write("**WATCH — NEED MORE EVIDENCE** → interesting signals, but insufficient edge for capital deployment.")
        st.write("**HIGH DOWNSIDE RISK** → Bear scenario indicates >45% downside; opportunity score cannot override this gate.")
        st.write("**DATA INSUFFICIENT** → insufficient fundamental/valuation data; no position should be inferred.")
        st.write("The Research Card is designed to answer four questions: **What is the decision? Why? What should confirm it? What would invalidate it?**")

    if section == "🔄 Sector Rotation":
        st.subheader("Sector Rotation")
        if sectors.empty:
            st.warning("Not enough sector data.")
        else:
            sector_cols = [
                "sector", "stocks", "sector_recovery_score", "sector_stage",
                "avg_60d_return_pct", "breadth_positive_pct",
                "avg_earnings_score", "avg_turnaround_score",
                "data_confidence_pct",
            ]
            st.dataframe(
                sectors[sector_cols].style.format({
                    "sector_recovery_score": "{:.1f}",
                    "avg_60d_return_pct": "{:.1f}%",
                    "breadth_positive_pct": "{:.0f}%",
                    "avg_earnings_score": "{:.1f}",
                    "avg_turnaround_score": "{:.1f}",
                    "data_confidence_pct": "{:.0f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )

    if section == "💰 Valuation":
        st.subheader("Valuation Context")
        st.caption(
            "Lower PE/cash-flow proxies score better within the scanned universe. "
            "This is relative screening, not intrinsic valuation."
        )
        val_cols = [
            "symbol", "company", "sector", "valuation_group",
            "valuation_method", "primary_multiple",
            "roe", "roce_for_valuation", "fcf_yield_pct",
            "valuation_growth_pct", "valuation_leverage",
            "valuation_data_quality_pct", "valuation_data_status",
            "valuation_warning", "valuation_disconnect",
        ]
        st.dataframe(
            scores[val_cols].style.format({
                "primary_multiple": "{:.1f}x",
                "roe": "{:.1%}",
                "roce_for_valuation": "{:.1f}%",
                "fcf_yield_pct": "{:.1f}%",
                "valuation_growth_pct": "{:.1f}%",
                "valuation_leverage": "{:.1f}x",
                "valuation_data_quality_pct": "{:.0f}%",
                "valuation_disconnect": "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.info(
            "V1.8.3 uses sector-aware valuation with explicit data-quality status. "
            "Financials → P/B + ROE; "
            "IT/Consumer → P/E with cash-flow/growth context; Manufacturing/Utilities → EV/EBITDA "
            "with ROCE/FCF; Cyclicals → P/E/FCF with an explicit normalized-earnings warning."
        )

        st.markdown("### Bear / Base / Bull Scenario")
        st.caption("V1.8 provides a transparent one-year research scenario, not a DCF, intrinsic-value estimate, or price prediction.")
        scenario_cols=["symbol","company","scenario_group","scenario_method","current_price","bear_value","base_value","bull_value","bear_return_pct","base_return_pct","bull_return_pct","scenario_reward_risk"]
        st.dataframe(
            scores[scenario_cols].style.format({
                "current_price":"₹{:.2f}","bear_value":"₹{:.2f}","base_value":"₹{:.2f}","bull_value":"₹{:.2f}",
                "bear_return_pct":"{:.1f}%","base_return_pct":"{:.1f}%","bull_return_pct":"{:.1f}%","scenario_reward_risk":"{:.2f}x"
            }),use_container_width=True,hide_index=True
        )

    if section == "🌐 Market Regime":
        st.subheader("Market Regime")
        st.caption(
            "A context layer for deciding how aggressively to deploy capital. "
            "It is not a prediction of the exact market bottom."
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Regime", market_regime.get("market_regime", "UNKNOWN"))
        m2.metric(
            "Score",
            f'{market_regime["market_regime_score"]:.0f}/100'
            if pd.notna(market_regime.get("market_regime_score"))
            else "N/A",
        )
        if pd.notna(market_regime.get("nifty_ma50")) and pd.notna(market_regime.get("nifty_close")):
            m3.metric(
                "Nifty vs 50DMA",
                f'{(market_regime["nifty_close"] / market_regime["nifty_ma50"] - 1) * 100:.1f}%',
            )
        else:
            m3.metric("Nifty vs 50DMA", "N/A")
        if pd.notna(market_regime.get("nifty_ma200")) and pd.notna(market_regime.get("nifty_close")):
            m4.metric(
                "Nifty vs 200DMA",
                f'{(market_regime["nifty_close"] / market_regime["nifty_ma200"] - 1) * 100:.1f}%',
            )
        else:
            m4.metric("Nifty vs 200DMA", "N/A")

        st.write("**Deployment bias:**", market_regime.get("deployment_bias", "N/A"))

        regime_table = pd.DataFrame([{
            "Regime Score": market_regime.get("market_regime_score"),
            "60D Return %": market_regime.get("market_return_60d_pct"),
            "200D Return %": market_regime.get("market_return_200d_pct"),
            "50DMA Slope %": market_regime.get("ma50_slope_pct"),
            "Scanned-Universe Breadth %": market_regime.get("breadth_pct"),
            "India VIX": market_regime.get("vix"),
        }])
        st.dataframe(regime_table, use_container_width=True, hide_index=True)

        st.markdown("**Regime interpretation**")
        st.write(
            "BEAR <35 · BOTTOMING 35–49 · EARLY RECOVERY 50–64 · "
            "RECOVERY 65–79 · BULL 80+"
        )

    if section == "🔎 Stock Diagnostics":
        st.subheader("Stock Diagnostics")
        st.caption(
            "Select any company. The scan is stored in session state, so changing this dropdown "
            "does not rerun the expensive scan or return you to Opportunities."
        )
        st.caption(
            "Price comes from the latest downloaded market close. Market cap uses yfinance "
            "quote fallbacks and remains N/A if the available public data is insufficient."
        )

        diagnostic_symbols = scores["symbol"].dropna().astype(str).tolist()
        selected = st.selectbox(
            "Select company",
            diagnostic_symbols,
            index=0,
            format_func=lambda s: (
                f"{s} — {scores.loc[scores['symbol'] == s, 'company'].iloc[0]}"
                if "company" in scores.columns
                and not scores.loc[scores["symbol"] == s].empty
                else s
            ),
            key="diagnostic_company",
        )

        row = scores.loc[scores["symbol"] == selected].iloc[0]

        st.markdown("### Company Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Company", row.get("company", selected))
        c2.metric("Sector", row.get("sector", "N/A"))
        c3.metric(
            "Market Cap",
            f"₹{float(row['market_cap']):,.0f}"
            if pd.notna(row.get("market_cap"))
            else "N/A",
        )
        price_value = row.get("current_price", row.get("price", np.nan))
        c4.metric(
            "Price",
            f"₹{float(price_value):,.2f}"
            if pd.notna(price_value)
            else "N/A",
        )

        st.markdown("### Investment Scores")
        score_fields = [
            ("Opportunity", "opportunity_score"),
            ("Turnaround", "turnaround_score"),
            ("Foundation", "foundation_score"),
            ("Sector", "sector_recovery_score"),
            ("Valuation", "valuation_disconnect"),
            ("Data", "data_completeness_pct"),
        ]
        score_cols = st.columns(6)
        for col, (label, field) in zip(score_cols, score_fields):
            value = row.get(field, np.nan)
            if pd.notna(value):
                suffix = "%" if field == "data_completeness_pct" else "/100"
                col.metric(label, f"{float(value):.0f}{suffix}")
            else:
                col.metric(label, "N/A")

        a, b = st.columns(2)
        a.metric("Turnaround Stage", row.get("turnaround_stage", "N/A"))
        b.metric("Sector Stage", row.get("sector_stage", "N/A"))

        st.markdown("### Turnaround Evidence")
        evidence_fields = [
            ("Revenue Growth %", "revenue_growth_pct"),
            ("Revenue Trend", "revenue_trend"),
            ("Net Income Growth %", "net_income_growth_pct"),
            ("Earnings Inflection", "earnings_inflection"),
            ("EBITDA Margin %", "ebitda_margin_pct"),
            ("Margin Inflection", "margin_inflection"),
            ("CFO Growth %", "cfo_growth_pct"),
            ("CFO Inflection", "cfo_inflection"),
            ("FCF Inflection", "fcf_inflection"),
            ("Debt Repair", "debt_repair"),
            ("ROCE Proxy %", "roce_proxy"),
            ("Debt / Equity", "debt_equity"),
            ("CFO / PAT", "cfo_pat"),
            ("60D Momentum %", "momentum_60d_pct"),
        ]
        evidence_rows = []
        for label, field in evidence_fields:
            value = row.get(field, np.nan)
            if pd.isna(value):
                display = "N/A"
            elif "%" in label:
                display = f"{float(value):.2f}%"
            elif label in {"Debt / Equity", "CFO / PAT"}:
                display = f"{float(value):.2f}x"
            else:
                display = f"{float(value):,.2f}"
            evidence_rows.append({"Signal": label, "Value": display})
        st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)

        st.markdown("### Valuation Diagnostics")
        valuation_fields = [
            ("PE Proxy", "pe_proxy"),
            ("Price / CFO", "price_to_cfo_proxy"),
            ("Price / FCF", "price_to_fcf_proxy"),
            ("Valuation Score", "valuation_disconnect"),
        ]
        valuation_rows = []
        for label, field in valuation_fields:
            value = row.get(field, np.nan)
            valuation_rows.append({
                "Metric": label,
                "Value": "N/A" if pd.isna(value) else f"{float(value):.2f}",
            })
        st.dataframe(pd.DataFrame(valuation_rows), use_container_width=True, hide_index=True)

        st.markdown("### Bear / Base / Bull Valuation")
        scols=st.columns(4)
        scols[0].metric("Bear", "N/A" if pd.isna(row.get("bear_value")) else f"₹{float(row['bear_value']):,.2f}")
        scols[1].metric("Base", "N/A" if pd.isna(row.get("base_value")) else f"₹{float(row['base_value']):,.2f}")
        scols[2].metric("Bull", "N/A" if pd.isna(row.get("bull_value")) else f"₹{float(row['bull_value']):,.2f}")
        scols[3].metric("Base Upside", "N/A" if pd.isna(row.get("base_return_pct")) else f"{float(row['base_return_pct']):.1f}%")
        growth_used = row.get("scenario_growth_used_pct", np.nan)
        growth_display = "N/A" if pd.isna(growth_used) else f"{float(growth_used):.1f}%"
        st.caption(
            f"Method: {row.get('scenario_method','N/A')} · Growth used: {growth_display} · "
            "Scenario values are research estimates, not guaranteed targets."
        )

        st.markdown("### Diagnostic Interpretation")
        positive, watch, risk = [], [], []
        for name, field in [
            ("Earnings inflection", "earnings_inflection"),
            ("Margin inflection", "margin_inflection"),
            ("CFO inflection", "cfo_inflection"),
            ("FCF inflection", "fcf_inflection"),
            ("Debt repair", "debt_repair"),
            ("Revenue trend", "revenue_trend"),
        ]:
            value = row.get(field, np.nan)
            if pd.notna(value):
                if float(value) >= 0.25:
                    positive.append(name)
                elif float(value) <= -0.25:
                    risk.append(name)
                else:
                    watch.append(name)

        momentum = row.get("momentum_60d_pct", np.nan)
        if pd.notna(momentum):
            (positive if float(momentum) > 0 else watch).append("60D price momentum")

        pcol, wcol, rcol = st.columns(3)
        with pcol:
            st.markdown("**🟢 Positive**")
            st.write(", ".join(positive) if positive else "No strong positive signal detected.")
        with wcol:
            st.markdown("**🟡 Watch**")
            st.write(", ".join(watch) if watch else "No major watch signal.")
        with rcol:
            st.markdown("**🔴 Risk / Weak**")
            st.write(", ".join(risk) if risk else "No major weak signal detected.")

        with st.expander("Show all calculated fields"):
            st.dataframe(
                pd.DataFrame([row]).T.rename(columns={row.name: "Value"}),
                use_container_width=True,
            )

    if section == "📈 Backtesting":
        st.subheader("Historical Backtesting")
        st.caption(
            "Walk-forward evaluation of point-in-time rankings saved by the screener. "
            "The engine does not reconstruct old fundamentals from today's data."
        )

        snapshots = load_snapshots()
        if snapshots.empty:
            st.info(
                "No scan snapshots are available yet. Each successful Run Scan saves a dated "
                "snapshot under data/snapshots/. Build history across multiple dates before judging the engine."
            )
        else:
            dates = pd.to_datetime(snapshots["as_of_date"], errors="coerce").dropna()
            st.write(
                f"Snapshots available: **{dates.dt.date.nunique()} dates** "
                f"({dates.min().date()} to {dates.max().date()})"
            )

            bt1, bt2 = st.columns(2)
            top_n = bt1.selectbox(
                "Top N stocks",
                [5, 10, 15, 20],
                index=1,
                key="bt_top_n",
            )
            horizon = bt2.selectbox(
                "Forward horizon",
                [3, 6, 12],
                index=1,
                key="bt_horizon",
            )

            tickers = sorted({
                str(s).upper() if str(s).upper().endswith(".NS")
                else f"{str(s).upper()}.NS"
                for s in snapshots["symbol"].dropna()
            })

            run_bt = st.button("▶ Run Backtest", type="primary", key="run_backtest")
            if run_bt:
                with st.spinner("Downloading historical prices..."):
                    px = download_backtest_prices(tickers, start=str(dates.min().date()))

                results = backtest_snapshots(
                    snapshots,
                    px,
                    top_n=top_n,
                    horizons=(horizon,),
                )

                completed = results.dropna(subset=["avg_return_pct"]) if not results.empty else pd.DataFrame()

                if completed.empty:
                    st.info(
                        "The snapshots exist, but the selected forward horizon has not completed "
                        "for any snapshot yet."
                    )
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Completed Test Dates", str(len(completed)))
                    m2.metric("Average Return", f"{completed['avg_return_pct'].mean():.1f}%")
                    m3.metric("Average Win Rate", f"{completed['win_rate_pct'].mean():.1f}%")
                    m4.metric("Average Max Drawdown", f"{completed['avg_max_drawdown_pct'].mean():.1f}%")

                    st.dataframe(
                        completed.style.format({
                            "avg_return_pct": "{:.1f}%",
                            "median_return_pct": "{:.1f}%",
                            "win_rate_pct": "{:.1f}%",
                            "worst_stock_return_pct": "{:.1f}%",
                            "best_stock_return_pct": "{:.1f}%",
                            "avg_max_drawdown_pct": "{:.1f}%",
                            "worst_max_drawdown_pct": "{:.1f}%",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

            with st.expander("How to interpret this"):
                st.write(
                    "A few successful observations are not enough to establish an edge. "
                    "We want many snapshots across bear, bottoming, recovery and bull regimes. "
                    "Entry is the first trading close after the snapshot date, which reduces same-day look-ahead."
                )

    if section == "📊 Data":
        st.subheader("Underlying Data")
        st.dataframe(scores, use_container_width=True, hide_index=True)
else:
    st.warning("Click **Run Scan** to fetch fresh data and calculate the ranking.")

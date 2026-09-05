import numpy as np
import pandas as pd

def _score_high(series):
    valid = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out = pd.Series(50.0, index=series.index)
    if valid.empty:
        return out
    out.loc[valid.index] = valid.rank(pct=True) * 100
    return out

def _score_low(series):
    return 100 - _score_high(series)

def _stage(row):
    s = row["turnaround_score"]
    if pd.isna(s):
        return "INSUFFICIENT DATA"
    if s >= 80:
        return "CONFIRMING"
    if s >= 65:
        return "EARLY"
    if s >= 50:
        return "WATCH"
    return "WEAK"

def calculate_scores(universe, fundamentals, prices):
    df = universe.merge(fundamentals, on="ticker", how="left")

    df["debt_equity"] = df["debt"] / df["equity"].replace(0, np.nan)
    df["cfo_pat"] = df["cfo"] / df["net_income"].replace(0, np.nan)
    capital = (df["equity"] + df["net_debt"]).replace(0, np.nan)
    df["roce_proxy"] = df["ebitda"] / capital * 100

    # Foundation / current-state quality.
    q1 = _score_high(df["roce_proxy"])
    q2 = _score_high(df["cfo_pat"].clip(-2, 5))
    q3 = _score_low(df["debt_equity"].clip(-1, 5))
    df["quality_score"] = 0.45*q1 + 0.30*q2 + 0.25*q3

    e1 = _score_high(df["revenue_growth_pct"])
    e2 = _score_high(df["net_income_growth_pct"])
    e3 = _score_high(df["ebitda_margin_pct"])
    e4 = _score_high(df["earnings_inflection"])
    df["earnings_score"] = 0.30*e1 + 0.25*e2 + 0.20*e3 + 0.25*e4

    c1 = _score_high(df["cfo"])
    c2 = _score_high(df["fcf"])
    c3 = _score_high(df["cfo_pat"].clip(-2, 5))
    c4 = _score_high(df["cfo_inflection"])
    df["cashflow_score"] = 0.30*c1 + 0.25*c2 + 0.20*c3 + 0.25*c4

    # Price momentum.
    momentum = []
    if prices is not None and not prices.empty:
        for ticker in df["ticker"]:
            p = prices.loc[prices["ticker"] == ticker].sort_values("date")["close"]
            if len(p) >= 60:
                momentum.append((ticker, (p.iloc[-1] / p.iloc[-60] - 1) * 100))
            else:
                momentum.append((ticker, np.nan))
    mom = pd.DataFrame(momentum, columns=["ticker", "momentum_60d_pct"])
    df = df.merge(mom, on="ticker", how="left")

    # Use the latest downloaded market close as the primary displayed price.
    # This is more reliable than depending on Yahoo quote metadata.
    current_prices = []
    if prices is not None and not prices.empty:
        for ticker in df["ticker"]:
            p = prices.loc[prices["ticker"] == ticker].sort_values("date")["close"]
            current_prices.append((ticker, float(p.iloc[-1]) if not p.empty else np.nan))
    price_df = pd.DataFrame(current_prices, columns=["ticker", "current_price"])
    df = df.merge(price_df, on="ticker", how="left")

    df["momentum_score"] = _score_high(df["momentum_60d_pct"])

    # Turnaround signals: change matters more than level.
    turnaround_components = {
        "earnings_inflection_score": _score_high(df["earnings_inflection"]),
        "margin_inflection_score": _score_high(df["margin_inflection"]),
        "debt_repair_score": _score_high(df["debt_repair"]),
        "cfo_inflection_score": _score_high(df["cfo_inflection"]),
        "fcf_inflection_score": _score_high(df["fcf_inflection"]),
        "revenue_trend_score": _score_high(df["revenue_trend"]),
    }
    for k, v in turnaround_components.items():
        df[k] = v

    df["turnaround_score"] = (
        0.25*df["earnings_inflection_score"] +
        0.20*df["margin_inflection_score"] +
        0.15*df["debt_repair_score"] +
        0.15*df["cfo_inflection_score"] +
        0.10*df["fcf_inflection_score"] +
        0.10*df["revenue_trend_score"] +
        0.05*df["momentum_score"]
    )

    df["foundation_score"] = (
        0.30*df["quality_score"] +
        0.25*df["earnings_score"] +
        0.20*df["cashflow_score"] +
        0.10*df["turnaround_score"] +
        0.15*df["momentum_score"]
    )

    df["data_completeness_pct"] = df[
        ["revenue", "ebitda", "net_income", "equity", "debt", "cfo"]
    ].notna().mean(axis=1) * 100

    # Keep display/valuation fields present even when a fundamentals provider
    # does not return quote metadata for a particular company.
    if "market_cap" not in df.columns:
        df["market_cap"] = np.nan
    if "current_price" not in df.columns:
        df["current_price"] = np.nan

    df["turnaround_stage"] = df.apply(_stage, axis=1)

    cols = [
        "symbol", "company", "sector", "ticker",
        "market_cap", "current_price",
        "foundation_score", "turnaround_score", "turnaround_stage",
        "quality_score", "earnings_score", "cashflow_score", "momentum_score",
        "revenue_growth_pct", "net_income_growth_pct", "ebitda_margin_pct",
        "earnings_inflection", "margin_inflection", "debt_repair",
        "cfo_growth_pct", "cfo_inflection", "fcf_inflection",
        "momentum_60d_pct", "roce_proxy", "debt_equity", "cfo_pat",
        "data_completeness_pct"
    ]
    return df[cols].sort_values(
        ["turnaround_score", "foundation_score"], ascending=False
    ).reset_index(drop=True)


def add_market_regime_modifier(df, regime_score: float, weight: float = 0.08):
    """Blend a small market-regime context into Opportunity Score.

    The modifier is intentionally small: stock-level fundamentals and turnaround
    evidence remain the primary drivers.
    """
    out = df.copy()
    if "Opportunity Score" in out.columns:
        base = pd.to_numeric(out["Opportunity Score"], errors="coerce")
    elif "opportunity_score" in out.columns:
        base = pd.to_numeric(out["opportunity_score"], errors="coerce")
    else:
        return out

    rs = float(regime_score) if np.isfinite(regime_score) else 50.0
    out["Market Regime Score"] = rs
    out["Opportunity Score Pre-Regime"] = base
    out["Opportunity Score"] = (base * (1 - weight) + rs * weight).clip(0, 100)
    return out

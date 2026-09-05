import numpy as np
import pandas as pd

def _rank(series):
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    out = pd.Series(50.0, index=s.index)
    if not valid.empty:
        out.loc[valid.index] = valid.rank(pct=True) * 100
    return out

def build_sector_scores(stock_scores, prices):
    """
    Build sector-level signals from the stocks already in the scan.

    This is intentionally constituent-based:
    - no paid sector feed
    - no hard dependency on a specific sector-index ticker
    - sector momentum = average constituent returns
    - breadth = % of constituents with positive return
    - earnings = average stock earnings score
    - turnaround = average stock turnaround score
    """
    df = stock_scores.copy()

    if df.empty or "sector" not in df.columns:
        return pd.DataFrame()

    rows = []
    for sector, g in df.groupby("sector", dropna=False):
        g = g.copy()
        sector = str(sector)

        returns = []
        if prices is not None and not prices.empty:
            for ticker in g["ticker"]:
                p = prices.loc[prices["ticker"] == ticker].sort_values("date")["close"]
                if len(p) >= 60:
                    returns.append((p.iloc[-1] / p.iloc[-60] - 1) * 100)

        avg_return = np.mean(returns) if returns else np.nan
        breadth = (np.array(returns) > 0).mean() * 100 if returns else np.nan

        rows.append({
            "sector": sector,
            "stocks": len(g),
            "avg_60d_return_pct": avg_return,
            "breadth_positive_pct": breadth,
            "avg_earnings_score": g["earnings_score"].mean(),
            "avg_turnaround_score": g["turnaround_score"].mean(),
            "avg_quality_score": g["quality_score"].mean(),
            "data_confidence_pct": g["data_completeness_pct"].mean(),
        })

    out = pd.DataFrame(rows)

    out["relative_strength_score"] = _rank(out["avg_60d_return_pct"])
    out["breadth_score"] = _rank(out["breadth_positive_pct"])
    out["earnings_score"] = _rank(out["avg_earnings_score"])
    out["turnaround_score"] = _rank(out["avg_turnaround_score"])
    out["quality_score"] = _rank(out["avg_quality_score"])

    # Sector recovery is driven primarily by price breadth and earnings improvement.
    out["sector_recovery_score"] = (
        0.30 * out["relative_strength_score"] +
        0.25 * out["breadth_score"] +
        0.25 * out["earnings_score"] +
        0.15 * out["turnaround_score"] +
        0.05 * out["quality_score"]
    )

    def stage(score):
        if pd.isna(score):
            return "INSUFFICIENT DATA"
        if score >= 80:
            return "LEADING"
        if score >= 65:
            return "RECOVERING"
        if score >= 50:
            return "WATCH"
        return "WEAK"

    out["sector_stage"] = out["sector_recovery_score"].apply(stage)

    return out.sort_values(
        ["sector_recovery_score", "breadth_positive_pct"],
        ascending=False
    ).reset_index(drop=True)

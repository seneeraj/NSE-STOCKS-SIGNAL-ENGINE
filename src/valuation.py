from __future__ import annotations

import numpy as np
import pandas as pd


FINANCIAL_SECTORS = {
    "financials", "bank", "banks", "nbfc", "insurance", "financial services"
}
IT_SECTORS = {"it", "information technology", "technology", "software"}
MANUFACTURING_SECTORS = {
    "manufacturing", "industrials", "capital goods", "engineering",
    "automobiles", "auto", "chemicals", "metals", "mining"
}
CONSUMER_SECTORS = {
    "consumer", "consumer discretionary", "consumer staples",
    "fmcg", "retail", "food & beverage"
}
UTILITY_SECTORS = {"utilities", "power", "oil & gas", "energy"}
CYCLICAL_SECTORS = {"metals", "mining", "commodities", "oil & gas", "chemicals", "cement"}


def _clean(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _sector_group(sector):
    s = str(sector or "").strip().lower()
    if s in FINANCIAL_SECTORS or any(k in s for k in ("bank", "nbfc", "insurance", "financial")):
        return "Financials"
    if s in IT_SECTORS or any(k in s for k in ("information technology", "software", "it services")):
        return "IT"
    if s in UTILITY_SECTORS or any(k in s for k in ("utility", "power", "energy")):
        return "Utilities"
    if s in CONSUMER_SECTORS or any(k in s for k in ("consumer", "fmcg", "retail")):
        return "Consumer"
    if s in CYCLICAL_SECTORS or any(k in s for k in ("metal", "mining", "commodity", "cement")):
        return "Cyclicals"
    if s in MANUFACTURING_SECTORS or any(k in s for k in ("industrial", "auto", "capital good", "engineering", "manufactur")):
        return "Manufacturing"
    return "General"


def _get(fundamentals, ticker):
    if fundamentals is None or fundamentals.empty or "ticker" not in fundamentals.columns:
        return {}
    row = fundamentals.loc[fundamentals["ticker"] == ticker]
    return row.iloc[0].to_dict() if not row.empty else {}


def _market_cap(row): return _clean(row.get("market_cap"))
def _net_income(row): return _clean(row.get("net_income"))
def _fcf(row): return _clean(row.get("fcf"))
def _equity(row): return _clean(row.get("equity"))
def _net_debt(row): return _clean(row.get("net_debt"))
def _roce(row): return _clean(row.get("roce_proxy"))


def _safe_ratio(num, den):
    if pd.isna(num) or pd.isna(den) or den == 0:
        return np.nan
    return num / den


def _intrinsic_proxy(row, group):
    """Sector-aware relative valuation inputs; deliberately not intrinsic value."""
    mcap = _market_cap(row)
    ni = _net_income(row)
    fcf = _fcf(row)
    equity = _equity(row)
    roce = _roce(row)
    net_debt = _net_debt(row)
    growth = _clean(row.get("revenue_growth_pct"))
    earnings_growth = _clean(row.get("net_income_growth_pct"))

    if group == "Financials":
        return {"primary_multiple": _safe_ratio(mcap, equity), "primary_name": "P/B", "roe": _safe_ratio(ni, equity)}

    if group == "IT":
        return {"primary_multiple": _safe_ratio(mcap, ni), "primary_name": "P/E", "fcf_yield": _safe_ratio(fcf, mcap)}

    if group == "Manufacturing":
        ebitda = _clean(row.get("ebitda"))
        ev = mcap + net_debt if pd.notna(mcap) and pd.notna(net_debt) else np.nan
        return {"primary_multiple": _safe_ratio(ev, ebitda), "primary_name": "EV/EBITDA", "roce": roce, "fcf_yield": _safe_ratio(fcf, mcap)}

    if group == "Consumer":
        return {"primary_multiple": _safe_ratio(mcap, ni), "primary_name": "P/E", "fcf_yield": _safe_ratio(fcf, mcap), "growth": growth}

    if group == "Utilities":
        ebitda = _clean(row.get("ebitda"))
        ev = mcap + net_debt if pd.notna(mcap) and pd.notna(net_debt) else np.nan
        return {"primary_multiple": _safe_ratio(ev, ebitda), "primary_name": "EV/EBITDA", "fcf_yield": _safe_ratio(fcf, mcap), "leverage": _safe_ratio(net_debt, equity)}

    if group == "Cyclicals":
        return {"primary_multiple": _safe_ratio(mcap, ni), "primary_name": "P/E (normalized needed)", "fcf_yield": _safe_ratio(fcf, mcap), "growth": earnings_growth if pd.notna(earnings_growth) else growth}

    return {"primary_multiple": _safe_ratio(mcap, ni), "primary_name": "P/E", "fcf_yield": _safe_ratio(fcf, mcap)}


def _quality_metadata(group, metrics):
    """Return only metrics that are actually applicable to the selected method."""
    applicable = {"Financials": ["primary_multiple", "roe"],
                  "IT": ["primary_multiple", "fcf_yield"],
                  "Manufacturing": ["primary_multiple", "roce", "fcf_yield"],
                  "Consumer": ["primary_multiple", "fcf_yield", "growth"],
                  "Utilities": ["primary_multiple", "fcf_yield", "leverage"],
                  "Cyclicals": ["primary_multiple", "fcf_yield", "growth"],
                  "General": ["primary_multiple", "fcf_yield"]}[group]
    available = [m for m in applicable if pd.notna(metrics.get(m, np.nan))]
    missing = [m for m in applicable if m not in available]
    quality = round(100 * len(available) / len(applicable), 1)
    warnings = []
    primary = metrics.get("primary_multiple", np.nan)
    growth = metrics.get("growth", np.nan)
    if pd.isna(primary):
        warnings.append("Primary valuation input unavailable")
    elif group in {"IT", "Consumer", "Cyclicals", "General"} and primary > 100:
        warnings.append("Very high P/E; earnings denominator may be temporarily depressed")
    elif group in {"Manufacturing", "Utilities"} and primary > 50:
        warnings.append("Very high EV/EBITDA; investigate earnings/capital-cycle normalization")
    if group == "Cyclicals":
        warnings.append("Cyclical earnings require normalization")
    if pd.notna(growth) and abs(growth) > 100:
        warnings.append("Extreme growth rate; likely base-effect distortion")
    if missing:
        warnings.append("Missing: " + ", ".join(missing))
    if quality >= 100:
        status = "COMPLETE"
    elif pd.notna(primary):
        status = "PARTIAL"
    else:
        status = "INSUFFICIENT"
    return applicable, available, quality, status, " | ".join(warnings)


def add_valuation_to_ranking(scores: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Add sector-aware relative valuation scores without fabricating missing data."""
    out = scores.copy()
    if out.empty:
        return out

    rows = []
    for _, stock in out.iterrows():
        ticker = stock.get("ticker", stock.get("symbol"))
        row = {**_get(fundamentals, ticker), **stock}
        group = _sector_group(stock.get("sector"))
        metrics = _intrinsic_proxy(row, group)
        applicable, available, quality, status, warning = _quality_metadata(group, metrics)
        rows.append({
            "symbol": stock.get("symbol"), "valuation_group": group,
            "valuation_method": metrics.get("primary_name", "P/E"),
            "primary_multiple": metrics.get("primary_multiple", np.nan),
            "roe": metrics.get("roe", np.nan),
            "roce_for_valuation": metrics.get("roce", np.nan),
            "fcf_yield_pct": metrics.get("fcf_yield", np.nan) * 100 if pd.notna(metrics.get("fcf_yield", np.nan)) else np.nan,
            "valuation_growth_pct": metrics.get("growth", np.nan),
            "valuation_leverage": metrics.get("leverage", np.nan),
            "valuation_applicable_metrics": ", ".join(applicable),
            "valuation_available_metrics": ", ".join(available),
            "valuation_data_quality_pct": quality,
            "valuation_data_status": status,
            "valuation_warning": warning,
        })

    v = pd.DataFrame(rows)
    out = out.merge(v, on="symbol", how="left")

    def rank_lower(s):
        valid = s.notna()
        result = pd.Series(np.nan, index=s.index, dtype=float)
        if valid.any(): result.loc[valid] = s.loc[valid].rank(pct=True, ascending=False) * 100
        return result

    def rank_higher(s):
        valid = s.notna()
        result = pd.Series(np.nan, index=s.index, dtype=float)
        if valid.any(): result.loc[valid] = s.loc[valid].rank(pct=True, ascending=True) * 100
        return result

    primary = rank_lower(out["primary_multiple"])
    fcf = rank_higher(out["fcf_yield_pct"])
    roe = rank_higher(out["roe"])
    roce = rank_higher(out["roce_for_valuation"])
    growth = rank_higher(out["valuation_growth_pct"])

    scores_list = []
    for i in out.index:
        group = out.at[i, "valuation_group"]
        parts = []
        if pd.notna(primary.at[i]): parts.append((primary.at[i], 0.60))
        if group == "Financials" and pd.notna(roe.at[i]): parts.append((roe.at[i], 0.40))
        elif group in {"IT", "Consumer", "Cyclicals"}:
            if pd.notna(fcf.at[i]): parts.append((fcf.at[i], 0.25))
            if group == "Consumer" and pd.notna(growth.at[i]): parts.append((growth.at[i], 0.15))
        elif group in {"Manufacturing", "Utilities"}:
            if pd.notna(roce.at[i]): parts.append((roce.at[i], 0.20))
            if pd.notna(fcf.at[i]): parts.append((fcf.at[i], 0.20))
        elif pd.notna(fcf.at[i]): parts.append((fcf.at[i], 0.40))
        scores_list.append(sum(x*w for x, w in parts) / sum(w for _, w in parts) if parts else np.nan)

    out["valuation_disconnect"] = scores_list
    out["valuation_data_confidence_pct"] = out["valuation_data_quality_pct"]
    out["pe_proxy"] = np.where(out["valuation_method"].str.contains("P/E", na=False), out["primary_multiple"], np.nan)
    out["price_to_cfo_proxy"] = np.nan
    out["price_to_fcf_proxy"] = np.where(out["fcf_yield_pct"].notna() & (out["fcf_yield_pct"] != 0), 100 / out["fcf_yield_pct"], np.nan)
    return out

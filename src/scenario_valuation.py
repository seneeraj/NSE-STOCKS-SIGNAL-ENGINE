from __future__ import annotations

import numpy as np
import pandas as pd


def _num(x):
    try:
        x=float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _ratio(a,b):
    a,b=_num(a),_num(b)
    return a/b if pd.notna(a) and pd.notna(b) and b != 0 else np.nan


def _scenario_targets(base_value, method, group):
    """Return deliberately conservative/central/aggressive target assumptions."""
    if group=="Financials":
        return {
            "Bear": base_value*0.75,
            "Base": base_value*1.00,
            "Bull": base_value*1.25,
        }
    if group=="IT":
        return {
            "Bear": base_value*0.75,
            "Base": base_value*1.05,
            "Bull": base_value*1.30,
        }
    if group in {"Manufacturing","Utilities"}:
        return {
            "Bear": base_value*0.70,
            "Base": base_value*1.05,
            "Bull": base_value*1.35,
        }
    if group=="Consumer":
        return {
            "Bear": base_value*0.80,
            "Base": base_value*1.05,
            "Bull": base_value*1.25,
        }
    if group=="Cyclicals":
        return {
            "Bear": base_value*0.65,
            "Base": base_value*1.05,
            "Bull": base_value*1.45,
        }
    return {"Bear":base_value*0.75,"Base":base_value*1.05,"Bull":base_value*1.30}


def build_scenario_valuation(row):
    """Build a one-year Bear/Base/Bull per-share valuation.

    This is a research scenario model, not a DCF or price prediction.
    Inputs are deliberately transparent and conservative. The base operating
    metric is grown modestly using available company trends; valuation
    multiples are anchored to current sector-aware multiples.
    """
    group=str(row.get("valuation_group","General"))
    method=str(row.get("valuation_method","P/E"))
    price=_num(row.get("current_price"))
    mcap=_num(row.get("market_cap"))
    ni=_num(row.get("net_income"))
    equity=_num(row.get("equity"))
    ebitda=_num(row.get("ebitda"))
    net_debt=_num(row.get("net_debt"))
    fcf=_num(row.get("fcf"))
    rev_growth=_num(row.get("revenue_growth_pct"))
    earn_growth=_num(row.get("net_income_growth_pct"))
    roce=_num(row.get("roce_proxy"))

    shares=_ratio(mcap,price)
    if pd.isna(shares) or shares <= 0:
        shares=np.nan

    current_eps=_ratio(ni,shares)
    current_bvps=_ratio(equity,shares)

    growth=earn_growth if pd.notna(earn_growth) else rev_growth
    if pd.isna(growth):
        growth=8.0
    # Avoid one-period outliers dominating the scenario.
    growth=float(np.clip(growth,-10,25))
    normalized_eps=current_eps*(1+growth/100) if pd.notna(current_eps) else np.nan

    if group=="Financials":
        current_pb=_ratio(mcap,equity)
        if pd.isna(current_pb): return _empty_result(group,method,price,"Insufficient market cap/equity data")
        roe=_ratio(ni,equity)
        roe_pct=roe*100 if pd.notna(roe) else np.nan
        # Higher sustainable ROE supports a somewhat higher P/B, but keep
        # scenario multiples bounded to avoid false precision.
        anchor=max(0.8,min(3.5,current_pb))
        if pd.notna(roe_pct):
            anchor=max(0.8,min(3.5,0.8+roe_pct/10))
        multiples={"Bear":max(0.7,anchor*0.75),"Base":anchor,"Bull":anchor*1.25}
        targets={k:current_bvps*v if pd.notna(current_bvps) else np.nan for k,v in multiples.items()}
        metric="P/B"
        anchor_display=anchor

    elif group in {"Manufacturing","Utilities"}:
        if pd.isna(ebitda) or pd.isna(mcap) or pd.isna(net_debt) or pd.isna(shares):
            return _empty_result(group,method,price,"Insufficient EBITDA/debt/share data")
        current_ev=mcap+net_debt
        current_ev_ebitda=_ratio(current_ev,ebitda)
        if pd.isna(current_ev_ebitda) or current_ev_ebitda<=0:
            return _empty_result(group,method,price,"Invalid EV/EBITDA")
        # Use a modest operating-growth adjustment; margins are not projected
        # independently because historical margin series are already noisy.
        op_growth=float(np.clip(rev_growth if pd.notna(rev_growth) else growth,-5,20))
        future_ebitda=ebitda*(1+op_growth/100)
        multiples={"Bear":current_ev_ebitda*0.80,"Base":current_ev_ebitda*1.00,"Bull":current_ev_ebitda*1.20}
        targets={}
        for k,m in multiples.items():
            future_ev=future_ebitda*m
            equity_value=future_ev-net_debt
            targets[k]=equity_value/shares if pd.notna(shares) else np.nan
        metric="EV/EBITDA"
        anchor_display=current_ev_ebitda

    else:
        if pd.isna(current_eps) or current_eps<=0:
            return _empty_result(group,method,price,"Current earnings not positive/available")
        current_pe=_ratio(mcap,ni)
        if pd.isna(current_pe) or current_pe<=0:
            return _empty_result(group,method,price,"Invalid P/E")
        if group=="Consumer":
            multiples={"Bear":current_pe*0.85,"Base":current_pe*1.00,"Bull":current_pe*1.15}
        elif group=="Cyclicals":
            # Current-cycle P/E is explicitly only an anchor. Use a wider range.
            multiples={"Bear":current_pe*0.70,"Base":current_pe*1.00,"Bull":current_pe*1.30}
        else:
            multiples={"Bear":current_pe*0.75,"Base":current_pe*1.00,"Bull":current_pe*1.25}
        targets={k:normalized_eps*m for k,m in multiples.items()}
        metric="P/E"
        anchor_display=current_pe

    result={
        "scenario_group":group,
        "scenario_method":metric,
        "scenario_anchor_multiple":anchor_display,
        "scenario_growth_used_pct":growth,
        "bear_value":targets.get("Bear",np.nan),
        "base_value":targets.get("Base",np.nan),
        "bull_value":targets.get("Bull",np.nan),
        "current_price":price,
        "bear_return_pct":_ratio(targets.get("Bear",np.nan),price)*100-100 if pd.notna(targets.get("Bear",np.nan)) and pd.notna(price) else np.nan,
        "base_return_pct":_ratio(targets.get("Base",np.nan),price)*100-100 if pd.notna(targets.get("Base",np.nan)) and pd.notna(price) else np.nan,
        "bull_return_pct":_ratio(targets.get("Bull",np.nan),price)*100-100 if pd.notna(targets.get("Bull",np.nan)) and pd.notna(price) else np.nan,
        "scenario_confidence": "Research scenario — not intrinsic value",
    }
    return result


def _empty_result(group,method,price,reason):
    return {
        "scenario_group":group,
        "scenario_method":method,
        "scenario_anchor_multiple":np.nan,
        "scenario_growth_used_pct":np.nan,
        "bear_value":np.nan,"base_value":np.nan,"bull_value":np.nan,
        "current_price":price,
        "bear_return_pct":np.nan,"base_return_pct":np.nan,"bull_return_pct":np.nan,
        "scenario_confidence":reason,
    }


def add_scenario_valuation(scores):
    if scores is None or scores.empty:
        return scores.copy()
    out=scores.copy()
    rows=[build_scenario_valuation(row) for _,row in out.iterrows()]
    scen=pd.DataFrame(rows,index=out.index)
    for col in scen.columns:
        out[col]=scen[col]
    # Risk/reward score is deliberately separate from the existing valuation score.
    upside=out["base_return_pct"]
    downside=out["bear_return_pct"]
    asymmetry=np.where(
        pd.notna(upside)&pd.notna(downside)&(downside<0),
        upside/np.abs(downside),
        np.nan,
    )
    out["scenario_reward_risk"]=asymmetry
    return out

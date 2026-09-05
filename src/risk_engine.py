from __future__ import annotations

import numpy as np
import pandas as pd


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _clip(x, lo=0.0, hi=100.0):
    return float(np.clip(_num(x) if pd.notna(_num(x)) else 50.0, lo, hi))


def _survival_score(row):
    """Capital-preservation score. Missing inputs are neutral, not fabricated."""
    parts = []
    de = _num(row.get("debt_equity"))
    if pd.notna(de):
        parts.append(np.clip(100 - (de / 2.0) * 100, 0, 100))
    cp = _num(row.get("cfo_pat"))
    if pd.notna(cp):
        parts.append(np.clip((cp + 1) / 2.0 * 100, 0, 100))
    dc = _num(row.get("data_completeness_pct"))
    if pd.notna(dc):
        parts.append(np.clip(dc, 0, 100))
    return float(np.mean(parts)) if parts else 50.0


def _scenario_score(row):
    base = _num(row.get("base_return_pct"))
    bear = _num(row.get("bear_return_pct"))
    rr = _num(row.get("scenario_reward_risk"))
    if pd.isna(base):
        return 50.0
    upside = np.clip(base, -50, 100) / 100 * 50 + 50
    downside = 50.0 if pd.isna(bear) else np.clip(100 + bear, 0, 100)
    asym = 50.0 if pd.isna(rr) else np.clip(rr / 3.0 * 100, 0, 100)
    return float(0.45 * upside + 0.35 * downside + 0.20 * asym)


def _stage_score(stage):
    return {"CONFIRMING": 100, "EARLY": 80, "WATCH": 55, "WEAK": 25}.get(str(stage), 50)


def _decision(row):
    opp = _num(row.get("opportunity_score"))
    risk = _num(row.get("risk_score"))
    stage = str(row.get("turnaround_stage", ""))
    quality = str(row.get("valuation_data_status", ""))
    base = _num(row.get("base_return_pct"))
    bear = _num(row.get("bear_return_pct"))

    # Hard gates protect the ranking from becoming a "cheap stock" list.
    if quality == "INSUFFICIENT" or _num(row.get("data_completeness_pct")) < 50:
        return "DATA INSUFFICIENT"
    if pd.notna(bear) and bear < -45:
        return "HIGH DOWNSIDE RISK"
    if stage == "WEAK" and pd.notna(base) and base < 10:
        return "AVOID / NO EDGE"

    if opp >= 75 and risk >= 70 and stage in {"EARLY", "CONFIRMING"} and (pd.isna(base) or base >= 10):
        return "ATTRACTIVE RISK/REWARD"
    if opp >= 65 and risk >= 55:
        return "PROMISING — WATCH / CONFIRM"
    if opp >= 55:
        return "WATCH — NEED MORE EVIDENCE"
    return "LOW PRIORITY"



def _decision_rationale(row):
    decision = str(row.get("investment_decision", ""))
    stage = str(row.get("turnaround_stage", ""))
    opp = _num(row.get("opportunity_score"))
    risk = _num(row.get("risk_score"))
    base = _num(row.get("base_return_pct"))
    bear = _num(row.get("bear_return_pct"))
    flags = str(row.get("risk_flags", "NONE"))

    if decision == "ATTRACTIVE RISK/REWARD":
        return "Strong opportunity and risk evidence with an EARLY/CONFIRMING turnaround; consider only staged entry after validating the thesis."
    if decision == "PROMISING — WATCH / CONFIRM":
        return "The opportunity is promising, but evidence is not yet strong enough for a higher-conviction decision."
    if decision == "WATCH — NEED MORE EVIDENCE":
        return "Interesting signals exist, but the current evidence does not provide enough edge for capital deployment."
    if decision == "HIGH DOWNSIDE RISK":
        return "The modeled Bear case implies severe downside; wait for better valuation, stronger evidence, or a change in the thesis."
    if decision == "DATA INSUFFICIENT":
        return "The available data is insufficient for a reliable investment conclusion."
    if decision == "AVOID / NO EDGE":
        return "The current turnaround/valuation evidence does not justify taking capital risk."
    return "The stock is not currently a high-priority opportunity under the screening framework."


def _research_priority(row):
    decision = str(row.get("investment_decision", ""))
    flags = str(row.get("risk_flags", "NONE"))
    if decision == "ATTRACTIVE RISK/REWARD":
        return "HIGH"
    if decision == "PROMISING — WATCH / CONFIRM":
        return "HIGH"
    if decision == "WATCH — NEED MORE EVIDENCE":
        return "MEDIUM"
    if decision == "DATA INSUFFICIENT":
        return "DATA"
    if "HIGH DEBT/EQUITY" in flags or "WEAK CFO/PAT" in flags or "BEAR DOWNSIDE >30%" in flags:
        return "HIGH"
    return "LOW"


def _next_confirmation(row):
    stage = str(row.get("turnaround_stage", ""))
    flags = str(row.get("risk_flags", "NONE"))
    checks = []
    if stage in {"WATCH", "EARLY"}:
        checks.append("Confirm revenue/earnings and margin improvement across the next reported periods")
    if "WEAK CFO/PAT" in flags:
        checks.append("Confirm operating cash flow catches up with reported profit")
    if "HIGH DEBT/EQUITY" in flags:
        checks.append("Confirm debt reduction and improving interest coverage")
    if "BEAR DOWNSIDE >30%" in flags:
        checks.append("Reassess entry valuation and downside protection")
    if "PARTIAL VALUATION DATA" in flags or str(row.get("valuation_data_status","")) != "COMPLETE":
        checks.append("Resolve missing valuation inputs")
    if not checks:
        checks.append("Revalidate the thesis at the next quarterly results")
    return " | ".join(dict.fromkeys(checks))


def _thesis_invalidation(row):
    flags = str(row.get("risk_flags", "NONE"))
    items = []
    if "WEAK CFO/PAT" in flags:
        items.append("cash conversion remains weak")
    if "HIGH DEBT/EQUITY" in flags:
        items.append("leverage fails to improve")
    if "TURNAROUND NOT YET CONFIRMED" in flags:
        items.append("earnings/margins fail to improve")
    if "Cyclical earnings require normalization" in flags:
        items.append("normalized earnings deteriorate")
    if not items:
        items.append("the expected fundamental improvement reverses")
    return " or ".join(items)


def add_decision_matrix_details(out):
    out["decision_rationale"] = out.apply(_decision_rationale, axis=1)
    out["research_priority"] = out.apply(_research_priority, axis=1)
    out["next_confirmation"] = out.apply(_next_confirmation, axis=1)
    out["thesis_invalidation"] = out.apply(_thesis_invalidation, axis=1)
    return out

def add_risk_engine(scores: pd.DataFrame) -> pd.DataFrame:
    """Add a transparent investment-decision layer above the existing scores.

    This is a screening/risk framework, not individualized investment advice.
    It deliberately uses hard data gates and never converts missing data to a
    bullish assumption.
    """
    out = scores.copy()
    if out.empty:
        return out

    survival = out.apply(_survival_score, axis=1)
    scenario = out.apply(_scenario_score, axis=1)
    def _series(name):
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(50)
        return pd.Series(50.0, index=out.index)

    valuation = _series("valuation_disconnect")
    turnaround = _series("turnaround_score")
    foundation = _series("foundation_score")
    sector = _series("sector_recovery_score")
    regime = _series("market_regime_score")

    out["survival_score"] = survival
    out["scenario_risk_reward_score"] = scenario
    out["risk_score"] = (
        0.30 * survival
        + 0.25 * scenario
        + 0.15 * valuation
        + 0.15 * foundation
        + 0.10 * sector
        + 0.05 * regime
    ).clip(0, 100)

    out["investment_decision"] = out.apply(_decision, axis=1)
    out["position_bias"] = out["investment_decision"].map({
        "ATTRACTIVE RISK/REWARD": "START SMALL / STAGE",
        "PROMISING — WATCH / CONFIRM": "WATCHLIST",
        "WATCH — NEED MORE EVIDENCE": "WATCHLIST",
        "HIGH DOWNSIDE RISK": "AVOID / WAIT",
        "AVOID / NO EDGE": "AVOID",
        "DATA INSUFFICIENT": "NO POSITION",
        "LOW PRIORITY": "NO PRIORITY",
    }).fillna("WATCHLIST")
    out["risk_flags"] = out.apply(_flags, axis=1)
    out = add_decision_matrix_details(out)
    return out


def _flags(row):
    flags = []
    de = _num(row.get("debt_equity"))
    cp = _num(row.get("cfo_pat"))
    bear = _num(row.get("bear_return_pct"))
    quality = str(row.get("valuation_data_status", ""))
    if pd.notna(de) and de > 1:
        flags.append("HIGH DEBT/EQUITY")
    if pd.notna(cp) and cp < 0.8:
        flags.append("WEAK CFO/PAT")
    if pd.notna(bear) and bear < -30:
        flags.append("BEAR DOWNSIDE >30%")
    if quality == "PARTIAL":
        flags.append("PARTIAL VALUATION DATA")
    warning = row.get("valuation_warning")
    if pd.notna(warning) and str(warning).strip():
        flags.append(str(warning))
    if str(row.get("turnaround_stage", "")) == "WATCH":
        flags.append("TURNAROUND NOT YET CONFIRMED")
    return " | ".join(dict.fromkeys(flags)) if flags else "NONE"

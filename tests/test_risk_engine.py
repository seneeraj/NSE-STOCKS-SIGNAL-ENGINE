import pandas as pd
from src.risk_engine import add_risk_engine


def test_attractive_risk_reward():
    df = pd.DataFrame([{
        "symbol": "ABC", "opportunity_score": 82, "turnaround_score": 85,
        "foundation_score": 80, "sector_recovery_score": 75, "market_regime_score": 60,
        "valuation_disconnect": 80, "data_completeness_pct": 100,
        "debt_equity": 0.3, "cfo_pat": 1.2, "turnaround_stage": "EARLY",
        "valuation_data_status": "COMPLETE", "base_return_pct": 30,
        "bear_return_pct": -15, "scenario_reward_risk": 2.0,
    }])
    out = add_risk_engine(df)
    assert out.iloc[0]["investment_decision"] == "ATTRACTIVE RISK/REWARD"
    assert out.iloc[0]["risk_score"] > 70


def test_insufficient_data_is_gated():
    df = pd.DataFrame([{
        "symbol": "BAD", "opportunity_score": 90, "data_completeness_pct": 40,
        "valuation_data_status": "INSUFFICIENT", "turnaround_stage": "CONFIRMING"
    }])
    out = add_risk_engine(df)
    assert out.iloc[0]["investment_decision"] == "DATA INSUFFICIENT"
    assert out.iloc[0]["position_bias"] == "NO POSITION"


def test_large_bear_downside_is_gated():
    df = pd.DataFrame([{
        "symbol": "RISK", "opportunity_score": 90, "foundation_score": 80,
        "turnaround_score": 85, "sector_recovery_score": 80, "valuation_disconnect": 90,
        "market_regime_score": 70, "data_completeness_pct": 100,
        "debt_equity": 0.2, "cfo_pat": 1.5, "turnaround_stage": "CONFIRMING",
        "valuation_data_status": "COMPLETE", "base_return_pct": 50,
        "bear_return_pct": -50, "scenario_reward_risk": 1.0,
    }])
    out = add_risk_engine(df)
    assert out.iloc[0]["investment_decision"] == "HIGH DOWNSIDE RISK"


def test_flags_are_transparent():
    df = pd.DataFrame([{
        "symbol": "FLAG", "opportunity_score": 60, "foundation_score": 60,
        "turnaround_score": 55, "sector_recovery_score": 50, "valuation_disconnect": 60,
        "market_regime_score": 50, "data_completeness_pct": 100,
        "debt_equity": 1.5, "cfo_pat": 0.5, "turnaround_stage": "WATCH",
        "valuation_data_status": "PARTIAL", "valuation_warning": "Very high P/E",
    }])
    out = add_risk_engine(df)
    flags = out.iloc[0]["risk_flags"]
    assert "HIGH DEBT/EQUITY" in flags
    assert "WEAK CFO/PAT" in flags
    assert "PARTIAL VALUATION DATA" in flags
    assert "TURNAROUND NOT YET CONFIRMED" in flags

from src.risk_engine import add_decision_matrix_details

def test_decision_matrix_details():
    df = pd.DataFrame([{
        "investment_decision": "PROMISING — WATCH / CONFIRM",
        "turnaround_stage": "EARLY",
        "opportunity_score": 70,
        "risk_score": 60,
        "base_return_pct": 20,
        "bear_return_pct": -25,
        "risk_flags": "WEAK CFO/PAT",
        "valuation_data_status": "COMPLETE",
    }])
    out = add_decision_matrix_details(df)
    for col in ["decision_rationale","research_priority","next_confirmation","thesis_invalidation"]:
        assert col in out.columns
        assert str(out.iloc[0][col]).strip()

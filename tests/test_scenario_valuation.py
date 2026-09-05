import pandas as pd
from src.scenario_valuation import build_scenario_valuation, add_scenario_valuation

def test_bank_scenario_uses_pb():
    row={"valuation_group":"Financials","valuation_method":"P/B","current_price":100,
         "market_cap":1000,"net_income":100,"equity":500}
    r=build_scenario_valuation(row)
    assert r["scenario_method"]=="P/B"
    assert r["base_value"]>0

def test_manufacturing_scenario_uses_ev_ebitda():
    row={"valuation_group":"Manufacturing","valuation_method":"EV/EBITDA","current_price":100,
         "market_cap":1000,"net_debt":200,"ebitda":150,"revenue_growth_pct":10}
    r=build_scenario_valuation(row)
    assert r["scenario_method"]=="EV/EBITDA"
    assert r["base_value"]>0

def test_insufficient_data_is_not_fabricated():
    row={"valuation_group":"Financials","valuation_method":"P/B","current_price":100}
    r=build_scenario_valuation(row)
    assert pd.isna(r["base_value"])
    assert "Insufficient" in r["scenario_confidence"]

def test_add_scenario_preserves_rows():
    scores=pd.DataFrame([{"symbol":"ABC","valuation_group":"IT","valuation_method":"P/E",
                          "current_price":100,"market_cap":1000,"net_income":50,
                          "revenue_growth_pct":10,"net_income_growth_pct":12}])
    out=add_scenario_valuation(scores)
    assert len(out)==1 and "base_value" in out.columns


def test_scenario_works_with_full_fundamental_inputs():
    scores = pd.DataFrame([{
        "symbol":"MFG","ticker":"MFG","sector":"Manufacturing",
        "valuation_group":"Manufacturing","valuation_method":"EV/EBITDA",
        "current_price":100,"market_cap":1000
    }])
    fundamentals = pd.DataFrame([{
        "ticker":"MFG","revenue_growth_pct":10,"net_income_growth_pct":12,
        "net_income":50,"equity":500,"ebitda":150,"net_debt":200,
        "fcf":80,"roce_proxy":18
    }])
    merged=scores.merge(fundamentals,on="ticker",how="left")
    out=add_scenario_valuation(merged)
    assert out.iloc[0]["base_value"] > 0
    assert pd.notna(out.iloc[0]["base_return_pct"])

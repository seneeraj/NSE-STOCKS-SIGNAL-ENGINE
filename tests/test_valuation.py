import numpy as np
import pandas as pd
from src.valuation import add_valuation_to_ranking, _sector_group


def test_sector_groups():
    assert _sector_group("Financials") == "Financials"
    assert _sector_group("Information Technology") == "IT"
    assert _sector_group("Automobiles") == "Manufacturing"
    assert _sector_group("Consumer Discretionary") == "Consumer"
    assert _sector_group("Utilities") == "Utilities"


def test_financials_use_pb_and_roe():
    scores=pd.DataFrame([{"symbol":"BANK","ticker":"BANK","sector":"Financials","market_cap":500,"net_income":50,"equity":250}])
    fundamentals=pd.DataFrame([{"ticker":"BANK","market_cap":500,"net_income":50,"equity":250}])
    out=add_valuation_to_ranking(scores,fundamentals)
    assert out.iloc[0]["valuation_method"]=="P/B"
    assert out.iloc[0]["primary_multiple"]==2.0
    assert out.iloc[0]["roe"]==0.2
    assert out.iloc[0]["valuation_data_status"]=="COMPLETE"
    assert out.iloc[0]["valuation_data_quality_pct"]==100.0


def test_manufacturing_uses_ev_ebitda():
    scores=pd.DataFrame([{"symbol":"MFG","ticker":"MFG","sector":"Manufacturing","market_cap":1000,"net_debt":200,"ebitda":150,"fcf":80,"roce_proxy":18}])
    fundamentals=pd.DataFrame([{"ticker":"MFG","market_cap":1000,"net_debt":200,"ebitda":150,"fcf":80,"roce_proxy":18}])
    out=add_valuation_to_ranking(scores,fundamentals)
    assert out.iloc[0]["valuation_method"]=="EV/EBITDA"
    assert abs(out.iloc[0]["primary_multiple"]-8.0)<1e-9
    assert out.iloc[0]["valuation_data_status"]=="COMPLETE"


def test_missing_applicable_metric_is_flagged_not_filled():
    scores=pd.DataFrame([{"symbol":"MFG","ticker":"MFG","sector":"Manufacturing","market_cap":1000,"net_debt":200,"ebitda":150}])
    fundamentals=pd.DataFrame([{"ticker":"MFG","market_cap":1000,"net_debt":200,"ebitda":150}])
    out=add_valuation_to_ranking(scores,fundamentals)
    row=out.iloc[0]
    assert pd.isna(row["roce_for_valuation"])
    assert pd.isna(row["fcf_yield_pct"])
    assert row["valuation_data_status"] == "PARTIAL"
    assert "roce" in row["valuation_warning"]
    assert "fcf_yield" in row["valuation_warning"]


def test_extreme_pe_gets_warning():
    scores=pd.DataFrame([{"symbol":"ITX","ticker":"ITX","sector":"IT","market_cap":1000,"net_income":5,"fcf":20}])
    fundamentals=pd.DataFrame([{"ticker":"ITX","market_cap":1000,"net_income":5,"fcf":20}])
    out=add_valuation_to_ranking(scores,fundamentals)
    assert out.iloc[0]["primary_multiple"]==200
    assert "Very high P/E" in out.iloc[0]["valuation_warning"]


def test_extreme_growth_gets_warning():
    scores=pd.DataFrame([{"symbol":"CYC","ticker":"CYC","sector":"Cement","market_cap":1000,"net_income":50,"fcf":50,"net_income_growth_pct":250}])
    fundamentals=pd.DataFrame([{"ticker":"CYC","market_cap":1000,"net_income":50,"fcf":50,"net_income_growth_pct":250}])
    out=add_valuation_to_ranking(scores,fundamentals)
    assert "Extreme growth rate" in out.iloc[0]["valuation_warning"]


def test_empty_input():
    out=add_valuation_to_ranking(pd.DataFrame(),pd.DataFrame())
    assert out.empty

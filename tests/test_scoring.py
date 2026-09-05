import pandas as pd
from src.scoring import calculate_scores

def test_turnaround_ranking_returns_expected_columns():
    universe = pd.DataFrame([
        {"symbol": "AAA", "company": "A", "sector": "Test", "ticker": "AAA.NS"},
        {"symbol": "BBB", "company": "B", "sector": "Test", "ticker": "BBB.NS"},
    ])
    fundamentals = pd.DataFrame([
        {"ticker":"AAA.NS","revenue":100,"revenue_growth_pct":10,"revenue_trend":20,
         "ebitda":20,"ebitda_margin_pct":20,"margin_inflection":5,
         "net_income":10,"net_income_growth_pct":15,"earnings_inflection":4,
         "equity":50,"debt":10,"cash":5,"net_debt":5,"debt_repair":4,
         "cfo":12,"cfo_growth_pct":20,"cfo_inflection":5,"capex":-3,"fcf":9,"fcf_inflection":4},
        {"ticker":"BBB.NS","revenue":100,"revenue_growth_pct":2,"revenue_trend":2,
         "ebitda":10,"ebitda_margin_pct":10,"margin_inflection":-2,
         "net_income":5,"net_income_growth_pct":1,"earnings_inflection":-1,
         "equity":50,"debt":30,"cash":2,"net_debt":28,"debt_repair":-3,
         "cfo":5,"cfo_growth_pct":2,"cfo_inflection":-2,"capex":-4,"fcf":1,"fcf_inflection":-2},
    ])
    prices = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=60),
        "ticker": ["AAA.NS"]*60,
        "close": range(100,160)
    })
    result = calculate_scores(universe, fundamentals, prices)
    assert "turnaround_score" in result.columns
    assert "turnaround_stage" in result.columns
    assert result.iloc[0]["symbol"] == "AAA"



def test_current_price_comes_from_latest_market_close():
    universe = pd.DataFrame([{"symbol":"ABC","ticker":"ABC","company":"ABC","sector":"IT"}])
    fundamentals = pd.DataFrame([{
        "ticker":"ABC","market_cap":1000,
        "revenue":100,"revenue_growth_pct":10,"revenue_trend":1,
        "ebitda":20,"ebitda_margin_pct":20,"margin_inflection":1,
        "net_income":10,"net_income_growth_pct":10,"earnings_inflection":1,
        "equity":100,"debt":20,"cash":10,"net_debt":10,"debt_repair":1,
        "cfo":12,"cfo_growth_pct":10,"cfo_inflection":1,
        "fcf":8,"fcf_inflection":1
    }])
    prices = pd.DataFrame([
        {"date":"2026-09-03","ticker":"ABC","close":120},
        {"date":"2026-09-04","ticker":"ABC","close":125},
    ])
    out = calculate_scores(universe, fundamentals, prices)
    assert out.iloc[0]["current_price"] == 125
    assert out.iloc[0]["market_cap"] == 1000

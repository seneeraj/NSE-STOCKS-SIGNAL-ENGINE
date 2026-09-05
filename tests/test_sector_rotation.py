import pandas as pd
from src.sector_rotation import build_sector_scores

def test_sector_score_has_stage():
    stocks = pd.DataFrame([
        {"ticker":"AAA.NS","sector":"Auto","earnings_score":80,"turnaround_score":85,"quality_score":75,"data_completeness_pct":100},
        {"ticker":"BBB.NS","sector":"Auto","earnings_score":70,"turnaround_score":75,"quality_score":70,"data_completeness_pct":100},
        {"ticker":"CCC.NS","sector":"IT","earnings_score":40,"turnaround_score":45,"quality_score":55,"data_completeness_pct":100},
    ])
    prices = pd.DataFrame({
        "date": list(pd.date_range("2026-01-01", periods=60))*3,
        "ticker": ["AAA.NS"]*60 + ["BBB.NS"]*60 + ["CCC.NS"]*60,
        "close": list(range(100,160)) + list(range(100,160)) + list(range(160,100,-1))
    })
    out = build_sector_scores(stocks, prices)
    assert "sector_recovery_score" in out.columns
    assert "sector_stage" in out.columns
    assert out.iloc[0]["sector"] == "Auto"

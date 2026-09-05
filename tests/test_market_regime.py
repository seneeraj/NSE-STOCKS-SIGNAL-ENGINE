import numpy as np
import pandas as pd

from src.market_regime import calculate_market_regime


def test_bullish_synthetic_market_scores_higher():
    dates = pd.date_range("2024-01-01", periods=260, freq="B")
    prices = pd.Series(np.linspace(100, 150, len(dates)), index=dates)
    result = calculate_market_regime(prices, breadth_pct=80)
    assert result["market_regime_score"] > 65
    assert result["market_regime"] in {"RECOVERY", "BULL"}


def test_bearish_synthetic_market_scores_lower():
    dates = pd.date_range("2024-01-01", periods=260, freq="B")
    prices = pd.Series(np.linspace(150, 100, len(dates)), index=dates)
    result = calculate_market_regime(prices, breadth_pct=20)
    assert result["market_regime_score"] < 50
    assert result["market_regime"] in {"BEAR", "BOTTOMING"}

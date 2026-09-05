from __future__ import annotations

import numpy as np
import pandas as pd
try:
    import yfinance as yf
except ImportError:  # Allows pure scoring tests without market-data dependencies.
    yf = None


def _score_band(value: float, low: float, high: float) -> float:
    if not np.isfinite(value):
        return 50.0
    return float(np.clip((value - low) / (high - low) * 100.0, 0, 100))


def calculate_market_regime(
    prices: pd.Series | pd.DataFrame,
    breadth_pct: float | None = None,
    index_symbol: str = "^NSEI",
    vix_symbol: str = "^INDIAVIX",
) -> dict:
    """Build a 0-100 market-regime score from trend, momentum, breadth and volatility.

    `prices` may be a Nifty price series or a DataFrame containing one. If a DataFrame
    is supplied, the first numeric column is used.
    """
    if isinstance(prices, pd.DataFrame):
        numeric = prices.select_dtypes(include="number")
        if numeric.empty:
            return _empty_result()
        series = numeric.iloc[:, 0].dropna()
    else:
        series = pd.Series(prices).dropna()

    if len(series) < 210:
        return _empty_result()

    close = series.astype(float)
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    slope_window = min(20, len(close) - 1)
    ma50_slope = (ma50.iloc[-1] / ma50.iloc[-1-slope_window] - 1) * 100 if ma50.iloc[-1-slope_window] else 0
    ret60 = (close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) > 61 else np.nan
    ret200 = (close.iloc[-1] / close.iloc[-201] - 1) * 100 if len(close) > 201 else np.nan

    trend = 0
    trend += 50 if close.iloc[-1] > ma50.iloc[-1] else 0
    trend += 50 if close.iloc[-1] > ma200.iloc[-1] else 0

    momentum = (
        0.55 * _score_band(ret60, -12, 12)
        + 0.45 * _score_band(ret200, -20, 25)
    )
    slope_score = _score_band(ma50_slope, -3, 3)

    if breadth_pct is None or not np.isfinite(breadth_pct):
        breadth_score = 50.0
    else:
        breadth_score = float(np.clip(breadth_pct, 0, 100))

    vix = np.nan
    try:
        if yf is None:
            raise ImportError("yfinance is not installed")
        v = yf.download(vix_symbol, period="1y", progress=False, auto_adjust=False)
        if isinstance(v.columns, pd.MultiIndex):
            v = v.iloc[:, 0] if v.shape[1] else pd.DataFrame()
        if not v.empty:
            vix = float(v["Close"].dropna().iloc[-1])
    except Exception:
        pass

    # Lower volatility is generally more supportive, but missing VIX should not punish the score.
    vix_score = 50.0 if not np.isfinite(vix) else float(np.clip(100 - (vix - 10) * 3.5, 0, 100))

    score = (
        0.30 * trend
        + 0.25 * momentum
        + 0.15 * slope_score
        + 0.20 * breadth_score
        + 0.10 * vix_score
    )
    score = float(np.clip(score, 0, 100))

    if score < 35:
        stage = "BEAR"
        bias = "Capital preservation / selective watchlist"
    elif score < 50:
        stage = "BOTTOMING"
        bias = "Small staged entries; wait for confirmation"
    elif score < 65:
        stage = "EARLY RECOVERY"
        bias = "Increase exposure selectively as confirmation improves"
    elif score < 80:
        stage = "RECOVERY"
        bias = "Favorable environment for staged deployment"
    else:
        stage = "BULL"
        bias = "Trend-following environment; avoid chasing excess valuation"

    return {
        "market_regime_score": score,
        "market_regime": stage,
        "deployment_bias": bias,
        "nifty_close": float(close.iloc[-1]),
        "nifty_ma50": float(ma50.iloc[-1]),
        "nifty_ma200": float(ma200.iloc[-1]),
        "ma50_slope_pct": float(ma50_slope),
        "market_return_60d_pct": float(ret60),
        "market_return_200d_pct": float(ret200),
        "breadth_pct": float(breadth_pct) if breadth_pct is not None and np.isfinite(breadth_pct) else np.nan,
        "vix": vix,
    }


def fetch_market_regime(
    breadth_pct: float | None = None,
    index_symbol: str = "^NSEI",
    vix_symbol: str = "^INDIAVIX",
) -> dict:
    if yf is None:
        return _empty_result()
    data = yf.download(index_symbol, period="2y", progress=False, auto_adjust=False)
    if data.empty:
        return _empty_result()
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
    else:
        close = data["Close"]
    return calculate_market_regime(
        close,
        breadth_pct=breadth_pct,
        index_symbol=index_symbol,
        vix_symbol=vix_symbol,
    )


def _empty_result() -> dict:
    return {
        "market_regime_score": np.nan,
        "market_regime": "UNKNOWN",
        "deployment_bias": "Insufficient market data",
        "nifty_close": np.nan,
        "nifty_ma50": np.nan,
        "nifty_ma200": np.nan,
        "ma50_slope_pct": np.nan,
        "market_return_60d_pct": np.nan,
        "market_return_200d_pct": np.nan,
        "breadth_pct": np.nan,
        "vix": np.nan,
    }

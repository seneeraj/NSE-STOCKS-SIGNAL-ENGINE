import pandas as pd
import yfinance as yf

def download_prices(tickers, period="2y"):
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        threads=True,
        progress=False,
    )

    if data.empty:
        return pd.DataFrame()

    # Normalize single- and multi-ticker outputs into long form.
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].copy()
        close.columns = [tickers[0]]

    long = close.stack(future_stack=False).reset_index()
    long.columns = ["date", "ticker", "close"]
    long["close"] = pd.to_numeric(long["close"], errors="coerce")
    return long.dropna(subset=["close"])

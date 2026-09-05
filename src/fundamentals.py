import numpy as np
import pandas as pd
import yfinance as yf

def _series(stmt, names):
    if stmt is None or stmt.empty:
        return pd.Series(dtype="float64")
    for name in names:
        if name in stmt.index:
            s = pd.to_numeric(stmt.loc[name], errors="coerce").dropna()
            if not s.empty:
                return s
    return pd.Series(dtype="float64")

def _latest(stmt, names):
    s = _series(stmt, names)
    return s.iloc[0] if not s.empty else np.nan

def _growth(stmt, names):
    s = _series(stmt, names)
    if len(s) < 2 or s.iloc[1] == 0:
        return np.nan
    return (s.iloc[0] / s.iloc[1] - 1) * 100

def _trend_score(s, higher_is_better=True):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 3:
        return np.nan
    # Yahoo returns newest -> oldest. Compare latest with oldest and recent slope.
    x = np.arange(len(s))
    slope = np.polyfit(x, s.values, 1)[0]
    change = s.iloc[0] - s.iloc[-1]
    score = change + slope * 2
    return score if higher_is_better else -score

def collect_fundamentals(tickers):
    rows = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            inc = t.income_stmt
            bs = t.balance_sheet
            cf = t.cashflow
            # yfinance can expose quote data through fast_info, but fields may
            # be missing depending on the Yahoo response. Keep multiple fallbacks.
            try:
                fast = t.fast_info
            except Exception:
                fast = {}

            try:
                info = t.info
            except Exception:
                info = {}

            current_price = np.nan
            for source, key in [
                (fast, "last_price"),
                (fast, "regular_market_price"),
                (info, "currentPrice"),
                (info, "regularMarketPrice"),
            ]:
                try:
                    value = source.get(key, np.nan) if hasattr(source, "get") else getattr(source, key, np.nan)
                except Exception:
                    value = np.nan
                if pd.notna(value):
                    current_price = float(value)
                    break

            market_cap = np.nan
            for source, key in [
                (fast, "market_cap"),
                (info, "marketCap"),
            ]:
                try:
                    value = source.get(key, np.nan) if hasattr(source, "get") else getattr(source, key, np.nan)
                except Exception:
                    value = np.nan
                if pd.notna(value):
                    market_cap = float(value)
                    break

            # Final fallback: market cap = current price × shares outstanding.
            if pd.isna(market_cap) and pd.notna(current_price):
                shares = info.get("sharesOutstanding", np.nan) if isinstance(info, dict) else np.nan
                if pd.notna(shares):
                    market_cap = current_price * float(shares)

            revenue_s = _series(inc, ["Total Revenue", "Operating Revenue"])
            ebitda_s = _series(inc, ["EBITDA", "Normalized EBITDA"])
            ni_s = _series(inc, ["Net Income", "Net Income Common Stockholders"])
            equity_s = _series(bs, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
            debt_s = _series(bs, ["Total Debt", "Long Term Debt"])
            cash_s = _series(bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])
            cfo_s = _series(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
            capex_s = _series(cf, ["Capital Expenditure", "Capital Expenditures"])

            margin_s = pd.Series(dtype="float64")
            if not revenue_s.empty and not ebitda_s.empty:
                common = revenue_s.index.intersection(ebitda_s.index)
                if len(common):
                    margin_s = (ebitda_s.loc[common] / revenue_s.loc[common]) * 100

            net_debt_s = pd.Series(dtype="float64")
            if not debt_s.empty and not cash_s.empty:
                common = debt_s.index.intersection(cash_s.index)
                if len(common):
                    net_debt_s = debt_s.loc[common] - cash_s.loc[common]

            fcf_s = pd.Series(dtype="float64")
            if not cfo_s.empty and not capex_s.empty:
                common = cfo_s.index.intersection(capex_s.index)
                if len(common):
                    fcf_s = cfo_s.loc[common] + capex_s.loc[common]

            rows.append({
                "ticker": ticker,
                "revenue": _latest(inc, ["Total Revenue", "Operating Revenue"]),
                "revenue_growth_pct": _growth(inc, ["Total Revenue", "Operating Revenue"]),
                "revenue_trend": _trend_score(revenue_s),
                "ebitda": _latest(inc, ["EBITDA", "Normalized EBITDA"]),
                "ebitda_margin_pct": margin_s.iloc[0] if not margin_s.empty else np.nan,
                "margin_inflection": _trend_score(margin_s),
                "net_income": _latest(inc, ["Net Income", "Net Income Common Stockholders"]),
                "net_income_growth_pct": _growth(inc, ["Net Income", "Net Income Common Stockholders"]),
                "earnings_inflection": _trend_score(ni_s),
                "equity": _latest(bs, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]),
                "debt": _latest(bs, ["Total Debt", "Long Term Debt"]),
                "cash": _latest(bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"]),
                "net_debt": net_debt_s.iloc[0] if not net_debt_s.empty else np.nan,
                "debt_repair": _trend_score(net_debt_s, higher_is_better=False),
                "cfo": _latest(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"]),
                "cfo_growth_pct": _growth(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"]),
                "cfo_inflection": _trend_score(cfo_s),
                "capex": _latest(cf, ["Capital Expenditure", "Capital Expenditures"]),
                "fcf": fcf_s.iloc[0] if not fcf_s.empty else np.nan,
                "fcf_inflection": _trend_score(fcf_s),
                "market_cap": market_cap,
                "current_price": current_price,
            })
        except Exception as exc:
            rows.append({"ticker": ticker, "error": str(exc)})

    return pd.DataFrame(rows)

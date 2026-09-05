from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import yfinance as yf
except ImportError:
    yf=None

SNAPSHOT_DIR=Path("data/snapshots")

def save_scan_snapshot(scores,as_of=None):
    if scores is None or scores.empty: raise ValueError("Cannot save an empty scan snapshot.")
    SNAPSHOT_DIR.mkdir(parents=True,exist_ok=True)
    d=as_of or pd.Timestamp.now().strftime("%Y-%m-%d")
    out=scores.copy()
    if "as_of_date" in out.columns: out=out.drop(columns=["as_of_date"])
    out.insert(0,"as_of_date",d)
    path=SNAPSHOT_DIR/f"scan_{d}.csv"; out.to_csv(path,index=False); return path

def load_snapshots(directory=SNAPSHOT_DIR):
    frames=[]
    for f in sorted(directory.glob("scan_*.csv")):
        try:
            df=pd.read_csv(f)
            if {"as_of_date","symbol"}.issubset(df.columns): frames.append(df)
        except Exception: pass
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def _series(px,symbol):
    t=str(symbol).upper(); t=t if t.endswith(".NS") else f"{t}.NS"
    return px.loc[px["ticker"]==t].sort_values("date").set_index("date")["close"].dropna()

def _forward_return(s,entry,months):
    if s.empty: return np.nan
    e=s.loc[s.index>pd.Timestamp(entry)]
    f=s.loc[s.index>=pd.Timestamp(entry)+pd.DateOffset(months=months)]
    if e.empty or f.empty: return np.nan
    return float((f.iloc[0]/e.iloc[0]-1)*100)

def _max_drawdown(s,entry,months):
    if s.empty: return np.nan
    w=s.loc[s.index>pd.Timestamp(entry)]
    w=w.loc[w.index<=pd.Timestamp(entry)+pd.DateOffset(months=months)]
    if w.empty: return np.nan
    return float((w/w.cummax()-1).min()*100)

def _portfolio_return(px,symbols,entry,months):
    vals=[]
    for x in symbols:
        r=_forward_return(_series(px,x),entry,months)
        if np.isfinite(r): vals.append(r)
    return float(np.mean(vals)) if vals else np.nan

def backtest_snapshots(snapshots,prices,top_n=10,horizons=(3,6,12)):
    req={"as_of_date","symbol","opportunity_score"}
    if snapshots.empty or not req.issubset(snapshots.columns): return pd.DataFrame()
    px=prices.copy()
    px["date"]=pd.to_datetime(px["date"]); px["ticker"]=px["ticker"].astype(str).str.upper()
    px["close"]=pd.to_numeric(px["close"],errors="coerce"); px=px.dropna(subset=["date","ticker","close"])
    rows=[]
    for as_of,g in snapshots.groupby("as_of_date"):
        g=g.copy(); g["opportunity_score"]=pd.to_numeric(g["opportunity_score"],errors="coerce")
        g=g.dropna(subset=["opportunity_score"]).sort_values("opportunity_score",ascending=False)
        top=g.head(top_n)
        if top.empty: continue
        for months in horizons:
            rs=[]; ds=[]
            for _,stock in top.iterrows():
                s=_series(px,stock["symbol"]); r=_forward_return(s,as_of,months); d=_max_drawdown(s,as_of,months)
                if np.isfinite(r): rs.append(r)
                if np.isfinite(d): ds.append(d)
            tr=float(np.mean(rs)) if rs else np.nan
            ur=_portfolio_return(px,g["symbol"].tolist(),as_of,months)
            rows.append({
                "as_of_date":as_of,"top_n":len(top),"horizon_months":months,
                "stocks_with_forward_data":len(rs),"avg_return_pct":tr,
                "median_return_pct":float(np.median(rs)) if rs else np.nan,
                "win_rate_pct":float(np.mean(np.array(rs)>0)*100) if rs else np.nan,
                "worst_stock_return_pct":float(np.min(rs)) if rs else np.nan,
                "best_stock_return_pct":float(np.max(rs)) if rs else np.nan,
                "avg_max_drawdown_pct":float(np.mean(ds)) if ds else np.nan,
                "worst_max_drawdown_pct":float(np.min(ds)) if ds else np.nan,
                "scanned_universe_return_pct":ur,
                "top_n_excess_return_pct":tr-ur if np.isfinite(tr) and np.isfinite(ur) else np.nan
            })
    return pd.DataFrame(rows)

def score_bucket_analysis(snapshots,prices,horizon=6,buckets=4):
    req={"as_of_date","symbol","opportunity_score"}
    if snapshots.empty or not req.issubset(snapshots.columns): return pd.DataFrame()
    px=prices.copy()
    px["date"]=pd.to_datetime(px["date"]); px["ticker"]=px["ticker"].astype(str).str.upper()
    px["close"]=pd.to_numeric(px["close"],errors="coerce"); px=px.dropna(subset=["date","ticker","close"])
    labels=["Highest Score","High-Mid","Low-Mid","Lowest Score"]; rows=[]
    for as_of,g in snapshots.groupby("as_of_date"):
        g=g.copy(); g["opportunity_score"]=pd.to_numeric(g["opportunity_score"],errors="coerce")
        g=g.dropna(subset=["opportunity_score"]).sort_values("opportunity_score",ascending=False)
        if len(g)<buckets: continue
        g["bucket_num"]=np.floor(np.arange(len(g))*buckets/len(g)).astype(int).clip(0,buckets-1)
        for b,bg in g.groupby("bucket_num"):
            vals=[]
            for _,stock in bg.iterrows():
                r=_forward_return(_series(px,stock["symbol"]),as_of,horizon)
                if np.isfinite(r): vals.append(r)
            rows.append({
                "as_of_date":as_of,"bucket":labels[int(b)],"stocks":len(bg),
                "avg_opportunity_score":float(bg["opportunity_score"].mean()),
                "forward_months":horizon,"stocks_with_forward_data":len(vals),
                "avg_return_pct":float(np.mean(vals)) if vals else np.nan,
                "median_return_pct":float(np.median(vals)) if vals else np.nan,
                "win_rate_pct":float(np.mean(np.array(vals)>0)*100) if vals else np.nan
            })
    return pd.DataFrame(rows)

def download_backtest_prices(tickers,start,end=None):
    if yf is None: raise ImportError("yfinance is required for downloading backtest prices.")
    tickers=list(dict.fromkeys(tickers))
    if not tickers: return pd.DataFrame(columns=["date","ticker","close"])
    data=yf.download(tickers,start=start,end=end,interval="1d",auto_adjust=True,group_by="column",threads=True,progress=False)
    if data.empty: return pd.DataFrame(columns=["date","ticker","close"])
    if isinstance(data.columns,pd.MultiIndex): close=data["Close"].copy()
    else: close=data[["Close"]].copy(); close.columns=[tickers[0]]
    long=close.stack(future_stack=False).reset_index()
    long.columns=["date","ticker","close"]; long["ticker"]=long["ticker"].astype(str).str.upper()
    long["close"]=pd.to_numeric(long["close"],errors="coerce")
    return long.dropna(subset=["close"])

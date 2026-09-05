import numpy as np
import pandas as pd
from src.backtesting import backtest_snapshots, score_bucket_analysis

def test_backtest_positive_return():
    d=pd.date_range("2024-01-01",periods=300,freq="B")
    p=pd.DataFrame({"date":d,"ticker":"ABC.NS","close":np.linspace(100,150,len(d))})
    s=pd.DataFrame([{"as_of_date":"2024-01-02","symbol":"ABC","opportunity_score":90}])
    r=backtest_snapshots(s,p,top_n=1,horizons=(3,))
    assert len(r)==1 and r.iloc[0]["avg_return_pct"]>0

def test_backtest_excess_return():
    d=pd.date_range("2024-01-01",periods=300,freq="B")
    p=pd.DataFrame({"date":list(d)*2,"ticker":["ABC.NS"]*len(d)+["XYZ.NS"]*len(d),
                    "close":list(np.linspace(100,160,len(d)))+list(np.linspace(100,120,len(d)))})
    s=pd.DataFrame([{"as_of_date":"2024-01-02","symbol":"ABC","opportunity_score":90},
                    {"as_of_date":"2024-01-02","symbol":"XYZ","opportunity_score":50}])
    r=backtest_snapshots(s,p,top_n=1,horizons=(3,))
    assert r.iloc[0]["top_n_excess_return_pct"]>0

def test_score_buckets():
    d=pd.date_range("2024-01-01",periods=300,freq="B"); rows=[]
    for i in range(4):
        rows.extend(zip(d,[f"S{i}.NS"]*len(d),np.linspace(100+i*5,140+i*15,len(d))))
    p=pd.DataFrame(rows,columns=["date","ticker","close"])
    s=pd.DataFrame([{"as_of_date":"2024-01-02","symbol":f"S{i}","opportunity_score":100-i*10} for i in range(4)])
    r=score_bucket_analysis(s,p,horizon=3,buckets=4)
    assert len(r)==4 and r.iloc[0]["bucket"]=="Highest Score"

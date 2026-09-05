from pathlib import Path
import pandas as pd

STARTER = [
    ("RELIANCE", "Reliance Industries", "Energy"),
    ("HDFCBANK", "HDFC Bank", "Financials"),
    ("ICICIBANK", "ICICI Bank", "Financials"),
    ("SBIN", "State Bank of India", "Financials"),
    ("AXISBANK", "Axis Bank", "Financials"),
    ("KOTAKBANK", "Kotak Mahindra Bank", "Financials"),
    ("BHARTIARTL", "Bharti Airtel", "Telecom"),
    ("TCS", "Tata Consultancy Services", "IT"),
    ("INFY", "Infosys", "IT"),
    ("HCLTECH", "HCL Technologies", "IT"),
    ("WIPRO", "Wipro", "IT"),
    ("LT", "Larsen & Toubro", "Capital Goods"),
    ("MARUTI", "Maruti Suzuki India", "Auto"),
    ("M&M", "Mahindra & Mahindra", "Auto"),
    ("TATAMOTORS", "Tata Motors", "Auto"),
    ("SUNPHARMA", "Sun Pharmaceutical Industries", "Healthcare"),
    ("CIPLA", "Cipla", "Healthcare"),
    ("DRREDDY", "Dr Reddy's Laboratories", "Healthcare"),
    ("ITC", "ITC", "FMCG"),
    ("HINDUNILVR", "Hindustan Unilever", "FMCG"),
    ("NESTLEIND", "Nestle India", "FMCG"),
    ("TITAN", "Titan Company", "Consumer"),
    ("ASIANPAINT", "Asian Paints", "Consumer"),
    ("ULTRACEMCO", "UltraTech Cement", "Cement"),
    ("GRASIM", "Grasim Industries", "Materials"),
    ("JSWSTEEL", "JSW Steel", "Metals"),
    ("TATASTEEL", "Tata Steel", "Metals"),
    ("HINDALCO", "Hindalco Industries", "Metals"),
    ("NTPC", "NTPC", "Utilities"),
    ("POWERGRID", "Power Grid Corporation of India", "Utilities"),
    ("ONGC", "Oil & Natural Gas Corporation", "Energy"),
    ("COALINDIA", "Coal India", "Energy"),
    ("ADANIPORTS", "Adani Ports and SEZ", "Infrastructure"),
    ("ADANIENT", "Adani Enterprises", "Infrastructure"),
    ("BEL", "Bharat Electronics", "Defence"),
    ("HAL", "Hindustan Aeronautics", "Defence"),
    ("TRENT", "Trent", "Retail"),
    ("ETERNAL", "Eternal", "Consumer Tech"),
    ("EICHERMOT", "Eicher Motors", "Auto"),
    ("BAJAJ-AUTO", "Bajaj Auto", "Auto"),
    ("HEROMOTOCO", "Hero MotoCorp", "Auto"),
    ("BAJFINANCE", "Bajaj Finance", "Financials"),
    ("INDUSINDBK", "IndusInd Bank", "Financials"),
    ("TECHM", "Tech Mahindra", "IT"),
    ("TATACONSUM", "Tata Consumer Products", "FMCG"),
    ("APOLLOHOSP", "Apollo Hospitals Enterprise", "Healthcare"),
    ("DABUR", "Dabur India", "FMCG"),
    ("PIDILITIND", "Pidilite Industries", "Chemicals"),
    ("SIEMENS", "Siemens", "Capital Goods"),
    ("ABB", "ABB India", "Capital Goods"),
    ("CUMMINSIND", "Cummins India", "Capital Goods"),
]

def load_universe(custom=False):
    if custom:
        path = Path("data/universe.csv")
        if not path.exists():
            raise FileNotFoundError("data/universe.csv not found.")
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(STARTER, columns=["symbol", "company", "sector"])

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["ticker"] = df["symbol"].apply(lambda x: x if x.endswith(".NS") else f"{x}.NS")
    return df.drop_duplicates("ticker").reset_index(drop=True)

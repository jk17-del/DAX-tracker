import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="DAX Trend Scanner (Stooq)", layout="wide")

st.title("📈 DAX Trend Scanner (stabile Version)")

# -----------------------------
# SETTINGS
# -----------------------------
st.sidebar.header("⚙️ Einstellungen")

ma_short = st.sidebar.slider("MA Kurzfristig", 10, 100, 50)
ma_long = st.sidebar.slider("MA Langfristig", 100, 300, 200)
momentum_days = st.sidebar.slider("Momentum Tage", 5, 60, 20)
top_n = st.sidebar.slider("Top N Aktien", 5, 20, 10)

# -----------------------------
# DAX TICKER (Stooq Format!)
# -----------------------------
DAX_TICKERS = [
    "ads.de","air.de","alv.de","bas.de","bayn.de","bei.de",
    "bmw.de","bnr.de","cbk.de","con.de","db1.de","dbk.de",
    "dhl.de","dtg.de","dte.de","enr.de","eoan.de","fme.de",
    "fre.de","hei.de","hen3.de","ifx.de","lin.de","mbg.de",
    "mrk.de","mtx.de","muv2.de","p911.de","qia.de","rwe.de",
    "sap.de","sie.de","sy1.de","vow3.de","zal.de","vna.de"
]

st.write(f"📊 {len(DAX_TICKERS)} DAX Aktien")

# -----------------------------
# DATA LOAD (STOOQ)
# -----------------------------
@st.cache_data(ttl=3600)
def load_data(ticker):
    url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
    df = pd.read_csv(url)

    if df.empty:
        return None

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df.set_index("Date")

    return df

# -----------------------------
# RSI
# -----------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# -----------------------------
# ANALYSE
# -----------------------------
def analyze_stock(ticker):
    try:
        df = load_data(ticker)

        if df is None or len(df) < ma_long:
            return None

        df["MA_short"] = df["Close"].rolling(ma_short).mean()
        df["MA_long"] = df["Close"].rolling(ma_long).mean()
        df["Momentum"] = df["Close"].pct_change(momentum_days)
        df["RSI"] = compute_rsi(df["Close"])

        df = df.dropna()

        if df.empty:
            return None

        latest = df.iloc[-1]

        trend = latest["Close"] > latest["MA_short"] > latest["MA_long"]
        momentum = latest["Momentum"] > 0
        ma_slope = df["MA_short"].iloc[-1] > df["MA_short"].iloc[-5]

        score = 0
        if trend:
            score += 3
        if momentum:
            score += 2
        if ma_slope:
            score += 2

        distance = (latest["Close"] - latest["MA_short"]) / latest["MA_short"]
        score += max(0, distance * 10)

        return {
            "Ticker": ticker.upper(),
            "Preis": round(latest["Close"], 2),
            "Momentum": round(latest["Momentum"], 3),
            "RSI": round(latest["RSI"], 1),
            "Score": round(score, 2)
        }

    except:
        return None

# -----------------------------
# SCAN
# -----------------------------
if st.button("🔍 Scan starten"):

    results = []
    progress = st.progress(0)

    for i, ticker in enumerate(DAX_TICKERS):
        res = analyze_stock(ticker)

        if res:
            results.append(res)

        progress.progress((i + 1) / len(DAX_TICKERS))

    if len(results) == 0:
        st.error("❌ Keine Daten verfügbar (selten bei Stooq)")
        st.stop()

    df_results = pd.DataFrame(results)

    st.subheader("📊 Alle Aktien")
    st.dataframe(df_results)

    df_sorted = df_results.sort_values("Score", ascending=False)
    top_df = df_sorted.head(top_n)

    st.subheader("🚀 Top Rising Stocks")
    st.dataframe(top_df)

    st.success(f"Top {len(top_df)} Aktien")

# -----------------------------
# CHART
# -----------------------------
st.subheader("📉 Einzelanalyse")

selected = st.selectbox("Wähle Aktie", DAX_TICKERS)

if selected:
    df = load_data(selected)

    if df is not None:
        df["MA_short"] = df["Close"].rolling(ma_short).mean()
        df["MA_long"] = df["Close"].rolling(ma_long).mean()

        st.line_chart(df[["Close", "MA_short", "MA_long"]])

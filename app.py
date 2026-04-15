import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="DAX Trend Scanner V2", layout="wide")

st.title("📈 DAX Trend Scanner V2")

# -----------------------------
# SETTINGS
# -----------------------------
st.sidebar.header("⚙️ Einstellungen")

ma_short = st.sidebar.slider("MA Kurzfristig", 10, 100, 50)
ma_long = st.sidebar.slider("MA Langfristig", 100, 300, 200)
momentum_days = st.sidebar.slider("Momentum Tage", 5, 60, 20)
top_n = st.sidebar.slider("Top N Aktien", 5, 20, 10)

# -----------------------------
# DAX TICKER (stabil)
# -----------------------------
@st.cache_data(ttl=86400)
def get_dax_tickers():
    return [
        "ADS.DE","AIR.DE","ALV.DE","BAS.DE","BAYN.DE","BEI.DE",
        "BMW.DE","BNR.DE","CBK.DE","CON.DE","DB1.DE","DBK.DE",
        "DHL.DE","DTG.DE","DTE.DE","ENR.DE","EOAN.DE","FME.DE",
        "FRE.DE","HEI.DE","HEN3.DE","IFX.DE","LIN.DE","MBG.DE",
        "MRK.DE","MTX.DE","MUV2.DE","P911.DE","QIA.DE","RWE.DE",
        "SAP.DE","SIE.DE","SY1.DE","VOW3.DE","ZAL.DE","VNA.DE"
    ]

dax_tickers = get_dax_tickers()

st.write(f"📊 {len(dax_tickers)} DAX Aktien im Universe")

# -----------------------------
# CACHE FÜR PREISDATEN
# -----------------------------
@st.cache_data(ttl=3600)
def load_data(ticker):
    return yf.download(ticker, period="1y", progress=False)

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

        if df is None or df.empty:
            return None

        # Indikatoren
        df["MA_short"] = df["Close"].rolling(ma_short).mean()
        df["MA_long"] = df["Close"].rolling(ma_long).mean()
        df["Momentum"] = df["Close"].pct_change(momentum_days)
        df["RSI"] = compute_rsi(df["Close"])

        df = df.dropna()

        if df.empty:
            return None

        latest = df.iloc[-1]

        # Kriterien
        trend = latest["Close"] > latest["MA_short"] > latest["MA_long"]
        momentum = latest["Momentum"] > 0
        ma_slope = df["MA_short"].iloc[-1] > df["MA_short"].iloc[-5]

        # Score
        score = 0

        if trend:
            score += 3
        if momentum:
            score += 2
        if ma_slope:
            score += 2

        # Stärke (Abstand zum MA)
        distance = (latest["Close"] - latest["MA_short"]) / latest["MA_short"]
        score += max(0, distance * 10)

        return {
            "Ticker": ticker,
            "Preis": round(latest["Close"], 2),
            "Momentum": round(latest["Momentum"], 3),
            "RSI": round(latest["RSI"], 1),
            "Score": round(score, 2),
            "Trend": trend
        }

    except:
        return None

# -----------------------------
# SCAN
# -----------------------------
if st.button("🔍 Scan starten"):

    results = []
    progress = st.progress(0)

    for i, ticker in enumerate(dax_tickers):
        res = analyze_stock(ticker)

        if res:
            results.append(res)

        progress.progress((i + 1) / len(dax_tickers))

    if len(results) == 0:
        st.error("❌ Keine Daten geladen – prüfe Internet/API")
        st.stop()

    df_results = pd.DataFrame(results)

    st.subheader("📊 Alle Aktien")
    st.dataframe(df_results)

    # Ranking
    df_sorted = df_results.sort_values("Score", ascending=False)
    top_df = df_sorted.head(top_n)

    st.subheader("🚀 Top Rising Stocks")
    st.dataframe(top_df)

    st.success(f"Top {len(top_df)} Aktien nach Score")

# -----------------------------
# EINZELCHART
# -----------------------------
st.subheader("📉 Einzelanalyse")

selected = st.selectbox("Wähle Aktie", dax_tickers)

if selected:
    data = load_data(selected)

    if data is not None and not data.empty:
        data["MA_short"] = data["Close"].rolling(ma_short).mean()
        data["MA_long"] = data["Close"].rolling(ma_long).mean()

        st.line_chart(data[["Close", "MA_short", "MA_long"]])
    else:
        st.warning("Keine Daten verfügbar")

# -----------------------------
# DEBUG / CACHE RESET
# -----------------------------
if st.sidebar.button("🔄 Cache leeren"):
    st.cache_data.clear()
    st.success("Cache geleert!")

with st.expander("🛠 Debug"):
    st.write(dax_tickers)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="SDAX Trend Scanner", layout="wide")

st.title("📈 SDAX Rising Stocks Scanner")

# -----------------------------
# SDAX Ticker Liste (Beispiel)
# -----------------------------
# Hinweis: Yahoo Finance nutzt .DE für deutsche Aktien
sdax_tickers = [
    "AT1.DE", "BVB.DE", "EVT.DE", "SIX2.DE", "S92.DE",
    "DRW3.DE", "FNTN.DE", "GFT.DE", "HFG.DE", "JEN.DE",
    "KWS.DE", "NEM.DE", "PNE3.DE", "RHM.DE", "SGL.DE"
]

# -----------------------------
# Parameter Sidebar
# -----------------------------
st.sidebar.header("⚙️ Einstellungen")

ma_short = st.sidebar.slider("MA Kurzfristig", 10, 100, 50)
ma_long = st.sidebar.slider("MA Langfristig", 100, 300, 200)
momentum_days = st.sidebar.slider("Momentum Tage", 5, 60, 20)

use_rsi = st.sidebar.checkbox("RSI Filter (<70)", value=True)

# -----------------------------
# Hilfsfunktionen
# -----------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_stock(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty:
            return None

        df["MA_short"] = df["Close"].rolling(ma_short).mean()
        df["MA_long"] = df["Close"].rolling(ma_long).mean()

        df["Momentum"] = df["Close"].pct_change(momentum_days)

        df["RSI"] = compute_rsi(df["Close"])

        latest = df.iloc[-1]

        # Trend Bedingungen
        trend = (
            latest["Close"] > latest["MA_short"] > latest["MA_long"]
        )

        # Momentum
        momentum = latest["Momentum"] > 0

        # MA Steigung
        ma_slope = df["MA_short"].iloc[-1] > df["MA_short"].iloc[-5]

        # RSI
        rsi_ok = True
        if use_rsi:
            rsi_ok = latest["RSI"] < 70

        is_rising = trend and momentum and ma_slope and rsi_ok

        return {
            "Ticker": ticker,
            "Preis": round(latest["Close"], 2),
            "Momentum": round(latest["Momentum"], 3),
            "RSI": round(latest["RSI"], 1),
            "Trend": trend,
            "Steigend": ma_slope,
            "Rising": is_rising
        }

    except Exception as e:
        return None

# -----------------------------
# Analyse starten
# -----------------------------
if st.button("🔍 Scan starten"):

    results = []

    progress = st.progress(0)

    for i, ticker in enumerate(sdax_tickers):
        res = analyze_stock(ticker)
        if res:
            results.append(res)

        progress.progress((i + 1) / len(sdax_tickers))

    df_results = pd.DataFrame(results)

    st.subheader("📊 Alle analysierten Aktien")
    st.dataframe(df_results)

    # Filter Rising Stocks
    rising_df = df_results[df_results["Rising"] == True]

    st.subheader("🚀 Rising Stocks")
    st.dataframe(rising_df)

    st.success(f"{len(rising_df)} Aktien im Aufwärtstrend gefunden")

# -----------------------------
# Zusatz: Chart anzeigen
# -----------------------------
st.subheader("📉 Einzelanalyse")

selected = st.selectbox("Wähle Aktie", sdax_tickers)

if selected:
    data = yf.download(selected, period="6mo", progress=False)

    data["MA_short"] = data["Close"].rolling(ma_short).mean()
    data["MA_long"] = data["Close"].rolling(ma_long).mean()

    st.line_chart(data[["Close", "MA_short", "MA_long"]])

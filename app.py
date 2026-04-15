import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="SDAX Trend Scanner V2", layout="wide")

st.title("📈 SDAX Trend Scanner V2")

# -----------------------------
# SETTINGS
# -----------------------------
st.sidebar.header("⚙️ Einstellungen")

ma_short = st.sidebar.slider("MA Kurzfristig", 10, 100, 50)
ma_long = st.sidebar.slider("MA Langfristig", 100, 300, 200)
momentum_days = st.sidebar.slider("Momentum Tage", 5, 60, 20)

top_n = st.sidebar.slider("Top N Aktien", 5, 20, 10)

# -----------------------------
# FALLBACK SDAX
# -----------------------------
FALLBACK_SDAX = [
    "1U1.DE", "ADN1.DE", "AOX.DE", "AT1.DE", "BVB.DE",
    "CE2.DE", "DUE.DE", "EKT.DE", "EVT.DE", "GFT.DE",
    "HDD.DE", "HFG.DE", "HBH.DE", "KWS.DE", "NEM.DE",
    "PNE3.DE", "S92.DE", "SGL.DE", "SIX2.DE"
]

# -----------------------------
# SCRAPER + CACHE
# -----------------------------
@st.cache_data(ttl=86400)
def get_sdax_tickers():
    try:
        url = "https://de.finance.yahoo.com/quote/%5ESDAXI/components/"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df["Symbol"].dropna().tolist()
        tickers = [t for t in tickers if ".DE" in t]

        if len(tickers) < 20:
            return FALLBACK_SDAX

        return tickers
    except:
        return FALLBACK_SDAX

sdax_tickers = get_sdax_tickers()

st.write(f"📊 {len(sdax_tickers)} Aktien im Universe")

# -----------------------------
# INDICATORS
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
        df = yf.download(ticker, period="1y", progress=False)

        if df.empty or len(df) < ma_long:
            return None

        df["MA_short"] = df["Close"].rolling(ma_short).mean()
        df["MA_long"] = df["Close"].rolling(ma_long).mean()
        df["Momentum"] = df["Close"].pct_change(momentum_days)
        df["RSI"] = compute_rsi(df["Close"])

        latest = df.iloc[-1]

        # --- Kriterien ---
        trend = latest["Close"] > latest["MA_short"] > latest["MA_long"]
        momentum = latest["Momentum"] > 0
        ma_slope = df["MA_short"].iloc[-1] > df["MA_short"].iloc[-5]

        # --- Score (NEU 🔥) ---
        score = 0

        if trend:
            score += 3
        if momentum:
            score += 2
        if ma_slope:
            score += 2

        # Abstand zum MA als Stärke
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
# RUN SCAN
# -----------------------------
if st.button("🔍 Scan starten"):

    results = []
    progress = st.progress(0)

    for i, ticker in enumerate(sdax_tickers):
        res = analyze_stock(ticker)
        if res:
            results.append(res)

        progress.progress((i + 1) / len(sdax_tickers))

    # 🔥 WICHTIG: Fehler-Fix
    if len(results) == 0:
        st.error("❌ Keine Daten geladen – prüfe Internet/API")
        st.stop()

    df_results = pd.DataFrame(results)

    st.subheader("📊 Alle Aktien")
    st.dataframe(df_results)

    # -----------------------------
    # RANKING statt Boolean Filter
    # -----------------------------
    df_sorted = df_results.sort_values("Score", ascending=False)

    top_df = df_sorted.head(top_n)

    st.subheader("🚀 Top Rising Stocks")
    st.dataframe(top_df)

    st.success(f"Top {len(top_df)} Aktien nach Score")

# -----------------------------
# EINZELCHART
# -----------------------------
st.subheader("📉 Einzelanalyse")

selected = st.selectbox("Wähle Aktie", sdax_tickers)

if selected:
    data = yf.download(selected, period="6mo", progress=False)

    if not data.empty:
        data["MA_short"] = data["Close"].rolling(ma_short).mean()
        data["MA_long"] = data["Close"].rolling(ma_long).mean()

        st.line_chart(data[["Close", "MA_short", "MA_long"]])

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="DAX Trend Scanner Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.signal-buy    { background:#0d3321; color:#00ff88; padding:4px 12px; border-radius:4px; font-weight:600; font-family:monospace; }
.signal-watch  { background:#2d2800; color:#ffd700; padding:4px 12px; border-radius:4px; font-weight:600; font-family:monospace; }
.signal-avoid  { background:#2d0d0d; color:#ff4444; padding:4px 12px; border-radius:4px; font-weight:600; font-family:monospace; }

.metric-card {
    background: #111;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 DAX Trend Scanner Pro")
st.caption("Technische Analyse · Kaufsignale · Momentum-Ranking")

# ── Sidebar Settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameter")
    ma_short     = st.slider("MA Kurzfristig",  10, 100, 50)
    ma_long      = st.slider("MA Langfristig", 100, 300, 200)
    momentum_days = st.slider("Momentum Tage",   5,  60,  20)
    top_n        = st.slider("Top N Aktien",     5,  20,  10)
    rsi_ob       = st.slider("RSI Überkauft",   60,  90,  70)
    rsi_os       = st.slider("RSI Überverkauft", 10,  40,  30)
    st.divider()
    if st.button("🔄 Cache leeren"):
        st.cache_data.clear()
        st.success("Cache geleert!")

# ── DAX Universe ──────────────────────────────────────────────────────────────
DAX_TICKERS = [
    "ADS.DE","AIR.DE","ALV.DE","BAS.DE","BAYN.DE","BEI.DE",
    "BMW.DE","BNR.DE","CBK.DE","CON.DE","DB1.DE","DBK.DE",
    "DHL.DE","DTG.DE","DTE.DE","ENR.DE","EOAN.DE","FME.DE",
    "FRE.DE","HEI.DE","HEN3.DE","IFX.DE","LIN.DE","MBG.DE",
    "MRK.DE","MTX.DE","MUV2.DE","P911.DE","QIA.DE","RWE.DE",
    "SAP.DE","SIE.DE","SY1.DE","VOW3.DE","ZAL.DE","VNA.DE"
]

# ── Indicator Functions ───────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = -delta.clip(upper=0).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger(series, period=20, std=2):
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return mid, mid + std * sigma, mid - std * sigma

def compute_atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all_data(tickers):
    """Download 1 year of daily data for all tickers in one shot."""
    raw = yf.download(tickers, period="1y", group_by="ticker", threads=False, progress=False)
    return raw

def extract_ticker(raw, ticker):
    """Safely extract single-ticker OHLCV from a possibly MultiIndex DataFrame."""
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw[ticker].copy()
        else:
            df = raw.copy()
        df = df.dropna(how="all")
        # Flatten column names if needed
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df if not df.empty else None
    except Exception:
        return None

# ── Analysis ──────────────────────────────────────────────────────────────────
def analyze(ticker, raw, ma_short, ma_long, momentum_days, rsi_ob, rsi_os):
    df = extract_ticker(raw, ticker)
    if df is None or len(df) < ma_long + 10:
        return None

    close = df["Close"]

    df["MA_short"]  = close.rolling(ma_short).mean()
    df["MA_long"]   = close.rolling(ma_long).mean()
    df["Momentum"]  = close.pct_change(momentum_days)
    df["RSI"]       = compute_rsi(close)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = compute_macd(close)
    df["BB_mid"], df["BB_upper"], df["BB_lower"]   = compute_bollinger(close)
    df["ATR"]       = compute_atr(df)
    df["Vol_avg"]   = df["Volume"].rolling(20).mean()

    df = df.dropna()
    if df.empty:
        return None

    r = df.iloc[-1]
    prev5 = df.iloc[-6:-1]

    # ── Boolean conditions ────────────────────────────────────────────────────
    trend_ok    = bool(r["Close"] > r["MA_short"] > r["MA_long"])
    mom_ok      = bool(r["Momentum"] > 0)
    ma_slope    = bool(r["MA_short"] > prev5["MA_short"].mean())
    rsi_ok      = bool(rsi_os < r["RSI"] < rsi_ob)
    macd_ok     = bool(r["MACD"] > r["MACD_signal"])
    macd_cross  = bool(
        r["MACD"] > r["MACD_signal"] and
        prev5.iloc[-1]["MACD"] < prev5.iloc[-1]["MACD_signal"]
    )
    above_bb_mid = bool(r["Close"] > r["BB_mid"])
    vol_spike    = bool(r["Volume"] > 1.5 * r["Vol_avg"])
    near_bb_low  = bool(r["Close"] < r["BB_mid"])

    # ── Composite score ───────────────────────────────────────────────────────
    score = 0
    score += 3 if trend_ok      else 0
    score += 2 if mom_ok        else 0
    score += 2 if ma_slope      else 0
    score += 2 if macd_ok       else 0
    score += 3 if macd_cross    else 0   # fresh crossover = strongest signal
    score += 1 if rsi_ok        else 0
    score += 1 if above_bb_mid  else 0
    score += 1 if vol_spike     else 0

    # Distance-to-MA bonus (max ±2)
    dist = (r["Close"] - r["MA_short"]) / r["MA_short"]
    score += float(np.clip(dist * 10, -2, 2))

    # ── Buy signal classification ─────────────────────────────────────────────
    buy_signals = []
    if macd_cross:           buy_signals.append("MACD Cross ↑")
    if trend_ok and mom_ok:  buy_signals.append("Trend + Mom")
    if near_bb_low and rsi_ok and r["RSI"] < 45: buy_signals.append("BB Bounce")
    if vol_spike and mom_ok: buy_signals.append("Vol Surge")

    if score >= 9:
        signal = "🟢 KAUFEN"
    elif score >= 6:
        signal = "🟡 BEOBACHTEN"
    else:
        signal = "🔴 MEIDEN"

    # 52w data
    high_52w = close.rolling(252).max().iloc[-1]
    low_52w  = close.rolling(252).min().iloc[-1]
    pct_from_high = (r["Close"] - high_52w) / high_52w * 100

    return {
        "Ticker"        : ticker,
        "Signal"        : signal,
        "Score"         : round(score, 1),
        "Preis"         : round(float(r["Close"]), 2),
        "RSI"           : round(float(r["RSI"]), 1),
        "MACD"          : round(float(r["MACD"]), 3),
        "Momentum %"    : round(float(r["Momentum"]) * 100, 1),
        "52w Hoch %"    : round(float(pct_from_high), 1),
        "Kaufsignale"   : ", ".join(buy_signals) if buy_signals else "—",
        "Trend"         : trend_ok,
        "MACD Cross"    : macd_cross,
        "_df"           : df,          # full df for charting
    }

# ── Chart ─────────────────────────────────────────────────────────────────────
def build_chart(result, ma_short, ma_long):
    df     = result["_df"]
    ticker = result["Ticker"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker} – Kurs & Indikatoren", "MACD", "RSI")
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00ff88", decreasing_line_color="#ff4444",
        name="Kurs"
    ), row=1, col=1)

    # Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_short"], name=f"MA{ma_short}",
                             line=dict(color="#00aaff", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_long"],  name=f"MA{ma_long}",
                             line=dict(color="#ff8800", width=1.5)), row=1, col=1)

    # Bollinger Bands (filled)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper",
                             line=dict(color="rgba(150,150,255,0.3)", width=1),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower",
                             line=dict(color="rgba(150,150,255,0.3)", width=1),
                             fill="tonexty",
                             fillcolor="rgba(150,150,255,0.07)",
                             showlegend=False), row=1, col=1)

    # MACD
    colors = ["#00ff88" if v >= 0 else "#ff4444" for v in df["MACD_hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Histogramm",
                         marker_color=colors, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
                             name="MACD", line=dict(color="#00aaff", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"],
                             name="Signal", line=dict(color="#ff8800", width=1.5)), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="#aa88ff", width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line=dict(color="#ff4444", dash="dash", width=1), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="#00ff88", dash="dash", width=1), row=3, col=1)
    fig.add_hline(y=50, line=dict(color="grey",    dash="dot",  width=0.5), row=3, col=1)

    fig.update_layout(
        height=700,
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0d0d0d",
        font=dict(color="#cccccc", family="IBM Plex Mono"),
        legend=dict(bgcolor="#111", bordercolor="#333", borderwidth=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    for axis in ["xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3"]:
        fig.update_layout(**{axis: dict(gridcolor="#1a1a1a", zerolinecolor="#333")})

    return fig

# ── Signal badge helper ───────────────────────────────────────────────────────
def signal_html(sig):
    if "KAUFEN"     in sig: return f'<span class="signal-buy">{sig}</span>'
    if "BEOBACHTEN" in sig: return f'<span class="signal-watch">{sig}</span>'
    return f'<span class="signal-avoid">{sig}</span>'

# ── Main UI ───────────────────────────────────────────────────────────────────
if st.button("🔍 Scan starten", type="primary"):

    with st.spinner("⏳ Lade Marktdaten für alle DAX-Aktien…"):
        raw = load_all_data(DAX_TICKERS)

    if raw is None or raw.empty:
        st.error("❌ Daten konnten nicht geladen werden.")
        st.stop()

    results  = []
    failed   = []
    progress = st.progress(0, text="Analysiere…")

    for i, ticker in enumerate(DAX_TICKERS):
        res = analyze(ticker, raw, ma_short, ma_long, momentum_days, rsi_ob, rsi_os)
        if res:
            results.append(res)
        else:
            failed.append(ticker)
        progress.progress((i + 1) / len(DAX_TICKERS), text=f"Analysiere {ticker}…")

    progress.empty()

    if not results:
        st.error("❌ Keine Aktien analysierbar.")
        st.stop()

    st.session_state["results"] = results
    st.session_state["failed"]  = failed

# ── Display results (persistent across reruns) ────────────────────────────────
if "results" in st.session_state:
    results = st.session_state["results"]

    df_all = pd.DataFrame([{k: v for k, v in r.items() if k != "_df"} for r in results])
    df_sorted = df_all.sort_values("Score", ascending=False)

    # Summary KPIs
    buys    = (df_all["Signal"].str.contains("KAUFEN")).sum()
    watches = (df_all["Signal"].str.contains("BEOBACHTEN")).sum()
    avoids  = (df_all["Signal"].str.contains("MEIDEN")).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analysiert",    len(results))
    c2.metric("🟢 Kaufen",     buys)
    c3.metric("🟡 Beobachten", watches)
    c4.metric("🔴 Meiden",     avoids)

    st.divider()

    # ── Top N Table ───────────────────────────────────────────────────────────
    st.subheader(f"🚀 Top {top_n} Aktien nach Score")

    top_df = df_sorted.head(top_n).reset_index(drop=True)
    display_cols = ["Ticker","Signal","Score","Preis","RSI","Momentum %","52w Hoch %","Kaufsignale"]

    st.dataframe(
        top_df[display_cols].reset_index(drop=True),
        use_container_width=True,
        column_config={
            "Score":       st.column_config.ProgressColumn("Score", min_value=0, max_value=15, format="%.1f"),
            "Preis":       st.column_config.NumberColumn("Preis", format="%.2f €"),
            "RSI":         st.column_config.NumberColumn("RSI", format="%.1f"),
            "Momentum %":  st.column_config.NumberColumn("Momentum %", format="%+.1f%%"),
            "52w Hoch %":  st.column_config.NumberColumn("52w Hoch %", format="%+.1f%%"),
        }
    )

    # ── Full table (collapsed) ────────────────────────────────────────────────
    with st.expander("📊 Alle Aktien anzeigen"):
        st.dataframe(
            df_sorted[display_cols].reset_index(drop=True),
            use_container_width=True
        )

    if st.session_state.get("failed"):
        st.warning(f"⚠️ Übersprungen: {', '.join(st.session_state['failed'])}")

    st.divider()

    # ── Chart Section ─────────────────────────────────────────────────────────
    st.subheader("📉 Chart-Analyse")

    # Quick-select top picks
    buy_tickers = df_sorted[df_sorted["Signal"].str.contains("KAUFEN")]["Ticker"].tolist()

    tab1, tab2 = st.tabs(["Top-Picks (Kaufen)", "Alle Aktien"])

    with tab1:
        if buy_tickers:
            sel = st.selectbox("Kaufsignal-Aktie wählen", buy_tickers, key="sel_buy")
        else:
            st.info("Keine Kaufsignale im aktuellen Scan.")
            sel = None

    with tab2:
        sel2 = st.selectbox("Aktie wählen", [r["Ticker"] for r in results], key="sel_all")
        sel  = sel2

    if sel:
        match = next((r for r in results if r["Ticker"] == sel), None)
        if match:
            # Signal badge + metrics row
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.markdown(signal_html(match["Signal"]), unsafe_allow_html=True)
            sc2.metric("Score",      match["Score"])
            sc3.metric("RSI",        match["RSI"])
            sc4.metric("MACD",       match["MACD"])
            sc5.metric("Momentum",   f"{match['Momentum %']:+.1f}%")

            if match["Kaufsignale"] != "—":
                st.success(f"📣 Aktive Kaufsignale: **{match['Kaufsignale']}**")

            st.plotly_chart(
                build_chart(match, ma_short, ma_long),
                use_container_width=True
            )

else:
    st.info("👆 Klicke **Scan starten** um die Analyse zu beginnen.")

    # Still allow manual chart lookup without scan
    st.divider()
    st.subheader("📉 Schnellanalyse (ohne Scan)")
    sel_quick = st.selectbox("Aktie wählen", DAX_TICKERS)
    if sel_quick and st.button("Chart laden"):
        with st.spinner("Lade Daten…"):
            df_q = yf.download(sel_quick, period="6mo", progress=False)
            if df_q is not None and not df_q.empty:
                df_q.columns = [c[0] if isinstance(c, tuple) else c for c in df_q.columns]
                df_q["MA_short"] = df_q["Close"].rolling(ma_short).mean()
                df_q["MA_long"]  = df_q["Close"].rolling(ma_long).mean()
                df_q["RSI"]      = compute_rsi(df_q["Close"])
                df_q["MACD"], df_q["MACD_signal"], df_q["MACD_hist"] = compute_macd(df_q["Close"])
                df_q["BB_mid"], df_q["BB_upper"], df_q["BB_lower"]   = compute_bollinger(df_q["Close"])
                df_q["ATR"] = compute_atr(df_q)
                df_q["Vol_avg"] = df_q["Volume"].rolling(20).mean()

                fake_result = {
                    "Ticker": sel_quick,
                    "Signal": "—",
                    "Score": "—",
                    "RSI": round(float(df_q["RSI"].iloc[-1]), 1),
                    "MACD": round(float(df_q["MACD"].iloc[-1]), 3),
                    "Momentum %": 0,
                    "Kaufsignale": "—",
                    "_df": df_q.dropna()
                }
                st.plotly_chart(build_chart(fake_result, ma_short, ma_long), use_container_width=True)
            else:
                st.warning("Keine Daten verfügbar.")

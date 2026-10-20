import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from plotly.subplots import make_subplots

st.set_page_config(page_title="SmallCap Radar", page_icon="🔭", layout="wide")

# ── Design ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap');
html, body, [class*="css"] { font-family:'Manrope',sans-serif; font-variant-numeric:tabular-nums; }
.block-container { padding-top:1.6rem; max-width:1400px; }
[data-testid="stSidebar"] { background:#14213d; }
[data-testid="stSidebar"] * { color:#e8edf7 !important; }
h1 { font-weight:800; letter-spacing:-0.02em; margin-bottom:0; }
.sub { color:#5b6b8a; margin:0 0 1.2rem 0; }
.regime { border-radius:10px; padding:12px 18px; margin-bottom:1rem; font-weight:600; }
.regime.on  { background:#dff3e8; color:#14532d; border-left:6px solid #1f9d55; }
.regime.off { background:#fde8e4; color:#7a1d10; border-left:6px solid #d9442b; }
.kpi { background:#fff; border-radius:10px; padding:14px 18px; border:1px solid #dde3ee; }
.kpi b { display:block; font-size:1.9rem; line-height:1.1; color:#14213d; }
.kpi span { color:#5b6b8a; font-size:.85rem; }
.pill { display:inline-block; padding:3px 12px; border-radius:99px; font-weight:700; font-size:.85rem; }
.p-buy { background:#1f9d55; color:#fff; } .p-watch { background:#f2b632; color:#3b2a00; } .p-avoid { background:#d9442b; color:#fff; }
</style>
""", unsafe_allow_html=True)

# ── Universum (Small Caps; ungültige Ticker werden automatisch übersprungen) ──
UNIVERSE = {
 "USA": ["SMCI","CELH","ONTO","AXON","FIX","MEDP","ANF","BOOT","CVLT","SM","MTDR","AIT","ENSG","HALO","CRDO",
         "IBP","ALKS","KNSL","LNTH","MOD","SKY","WIRE","BMI","ESE","FELE","GPI","CNX","PTEN","RMBS","SPSC",
         "QLYS","TMDX","VRNS","AAON","BCC","CALM","CPRX","CRVL","DY","EXLS","GMS","HRI","ITGR","JBT","KTB",
         "LGND","MGY","NSIT","OSIS","PI","RDNT","STRL","TGTX","UFPI","VCEL","WDFC","XPEL","YOU","ZWS"],
 "Deutschland": ["AIXA.DE","BC8.DE","COK.DE","DRW3.DE","EVD.DE","FNTN.DE","GBF.DE","HAG.DE","JUN3.DE","KBX.DE",
         "NDX1.DE","NEM.DE","PBB.DE","PSM.DE","RAA.DE","S92.DE","SFQ.DE","SGL.DE","TKA.DE","TLX.DE",
         "VBK.DE","WAF.DE","SHA.DE","LEG.DE","HOT.DE","G1A.DE","GXI.DE","DUE.DE","BFSA.DE","AG1.DE"],
 "Frankreich/Benelux": ["AM.PA","ALO.PA","ELIS.PA","IPS.PA","LR.PA","NXI.PA","OVH.PA","RCO.PA","SOI.PA","TRI.PA",
         "VIRP.PA","ADYEN.AS","BESI.AS","IMCD.AS","FLOW.AS","TKWY.AS","CRBN.AS","APAM.AS","SBMO.AS","FUR.AS",
         "ACKB.BR","BEKB.BR","UMI.BR","KIN.BR","MELE.BR"],
 "Skandinavien": ["EVO.ST","SINCH.ST","BALD-B.ST","INDT.ST","LIFCO-B.ST","NIBE-B.ST","SAAB-B.ST","SWEC-B.ST",
         "VOLCAR-B.ST","BONAV-B.ST","KOG.OL","NEL.OL","TOM.OL","KIT.OL","AKRBP.OL","NAS.OL","DEMANT.CO",
         "NETC.CO","RBREW.CO","SYDB.CO","TRYG.CO","SIM.CO","WDH.CO","FSKRS.HE","KEMIRA.HE","VALMT.HE","UPM.HE"],
 "Süd/Schweiz": ["AMP.MI","BC.MI","BGN.MI","DIA.MI","ERG.MI","IP.MI","REC.MI","SRG.MI","TIP.MI","CLN.SW",
         "BARN.SW","BUCN.SW","DAE.SW","GALE.SW","INRN.SW","SFSN.SW","SIGN.SW","VZUG.SW","ACX.MC","CIE.MC",
         "ENG.MC","GRF.MC","IDR.MC","MAP.MC","VIS.MC"],
}
BENCH = {"USA": "IWM", "Europa": "IEUS"}

# ── Daten ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load(tickers: tuple):
    raw = yf.download(list(tickers), period="2y", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    out = {}
    lvl = raw.columns.get_level_values(0).unique() if isinstance(raw.columns, pd.MultiIndex) else []
    for t in tickers:
        if t in lvl:
            d = raw[t].dropna(subset=["Close"])
            if len(d) >= 210:
                out[t] = d
    return out

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

def atr(d, n=14):
    tr = pd.concat([d.High - d.Low, (d.High - d.Close.shift()).abs(), (d.Low - d.Close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def ret(c, n):
    return c.iloc[-1] / c.iloc[-1 - n] - 1

# ── Kennzahlen je Aktie ───────────────────────────────────────────────────────
def metrics(t, d, region, bench_6m):
    c, v = d.Close, d.Volume
    r = c.pct_change().dropna()
    vol = r.tail(126).std() * np.sqrt(252)
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    hi52 = c.tail(252).max()
    dd = (c.tail(126) / c.tail(126).cummax() - 1).min()
    a = atr(d)
    return dict(
        Ticker=t, Region=region, Preis=c.iloc[-1],
        mom12_1=c.iloc[-22] / c.iloc[-min(253, len(c))] - 1,
        mom3m=ret(c, 63), mom1m=ret(c, 21), mom6m=ret(c, 126),
        RelStärke=ret(c, 126) - bench_6m,
        Vola=vol, Sharpe=(ret(c, 126) * 2) / vol if vol else 0,
        AbstandHoch=c.iloc[-1] / hi52 - 1, MaxDD=dd,
        RSI=rsi(c).iloc[-1], ATR=a.iloc[-1], ATRpct=a.iloc[-1] / c.iloc[-1],
        Liquidität=(c * v).tail(50).mean(),
        VolTrend=v.tail(10).mean() / max(v.tail(60).mean(), 1),
        ÜberMA50=c.iloc[-1] > ma50.iloc[-1], MA50ÜberMA200=ma50.iloc[-1] > ma200.iloc[-1],
        MA50Steigt=ma50.iloc[-1] > ma50.iloc[-11], ÜberMA200=c.iloc[-1] > ma200.iloc[-1],
    )

def score_all(df):
    rk = lambda s: s.rank(pct=True)
    trend = (df.ÜberMA50 * .35 + df.MA50ÜberMA200 * .35 + df.MA50Steigt * .3)
    s = 100 * (.25 * rk(df.mom12_1) + .15 * rk(df.mom3m) + .15 * rk(df.RelStärke) +
               .15 * trend + .15 * rk(df.Sharpe) + .10 * rk(df.AbstandHoch) + .05 * rk(df.VolTrend))
    s = s - np.where(df.RSI > 80, 6, 0) - np.where(df.MaxDD < -.35, 5, 0)
    s = np.where(df.ÜberMA200, s, np.minimum(s, 45))        # kein Kauf unter MA200
    df["Score"] = np.round(np.clip(s, 0, 100), 1)
    df["Signal"] = np.select([df.Score >= 70, df.Score >= 55], ["KAUFEN", "BEOBACHTEN"], "MEIDEN")
    setups = []
    for _, r in df.iterrows():
        x = []
        if r.AbstandHoch > -.03 and r.VolTrend > 1.2: x.append("Ausbruch")
        if r.ÜberMA200 and r.MA50ÜberMA200 and r.RSI < 45: x.append("Pullback")
        if r.Score >= 80: x.append("Leader")
        setups.append(", ".join(x) or "–")
    df["Setup"] = setups
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def run_scan(regions: tuple, extra: tuple):
    tick = {t: reg for reg in regions for t in UNIVERSE[reg]}
    tick.update({t: "Eigene" for t in extra})
    bench = {k: yf.download(v, period="1y", auto_adjust=True, progress=False) for k, v in BENCH.items()}
    bclose = {k: (b.Close.squeeze() if not b.empty else None) for k, b in bench.items()}
    data = load(tuple(list(tick) + list(BENCH.values())))
    rows, hist = [], {}
    for t, reg in tick.items():
        if t not in data:
            continue
        key = "USA" if reg == "USA" or ("." not in t and reg == "Eigene") else "Europa"
        b = bclose.get(key)
        rows.append(metrics(t, data[t], reg, ret(b, 126) if b is not None and len(b) > 130 else 0))
        hist[t] = data[t]
    if not rows:
        return None, {}, {}, list(tick)
    regime = {}
    for k, b in bclose.items():
        if b is not None and len(b) > 200:
            regime[k] = bool(b.iloc[-1] > b.rolling(200).mean().iloc[-1])
    return score_all(pd.DataFrame(rows)), hist, regime, [t for t in tick if t not in hist]

# ── Efficient Frontier ────────────────────────────────────────────────────────
def build_frontier(mu, cov, max_w, rf, n_pts=40):
    n = len(mu)
    bnds = [(0, max_w)] * n
    w0 = np.full(n, 1 / n)
    fix = {"type": "eq", "fun": lambda w: w.sum() - 1}
    vol_f = lambda w: np.sqrt(w @ cov @ w)

    mv = minimize(lambda w: w @ cov @ w, w0, bounds=bnds, constraints=[fix], method="SLSQP").x
    ms = minimize(lambda w: -(w @ mu - rf) / vol_f(w), w0, bounds=bnds, constraints=[fix], method="SLSQP").x

    # größter erreichbarer Ertrag unter der Gewichtsgrenze (gierig füllen)
    w_max, left = np.zeros(n), 1.0
    for i in np.argsort(-mu):
        w_max[i] = min(max_w, left); left -= w_max[i]
        if left <= 1e-9: break

    pts = []
    for tgt in np.linspace(mv @ mu, w_max @ mu, n_pts):
        cons = [fix, {"type": "eq", "fun": lambda w, t=tgt: w @ mu - t}]
        r = minimize(lambda w: w @ cov @ w, mv, bounds=bnds, constraints=cons, method="SLSQP")
        if r.success:
            pts.append((vol_f(r.x), tgt))
    return mv, ms, pd.DataFrame(pts, columns=["vol", "ret"])

# ── Chart ─────────────────────────────────────────────────────────────────────
def chart(d, row, ma_s, ma_l, stop):
    c = d.Close
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.62, .16, .22], vertical_spacing=.025)
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=c, name="Kurs",
                  increasing_line_color="#1f9d55", decreasing_line_color="#d9442b"), 1, 1)
    for n, col in [(ma_s, "#1f5fbf"), (ma_l, "#e08a00")]:
        fig.add_trace(go.Scatter(x=d.index, y=c.rolling(n).mean(), name=f"MA{n}", line=dict(color=col, width=1.6)), 1, 1)
    fig.add_hline(y=stop, line=dict(color="#d9442b", dash="dash", width=1), annotation_text="Stop", row=1, col=1)
    up = (d.Close >= d.Open)
    fig.add_trace(go.Bar(x=d.index, y=d.Volume, marker_color=np.where(up, "#9fd8b4", "#f0b3a8"), name="Volumen"), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=rsi(c), name="RSI", line=dict(color="#6b4fbb", width=1.5)), 3, 1)
    for y, col in [(70, "#d9442b"), (30, "#1f9d55")]:
        fig.add_hline(y=y, line=dict(color=col, dash="dot", width=1), row=3, col=1)
    fig.update_layout(height=680, xaxis_rangeslider_visible=False, paper_bgcolor="#f4f6fa", plot_bgcolor="#fff",
                      font=dict(family="Manrope", color="#14213d"), margin=dict(l=8, r=8, t=10, b=8),
                      legend=dict(orientation="h", y=1.04))
    fig.update_xaxes(range=[d.index[-180], d.index[-1]], gridcolor="#eef1f6")
    fig.update_yaxes(gridcolor="#eef1f6")
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filter")
    regions = st.multiselect("Märkte", list(UNIVERSE), default=["USA", "Deutschland", "Skandinavien"])
    extra = tuple(x.strip().upper() for x in st.text_input("Eigene Ticker (Komma)", "").split(",") if x.strip())
    min_liq = st.slider("Mindestumsatz pro Tag (Mio., lokal)", 0.0, 20.0, 1.0, 0.5)
    min_score = st.slider("Mindest-Score", 0, 90, 0)
    setup_f = st.multiselect("Setup", ["Ausbruch", "Pullback", "Leader"])
    st.header("Risiko")
    capital = st.number_input("Depotgröße", 1000, 10_000_000, 25_000, 1000)
    risk_pct = st.slider("Risiko pro Trade (%)", 0.25, 3.0, 1.0, 0.25)
    atr_mult = st.slider("Stop-Abstand (× ATR)", 1.5, 4.0, 2.5, 0.5)
    st.header("Chart")
    ma_s = st.slider("MA kurz", 20, 100, 50)
    ma_l = st.slider("MA lang", 100, 250, 200)
    if st.button("Daten neu laden"):
        st.cache_data.clear()

# ── Hauptseite ────────────────────────────────────────────────────────────────
st.title("🔭 SmallCap Radar")
st.markdown('<p class="sub">Europäische und amerikanische Small Caps, gerankt nach risikoadjustiertem Momentum, Relativer Stärke und Trendqualität.</p>', unsafe_allow_html=True)

if not regions and not extra:
    st.info("Wähle links mindestens einen Markt oder gib eigene Ticker ein.")
    st.stop()

with st.spinner("Lade Marktdaten und berechne Scores …"):
    df, hist, regime, failed = run_scan(tuple(regions), extra)
if df is None:
    st.error("Es konnten keine Daten geladen werden. Prüfe die Verbindung und die Ticker.")
    st.stop()

cols = st.columns(len(regime) or 1)
for col, (k, on) in zip(cols, regime.items()):
    name = "US Small Caps (IWM)" if k == "USA" else "Europa Small Caps (IEUS)"
    col.markdown(f'<div class="regime {"on" if on else "off"}">{name}: '
                 f'{"Aufwärtstrend – über MA200, Käufe erlaubt" if on else "unter MA200 – defensiv bleiben"}</div>', unsafe_allow_html=True)

f = df[(df.Liquidität >= min_liq * 1e6) & (df.Score >= min_score)]
if setup_f:
    f = f[f.Setup.apply(lambda s: any(x in s for x in setup_f))]
f = f.sort_values("Score", ascending=False).reset_index(drop=True)
f["Stop"] = f.Preis - atr_mult * f.ATR
f["Stück"] = np.floor((capital * risk_pct / 100) / (f.Preis - f.Stop).clip(lower=1e-9)).astype(int)
f["Positionswert"] = np.minimum(f.Stück * f.Preis, capital * .25)   # max. 25 % pro Position
f["Stück"] = np.floor(f.Positionswert / f.Preis).astype(int)

k = st.columns(4)
for col, (lab, val) in zip(k, [("Analysiert", len(f)), ("Kaufen", (f.Signal == "KAUFEN").sum()),
                               ("Beobachten", (f.Signal == "BEOBACHTEN").sum()), ("Median 12-1 Momentum", f"{f.mom12_1.median()*100:+.0f} %" if len(f) else "–")]):
    col.markdown(f'<div class="kpi"><b>{val}</b><span>{lab}</span></div>', unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["Ranking", "Chart & Positionsgröße", "Risiko-Rendite-Karte", "Rendite vs. Volatilität"])

with tab1:
    show = f[["Ticker", "Region", "Signal", "Score", "Setup", "Preis", "mom12_1", "mom3m", "RelStärke", "Vola", "RSI", "AbstandHoch", "Liquidität"]].copy()
    for c_ in ["mom12_1", "mom3m", "RelStärke", "Vola", "AbstandHoch"]:
        show[c_] *= 100
    show["Liquidität"] /= 1e6
    st.dataframe(show, use_container_width=True, hide_index=True, height=560, column_config={
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
        "mom12_1": st.column_config.NumberColumn("Momentum 12-1", format="%+.0f%%"),
        "mom3m": st.column_config.NumberColumn("3 Monate", format="%+.0f%%"),
        "RelStärke": st.column_config.NumberColumn("vs. Index (6M)", format="%+.0f%%"),
        "Vola": st.column_config.NumberColumn("Volatilität", format="%.0f%%"),
        "AbstandHoch": st.column_config.NumberColumn("Abstand 52W-Hoch", format="%.1f%%"),
        "Liquidität": st.column_config.NumberColumn("Umsatz/Tag (Mio.)", format="%.1f"),
        "Preis": st.column_config.NumberColumn(format="%.2f"), "RSI": st.column_config.NumberColumn(format="%.0f"),
    })
    if failed:
        st.caption(f"Keine ausreichenden Daten für: {', '.join(failed)}")

with tab2:
    if f.empty:
        st.info("Kein Treffer – lockere die Filter.")
    else:
        sel = st.selectbox("Aktie", f.Ticker, format_func=lambda t: f"{t}  ·  Score {f.loc[f.Ticker == t, 'Score'].iloc[0]:.0f}")
        r = f[f.Ticker == sel].iloc[0]
        cls = {"KAUFEN": "p-buy", "BEOBACHTEN": "p-watch", "MEIDEN": "p-avoid"}[r.Signal]
        st.markdown(f'<span class="pill {cls}">{r.Signal}</span> &nbsp; Setup: <b>{r.Setup}</b>', unsafe_allow_html=True)
        m = st.columns(5)
        m[0].metric("Kurs", f"{r.Preis:.2f}")
        m[1].metric("Stop-Loss", f"{r.Stop:.2f}", f"{(r.Stop / r.Preis - 1) * 100:.1f}%")
        m[2].metric("Stückzahl", int(r.Stück))
        m[3].metric("Positionswert", f"{r.Positionswert:,.0f}")
        m[4].metric("Max. Verlust", f"{r.Stück * (r.Preis - r.Stop):,.0f}")
        st.plotly_chart(chart(hist[sel], r, ma_s, ma_l, r.Stop), use_container_width=True)

with tab3:
    if f.empty:
        st.info("Kein Treffer – lockere die Filter.")
    else:
        sc = go.Figure(go.Scatter(
            x=f.Vola * 100, y=f.mom12_1 * 100, mode="markers+text", text=f.Ticker, textposition="top center",
            marker=dict(size=np.clip(np.log10(f.Liquidität.clip(lower=1e5)) * 4 - 15, 7, 30), color=f.Score,
                        colorscale="Blues", cmin=20, cmax=100, showscale=True, colorbar=dict(title="Score"),
                        line=dict(width=1, color="#14213d")),
            hovertemplate="%{text}<br>Vola %{x:.0f}%<br>Momentum %{y:+.0f}%<extra></extra>"))
        sc.add_hline(y=0, line=dict(color="#9aa6bd", dash="dot"))
        sc.update_layout(height=620, xaxis_title="Volatilität p.a. (%)", yaxis_title="Momentum 12-1 (%)",
                         paper_bgcolor="#f4f6fa", plot_bgcolor="#fff", font=dict(family="Manrope", color="#14213d"),
                         margin=dict(l=8, r=8, t=10, b=8))
        st.plotly_chart(sc, use_container_width=True)
        st.caption("Oben links ist ideal: hohes Momentum bei geringer Schwankung. Die Blasengröße zeigt die Liquidität.")

with tab4:
    st.markdown("**1-Jahres-Rendite gegen Volatilität, mit effizientem Rand (Markowitz, nur Long)**")
    c1, c2, c3 = st.columns(3)
    n_top = c1.slider("Anzahl Aktien (Top nach Score)", 8, 40, 20)
    max_w = c2.slider("Max. Gewicht pro Aktie (%)", 5, 50, 15) / 100
    rf = c3.slider("Risikofreier Zins (%)", 0.0, 6.0, 3.0, 0.25) / 100

    cand = [t for t in f.Ticker.head(n_top) if t in hist]
    px = pd.DataFrame({t: hist[t].Close for t in cand}).sort_index().ffill().tail(253)
    px = px.dropna(axis=1).dropna()
    if px.shape[1] < 5 or max_w * px.shape[1] < 1:
        st.info("Zu wenige Aktien für die Optimierung: erhöhe die Anzahl oder das maximale Gewicht.")
    else:
        rets = px.pct_change().dropna()
        mu = (px.iloc[-1] / px.iloc[0] - 1).values            # 1-Jahres-Rendite
        cov = rets.cov().values * 252
        vol = np.sqrt(np.diag(cov))
        names = list(px.columns)
        mv, ms, fr = build_frontier(mu, cov, max_w, rf)
        pv = lambda w: (np.sqrt(w @ cov @ w) * 100, (w @ mu) * 100)

        fig = go.Figure()
        sc_map = f.set_index("Ticker").Score
        fig.add_trace(go.Scatter(x=vol * 100, y=mu * 100, mode="markers+text", text=names, textposition="top center",
            textfont=dict(size=10), name="Einzelaktien",
            marker=dict(size=9, color=[sc_map[t] for t in names], colorscale="Blues", cmin=20, cmax=100,
                        showscale=True, colorbar=dict(title="Score"), line=dict(width=1, color="#14213d")),
            hovertemplate="%{text}<br>Vola %{x:.0f}%<br>Rendite %{y:+.0f}%<extra></extra>"))
        fig.add_trace(go.Scatter(x=fr.vol * 100, y=fr.ret * 100, mode="lines", name="Effizienter Rand",
                                 line=dict(color="#e08a00", width=3)))
        vs, rs = pv(ms)
        fig.add_trace(go.Scatter(x=[0, vs * 1.6], y=[rf * 100, rf * 100 + (rs - rf * 100) * 1.6], mode="lines",
                                 name="Kapitalmarktlinie", line=dict(color="#6b4fbb", dash="dot", width=1.5)))
        for w, nm, col in [(mv, "Min. Varianz", "#1f9d55"), (ms, "Max. Sharpe", "#d9442b")]:
            x_, y_ = pv(w)
            fig.add_trace(go.Scatter(x=[x_], y=[y_], mode="markers", name=nm,
                                     marker=dict(symbol="star", size=18, color=col, line=dict(width=1.5, color="#14213d"))))
        fig.add_hline(y=0, line=dict(color="#9aa6bd", dash="dot"))
        fig.update_layout(height=640, xaxis_title="Volatilität p.a. (%)", yaxis_title="Rendite letzte 12 Monate (%)",
                          paper_bgcolor="#f4f6fa", plot_bgcolor="#fff", font=dict(family="Manrope", color="#14213d"),
                          margin=dict(l=8, r=8, t=10, b=8), legend=dict(orientation="h", y=1.07))
        st.plotly_chart(fig, use_container_width=True)

        a, b = st.columns(2)
        for col, w, nm in [(a, ms, "Max. Sharpe"), (b, mv, "Min. Varianz")]:
            v_, r_ = pv(w)
            col.markdown(f"**{nm}**: Rendite {r_:+.0f} %, Vola {v_:.0f} %, Sharpe {(r_ / 100 - rf) / (v_ / 100):.2f}")
            wd = pd.DataFrame({"Aktie": names, "Gewicht": w * 100}).query("Gewicht > 0.5").sort_values("Gewicht", ascending=False)
            col.dataframe(wd, hide_index=True, use_container_width=True,
                          column_config={"Gewicht": st.column_config.ProgressColumn("Gewicht", min_value=0, max_value=max_w * 100, format="%.1f%%")})
        st.caption("Die Optimierung nutzt die Vergangenheit als Erwartung und ist damit stark an die Daten angepasst (Overfitting). "
                   "Sie zeigt, was im letzten Jahr effizient gewesen wäre, nicht was künftig effizient sein wird. "
                   "Kurse in Lokalwährung, Währungsrisiko ist nicht berücksichtigt.")

st.caption("Keine Anlageberatung. Daten via Yahoo Finance, ggf. verzögert. Umsätze in Lokalwährung.")

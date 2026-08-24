"""
Swing Trading Breakout Scanner — Nifty 500
--------------------------------------------
Weekly-trend-based breakout scanner with volume confirmation.
Run locally with:  streamlit run app.py
Deploy free on Streamlit Community Cloud (see README.md).
"""

import io
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Swing Breakout Scanner", page_icon="\U0001F4CA", layout="wide")

# ---------------------------------------------------------------------------
# TERMINAL THEME — fonts, header banner, ticker-board summary, table styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"]  { font-family: 'Space Grotesk', sans-serif; }

/* Numbers / data everywhere in monospace for a terminal feel */
[data-testid="stDataFrame"] * { font-family: 'JetBrains Mono', monospace !important; }

.app-header {
    border: 1px solid #2A323D;
    border-left: 4px solid #C89B3C;
    background: linear-gradient(90deg, #161D27 0%, #0F1419 100%);
    padding: 22px 26px;
    border-radius: 4px;
    margin-bottom: 22px;
}
.app-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.9rem;
    color: #E8E6DF;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.app-header p {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #8A93A0;
    margin: 0;
    letter-spacing: 0.2px;
}

/* Ticker-board summary strip */
.ticker-board {
    display: flex;
    gap: 1px;
    background: #2A323D;
    border: 1px solid #2A323D;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 24px;
}
.ticker-cell {
    flex: 1;
    background: #161D27;
    padding: 14px 18px;
}
.ticker-cell .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #8A93A0;
    margin-bottom: 4px;
}
.ticker-cell .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #E8E6DF;
}
.ticker-cell .value.accent { color: #C89B3C; }
.ticker-cell .value.bull { color: #4A9B7F; }

/* Section labels styled like a board eyebrow */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #C89B3C;
    border-bottom: 1px solid #2A323D;
    padding-bottom: 8px;
    margin: 28px 0 4px 0;
}
.section-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #8A93A0;
    margin-bottom: 14px;
}

/* Buttons */
div.stButton > button {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    background-color: #C89B3C;
    color: #0F1419;
    border: none;
    border-radius: 3px;
    letter-spacing: 0.3px;
}
div.stButton > button:hover {
    background-color: #E0B455;
    color: #0F1419;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 1. NIFTY 500 LIST
# ---------------------------------------------------------------------------

NSE_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_nifty500_symbols():
    """Try to pull the official Nifty 500 constituent list from NSE.
    Falls back to None if blocked (NSE sometimes rejects scripted requests)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        session = requests.Session()
        # NSE requires a warm-up hit to the homepage first to set cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        resp = session.get(NSE_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbols = df["Symbol"].astype(str).str.strip().tolist()
        return [s + ".NS" for s in symbols]
    except Exception:
        return None


def get_symbol_list(uploaded_file):
    """Priority: user-uploaded CSV > live NSE fetch > manual paste box."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        syms = df[col].astype(str).str.strip().tolist()
        return [s if s.endswith(".NS") else s + ".NS" for s in syms]

    live = fetch_nifty500_symbols()
    if live:
        return live

    return None


# ---------------------------------------------------------------------------
# 2. DATA DOWNLOAD
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def download_data(tickers, interval, period):
    """Bulk-download OHLCV for a list of tickers. Returns a dict {ticker: df}."""
    data = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            raw = yf.download(
                batch, period=period, interval=interval,
                group_by="ticker", threads=True, progress=False, auto_adjust=False,
            )
        except Exception:
            continue

        for t in batch:
            try:
                if len(batch) == 1:
                    df = raw
                else:
                    df = raw[t]
                df = df.dropna(how="all")
                if not df.empty and len(df) > 25:
                    data[t] = df
            except Exception:
                continue
    return data


# ---------------------------------------------------------------------------
# 3. INDICATOR / CONDITION LOGIC
# ---------------------------------------------------------------------------

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def evaluate_stock(symbol, daily, weekly, params):
    """Returns a dict of computed metrics + pass/fail flags, or None if data insufficient."""
    if daily is None or weekly is None:
        return None
    if len(daily) < 25 or len(weekly) < 55:
        return None

    d_close = daily["Close"]
    d_open = daily["Open"]
    d_vol = daily["Volume"]
    w_close = weekly["Close"]
    w_high = weekly["High"]
    w_vol = weekly["Volume"]

    try:
        latest_close = float(d_close.iloc[-1])
        latest_vol = float(d_vol.iloc[-1])

        # Weekly trend
        w_ema20 = ema(w_close, 20)
        w_ema50 = ema(w_close, 50)
        weekly_trend_ok = (w_close.iloc[-1] > w_ema20.iloc[-1]) and (w_ema20.iloc[-1] > w_ema50.iloc[-1])
        ema50_rising = w_ema50.iloc[-1] > w_ema50.iloc[-2]

        # Daily compression (last 20 days, based on Open, matching original Chartink logic)
        last20_open = d_open.iloc[-20:]
        compression_ratio = last20_open.max() / last20_open.min()

        # Near / at weekly breakout: compare latest weekly close to prior 20-week high (excluding current week)
        prior_20w_high = w_high.iloc[-21:-1].max()
        pct_from_high = (w_close.iloc[-1] / prior_20w_high - 1) * 100  # negative = below high

        # Weekly volume expansion
        w_vol_avg20 = w_vol.iloc[-21:-1].mean()
        vol_ratio = w_vol.iloc[-1] / w_vol_avg20 if w_vol_avg20 > 0 else np.nan

        # Extension above weekly EMA20 (avoid chasing)
        pct_above_ema20 = (w_close.iloc[-1] / w_ema20.iloc[-1] - 1) * 100

        row = {
            "Symbol": symbol.replace(".NS", ""),
            "Close": round(latest_close, 2),
            "Weekly Trend OK": weekly_trend_ok,
            "EMA50 Rising": ema50_rising,
            "Compression (20d)": round((compression_ratio - 1) * 100, 2),
            "% From 20W High": round(pct_from_high, 2),
            "Weekly Vol Ratio": round(vol_ratio, 2) if not np.isnan(vol_ratio) else None,
            "% Above Weekly EMA20": round(pct_above_ema20, 2),
            "Daily Volume": int(latest_vol),
        }

        # Pass flags for STRICT and WATCHLIST profiles
        for label, p in [("Strict", params["strict"]), ("Watchlist", params["watchlist"])]:
            passes = (
                latest_vol >= p["min_volume"]
                and latest_close >= p["min_price"]
                and weekly_trend_ok
                and (ema50_rising or not p["require_ema50_rising"])
                and compression_ratio <= p["compression_band"]
                and pct_from_high >= -p["near_high_pct"]
                and (vol_ratio is not None and not np.isnan(vol_ratio) and vol_ratio >= p["vol_multiple"])
                and pct_above_ema20 <= p["extension_cap"]
            )
            row[f"Pass {label}"] = passes

        return row
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. UI
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="app-header">
        <h1>SWING BREAKOUT SCANNER — NIFTY 500</h1>
        <p>WEEKLY TIMEFRAME &nbsp;·&nbsp; TREND + CONSOLIDATION + VOLUME-CONFIRMED BREAKOUT PROXIMITY
        &nbsp;·&nbsp; RUN FRIDAY / SATURDAY AFTER CLOSE</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Universe")
    uploaded = st.file_uploader(
        "Optional: upload Nifty 500 CSV (Symbol column). "
        "If skipped, the app tries to fetch it live from NSE.",
        type=["csv"],
    )
    limit = st.slider("Limit stocks scanned (for a faster test run)", 20, 500, 100, step=10)

    st.header("Strict scan thresholds")
    s_min_price = st.number_input("Strict: Min price", value=150, key="s_price")
    s_min_vol = st.number_input("Strict: Min daily volume", value=500000, step=50000, key="s_vol")
    s_compression = st.slider("Strict: Compression band (max/min open)", 1.00, 1.30, 1.10, 0.01, key="s_comp")
    s_near_high = st.slider("Strict: Within % of 20W high", 0.5, 10.0, 2.0, 0.5, key="s_high")
    s_vol_mult = st.slider("Strict: Weekly volume expansion (x avg)", 1.0, 3.0, 1.5, 0.1, key="s_volmult")
    s_extension = st.slider("Strict: Max % above weekly EMA20", 2.0, 25.0, 10.0, 1.0, key="s_ext")
    s_ema50_rising = st.checkbox("Strict: Require EMA50 rising", value=True, key="s_rising")

    st.header("Watchlist scan thresholds (looser)")
    w_min_price = st.number_input("Watchlist: Min price", value=150, key="w_price")
    w_min_vol = st.number_input("Watchlist: Min daily volume", value=500000, step=50000, key="w_vol")
    w_compression = st.slider("Watchlist: Compression band", 1.00, 1.40, 1.15, 0.01, key="w_comp")
    w_near_high = st.slider("Watchlist: Within % of 20W high", 0.5, 15.0, 5.0, 0.5, key="w_high")
    w_vol_mult = st.slider("Watchlist: Weekly volume expansion (x avg)", 1.0, 3.0, 1.25, 0.1, key="w_volmult")
    w_extension = st.slider("Watchlist: Max % above weekly EMA20", 2.0, 30.0, 15.0, 1.0, key="w_ext")
    w_ema50_rising = st.checkbox("Watchlist: Require EMA50 rising", value=False, key="w_rising")

    run_button = st.button("🔍 Run Scan", type="primary", use_container_width=True)

params = {
    "strict": dict(
        min_price=s_min_price, min_volume=s_min_vol, compression_band=s_compression,
        near_high_pct=s_near_high, vol_multiple=s_vol_mult, extension_cap=s_extension,
        require_ema50_rising=s_ema50_rising,
    ),
    "watchlist": dict(
        min_price=w_min_price, min_volume=w_min_vol, compression_band=w_compression,
        near_high_pct=w_near_high, vol_multiple=w_vol_mult, extension_cap=w_extension,
        require_ema50_rising=w_ema50_rising,
    ),
}

if run_button:
    with st.spinner("Fetching Nifty 500 symbol list..."):
        symbols = get_symbol_list(uploaded)

    if not symbols:
        st.error(
            "Could not fetch the Nifty 500 list from NSE (it often blocks scripted requests). "
            "Please download the list manually from "
            "https://www.niftyindices.com/indices/equity/broad-based-indices/nifty-500 "
            "and upload the CSV using the sidebar uploader, then click Run Scan again."
        )
        st.stop()

    symbols = symbols[:limit]
    status_msg = st.empty()
    status_msg.info(f"Scanning {len(symbols)} stocks...")

    progress = st.progress(0, text="Downloading daily data...")
    daily_data = download_data(symbols, interval="1d", period="6mo")
    progress.progress(50, text="Downloading weekly data...")
    weekly_data = download_data(symbols, interval="1wk", period="2y")
    progress.progress(80, text="Computing conditions...")

    results = []
    for sym in symbols:
        row = evaluate_stock(sym, daily_data.get(sym), weekly_data.get(sym), params)
        if row:
            results.append(row)
    progress.progress(100, text="Done")
    time.sleep(0.3)
    progress.empty()
    status_msg.empty()

    if not results:
        st.warning("No data could be evaluated. Try uploading the CSV manually or reducing the stock limit.")
        st.stop()

    df = pd.DataFrame(results)

    strict_df = df[df["Pass Strict"]].drop(columns=["Pass Strict", "Pass Watchlist"]).sort_values(
        "% From 20W High", ascending=False
    )
    watchlist_df = df[df["Pass Watchlist"] & ~df["Pass Strict"]].drop(
        columns=["Pass Strict", "Pass Watchlist"]
    ).sort_values("% From 20W High", ascending=False)

    now_str = pd.Timestamp.now().strftime("%d %b %Y, %H:%M")
    st.markdown(
        f"""
        <div class="ticker-board">
            <div class="ticker-cell">
                <div class="label">Scanned</div>
                <div class="value">{len(symbols)}</div>
            </div>
            <div class="ticker-cell">
                <div class="label">Strict Hits</div>
                <div class="value accent">{len(strict_df)}</div>
            </div>
            <div class="ticker-cell">
                <div class="label">Watchlist Hits</div>
                <div class="value bull">{len(watchlist_df)}</div>
            </div>
            <div class="ticker-cell">
                <div class="label">Last Run</div>
                <div class="value" style="font-size:1.05rem;">{now_str}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def style_table(t_df):
        """Color-code the key signal columns so strength is visible at a glance."""
        if t_df.empty:
            return t_df
        return (
            t_df.style
            .background_gradient(subset=["Weekly Vol Ratio"], cmap="Greens", vmin=1.0, vmax=3.0)
            .background_gradient(subset=["% From 20W High"], cmap="RdYlGn", vmin=-10, vmax=2)
            .format({
                "Close": "{:.2f}",
                "Compression (20d)": "{:.2f}",
                "% From 20W High": "{:.2f}",
                "Weekly Vol Ratio": "{:.2f}",
                "% Above Weekly EMA20": "{:.2f}",
                "Daily Volume": "{:,.0f}",
            })
        )

    st.markdown('<div class="section-label">STRICT BREAKOUT LIST</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Near/at breakout, confirmed by weekly volume expansion. Highest conviction.</div>',
        unsafe_allow_html=True,
    )
    if strict_df.empty:
        st.markdown("`empty — no stock currently meets every strict condition`")
    else:
        st.dataframe(style_table(strict_df), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">WATCHLIST</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Setting up but not yet meeting the strict criteria — track these for next week.</div>',
        unsafe_allow_html=True,
    )
    if watchlist_df.empty:
        st.markdown("`empty — nothing setting up right now`")
    else:
        st.dataframe(style_table(watchlist_df), use_container_width=True, hide_index=True)

    with st.expander("Show full scan data (all evaluated stocks, pass/fail included)"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download full results as CSV", csv, "scan_results.csv", "text/csv")

else:
    st.info("Set your thresholds in the sidebar, then click **Run Scan**.")

"""
Swing Trading Breakout Scanner — Nifty 500
--------------------------------------------
Weekly-trend-based breakout scanner with volume confirmation.
Run locally:   py -m streamlit run app.py
Deploy free:   Streamlit Community Cloud (see README.md)
"""

import io
import time
import concurrent.futures
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

st.set_page_config(
    page_title="Swing Breakout Scanner",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
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
.ticker-cell .value.bull   { color: #4A9B7F; }

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
# 1. SYMBOL LISTS
# ---------------------------------------------------------------------------

NSE_URL       = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
NSE_FULL_URL  = "https://archives.nseindia.com/content/equity/EQUITY_L.csv"
NSE_HEADERS   = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _nse_session():
    s = requests.Session()
    try:
        s.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
    except Exception:
        pass
    return s


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_nifty500_symbols():
    try:
        resp = _nse_session().get(NSE_URL, headers=NSE_HEADERS, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        return [s.strip() + ".NS" for s in df["Symbol"].astype(str).tolist()]
    except Exception:
        return None


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_full_nse_equity_list():
    try:
        resp = _nse_session().get(NSE_FULL_URL, headers=NSE_HEADERS, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        return [s.strip() + ".NS" for s in df["SYMBOL"].astype(str).tolist()]
    except Exception:
        return None


def _fetch_market_cap(ticker):
    try:
        fi = yf.Ticker(ticker).fast_info
        mc = getattr(fi, "market_cap", None) or getattr(fi, "marketCap", None)
        return ticker, mc
    except Exception:
        return ticker, None


@st.cache_data(ttl=60 * 60 * 24 * 7, show_spinner=False)
def rank_by_market_cap(symbols, top_n):
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        for t, mc in ex.map(_fetch_market_cap, symbols):
            if mc:
                results[t] = mc
    ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked[:top_n]]


def symbols_from_upload(uploaded_file):
    df = pd.read_csv(uploaded_file)
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    syms = df[col].astype(str).str.strip().tolist()
    return [s if s.endswith(".NS") else s + ".NS" for s in syms]


# ---------------------------------------------------------------------------
# 2. DATA DOWNLOAD — robust multi-ticker extraction
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def download_data(tickers: tuple, interval: str, period: str) -> dict:
    """
    Bulk-download OHLCV. Returns dict {ticker: DataFrame}.
    Handles both old (ticker-first) and new (price-first) yfinance MultiIndex layouts.
    """
    tickers = list(tickers)
    data = {}
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            raw = yf.download(
                batch,
                period=period,
                interval=interval,
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=True,   # auto_adjust=True removes Adj Close noise
            )
        except Exception:
            continue

        if len(batch) == 1:
            t = batch[0]
            df = raw.dropna(how="all")
            if not df.empty and len(df) > 10:
                data[t] = df
            continue

        # Multi-ticker: yfinance ≥0.2.x returns columns as MultiIndex (Price, Ticker)
        # Older versions return (Ticker, Price). Detect and handle both.
        if isinstance(raw.columns, pd.MultiIndex):
            top_level = raw.columns.get_level_values(0).unique().tolist()
            # New layout: top level = price names (Open, High, …)
            price_names = {"Open", "High", "Low", "Close", "Volume"}
            if set(top_level[:5]) & price_names:
                # columns are (Price, Ticker) → swap to get per-ticker slice
                raw = raw.swaplevel(axis=1)
            # Now columns are (Ticker, Price)
            for t in batch:
                try:
                    df = raw[t].dropna(how="all")
                    if not df.empty and len(df) > 10:
                        data[t] = df
                except KeyError:
                    continue
        else:
            # Flat columns — single ticker fell through somehow
            for t in batch:
                try:
                    df = raw.dropna(how="all")
                    if not df.empty and len(df) > 10:
                        data[t] = df
                except Exception:
                    continue

    return data


# ---------------------------------------------------------------------------
# 3. INDICATOR HELPERS
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _last_closed_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the current in-progress week so partial candles don't skew results.
    Works correctly whether the index is tz-aware or tz-naive.
    """
    now_ist = pd.Timestamp.now(tz=IST)
    this_monday_ist = (now_ist - pd.Timedelta(days=now_ist.weekday())).normalize()

    idx = weekly.index
    if idx.tzinfo is not None or (hasattr(idx, "tz") and idx.tz is not None):
        this_monday = this_monday_ist. astimezone(idx.tz)
    else:
        this_monday = this_monday_ist.replace(tzinfo=None)

    if weekly.index[-1] >= this_monday:
        weekly = weekly.iloc[:-1]
    return weekly


def _last_closed_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Drop today's partial daily bar if the NSE market hasn't fully closed yet
    (NSE closes 15:30 IST; add a 15-min buffer → 15:45 IST).
    """
    now_ist = pd.Timestamp.now(tz=IST)
    market_close_ist = now_ist.normalize().replace(hour=15, minute=45)

    idx = daily.index
    if idx.tzinfo is not None or (hasattr(idx, "tz") and idx.tz is not None):
        today_start = now_ist.normalize().astimezone(idx.tz)
    else:
        today_start = now_ist.normalize().replace(tzinfo=None)

    if now_ist < market_close_ist and daily.index[-1] >= today_start:
        daily = daily.iloc[:-1]
    return daily


# ---------------------------------------------------------------------------
# 4. STOCK EVALUATION
# ---------------------------------------------------------------------------

def evaluate_stock(symbol: str, daily: pd.DataFrame, weekly: pd.DataFrame, params: dict):
    """
    Returns a metrics dict (with Pass Strict / Pass Watchlist flags),
    or None if data is insufficient.
    """
    if daily is None or weekly is None:
        return None

    # Remove partial candles
    weekly = _last_closed_weekly(weekly)
    daily  = _last_closed_daily(daily)

    if len(daily) < 22 or len(weekly) < 55:
        return None

    try:
        d_close = daily["Close"].squeeze()
        d_open  = daily["Open"].squeeze()
        d_vol   = daily["Volume"].squeeze()
        w_close = weekly["Close"].squeeze()
        w_high  = weekly["High"].squeeze()
        w_vol   = weekly["Volume"].squeeze()

        # Sanity check — must be a Series after squeeze
        for s in (d_close, d_open, d_vol, w_close, w_high, w_vol):
            if not isinstance(s, pd.Series):
                return None

        latest_close = float(d_close.iloc[-1])
        latest_vol   = float(d_vol.iloc[-1])

        # Guard against NaN close
        if np.isnan(latest_close) or latest_close <= 0:
            return None

        # Weekly trend
        w_ema20 = _ema(w_close, 20)
        w_ema50 = _ema(w_close, 50)
        weekly_trend_ok = (
            float(w_close.iloc[-1]) > float(w_ema20.iloc[-1]) and
            float(w_ema20.iloc[-1]) > float(w_ema50.iloc[-1])
        )
        ema50_rising = float(w_ema50.iloc[-1]) > float(w_ema50.iloc[-2])

        # Daily 20-day open compression
        last20_open = d_open.iloc[-20:]
        mn, mx = float(last20_open.min()), float(last20_open.max())
        compression_ratio = mx / mn if mn > 0 else np.nan

        # Proximity to prior 20-week high (exclude current week)
        prior_slice  = w_high.iloc[-21:-1]
        prior_20w_hi = float(prior_slice.max())
        pct_from_high = (float(w_close.iloc[-1]) / prior_20w_hi - 1) * 100 if prior_20w_hi > 0 else np.nan

        # Weekly volume expansion vs prior 20 weeks
        vol_slice    = w_vol.iloc[-21:-1]
        w_vol_avg20  = float(vol_slice.mean())
        vol_ratio    = float(w_vol.iloc[-1]) / w_vol_avg20 if w_vol_avg20 > 0 else np.nan

        # Extension above weekly EMA20
        pct_above_ema20 = (float(w_close.iloc[-1]) / float(w_ema20.iloc[-1]) - 1) * 100

        row = {
            "Symbol":               symbol.replace(".NS", ""),
            "Close":                round(latest_close, 2),
            "Weekly Trend OK":      weekly_trend_ok,
            "EMA50 Rising":         ema50_rising,
            "Compression (20d)":    round((compression_ratio - 1) * 100, 2) if not np.isnan(compression_ratio) else None,
            "% From 20W High":      round(pct_from_high, 2) if not np.isnan(pct_from_high) else None,
            "Weekly Vol Ratio":     round(vol_ratio, 2) if not np.isnan(vol_ratio) else None,
            "% Above Weekly EMA20": round(pct_above_ema20, 2),
            "Daily Volume":         int(latest_vol),
        }

        # Evaluate pass conditions for both profiles
        for label, p in [("Strict", params["strict"]), ("Watchlist", params["watchlist"])]:
            vol_ok  = (row["Weekly Vol Ratio"] is not None and row["Weekly Vol Ratio"] >= p["vol_multiple"])
            high_ok = (row["% From 20W High"] is not None and row["% From 20W High"] >= -p["near_high_pct"])
            comp_ok = (row["Compression (20d)"] is not None and (row["Compression (20d)"] / 100 + 1) <= p["compression_band"])
            passes = (
                latest_vol   >= p["min_volume"]
                and latest_close >= p["min_price"]
                and weekly_trend_ok
                and (ema50_rising or not p["require_ema50_rising"])
                and comp_ok
                and high_ok
                and vol_ok
                and pct_above_ema20 <= p["extension_cap"]
            )
            row[f"Pass {label}"] = passes

        return row

    except Exception:
        return None


# ---------------------------------------------------------------------------
# 5. TABLE STYLING — no matplotlib
# ---------------------------------------------------------------------------

def _color_vol(val):
    try:
        v = float(val)
        if v >= 2.5:    return "background-color:#1a4d35; color:#E8E6DF"
        elif v >= 2.0:  return "background-color:#1d5c3e; color:#E8E6DF"
        elif v >= 1.5:  return "background-color:#236b48; color:#E8E6DF"
        elif v >= 1.25: return "background-color:#2a7a52; color:#E8E6DF"
        else:           return "background-color:#2a3d32; color:#8A93A0"
    except Exception:
        return ""


def _color_high(val):
    try:
        v = float(val)
        if v >= 0:      return "background-color:#1d5c3e; color:#E8E6DF"
        elif v >= -2:   return "background-color:#3a6b2a; color:#E8E6DF"
        elif v >= -5:   return "background-color:#7a6b1a; color:#E8E6DF"
        elif v >= -10:  return "background-color:#7a3a1a; color:#E8E6DF"
        else:           return "background-color:#5c2222; color:#8A93A0"
    except Exception:
        return ""


def style_table(t_df: pd.DataFrame):
    if t_df.empty:
        return t_df
    styled = t_df.style
    if "Weekly Vol Ratio" in t_df.columns:
        styled = styled.map(_color_vol, subset=["Weekly Vol Ratio"])
    if "% From 20W High" in t_df.columns:
        styled = styled.map(_color_high, subset=["% From 20W High"])
    fmt = {}
    for col, fmt_str in [
        ("Close",                "{:.2f}"),
        ("Compression (20d)",    "{:.2f}"),
        ("% From 20W High",      "{:.2f}"),
        ("Weekly Vol Ratio",     "{:.2f}"),
        ("% Above Weekly EMA20", "{:.2f}"),
        ("Daily Volume",         "{:,.0f}"),
    ]:
        if col in t_df.columns:
            fmt[col] = fmt_str
    return styled.format(fmt, na_rep="—")


# ---------------------------------------------------------------------------
# 6. UI
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
    universe_choice = st.radio(
        "Stock universe",
        ["Nifty 500 (fast)", "Top N by market cap (broader, slower first run)"],
        index=0,
    )
    top_n = 1000
    if universe_choice.startswith("Top N"):
        top_n = st.number_input(
            "How many stocks (by market cap)",
            min_value=100, max_value=2000, value=1000, step=100,
        )
        st.caption(
            "First run ranks ~2000 NSE stocks by market cap via live lookups — "
            "can take several minutes. Cached for 7 days after that."
        )

    uploaded = st.file_uploader(
        "Optional: upload your own symbol list (CSV with a Symbol column). "
        "Overrides the live fetch above.",
        type=["csv"],
    )

    max_limit = 2000 if universe_choice.startswith("Top N") else 500
    limit = st.slider(
        "Limit stocks scanned (lower = faster test run)",
        20, max_limit, min(500, max_limit), step=10,
    )

    st.header("Strict scan thresholds")
    s_min_price   = st.number_input("Min price (₹)", value=150, key="s_price")
    s_min_vol     = st.number_input("Min daily volume", value=500000, step=50000, key="s_vol")
    s_compression = st.slider("Compression band (max/min open)", 1.00, 1.30, 1.10, 0.01, key="s_comp")
    s_near_high   = st.slider("Within % of 20W high", 0.5, 10.0, 2.0, 0.5, key="s_high")
    s_vol_mult    = st.slider("Weekly volume expansion (×avg)", 1.0, 3.0, 1.5, 0.1, key="s_volmult")
    s_extension   = st.slider("Max % above weekly EMA20", 2.0, 25.0, 10.0, 1.0, key="s_ext")
    s_ema50_rising = st.checkbox("Require EMA50 rising", value=True, key="s_rising")

    st.header("Watchlist thresholds (looser)")
    w_min_price   = st.number_input("Min price (₹)", value=150, key="w_price")
    w_min_vol     = st.number_input("Min daily volume", value=500000, step=50000, key="w_vol")
    w_compression = st.slider("Compression band (max/min open)", 1.00, 1.40, 1.15, 0.01, key="w_comp")
    w_near_high   = st.slider("Within % of 20W high", 0.5, 15.0, 5.0, 0.5, key="w_high")
    w_vol_mult    = st.slider("Weekly volume expansion (×avg)", 1.0, 3.0, 1.25, 0.1, key="w_volmult")
    w_extension   = st.slider("Max % above weekly EMA20", 2.0, 30.0, 15.0, 1.0, key="w_ext")
    w_ema50_rising = st.checkbox("Require EMA50 rising", value=False, key="w_rising")

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

    # --- Resolve symbol list ---
    with st.spinner("Fetching stock universe..."):
        if uploaded is not None:
            symbols = symbols_from_upload(uploaded)
            if universe_choice.startswith("Top N"):
                symbols = symbols[:int(top_n)]
        elif universe_choice.startswith("Top N"):
            mc_msg = st.empty()
            pool = fetch_full_nse_equity_list()
            if pool:
                mc_msg.info(
                    f"Ranking {len(pool)} NSE stocks by market cap — "
                    "takes a few minutes on first run, then cached 7 days."
                )
                symbols = rank_by_market_cap(tuple(pool), int(top_n))
                mc_msg.empty()
            else:
                symbols = None
        else:
            symbols = fetch_nifty500_symbols()

    if not symbols:
        st.error(
            "Could not fetch the stock list from NSE — NSE often blocks automated requests. "
            "Please download the Nifty 500 list from "
            "https://www.niftyindices.com/indices/equity/broad-based-indices/nifty-500 "
            "and upload the CSV via the sidebar."
        )
        st.stop()

    symbols = symbols[:limit]

    # --- Download data ---
    progress = st.progress(0, text="Downloading daily data…")
    daily_data  = download_data(tuple(symbols), interval="1d",  period="6mo")
    progress.progress(50, text="Downloading weekly data…")
    weekly_data = download_data(tuple(symbols), interval="1wk", period="2y")
    progress.progress(85, text="Computing conditions…")

    # --- Evaluate ---
    results = []
    for sym in symbols:
        row = evaluate_stock(sym, daily_data.get(sym), weekly_data.get(sym), params)
        if row:
            results.append(row)
    progress.progress(100, text="Done ✓")
    time.sleep(0.4)
    progress.empty()

    if not results:
        st.warning(
            "No stocks could be evaluated — this usually means NSE data could not be "
            "downloaded for any ticker. Try uploading a Nifty 500 CSV to bypass the live fetch."
        )
        st.stop()

    df = pd.DataFrame(results)

    strict_df = (
        df[df["Pass Strict"]]
        .drop(columns=["Pass Strict", "Pass Watchlist"])
        .sort_values("% From 20W High", ascending=False)
        .reset_index(drop=True)
    )
    watchlist_df = (
        df[df["Pass Watchlist"] & ~df["Pass Strict"]]
        .drop(columns=["Pass Strict", "Pass Watchlist"])
        .sort_values("% From 20W High", ascending=False)
        .reset_index(drop=True)
    )

    # IST timestamp — accurate regardless of server timezone
    now_ist = pd.Timestamp.now(tz=IST).strftime("%d %b %Y, %H:%M IST")

    st.markdown(
        f"""
        <div class="ticker-board">
            <div class="ticker-cell">
                <div class="label">Scanned</div>
                <div class="value">{len(results)}</div>
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
                <div class="label">Last Run (IST)</div>
                <div class="value" style="font-size:0.95rem;">{now_ist}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">STRICT BREAKOUT LIST</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Near/at breakout · weekly volume confirmed · highest conviction</div>',
        unsafe_allow_html=True,
    )
    if strict_df.empty:
        st.markdown("`No stocks meet every strict condition right now — check back on Friday/Saturday.`")
    else:
        st.dataframe(style_table(strict_df), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">WATCHLIST</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Setting up but not yet at the trigger — track these for next week</div>',
        unsafe_allow_html=True,
    )
    if watchlist_df.empty:
        st.markdown("`Nothing setting up right now.`")
    else:
        st.dataframe(style_table(watchlist_df), use_container_width=True, hide_index=True)

    with st.expander("Show full scan data (all stocks, pass/fail columns included)"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download full results as CSV",
        csv_bytes, "scan_results.csv", "text/csv",
    )

else:
    st.info("Configure thresholds in the sidebar, then click **Run Scan**.")

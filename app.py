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


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range — measures real volatility including gaps."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _last_closed_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    """Drop the current in-progress week (partial candle)."""
    now_ist = pd.Timestamp.now(tz=IST)
    this_monday_ist = (now_ist - pd.Timedelta(days=now_ist.weekday())).normalize()
    idx = weekly.index
    if idx.tzinfo is not None or (hasattr(idx, "tz") and idx.tz is not None):
        this_monday = this_monday_ist.astimezone(idx.tz)
    else:
        this_monday = this_monday_ist.replace(tzinfo=None)
    if weekly.index[-1] >= this_monday:
        weekly = weekly.iloc[:-1]
    return weekly


def _last_closed_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Drop today's bar if NSE hasn't closed yet (15:45 IST buffer)."""
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
# 4. STOCK EVALUATION  (price-action grade)
# ---------------------------------------------------------------------------

def evaluate_stock(symbol: str, daily: pd.DataFrame, weekly: pd.DataFrame, params: dict):
    """
    Price-action-grade evaluation. Computes:

    TREND (weekly)
      • Close > EMA20 > EMA50, EMA50 slope rising

    CONSOLIDATION QUALITY (daily, last 20 days)
      • ATR compression: 10d ATR / 50d ATR < threshold  (tight = contracting volatility)
      • Higher-low structure: lowest low of last 10d > lowest low of prior 10d

    BREAKOUT PROXIMITY (weekly)
      • Within N% of prior 20-week high
      • Within N% of 52-week high

    CANDLE QUALITY (last weekly candle)
      • Close in upper 40%+ of the weekly candle range (not a rejection/wick candle)

    VOLUME EXPANSION (weekly)
      • This week's volume > X × 10-week median (median is more robust than mean — ignores prior spikes)

    EXTENSION GUARD
      • Close not more than N% above weekly EMA20

    Returns a dict with all metrics + Pass Strict / Pass Watchlist flags, or None.
    """
    if daily is None or weekly is None:
        return None

    weekly = _last_closed_weekly(weekly)
    daily  = _last_closed_daily(daily)

    # 60 daily bars covers ATR50 + buffer; 55 weekly bars covers EMA50
    if len(daily) < 60 or len(weekly) < 55:
        return None

    try:
        d_close  = daily["Close"].squeeze()
        d_open   = daily["Open"].squeeze()
        d_high   = daily["High"].squeeze()
        d_low    = daily["Low"].squeeze()
        d_vol    = daily["Volume"].squeeze()
        w_close  = weekly["Close"].squeeze()
        w_open   = weekly["Open"].squeeze()
        w_high   = weekly["High"].squeeze()
        w_low    = weekly["Low"].squeeze()
        w_vol    = weekly["Volume"].squeeze()

        for s in (d_close, d_open, d_high, d_low, d_vol,
                  w_close, w_open, w_high, w_low, w_vol):
            if not isinstance(s, pd.Series):
                return None

        latest_close = float(d_close.iloc[-1])
        latest_vol   = float(d_vol.iloc[-1])

        if np.isnan(latest_close) or latest_close <= 0:
            return None

        # ── WEEKLY TREND ──────────────────────────────────────────────────
        w_ema20 = _ema(w_close, 20)
        w_ema50 = _ema(w_close, 50)
        weekly_trend_ok = (
            float(w_close.iloc[-1]) > float(w_ema20.iloc[-1]) and
            float(w_ema20.iloc[-1]) > float(w_ema50.iloc[-1])
        )
        ema50_rising = float(w_ema50.iloc[-1]) > float(w_ema50.iloc[-2])

        # ── ATR COMPRESSION (daily) ───────────────────────────────────────
        # 10d ATR / 50d ATR — ratio < 1 means current volatility is
        # contracting vs longer history. Good base = ratio < ~0.75.
        d_atr10 = _atr(d_high, d_low, d_close, 10)
        d_atr50 = _atr(d_high, d_low, d_close, 50)
        atr10_val = float(d_atr10.iloc[-1])
        atr50_val = float(d_atr50.iloc[-1])
        atr_compression = atr10_val / atr50_val if atr50_val > 0 else np.nan

        # ── HIGHER-LOW STRUCTURE (daily) ──────────────────────────────────
        # Genuine accumulation = 3 ascending 10-day low windows (a staircase).
        # One window comparison is too easily faked by a single gap-up day.
        # All three windows must be ascending: window3 > window2 > window1.
        low_w1 = float(d_low.iloc[-30:-20].min())  # oldest
        low_w2 = float(d_low.iloc[-20:-10].min())  # middle
        low_w3 = float(d_low.iloc[-10:].min())     # most recent
        higher_lows = (low_w3 > low_w2) and (low_w2 > low_w1)

        # ── BREAKOUT PROXIMITY ────────────────────────────────────────────
        # Prior 20-week high (excluding current week)
        prior_20w_hi = float(w_high.iloc[-21:-1].max())
        pct_from_20w_high = (
            (float(w_close.iloc[-1]) / prior_20w_hi - 1) * 100
            if prior_20w_hi > 0 else np.nan
        )

        # 52-week high proximity (using daily data — more granular)
        high_52w = float(d_high.iloc[-252:].max()) if len(d_high) >= 252 else float(d_high.max())
        pct_from_52w_high = (latest_close / high_52w - 1) * 100 if high_52w > 0 else np.nan

        # ── WEEKLY CANDLE QUALITY ─────────────────────────────────────────
        # Close position within the weekly range.
        # 1.0 = closed at the very top, 0.0 = closed at the very bottom.
        # Require ≥ 0.4 to filter out rejection/wick candles.
        w_range = float(w_high.iloc[-1]) - float(w_low.iloc[-1])
        if w_range > 0:
            weekly_close_position = (float(w_close.iloc[-1]) - float(w_low.iloc[-1])) / w_range
        else:
            weekly_close_position = 0.5

        # ── WEEKLY VOLUME EXPANSION ───────────────────────────────────────
        # Use 10-week MEDIAN of prior weeks (more robust than mean — ignores
        # prior spike weeks that inflate the average and hide real expansion).
        prior_10w_vol_median = float(w_vol.iloc[-11:-1].median())
        vol_ratio = (
            float(w_vol.iloc[-1]) / prior_10w_vol_median
            if prior_10w_vol_median > 0 else np.nan
        )

        # ── EXTENSION GUARD ───────────────────────────────────────────────
        pct_above_ema20 = (float(w_close.iloc[-1]) / float(w_ema20.iloc[-1]) - 1) * 100

        # ── RELATIVE STRENGTH vs NIFTY 50 (12-week, display only) ────────
        # RS = stock 12W return / Nifty50 12W return.
        # > 1.0 = outperforming index. We pass nifty_close via params.
        rs_12w = None
        try:
            nifty = params.get("nifty_weekly")
            if nifty is not None and len(nifty) >= 13:
                nifty_now  = float(nifty.iloc[-1])
                nifty_12w  = float(nifty.iloc[-13])
                stock_now  = float(w_close.iloc[-1])
                stock_12w  = float(w_close.iloc[-13])
                if nifty_12w > 0 and stock_12w > 0:
                    stock_ret  = stock_now / stock_12w
                    nifty_ret  = nifty_now / nifty_12w
                    rs_12w     = round(stock_ret / nifty_ret, 2)
        except Exception:
            rs_12w = None

        # ── BUILD ROW ─────────────────────────────────────────────────────
        row = {
            "Symbol":                symbol.replace(".NS", ""),
            "Close":                 round(latest_close, 2),
            # Trend
            "Weekly Trend":          weekly_trend_ok,
            "EMA50 Rising":          ema50_rising,
            "RS vs Nifty (12W)":     rs_12w,
            # Consolidation quality
            "ATR Compression":       round(atr_compression, 2) if not np.isnan(atr_compression) else None,
            "Higher Lows (3W)":      higher_lows,
            # Breakout proximity
            "% From 20W High":       round(pct_from_20w_high, 2) if not np.isnan(pct_from_20w_high) else None,
            "% From 52W High":       round(pct_from_52w_high, 2) if not np.isnan(pct_from_52w_high) else None,
            # Candle quality
            "Wkly Close Position":   round(weekly_close_position, 2),
            # Volume
            "Weekly Vol Ratio":      round(vol_ratio, 2) if not np.isnan(vol_ratio) else None,
            # Extension guard
            "% Above EMA20":         round(pct_above_ema20, 2),
            # Liquidity
            "Daily Volume":          int(latest_vol),
        }

        # ── PASS FLAGS + FAILURE DIAGNOSTICS ─────────────────────────────
        for label, p in [("Strict", params["strict"]), ("Watchlist", params["watchlist"])]:
            vol_ok      = row["Weekly Vol Ratio"]   is not None and row["Weekly Vol Ratio"]   >= p["vol_multiple"]
            high_20w_ok = row["% From 20W High"]    is not None and row["% From 20W High"]    >= -p["near_high_pct"]
            high_52w_ok = row["% From 52W High"]    is not None and row["% From 52W High"]    >= -p["near_52w_pct"]
            atr_ok      = row["ATR Compression"]    is not None and row["ATR Compression"]    <= p["atr_compression"]
            candle_ok   = row["Wkly Close Position"]                                          >= p["min_close_position"]
            liq_ok      = latest_vol   >= p["min_volume"]
            price_ok    = latest_close >= p["min_price"]
            ext_ok      = pct_above_ema20 <= p["extension_cap"]
            hl_ok       = higher_lows or not p["require_higher_lows"]
            trend_ok    = weekly_trend_ok
            ema50_ok    = ema50_rising or not p["require_ema50_rising"]

            passes = (
                liq_ok and price_ok and trend_ok and ema50_ok
                and atr_ok and hl_ok
                and high_20w_ok and high_52w_ok
                and candle_ok and vol_ok and ext_ok
            )
            row[f"Pass {label}"] = passes

            # Store per-condition flags for the diagnostic breakdown (Watchlist only)
            if label == "Watchlist":
                row["_trend_ok"]   = trend_ok
                row["_ema50_ok"]   = ema50_ok
                row["_atr_ok"]     = atr_ok
                row["_hl_ok"]      = hl_ok
                row["_20w_ok"]     = high_20w_ok
                row["_52w_ok"]     = high_52w_ok
                row["_candle_ok"]  = candle_ok
                row["_vol_ok"]     = vol_ok
                row["_ext_ok"]     = ext_ok
                row["_liq_ok"]     = liq_ok
                row["_price_ok"]   = price_ok

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


def _color_atr(val):
    """Green = tight (good base), red = wide (volatile/extended)."""
    try:
        v = float(val)
        if v <= 0.60:   return "background-color:#1a4d35; color:#E8E6DF"
        elif v <= 0.75: return "background-color:#236b48; color:#E8E6DF"
        elif v <= 0.90: return "background-color:#7a6b1a; color:#E8E6DF"
        else:           return "background-color:#5c2222; color:#8A93A0"
    except Exception:
        return ""


def _color_close_pos(val):
    """Green = closed near top of range (bullish), red = closed near bottom (rejection)."""
    try:
        v = float(val)
        if v >= 0.7:    return "background-color:#1d5c3e; color:#E8E6DF"
        elif v >= 0.5:  return "background-color:#3a6b2a; color:#E8E6DF"
        elif v >= 0.35: return "background-color:#7a6b1a; color:#E8E6DF"
        else:           return "background-color:#5c2222; color:#8A93A0"
    except Exception:
        return ""


def _color_rs(val):
    """Green = outperforming Nifty, red = underperforming."""
    try:
        v = float(val)
        if v >= 1.20:   return "background-color:#1a4d35; color:#E8E6DF"
        elif v >= 1.05: return "background-color:#236b48; color:#E8E6DF"
        elif v >= 0.95: return "background-color:#7a6b1a; color:#E8E6DF"
        else:           return "background-color:#5c2222; color:#8A93A0"
    except Exception:
        return ""


def style_table(t_df: pd.DataFrame):
    if t_df.empty:
        return t_df
    styled = t_df.style
    for col, fn in [
        ("Weekly Vol Ratio",    _color_vol),
        ("% From 20W High",     _color_high),
        ("% From 52W High",     _color_high),
        ("ATR Compression",     _color_atr),
        ("Wkly Close Position", _color_close_pos),
        ("RS vs Nifty (12W)",   _color_rs),
    ]:
        if col in t_df.columns:
            styled = styled.map(fn, subset=[col])
    fmt = {}
    for col, fmt_str in [
        ("Close",               "{:.2f}"),
        ("ATR Compression",     "{:.2f}"),
        ("% From 20W High",     "{:.2f}"),
        ("% From 52W High",     "{:.2f}"),
        ("Wkly Close Position", "{:.2f}"),
        ("Weekly Vol Ratio",    "{:.2f}"),
        ("RS vs Nifty (12W)",   "{:.2f}"),
        ("% Above EMA20",       "{:.2f}"),
        ("Daily Volume",        "{:,.0f}"),
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
    s_min_price       = st.number_input("Min price (₹)", value=150, key="s_price")
    s_min_vol         = st.number_input("Min daily volume", value=500000, step=50000, key="s_vol")
    s_atr_compression = st.slider("ATR compression (10d÷50d, lower = tighter base)", 0.40, 1.20, 0.80, 0.05, key="s_atr",
                                   help="Ratio of short-term to long-term ATR. <0.75 = very tight base. 0.80 is a good default.")
    s_near_high       = st.slider("Within % of 20W high", 0.5, 10.0, 3.0, 0.5, key="s_high")
    s_near_52w        = st.slider("Within % of 52W high", 1.0, 40.0, 20.0, 1.0, key="s_52w",
                                   help="Stock must be near its 52-week high — ensures real momentum, not a dead-cat bounce.")
    s_vol_mult        = st.slider("Weekly vol expansion (×10W median)", 1.0, 3.0, 1.5, 0.1, key="s_volmult")
    s_close_pos       = st.slider("Min weekly close position (0=bottom, 1=top)", 0.0, 1.0, 0.40, 0.05, key="s_cpos",
                                   help="Filters out rejection/wick candles. 0.4 = close must be in upper 60% of the weekly range.")
    s_extension       = st.slider("Max % above weekly EMA20", 2.0, 25.0, 10.0, 1.0, key="s_ext")
    s_ema50_rising    = st.checkbox("Require EMA50 rising", value=True, key="s_rising")
    s_higher_lows     = st.checkbox("Require higher lows (accumulation structure)", value=True, key="s_hl",
                                     help="Last 10d lowest low > prior 10d lowest low — confirms accumulation, not distribution.")

    st.header("Watchlist thresholds (looser)")
    w_min_price       = st.number_input("Min price (₹)", value=150, key="w_price")
    w_min_vol         = st.number_input("Min daily volume", value=300000, step=50000, key="w_vol")
    w_atr_compression = st.slider("ATR compression (10d÷50d)", 0.40, 1.50, 1.20, 0.05, key="w_atr")
    w_near_high       = st.slider("Within % of 20W high", 0.5, 20.0, 10.0, 0.5, key="w_high")
    w_near_52w        = st.slider("Within % of 52W high", 1.0, 60.0, 40.0, 1.0, key="w_52w")
    w_vol_mult        = st.slider("Weekly vol expansion (×10W median)", 0.8, 3.0, 1.0, 0.1, key="w_volmult")
    w_close_pos       = st.slider("Min weekly close position", 0.0, 1.0, 0.25, 0.05, key="w_cpos")
    w_extension       = st.slider("Max % above weekly EMA20", 2.0, 40.0, 20.0, 1.0, key="w_ext")
    w_ema50_rising    = st.checkbox("Require EMA50 rising", value=False, key="w_rising")
    w_higher_lows     = st.checkbox("Require higher lows", value=False, key="w_hl")

    run_button = st.button("🔍 Run Scan", type="primary", use_container_width=True)

params = {
    "strict": dict(
        min_price=s_min_price, min_volume=s_min_vol,
        atr_compression=s_atr_compression,
        near_high_pct=s_near_high, near_52w_pct=s_near_52w,
        vol_multiple=s_vol_mult, min_close_position=s_close_pos,
        extension_cap=s_extension,
        require_ema50_rising=s_ema50_rising, require_higher_lows=s_higher_lows,
    ),
    "watchlist": dict(
        min_price=w_min_price, min_volume=w_min_vol,
        atr_compression=w_atr_compression,
        near_high_pct=w_near_high, near_52w_pct=w_near_52w,
        vol_multiple=w_vol_mult, min_close_position=w_close_pos,
        extension_cap=w_extension,
        require_ema50_rising=w_ema50_rising, require_higher_lows=w_higher_lows,
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
    progress = st.progress(0, text="Downloading Nifty 50 benchmark…")
    nifty_weekly_close = None
    try:
        nifty_raw = yf.download(
            "^NSEI", period="2y", interval="1wk",
            progress=False, auto_adjust=True,
        )
        if not nifty_raw.empty:
            nc = nifty_raw["Close"].squeeze()
            if isinstance(nc, pd.Series):
                nifty_weekly_close = _last_closed_weekly(
                    nifty_raw[["Close"]]
                )["Close"].squeeze()
    except Exception:
        nifty_weekly_close = None

    progress.progress(10, text="Downloading daily data…")
    daily_data  = download_data(tuple(symbols), interval="1d",  period="1y")
    progress.progress(55, text="Downloading weekly data…")
    weekly_data = download_data(tuple(symbols), interval="1wk", period="2y")
    progress.progress(85, text="Computing conditions…")

    # Inject Nifty benchmark into params so evaluate_stock can compute RS
    params["nifty_weekly"] = nifty_weekly_close

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
        display_cols = [c for c in df.columns if not c.startswith("_")]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    # ── DIAGNOSTIC BREAKDOWN ──────────────────────────────────────────────
    with st.expander("🔍 Why are stocks failing? (Condition breakdown)"):
        diag_cols = {
            "_trend_ok":  "Weekly Trend (Close>EMA20>EMA50)",
            "_ema50_ok":  "EMA50 Rising",
            "_atr_ok":    "ATR Compression",
            "_hl_ok":     "Higher Lows (3-window)",
            "_20w_ok":    "Near 20W High",
            "_52w_ok":    "Near 52W High",
            "_candle_ok": "Weekly Candle Close Position",
            "_vol_ok":    "Weekly Volume Expansion",
            "_ext_ok":    "Not Extended (EMA20 cap)",
            "_liq_ok":    "Min Daily Volume",
            "_price_ok":  "Min Price",
        }
        total = len(df)
        st.markdown(f"**{total} stocks evaluated.** For each condition, this shows how many stocks pass it (Watchlist thresholds).")
        diag_rows = []
        for col, label in diag_cols.items():
            if col in df.columns:
                passing = int(df[col].sum())
                failing = total - passing
                pct     = passing / total * 100 if total > 0 else 0
                diag_rows.append({
                    "Condition": label,
                    "Passing": passing,
                    "Failing": failing,
                    "Pass Rate": f"{pct:.0f}%",
                })
        if diag_rows:
            diag_df = pd.DataFrame(diag_rows)
            # Color pass rate: green if >50%, red if <20%
            def _color_pass_rate(val):
                try:
                    v = float(val.replace("%",""))
                    if v >= 60:   return "background-color:#1d5c3e; color:#E8E6DF"
                    elif v >= 35: return "background-color:#7a6b1a; color:#E8E6DF"
                    else:         return "background-color:#5c2222; color:#E8E6DF"
                except Exception:
                    return ""
            st.dataframe(
                diag_df.style.map(_color_pass_rate, subset=["Pass Rate"]),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "The condition with the lowest pass rate is your bottleneck. "
                "Loosen that slider first before loosening others."
            )

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download full results as CSV",
        csv_bytes, "scan_results.csv", "text/csv",
    )

else:
    st.info("Configure thresholds in the sidebar, then click **Run Scan**.")

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
# 4. STOCK EVALUATION — setup + breakout engine
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def evaluate_stock(symbol: str, daily: pd.DataFrame, weekly: pd.DataFrame, params: dict):
    """
    Higher-quality weekly swing scanner.

    Design:
      1) Trend regime: weekly Close > EMA20 > EMA50 + rising EMA50.
      2) Setup quality: tight weekly base, ATR compression, higher lows.
      3) Breakout engine: separates "near breakout" from "recent confirmed breakout".
         This fixes the old problem where a good breakout was rejected simply because
         the CURRENT week no longer had breakout-level volume.
      4) Confirmation: breakout week must close strongly and expand volume.
      5) Daily confirmation: price structure, RSI and ADX/DI.
      6) Relative strength: stock's 12W return minus Nifty 50's 12W return.
      7) Ranking score: candidates are ranked rather than relying only on one
         all-or-nothing filter.
    """
    if daily is None or weekly is None:
        return None

    weekly = _last_closed_weekly(weekly)
    daily = _last_closed_daily(daily)

    if len(daily) < 260 or len(weekly) < 65:
        return None

    try:
        d_close = daily["Close"].squeeze()
        d_open = daily["Open"].squeeze()
        d_high = daily["High"].squeeze()
        d_low = daily["Low"].squeeze()
        d_vol = daily["Volume"].squeeze()
        w_close = weekly["Close"].squeeze()
        w_open = weekly["Open"].squeeze()
        w_high = weekly["High"].squeeze()
        w_low = weekly["Low"].squeeze()
        w_vol = weekly["Volume"].squeeze()

        series = [d_close, d_open, d_high, d_low, d_vol,
                  w_close, w_open, w_high, w_low, w_vol]
        if not all(isinstance(s, pd.Series) for s in series):
            return None

        latest_close = float(d_close.iloc[-1])
        latest_vol = float(d_vol.iloc[-1])
        if not np.isfinite(latest_close) or latest_close <= 0:
            return None

        # ----- WEEKLY TREND -----
        w_ema20 = _ema(w_close, 20)
        w_ema50 = _ema(w_close, 50)
        weekly_trend_ok = (
            w_close.iloc[-1] > w_ema20.iloc[-1] > w_ema50.iloc[-1]
        )
        ema50_slope_pct = (w_ema50.iloc[-1] / w_ema50.iloc[-5] - 1) * 100

        # ----- DAILY TREND / MOMENTUM -----
        d_ema20 = _ema(d_close, 20)
        d_ema50 = _ema(d_close, 50)
        daily_trend_ok = d_close.iloc[-1] > d_ema20.iloc[-1] > d_ema50.iloc[-1]
        rsi14 = float(_rsi(d_close, 14).iloc[-1])
        adx14 = float(_adx(d_high, d_low, d_close, 14).iloc[-1])

        # ----- DAILY VOLATILITY / STRUCTURE -----
        d_atr10 = _atr(d_high, d_low, d_close, 10)
        d_atr50 = _atr(d_high, d_low, d_close, 50)
        atr10_val = float(d_atr10.iloc[-1])
        atr50_val = float(d_atr50.iloc[-1])
        atr_compression = atr10_val / atr50_val if atr50_val > 0 else np.nan
        atr_pct = (atr10_val / latest_close) * 100 if latest_close > 0 else np.nan

        low_w1 = float(d_low.iloc[-30:-20].min())
        low_w2 = float(d_low.iloc[-20:-10].min())
        low_w3 = float(d_low.iloc[-10:].min())
        higher_lows = low_w3 > low_w2 > low_w1

        # ----- WEEKLY BASE QUALITY -----
        prior_20w_hi = float(w_high.iloc[-21:-1].max())
        prior_20w_lo = float(w_low.iloc[-21:-1].min())
        base_width_pct = ((prior_20w_hi / prior_20w_lo) - 1) * 100 if prior_20w_lo > 0 else np.nan

        # 8-week range is useful for identifying a tight shelf near resistance.
        prior_8w_hi = float(w_high.iloc[-9:-1].max())
        prior_8w_lo = float(w_low.iloc[-9:-1].min())
        range_8w_pct = ((prior_8w_hi / prior_8w_lo) - 1) * 100 if prior_8w_lo > 0 else np.nan

        pct_from_20w_high = (
            (latest_close / prior_20w_hi - 1) * 100 if prior_20w_hi > 0 else np.nan
        )

        # Exclude the current daily bar from the reference high when possible.
        prior_52w_high = float(d_high.iloc[-253:-1].max())
        pct_from_52w_high = (
            (latest_close / prior_52w_high - 1) * 100 if prior_52w_high > 0 else np.nan
        )

        # ----- WEEKLY CANDLE QUALITY -----
        w_range = float(w_high.iloc[-1] - w_low.iloc[-1])
        weekly_close_position = (
            (float(w_close.iloc[-1]) - float(w_low.iloc[-1])) / w_range
            if w_range > 0 else 0.5
        )
        weekly_body_pct = (
            abs(float(w_close.iloc[-1] - w_open.iloc[-1])) / w_range
            if w_range > 0 else 0.0
        )

        # ----- WEEKLY VOLUME -----
        prior_10w_median = float(w_vol.iloc[-11:-1].median())
        current_vol_ratio = (
            float(w_vol.iloc[-1]) / prior_10w_median
            if prior_10w_median > 0 else np.nan
        )
        avg_4w_vol = float(w_vol.iloc[-5:-1].mean())
        avg_4w_vol_ratio = (
            avg_4w_vol / prior_10w_median if prior_10w_median > 0 else np.nan
        )

        # ----- RECENT BREAKOUT DETECTION -----
        # Search the last 4 closed weeks. A breakout is valid only when:
        # close > the 20W resistance that existed BEFORE that week AND
        # breakout volume >= threshold AND candle closes strongly.
        breakout_window = min(4, len(w_close) - 21)
        breakout_events = []
        breakout_volume_threshold = params["breakout_vol_multiple"]

        for j in range(len(w_close) - breakout_window, len(w_close)):
            resistance = float(w_high.iloc[j-20:j].max())
            med_vol = float(w_vol.iloc[max(0, j-10):j].median())
            if resistance <= 0 or med_vol <= 0:
                continue
            c = float(w_close.iloc[j])
            h = float(w_high.iloc[j])
            l = float(w_low.iloc[j])
            v = float(w_vol.iloc[j])
            rng = h - l
            close_pos = (c - l) / rng if rng > 0 else 0.5
            vr = v / med_vol

            if c > resistance and vr >= breakout_volume_threshold and close_pos >= 0.65:
                breakout_events.append({
                    "index": j,
                    "resistance": resistance,
                    "volume_ratio": vr,
                    "close_position": close_pos,
                })

        latest_breakout = breakout_events[-1] if breakout_events else None
        breakout_weeks_ago = (
            len(w_close) - 1 - latest_breakout["index"]
            if latest_breakout else None
        )

        if latest_breakout:
            breakout_level = latest_breakout["resistance"]
            breakout_vol_ratio = latest_breakout["volume_ratio"]
            breakout_close_position = latest_breakout["close_position"]
            # Price should generally hold above the breakout level after the event.
            recent_breakout_ok = (
                breakout_weeks_ago <= params["recent_breakout_weeks"] and
                latest_close >= breakout_level * (1 - params["retest_tolerance_pct"] / 100)
            )
            setup_type = (
                "Fresh Breakout" if breakout_weeks_ago == 0
                else f"Breakout {breakout_weeks_ago}W ago"
            )
        else:
            breakout_level = prior_20w_hi
            breakout_vol_ratio = np.nan
            breakout_close_position = np.nan
            recent_breakout_ok = False
            setup_type = "Near Breakout"

        near_breakout_ok = (
            latest_close >= prior_20w_hi * (1 - params["near_breakout_pct"] / 100)
            and latest_close <= prior_20w_hi * (1 + params["max_breakout_extension_pct"] / 100)
        )

        # A "setup" does not need breakout volume yet; a confirmed breakout does.
        # This avoids demanding today's volume from a stock that broke out 1-3 weeks ago.
        setup_ok = near_breakout_ok or recent_breakout_ok

        # ----- EXTENSION / RISK GUARD -----
        pct_above_ema20 = (latest_close / float(w_ema20.iloc[-1]) - 1) * 100
        extension_ok = pct_above_ema20 <= params["extension_cap"]

        # ----- RELATIVE STRENGTH -----
        rs_excess_12w = None
        try:
            nifty = params.get("nifty_weekly")
            if nifty is not None and len(nifty) >= 13:
                n_now, n_old = float(nifty.iloc[-1]), float(nifty.iloc[-13])
                s_now, s_old = float(w_close.iloc[-1]), float(w_close.iloc[-13])
                if n_old > 0 and s_old > 0:
                    rs_excess_12w = ((s_now / s_old) - (n_now / n_old)) * 100
        except Exception:
            rs_excess_12w = None

        # ----- SCORE -----
        p = params
        score = 0
        score += 15 if weekly_trend_ok else 0
        score += 10 if ema50_slope_pct > 0 else 0
        score += 10 if daily_trend_ok else 0
        score += 10 if higher_lows else 0
        score += 10 if np.isfinite(atr_compression) and atr_compression <= p["atr_compression"] else 0
        score += 10 if np.isfinite(base_width_pct) and base_width_pct <= p["max_base_width_pct"] else 0
        score += 10 if setup_ok else 0
        score += 15 if recent_breakout_ok else 0
        score += 10 if (
            (breakout_vol_ratio is not None and np.isfinite(breakout_vol_ratio)
             and breakout_vol_ratio >= breakout_volume_threshold)
            or (not recent_breakout_ok and np.isfinite(current_vol_ratio)
                and current_vol_ratio >= p["setup_vol_multiple"])
        ) else 0
        if rs_excess_12w is not None:
            score += 10 if rs_excess_12w >= p["rs_min"] else 0
        if np.isfinite(rsi14):
            score += 5 if 50 <= rsi14 <= 70 else 0
        if np.isfinite(adx14):
            score += 5 if adx14 >= p["adx_min"] else 0
        score = min(score, 120)

        # High-conviction classification:
        # confirmed breakout OR very close setup, with trend + momentum + volume quality.
        volume_confirmation_ok = (
            recent_breakout_ok and np.isfinite(breakout_vol_ratio)
            and breakout_vol_ratio >= breakout_volume_threshold
        ) or (
            not recent_breakout_ok and np.isfinite(current_vol_ratio)
            and current_vol_ratio >= p["setup_vol_multiple"]
        )

        # A fresh breakout naturally expands volatility. Therefore ATR compression
        # is mandatory for a near-breakout setup, but is relaxed for a confirmed
        # recent breakout so we do not reject the very move we are trying to find.
        volatility_ok = (
            np.isfinite(atr_compression)
            and (
                atr_compression <= p["atr_compression"]
                if not recent_breakout_ok
                else atr_compression <= p["atr_compression"] + 0.25
            )
        )

        strict_ok = (
            latest_close >= p["min_price"]
            and latest_vol >= p["min_volume"]
            and weekly_trend_ok
            and ema50_slope_pct > 0
            and daily_trend_ok
            and setup_ok
            and volatility_ok
            and higher_lows
            and np.isfinite(base_width_pct)
            and base_width_pct <= p["max_base_width_pct"]
            and extension_ok
            and volume_confirmation_ok
            and np.isfinite(rsi14) and 50 <= rsi14 <= 70
            and (rs_excess_12w is None or rs_excess_12w >= p["rs_min"])
        )

        watch_ok = (
            latest_close >= p["min_price"]
            and latest_vol >= p["watch_min_volume"]
            and weekly_trend_ok
            and setup_ok
            and extension_ok
            and score >= p["watch_score"]
        )

        row = {
            "Symbol": symbol.replace(".NS", ""),
            "Setup": setup_type,
            "Score": int(score),
            "Close": round(latest_close, 2),
            "Breakout Level": round(breakout_level, 2),
            "% From 20W High": round(pct_from_20w_high, 2),
            "% From 52W High": round(pct_from_52w_high, 2),
            "Weekly Trend": weekly_trend_ok,
            "Daily Trend": daily_trend_ok,
            "EMA50 Slope %": round(ema50_slope_pct, 2),
            "RS Excess 12W %": round(rs_excess_12w, 2) if rs_excess_12w is not None else None,
            "RSI14": round(rsi14, 1),
            "ADX14": round(adx14, 1),
            "ATR Compression": round(atr_compression, 2) if np.isfinite(atr_compression) else None,
            "ATR %": round(atr_pct, 2) if np.isfinite(atr_pct) else None,
            "Base Width 20W %": round(base_width_pct, 2) if np.isfinite(base_width_pct) else None,
            "8W Range %": round(range_8w_pct, 2) if np.isfinite(range_8w_pct) else None,
            "Higher Lows": higher_lows,
            "Weekly Close Position": round(weekly_close_position, 2),
            "Weekly Body %": round(weekly_body_pct, 2),
            "Current Wk Vol ×": round(current_vol_ratio, 2) if np.isfinite(current_vol_ratio) else None,
            "4W Avg Vol ×": round(avg_4w_vol_ratio, 2) if np.isfinite(avg_4w_vol_ratio) else None,
            "Breakout Vol ×": round(breakout_vol_ratio, 2) if np.isfinite(breakout_vol_ratio) else None,
            "Breakout Wks Ago": breakout_weeks_ago,
            "% Above W EMA20": round(pct_above_ema20, 2),
            "Daily Volume": int(latest_vol),
            "Pass Strict": strict_ok,
            "Pass Watchlist": watch_ok,
            "_trend_ok": weekly_trend_ok,
            "_daily_trend_ok": daily_trend_ok,
            "_setup_ok": setup_ok,
            "_breakout_ok": recent_breakout_ok,
            "_volume_ok": volume_confirmation_ok,
            "_atr_ok": volatility_ok,
            "_base_ok": np.isfinite(base_width_pct) and base_width_pct <= p["max_base_width_pct"],
            "_rs_ok": rs_excess_12w is None or rs_excess_12w >= p["rs_min"],
            "_rsi_ok": np.isfinite(rsi14) and 50 <= rsi14 <= 70,
            "_extension_ok": extension_ok,
            "_liq_ok": latest_vol >= p["watch_min_volume"],
        }
        return row

    except Exception:
        return None


# ---------------------------------------------------------------------------
# 5. TABLE STYLING
# ---------------------------------------------------------------------------

def _color_score(val):
    try:
        v = float(val)
        if v >= 95: return "background-color:#1a4d35; color:#E8E6DF"
        if v >= 80: return "background-color:#236b48; color:#E8E6DF"
        if v >= 65: return "background-color:#7a6b1a; color:#E8E6DF"
        return "background-color:#5c2222; color:#8A93A0"
    except Exception:
        return ""


def _color_vol(val):
    try:
        v = float(val)
        if v >= 2.5: return "background-color:#1a4d35; color:#E8E6DF"
        if v >= 2.0: return "background-color:#1d5c3e; color:#E8E6DF"
        if v >= 1.5: return "background-color:#236b48; color:#E8E6DF"
        if v >= 1.0: return "background-color:#7a6b1a; color:#E8E6DF"
        return "background-color:#2a3d32; color:#8A93A0"
    except Exception:
        return ""


def _color_high(val):
    try:
        v = float(val)
        if v >= 0: return "background-color:#1d5c3e; color:#E8E6DF"
        if v >= -2: return "background-color:#3a6b2a; color:#E8E6DF"
        if v >= -5: return "background-color:#7a6b1a; color:#E8E6DF"
        if v >= -10: return "background-color:#7a3a1a; color:#E8E6DF"
        return "background-color:#5c2222; color:#8A93A0"
    except Exception:
        return ""


def _color_rs(val):
    try:
        v = float(val)
        if v >= 10: return "background-color:#1a4d35; color:#E8E6DF"
        if v >= 5: return "background-color:#236b48; color:#E8E6DF"
        if v >= 0: return "background-color:#7a6b1a; color:#E8E6DF"
        return "background-color:#5c2222; color:#8A93A0"
    except Exception:
        return ""


def style_table(t_df: pd.DataFrame):
    if t_df.empty:
        return t_df
    styled = t_df.style
    for col, fn in [
        ("Score", _color_score),
        ("Current Wk Vol ×", _color_vol),
        ("Breakout Vol ×", _color_vol),
        ("% From 20W High", _color_high),
        ("% From 52W High", _color_high),
        ("RS Excess 12W %", _color_rs),
    ]:
        if col in t_df.columns:
            styled = styled.map(fn, subset=[col])

    fmt = {}
    for col, fmt_str in [
        ("Close", "{:.2f}"),
        ("Breakout Level", "{:.2f}"),
        ("Score", "{:.0f}"),
        ("% From 20W High", "{:.2f}"),
        ("% From 52W High", "{:.2f}"),
        ("EMA50 Slope %", "{:.2f}"),
        ("RS Excess 12W %", "{:.2f}"),
        ("RSI14", "{:.1f}"),
        ("ADX14", "{:.1f}"),
        ("ATR Compression", "{:.2f}"),
        ("ATR %", "{:.2f}"),
        ("Base Width 20W %", "{:.2f}"),
        ("8W Range %", "{:.2f}"),
        ("Weekly Close Position", "{:.2f}"),
        ("Weekly Body %", "{:.2f}"),
        ("Current Wk Vol ×", "{:.2f}"),
        ("4W Avg Vol ×", "{:.2f}"),
        ("Breakout Vol ×", "{:.2f}"),
        ("% Above W EMA20", "{:.2f}"),
        ("Daily Volume", "{:,.0f}"),
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
        <p>WEEKLY SETUP + RECENT BREAKOUT + VOLUME CONFIRMATION · RUN FRIDAY / SATURDAY AFTER CLOSE</p>
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
        st.caption("First run ranks NSE stocks by market cap. Cached for 7 days.")

    uploaded = st.file_uploader(
        "Optional: upload your own symbol list (CSV with a Symbol column).",
        type=["csv"],
    )

    max_limit = 2000 if universe_choice.startswith("Top N") else 500
    limit = st.slider(
        "Limit stocks scanned (lower = faster test run)",
        20, max_limit, min(500, max_limit), step=10,
    )

    st.header("Core filters")
    s_min_price = st.number_input("Min price (₹)", value=150, key="s_price")
    s_min_vol = st.number_input("Min daily volume", value=500000, step=50000, key="s_vol")

    st.subheader("Breakout")
    s_near_breakout = st.slider("Near 20W breakout (below/above %)", 0.5, 8.0, 2.0, 0.5)
    s_recent_breakout = st.slider("Recent breakout lookback (weeks)", 1, 4, 3, 1)
    s_breakout_vol = st.slider("Breakout volume ≥ × prior 10W median", 1.2, 3.5, 1.5, 0.1)
    s_setup_vol = st.slider("Near-breakout current volume ≥ × median", 0.7, 2.0, 1.0, 0.1)
    s_retest = st.slider("Allowed post-breakout retest below level (%)", 0.0, 5.0, 2.0, 0.5)

    st.subheader("Base / trend quality")
    s_atr = st.slider("ATR compression (10D ÷ 50D)", 0.50, 1.20, 0.85, 0.05)
    s_base_width = st.slider("Max 20W base width (%)", 10.0, 50.0, 30.0, 1.0)
    s_rs = st.slider("Minimum 12W excess return vs Nifty (%)", -10.0, 20.0, 0.0, 1.0)
    s_adx = st.slider("Minimum ADX14", 10, 35, 18, 1)
    s_extension = st.slider("Max % above weekly EMA20", 5.0, 25.0, 12.0, 1.0)

    st.subheader("Watchlist")
    w_min_vol = st.number_input("Watchlist min daily volume", value=300000, step=50000)
    w_score = st.slider("Watchlist minimum score", 50, 110, 70, 5)

    run_button = st.button("🔍 Run Scan", type="primary", use_container_width=True)

params = {
    "min_price": s_min_price,
    "min_volume": s_min_vol,
    "watch_min_volume": w_min_vol,
    "near_breakout_pct": s_near_breakout,
    "recent_breakout_weeks": s_recent_breakout,
    "breakout_vol_multiple": s_breakout_vol,
    "setup_vol_multiple": s_setup_vol,
    "retest_tolerance_pct": s_retest,
    "atr_compression": s_atr,
    "max_base_width_pct": s_base_width,
    "rs_min": s_rs,
    "adx_min": s_adx,
    "extension_cap": s_extension,
    "watch_score": w_score,
    "max_breakout_extension_pct": 5.0,
}

if run_button:
    with st.spinner("Fetching stock universe..."):
        if uploaded is not None:
            symbols = symbols_from_upload(uploaded)
            if universe_choice.startswith("Top N"):
                symbols = symbols[:int(top_n)]
        elif universe_choice.startswith("Top N"):
            mc_msg = st.empty()
            pool = fetch_full_nse_equity_list()
            if pool:
                mc_msg.info(f"Ranking {len(pool)} NSE stocks by market cap — first run may take a few minutes.")
                symbols = rank_by_market_cap(tuple(pool), int(top_n))
                mc_msg.empty()
            else:
                symbols = None
        else:
            symbols = fetch_nifty500_symbols()

    if not symbols:
        st.error(
            "Could not fetch the stock list from NSE. Upload a Nifty 500 CSV from the sidebar if NSE blocks the request."
        )
        st.stop()

    symbols = symbols[:limit]

    progress = st.progress(0, text="Downloading Nifty 50 benchmark…")
    nifty_weekly_close = None
    try:
        nifty_raw = yf.download(
            "^NSEI", period="2y", interval="1wk",
            progress=False, auto_adjust=True,
        )
        if not nifty_raw.empty:
            nifty_weekly_close = _last_closed_weekly(
                nifty_raw[["Close"]]
            )["Close"].squeeze()
    except Exception:
        nifty_weekly_close = None

    progress.progress(10, text="Downloading daily data…")
    daily_data = download_data(tuple(symbols), interval="1d", period="2y")
    progress.progress(55, text="Downloading weekly data…")
    weekly_data = download_data(tuple(symbols), interval="1wk", period="3y")
    progress.progress(85, text="Evaluating trend, setups and breakouts…")

    params["nifty_weekly"] = nifty_weekly_close

    results = []
    for sym in symbols:
        row = evaluate_stock(sym, daily_data.get(sym), weekly_data.get(sym), params)
        if row:
            results.append(row)

    progress.progress(100, text="Done ✓")
    time.sleep(0.3)
    progress.empty()

    if not results:
        st.warning("No stocks could be evaluated. Try a smaller universe or upload a symbol CSV.")
        st.stop()

    df = pd.DataFrame(results)

    strict_df = (
        df[df["Pass Strict"]]
        .drop(columns=["Pass Strict", "Pass Watchlist"])
        .sort_values(["Score", "Setup"], ascending=[False, True])
        .reset_index(drop=True)
    )
    watchlist_df = (
        df[df["Pass Watchlist"] & ~df["Pass Strict"]]
        .drop(columns=["Pass Strict", "Pass Watchlist"])
        .sort_values(["Score", "% From 20W High"], ascending=[False, False])
        .reset_index(drop=True)
    )

    now_ist = pd.Timestamp.now(tz=IST).strftime("%d %b %Y, %H:%M IST")

    st.markdown(
        f"""
        <div class="ticker-board">
            <div class="ticker-cell">
                <div class="label">Evaluated</div>
                <div class="value">{len(results)}</div>
            </div>
            <div class="ticker-cell">
                <div class="label">High Conviction</div>
                <div class="value accent">{len(strict_df)}</div>
            </div>
            <div class="ticker-cell">
                <div class="label">Watchlist</div>
                <div class="value bull">{len(watchlist_df)}</div>
            </div>
            <div class="ticker-cell">
                <div class="label">Last Run</div>
                <div class="value" style="font-size:0.95rem;">{now_ist}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">HIGH-CONVICTION SETUPS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Recent volume-confirmed breakouts OR very-close breakout setups with strong trend/structure.</div>',
        unsafe_allow_html=True,
    )
    if strict_df.empty:
        st.markdown("`No high-conviction candidates this week. That is preferable to forcing weak trades.`")
    else:
        st.dataframe(style_table(strict_df), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">EARLY WATCHLIST</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Good trend/setup candidates that still need a stronger trigger or confirmation.</div>',
        unsafe_allow_html=True,
    )
    if watchlist_df.empty:
        st.markdown("`No watchlist candidates right now.`")
    else:
        st.dataframe(style_table(watchlist_df), use_container_width=True, hide_index=True)

    with st.expander("📌 How to use the results"):
        st.markdown(
            """
            **Near Breakout:** price is within the configured distance of the prior 20-week high.
            Wait for a weekly close above the breakout level with strong volume rather than buying only because it is close.

            **Fresh/Recent Breakout:** the scanner found a prior weekly candle that closed above its own
            20-week resistance with the required volume expansion. The current price is allowed to be slightly
            below that level to accommodate a controlled retest.

            **Score:** ranking aid, not a probability of profit. Prefer the highest-quality names and then
            manually inspect the weekly chart before entry.

            **Important:** the scanner is a filter, not a guarantee. Backtest the rules on historical data
            before using real money.
            """
        )

    with st.expander("🔍 Why are stocks failing?"):
        diag_cols = {
            "_trend_ok": "Weekly trend",
            "_daily_trend_ok": "Daily trend",
            "_setup_ok": "Near/recent breakout",
            "_breakout_ok": "Recent confirmed breakout",
            "_volume_ok": "Volume confirmation",
            "_atr_ok": "ATR compression",
            "_base_ok": "Base width",
            "_rs_ok": "Relative strength",
            "_rsi_ok": "RSI 50–70",
            "_extension_ok": "Not extended",
            "_liq_ok": "Liquidity",
        }
        total = len(df)
        diag_rows = []
        for col, label in diag_cols.items():
            if col in df.columns:
                passing = int(df[col].fillna(False).sum())
                diag_rows.append({
                    "Condition": label,
                    "Passing": passing,
                    "Failing": total - passing,
                    "Pass Rate": f"{passing / total * 100:.0f}%" if total else "0%",
                })
        if diag_rows:
            st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)

    with st.expander("Show all evaluated stocks"):
        display_cols = [c for c in df.columns if not c.startswith("_")]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download full results as CSV",
        csv_bytes, "swing_scanner_results.csv", "text/csv",
    )

else:
    st.info("Configure the scanner in the sidebar, then click **Run Scan**.")

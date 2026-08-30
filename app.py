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
# ---------------------------------------------------------------------------
# 4. STOCK EVALUATION — 5-YEAR WEEKLY RESISTANCE BREAKOUT
# ---------------------------------------------------------------------------

def _safe_float(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _weekly_resistance(weekly: pd.DataFrame, lookback_weeks: int):
    """Highest weekly high in completed weeks BEFORE the signal week."""
    if len(weekly) < lookback_weeks + 5:
        return np.nan
    prior = weekly.iloc[-(lookback_weeks + 1):-1]
    return _safe_float(prior["High"].max()) if not prior.empty else np.nan


def evaluate_stock(symbol: str, daily: pd.DataFrame, weekly: pd.DataFrame, params: dict):
    """
    5-year weekly resistance breakout engine.

    Primary:
      • Last completed weekly close above the highest high of the preceding
        ~5 years.
      • Breakout-week volume >= configured multiple of prior weekly median.
      • Strong weekly candle.
      • Weekly trend: Close > EMA20 > EMA50 and EMA50 rising.
      • Daily trend confirmation.
      • Excessive extension is filtered.

    Secondary:
      • Recent confirmed breakouts from the previous 1–N completed weeks.
      • Breakout Watch candidates just below 5-year resistance.
    """
    if daily is None or weekly is None:
        return None

    weekly = _last_closed_weekly(weekly)
    daily = _last_closed_daily(daily)

    lookback = int(params["lookback_weeks"])
    min_weekly = lookback + 60
    if len(weekly) < min_weekly or len(daily) < 80:
        return None

    try:
        def series(df, col):
            return df[col].squeeze()

        d = pd.concat([
            series(daily, "Open").rename("Open"),
            series(daily, "High").rename("High"),
            series(daily, "Low").rename("Low"),
            series(daily, "Close").rename("Close"),
            series(daily, "Volume").rename("Volume"),
        ], axis=1).dropna()

        w = pd.concat([
            series(weekly, "Open").rename("Open"),
            series(weekly, "High").rename("High"),
            series(weekly, "Low").rename("Low"),
            series(weekly, "Close").rename("Close"),
            series(weekly, "Volume").rename("Volume"),
        ], axis=1).dropna()

        if len(w) < min_weekly or len(d) < 80:
            return None

        latest_close = _safe_float(d["Close"].iloc[-1])
        latest_dvol = _safe_float(d["Volume"].iloc[-1])
        wc = _safe_float(w["Close"].iloc[-1])
        if not np.isfinite(latest_close) or latest_close <= 0:
            return None

        # WEEKLY TREND
        w_ema20 = _ema(w["Close"], 20)
        w_ema50 = _ema(w["Close"], 50)
        ema20 = _safe_float(w_ema20.iloc[-1])
        ema50 = _safe_float(w_ema50.iloc[-1])
        weekly_trend_ok = wc > ema20 > ema50
        ema50_rising = _safe_float(w_ema50.iloc[-1]) > _safe_float(w_ema50.iloc[-5])

        # DAILY TREND
        d_ema20 = _ema(d["Close"], 20)
        d_ema50 = _ema(d["Close"], 50)
        daily_trend_ok = (
            latest_close > _safe_float(d_ema20.iloc[-1])
            > _safe_float(d_ema50.iloc[-1])
        )

        # 5-YEAR RESISTANCE
        resistance = _weekly_resistance(w, lookback)
        if not np.isfinite(resistance) or resistance <= 0:
            return None
        pct_vs_resistance = (wc / resistance - 1) * 100
        breakout_now = wc > resistance

        # VOLUME
        vp = int(params["volume_baseline_weeks"])
        baseline = _safe_float(w["Volume"].iloc[-(vp + 1):-1].median())
        current_wvol = _safe_float(w["Volume"].iloc[-1])
        vol_ratio = current_wvol / baseline if baseline > 0 else np.nan
        volume_confirmed = np.isfinite(vol_ratio) and vol_ratio >= params["volume_multiple"]

        # WEEKLY CANDLE
        wrange = _safe_float(w["High"].iloc[-1] - w["Low"].iloc[-1])
        wclose_pos = ((wc - _safe_float(w["Low"].iloc[-1])) / wrange) if wrange > 0 else 0.5
        body_pct = (
            abs(wc - _safe_float(w["Open"].iloc[-1])) / wrange
            if wrange > 0 else 0
        )
        candle_ok = (
            wclose_pos >= params["min_close_position"]
            and body_pct >= params["min_body_pct"]
            and wc >= _safe_float(w["Open"].iloc[-1])
        )

        # RECENT CONFIRMED BREAKOUT SEARCH
        recent_n = int(params["recent_breakout_weeks"])
        recent_events = []
        first_i = max(lookback, len(w) - recent_n - 1)
        for i in range(first_i, len(w)):
            prior_high = _safe_float(w["High"].iloc[i-lookback:i].max())
            c = _safe_float(w["Close"].iloc[i])
            v = _safe_float(w["Volume"].iloc[i])
            b0 = max(0, i - vp)
            vbase = _safe_float(w["Volume"].iloc[b0:i].median())
            vr = v / vbase if vbase > 0 else np.nan
            rng = _safe_float(w["High"].iloc[i] - w["Low"].iloc[i])
            pos = ((c - _safe_float(w["Low"].iloc[i])) / rng) if rng > 0 else 0.5
            bull = c >= _safe_float(w["Open"].iloc[i])

            if (
                np.isfinite(prior_high) and prior_high > 0
                and c > prior_high
                and np.isfinite(vr) and vr >= params["volume_multiple"]
            ):
                recent_events.append({
                    "index": i, "resistance": prior_high,
                    "volume_ratio": vr, "close_position": pos, "bullish": bull
                })

        current_event = next(
            (e for e in recent_events if e["index"] == len(w) - 1), None
        )
        prior_event = None
        weeks_since = None
        for e in reversed(recent_events):
            age = len(w) - 1 - e["index"]
            if 1 <= age <= recent_n:
                prior_event, weeks_since = e, age
                break

        fresh_breakout = (
            current_event is not None
            and current_event["close_position"] >= params["min_close_position"]
            and current_event["bullish"]
        )
        recent_breakout = prior_event is not None

        # MOMENTUM
        delta = w["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = _safe_float((100 - 100 / (1 + rs)).iloc[-1])

        up = w["High"].diff()
        down = -w["Low"].diff()
        plus_dm = up.where((up > down) & (up > 0), 0.0)
        minus_dm = down.where((down > up) & (down > 0), 0.0)
        tr = pd.concat([
            w["High"] - w["Low"],
            (w["High"] - w["Close"].shift(1)).abs(),
            (w["Low"] - w["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = _safe_float(dx.rolling(14).mean().iloc[-1])

        # RELATIVE STRENGTH
        rs_excess = None
        try:
            nifty = params.get("nifty_weekly")
            if nifty is not None and len(nifty) >= 13 and len(w) >= 13:
                n_now, n_old = _safe_float(nifty.iloc[-1]), _safe_float(nifty.iloc[-13])
                s_old = _safe_float(w["Close"].iloc[-13])
                if n_now > 0 and n_old > 0 and s_old > 0:
                    rs_excess = ((wc / s_old - 1) - (n_now / n_old - 1)) * 100
        except Exception:
            pass

        # ATR %
        atr_pct = _safe_float(atr14.iloc[-1]) / wc * 100 if wc > 0 else np.nan

        # BASE WIDTH
        bw = int(params["base_weeks"])
        base_high = _safe_float(w["High"].iloc[-bw-1:-1].max())
        base_low = _safe_float(w["Low"].iloc[-bw-1:-1].min())
        base_width = (base_high / base_low - 1) * 100 if base_low > 0 else np.nan

        not_overextended = pct_vs_resistance <= params["max_extension_pct"]

        # QUALITY SCORE — ranking, not a guarantee of profitability
        score = 0
        score += 20 if breakout_now else 0
        score += 20 if volume_confirmed else 0
        score += 10 if candle_ok else 0
        score += 10 if weekly_trend_ok else 0
        score += 8 if ema50_rising else 0
        score += 8 if daily_trend_ok else 0
        score += 6 if np.isfinite(rsi) and params["rsi_min"] <= rsi <= params["rsi_max"] else 0
        score += 6 if np.isfinite(adx) and adx >= params["adx_min"] else 0
        score += 6 if rs_excess is not None and rs_excess > 0 else 0
        score += 6 if not_overextended else 0

        if fresh_breakout and volume_confirmed and weekly_trend_ok:
            signal = "CONFIRMED BREAKOUT"
        elif recent_breakout and not_overextended:
            signal = f"RECENT BREAKOUT ({weeks_since}W AGO)"
        elif -params["watch_distance_pct"] <= pct_vs_resistance <= 0 and weekly_trend_ok:
            signal = "BREAKOUT WATCH"
        else:
            signal = "OTHER"

        confirmed = (
            breakout_now and volume_confirmed and candle_ok
            and weekly_trend_ok and ema50_rising and not_overextended
            and latest_close >= params["min_price"]
            and latest_dvol >= params["min_volume"]
        )
        watch = (
            signal == "BREAKOUT WATCH"
            and latest_close >= params["min_price"]
            and latest_dvol >= params["min_volume"]
        )

        return {
            "Symbol": symbol.replace(".NS", ""),
            "Signal": signal,
            "Score": int(score),
            "Close": round(latest_close, 2),
            "5Y Resistance": round(resistance, 2),
            "% vs 5Y Resistance": round(pct_vs_resistance, 2),
            "Weekly Vol Ratio": round(vol_ratio, 2) if np.isfinite(vol_ratio) else None,
            "Weekly Close Position": round(wclose_pos, 2),
            "Weekly Body %": round(body_pct * 100, 1),
            "Weekly Trend": weekly_trend_ok,
            "EMA50 Rising": ema50_rising,
            "Daily Trend": daily_trend_ok,
            "RSI14": round(rsi, 1) if np.isfinite(rsi) else None,
            "ADX14": round(adx, 1) if np.isfinite(adx) else None,
            "RS vs Nifty 12W": round(rs_excess, 1) if rs_excess is not None else None,
            "ATR %": round(atr_pct, 2) if np.isfinite(atr_pct) else None,
            "Base Width %": round(base_width, 1) if np.isfinite(base_width) else None,
            "Daily Volume": int(latest_dvol),
            "Breakout Confirmed": confirmed,
            "Breakout Watch": watch,
        }

    except Exception:
        return None


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
        ("Weekly Vol Ratio", _color_vol),
        ("% vs 5Y Resistance", _color_high),
        ("Weekly Close Position", _color_close_pos),
        ("RS vs Nifty 12W", _color_rs),
    ]:
        if col in t_df.columns:
            styled = styled.map(fn, subset=[col])
    fmt = {}
    for col, fmt_str in [
        ("Close", "{:.2f}"),
        ("5Y Resistance", "{:.2f}"),
        ("% vs 5Y Resistance", "{:.2f}"),
        ("Weekly Vol Ratio", "{:.2f}"),
        ("Weekly Close Position", "{:.2f}"),
        ("Weekly Body %", "{:.1f}"),
        ("RSI14", "{:.1f}"),
        ("ADX14", "{:.1f}"),
        ("RS vs Nifty 12W", "{:.1f}"),
        ("ATR %", "{:.2f}"),
        ("Base Width %", "{:.1f}"),
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
        <h1>5-YEAR RESISTANCE BREAKOUT SCANNER — NIFTY 500</h1>
        <p>WEEKLY TIMEFRAME &nbsp;·&nbsp; 5-YEAR HIGH BREAKOUT + VOLUME CONFIRMATION
        &nbsp;·&nbsp; RUN FRIDAY / SATURDAY AFTER WEEKLY CLOSE</p>
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

    st.header("5-Year breakout settings")
    s_lookback_years = st.slider("Resistance lookback (years)", 3, 7, 5, 1)
    s_vol_baseline = st.slider("Weekly volume baseline (weeks)", 10, 30, 20, 5)
    s_vol_mult = st.slider("Breakout volume multiple", 1.0, 3.0, 1.5, 0.1)
    s_min_close = st.slider("Minimum weekly close position", 0.50, 0.90, 0.65, 0.05)
    s_min_body = st.slider("Minimum candle body (% of range)", 10, 80, 30, 5)
    s_recent = st.slider("Recent breakout window (weeks)", 1, 6, 3, 1)
    s_watch = st.slider("Breakout watch distance below resistance (%)", 0.5, 5.0, 2.0, 0.5)
    s_extension = st.slider("Maximum extension above resistance (%)", 2.0, 20.0, 10.0, 1.0)
    s_min_price = st.number_input("Minimum price (₹)", value=150, step=10)
    s_min_vol = st.number_input("Minimum daily volume", value=500000, step=50000)
    s_rsi_min = st.slider("Minimum weekly RSI", 40, 70, 50, 1)
    s_rsi_max = st.slider("Maximum weekly RSI", 65, 90, 80, 1)
    s_adx = st.slider("Minimum weekly ADX", 10, 40, 18, 1)
    s_base_weeks = st.slider("Base-width measurement (weeks)", 8, 30, 12, 1)

    run_button = st.button("🔍 Run Scan", type="primary", use_container_width=True)

params = {
    "lookback_weeks": int(s_lookback_years * 52),
    "volume_baseline_weeks": int(s_vol_baseline),
    "volume_multiple": float(s_vol_mult),
    "min_close_position": float(s_min_close),
    "min_body_pct": float(s_min_body) / 100.0,
    "recent_breakout_weeks": int(s_recent),
    "watch_distance_pct": float(s_watch),
    "max_extension_pct": float(s_extension),
    "min_price": float(s_min_price),
    "min_volume": float(s_min_vol),
    "rsi_min": float(s_rsi_min),
    "rsi_max": float(s_rsi_max),
    "adx_min": float(s_adx),
    "base_weeks": int(s_base_weeks),
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
            "^NSEI", period="7y", interval="1wk",
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
    weekly_data = download_data(tuple(symbols), interval="1wk", period="7y")
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

    confirmed_df = (
        df[df["Breakout Confirmed"]]
        .sort_values(["Score", "Weekly Vol Ratio"], ascending=[False, False])
        .reset_index(drop=True)
    )
    recent_df = (
        df[df["Signal"].str.startswith("RECENT BREAKOUT", na=False)]
        .sort_values(["Score", "Weekly Vol Ratio"], ascending=[False, False])
        .reset_index(drop=True)
    )
    watchlist_df = (
        df[df["Breakout Watch"]]
        .sort_values(["Score", "% vs 5Y Resistance"], ascending=[False, False])
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
                <div class="value accent">{len(confirmed_df)}</div>
            </div>
            <div class="ticker-cell">
                <div class="label">Watchlist Hits</div>
                <div class="value bull">{len(watchlist_df) + len(recent_df)}</div>
            </div>
            <div class="ticker-cell">
                <div class="label">Last Run (IST)</div>
                <div class="value" style="font-size:0.95rem;">{now_ist}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">CONFIRMED 5-YEAR BREAKOUTS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Weekly close above the highest resistance of the preceding 5 years + required volume confirmation</div>',
        unsafe_allow_html=True,
    )
    if confirmed_df.empty:
        st.markdown("`No confirmed 5-year breakouts meet all core conditions.`")
    else:
        st.dataframe(style_table(confirmed_df), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">RECENT CONFIRMED BREAKOUTS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Breakout occurred within the recent window and had the required weekly volume</div>',
        unsafe_allow_html=True,
    )
    if recent_df.empty:
        st.markdown("`No recent confirmed breakouts found.`")
    else:
        st.dataframe(style_table(recent_df), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">BREAKOUT WATCH</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Within the configured distance below 5-year resistance — monitor for a confirmed weekly close</div>',
        unsafe_allow_html=True,
    )
    if watchlist_df.empty:
        st.markdown("`Nothing close enough to 5-year resistance right now.`")
    else:
        st.dataframe(style_table(watchlist_df), use_container_width=True, hide_index=True)

    with st.expander("Show full scan data (all stocks, pass/fail columns included)"):
        display_cols = [c for c in df.columns if not c.startswith("_")]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    with st.expander("🔍 Scanner diagnostics"):
        st.markdown(f"**{len(df)} stocks evaluated.**")
        diag = pd.DataFrame([
            {"Metric": "Above 5Y resistance", "Count": int((df["% vs 5Y Resistance"] > 0).sum())},
            {"Metric": "Volume ≥ configured multiple", "Count": int((df["Weekly Vol Ratio"] >= s_vol_mult).sum())},
            {"Metric": "Weekly trend confirmed", "Count": int(df["Weekly Trend"].sum())},
            {"Metric": "EMA50 rising", "Count": int(df["EMA50 Rising"].sum())},
            {"Metric": "Daily trend confirmed", "Count": int(df["Daily Trend"].sum())},
            {"Metric": "Confirmed breakouts", "Count": len(confirmed_df)},
            {"Metric": "Recent breakouts", "Count": len(recent_df)},
            {"Metric": "Breakout watch", "Count": len(watchlist_df)},
        ])
        st.dataframe(diag, use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download full results as CSV",
        csv_bytes, "scan_results.csv", "text/csv",
    )

else:
    st.info("Configure thresholds in the sidebar, then click **Run Scan**.")

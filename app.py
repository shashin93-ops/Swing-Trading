"""
Swing Breakout Scanner — Stage 2 Breakout from 6-Month+ Base
=============================================================
Finds Nifty 500 stocks that:
  1. Had a prior bullish advance (Stage 1 → Stage 2 move)
  2. Consolidated/ranged sideways for 6+ months (Stage 2 base / flat base)
  3. Are NOW breaking above that multi-month resistance on expanding weekly volume
  4. Have moved less than 10% from the breakout level (early entry, not chasing)

Weekly candle timeframe. Run every Friday / Saturday after NSE close.
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

st.set_page_config(page_title="Stage 2 Breakout Scanner", page_icon="📈", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
[data-testid="stDataFrame"] * { font-family: 'JetBrains Mono', monospace !important; }
.app-header {
    border: 1px solid #2A323D; border-left: 4px solid #C89B3C;
    background: linear-gradient(90deg, #161D27 0%, #0F1419 100%);
    padding: 22px 26px; border-radius: 4px; margin-bottom: 22px;
}
.app-header h1 { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.75rem;
    color:#E8E6DF; margin:0 0 6px 0; letter-spacing:-0.5px; }
.app-header p  { font-family:'JetBrains Mono',monospace; font-size:0.78rem;
    color:#8A93A0; margin:0; letter-spacing:0.2px; }
.ticker-board  { display:flex; gap:1px; background:#2A323D; border:1px solid #2A323D;
    border-radius:4px; overflow:hidden; margin-bottom:24px; }
.ticker-cell   { flex:1; background:#161D27; padding:14px 18px; }
.ticker-cell .label { font-family:'JetBrains Mono',monospace; font-size:0.65rem;
    text-transform:uppercase; letter-spacing:1.2px; color:#8A93A0; margin-bottom:4px; }
.ticker-cell .value { font-family:'JetBrains Mono',monospace; font-size:1.4rem;
    font-weight:700; color:#E8E6DF; }
.ticker-cell .value.accent { color:#C89B3C; }
.ticker-cell .value.bull   { color:#4A9B7F; }
.ticker-cell .value.sm     { font-size:0.95rem; }
.section-label { font-family:'JetBrains Mono',monospace; font-size:0.70rem;
    text-transform:uppercase; letter-spacing:1.5px; color:#C89B3C;
    border-bottom:1px solid #2A323D; padding-bottom:8px; margin:28px 0 4px 0; }
.section-sub   { font-family:'JetBrains Mono',monospace; font-size:0.75rem;
    color:#8A93A0; margin-bottom:14px; }
div.stButton > button { font-family:'Space Grotesk',sans-serif; font-weight:700;
    background-color:#C89B3C; color:#0F1419; border:none; border-radius:3px; }
div.stButton > button:hover { background-color:#E0B455; color:#0F1419; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>STAGE 2 BREAKOUT SCANNER — NIFTY 500</h1>
  <p>PRIOR ADVANCE → 6-MONTH+ BASE → FRESH BREAKOUT ABOVE RESISTANCE · WEEKLY CANDLES · RUN FRIDAY / SATURDAY</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYMBOL FETCHING
# ─────────────────────────────────────────────────────────────────────────────
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nifty500():
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        r = s.get("https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
                  headers=NSE_HEADERS, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        return [x.strip() + ".NS" for x in df["Symbol"].astype(str).tolist()]
    except Exception:
        return None

def symbols_from_upload(f):
    df = pd.read_csv(f)
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    syms = df[col].astype(str).str.strip().tolist()
    return [s if s.endswith(".NS") else s + ".NS" for s in syms]

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=21600, show_spinner=False)
def download_weekly(tickers: tuple) -> dict:
    """Download 3 years of weekly OHLCV. Returns {ticker: DataFrame}."""
    tickers = list(tickers)
    out = {}
    batch = 50
    for i in range(0, len(tickers), batch):
        grp = tickers[i:i+batch]
        try:
            raw = yf.download(grp, period="3y", interval="1wk",
                              group_by="ticker", threads=True,
                              progress=False, auto_adjust=True)
        except Exception:
            continue
        if len(grp) == 1:
            df = raw.dropna(how="all")
            if len(df) > 30:
                out[grp[0]] = df
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0).unique().tolist()
            price_names = {"Open","High","Low","Close","Volume"}
            if set(str(x) for x in lvl0[:5]) & price_names:
                raw = raw.swaplevel(axis=1)
            for t in grp:
                try:
                    df = raw[t].dropna(how="all")
                    if len(df) > 30:
                        out[t] = df
                except Exception:
                    pass
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _last_closed_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the current in-progress weekly bar."""
    now = pd.Timestamp.now(tz=IST)
    monday = (now - pd.Timedelta(days=now.weekday())).normalize()
    idx = df.index
    ref = monday.astimezone(idx.tz) if (hasattr(idx,'tz') and idx.tz) else monday.replace(tzinfo=None)
    return df.iloc[:-1] if df.index[-1] >= ref else df

# ─────────────────────────────────────────────────────────────────────────────
# 4. CORE EVALUATION — Stage 2 Breakout Logic
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(symbol: str, raw: pd.DataFrame, p: dict) -> dict | None:
    """
    Stage 2 Breakout pattern on weekly candles:

    STEP 1 — PRIOR ADVANCE (bullish phase before the base)
        The stock must have had a meaningful run-up before consolidating.
        We measure: highest close in weeks [-base_weeks-26 : -base_weeks]
                    vs lowest close in that same prior window.
        The advance must be >= prior_advance_pct (default 20%).
        This proves the stock was in Stage 2, not breaking out of a downtrend.

    STEP 2 — MULTI-MONTH BASE (sideways consolidation)
        Over the last `base_weeks` weeks (default 26 = ~6 months):
        a) Range tightness: (max_high - min_low) / min_low <= base_range_pct
           This measures how "flat" the base is. A wide choppy range is not a base.
        b) Declining volume trend inside the base (drying up = accumulation).
           First-half avg volume > second-half avg volume (volume contracting).
        c) Price held above the weekly EMA40 for most of the base.
           This proves the long-term trend is still intact underneath.

    STEP 3 — FRESH BREAKOUT
        The latest weekly close must be ABOVE the base resistance
        (defined as the highest weekly HIGH during the base period).
        "Fresh" means the close is within `max_breakout_pct` (default 10%)
        above that resistance — we don't chase stocks already 15%+ above it.

    STEP 4 — VOLUME CONFIRMATION
        The breakout week's volume must be >= `vol_expansion_x` × the
        10-week median volume inside the base.
        This is the most important false-breakout filter: price alone lies,
        volume doesn't.

    STEP 5 — WEEKLY CANDLE QUALITY
        The breakout candle must close in the upper half of its range
        (close position >= 0.5). A breakout that closes near its lows
        is a failed/suspicious breakout.

    STEP 6 — LIQUIDITY
        Min weekly volume and min price filters.
    """
    if raw is None:
        return None
    try:
        df = _last_closed_weekly(raw)
        df = df.dropna(subset=["Close","High","Low","Open","Volume"])
        if len(df) < p["base_weeks"] + 30:
            return None

        closes  = df["Close"].squeeze()
        highs   = df["High"].squeeze()
        lows    = df["Low"].squeeze()
        volumes = df["Volume"].squeeze()

        if not isinstance(closes, pd.Series):
            return None

        base_weeks  = p["base_weeks"]          # ~26 (6 months)
        base_slice  = slice(-base_weeks, None)  # last N weeks = the base
        prior_slice = slice(-(base_weeks + 26), -base_weeks)  # 26 weeks before base

        base_highs   = highs.iloc[base_slice]
        base_lows    = lows.iloc[base_slice]
        base_closes  = closes.iloc[base_slice]
        base_vols    = volumes.iloc[base_slice]
        prior_closes = closes.iloc[prior_slice]

        if len(prior_closes) < 10:
            return None

        # ── STEP 1: PRIOR ADVANCE ────────────────────────────────────────
        prior_high = float(prior_closes.max())
        prior_low  = float(prior_closes.min())
        prior_advance = (prior_high - prior_low) / prior_low * 100 if prior_low > 0 else 0
        had_advance = prior_advance >= p["prior_advance_pct"]

        # ── STEP 2a: BASE RANGE TIGHTNESS ────────────────────────────────
        base_max_high = float(base_highs.max())   # resistance ceiling
        base_min_low  = float(base_lows.min())    # support floor
        base_range_pct = (base_max_high - base_min_low) / base_min_low * 100 if base_min_low > 0 else 999
        tight_base = base_range_pct <= p["base_range_pct"]

        # ── STEP 2b: VOLUME DRYING UP INSIDE BASE ────────────────────────
        half = base_weeks // 2
        vol_first_half  = float(base_vols.iloc[:half].mean())
        vol_second_half = float(base_vols.iloc[half:].mean())
        volume_drying   = vol_second_half < vol_first_half  # volume contracting into base end

        # ── STEP 2c: PRICE ABOVE EMA40 during base ────────────────────────
        ema40_full  = _ema(closes, 40)
        ema40_base  = ema40_full.iloc[base_slice]
        pct_above_ema40 = (base_closes > ema40_base).mean()  # fraction of base weeks above EMA40
        above_ema40 = pct_above_ema40 >= 0.6   # at least 60% of base weeks above EMA40

        # ── STEP 3: FRESH BREAKOUT ────────────────────────────────────────
        # Resistance = highest HIGH during the base (excluding the last week = the breakout candle)
        resistance = float(base_highs.iloc[:-1].max())
        latest_close = float(closes.iloc[-1])
        latest_high  = float(highs.iloc[-1])

        # Breakout = latest close is ABOVE the resistance
        broke_out = latest_close > resistance

        # How far above resistance (breakout extension)
        pct_above_resistance = (latest_close / resistance - 1) * 100 if resistance > 0 else 999

        # Still early = within max_breakout_pct of resistance
        still_early = broke_out and pct_above_resistance <= p["max_breakout_pct"]

        # ── STEP 4: BREAKOUT WEEK VOLUME ─────────────────────────────────
        breakout_week_vol = float(volumes.iloc[-1])
        base_vol_median   = float(base_vols.iloc[:-1].median())  # exclude breakout week itself
        vol_expansion     = breakout_week_vol / base_vol_median if base_vol_median > 0 else 0
        vol_confirmed     = vol_expansion >= p["vol_expansion_x"]

        # ── STEP 5: BREAKOUT CANDLE QUALITY ──────────────────────────────
        w_range = float(highs.iloc[-1]) - float(lows.iloc[-1])
        if w_range > 0:
            candle_close_pos = (latest_close - float(lows.iloc[-1])) / w_range
        else:
            candle_close_pos = 0.5
        good_candle = candle_close_pos >= p["min_close_position"]

        # ── STEP 6: LIQUIDITY ─────────────────────────────────────────────
        latest_weekly_vol = float(volumes.iloc[-1])
        price_ok  = latest_close >= p["min_price"]
        vol_liq   = latest_weekly_vol >= p["min_weekly_volume"]

        # ── WEEKLY TREND CONTEXT (EMA20 > EMA40) ─────────────────────────
        ema20_val = float(_ema(closes, 20).iloc[-1])
        ema40_val = float(_ema(closes, 40).iloc[-1])
        trend_ok  = ema20_val > ema40_val

        # ── RELATIVE STRENGTH vs NIFTY ────────────────────────────────────
        rs_12w = None
        try:
            nifty = p.get("nifty_weekly")
            if nifty is not None and len(nifty) >= 13:
                sr = float(closes.iloc[-1]) / float(closes.iloc[-13]) if float(closes.iloc[-13]) > 0 else 1
                nr = float(nifty.iloc[-1])  / float(nifty.iloc[-13]) if float(nifty.iloc[-13])  > 0 else 1
                rs_12w = round(sr / nr, 2) if nr > 0 else None
        except Exception:
            rs_12w = None

        # ── PASS CONDITION ────────────────────────────────────────────────
        # STRICT: all conditions must pass
        passes_strict = (
            had_advance
            and tight_base
            and above_ema40
            and broke_out
            and still_early
            and vol_confirmed
            and good_candle
            and price_ok
            and vol_liq
        )

        # WATCHLIST: relaxed — base forming, near breakout but not yet above resistance
        # OR just broke out but volume not yet confirmed
        near_breakout = (
            had_advance
            and tight_base
            and above_ema40
            and price_ok
            and vol_liq
            and (
                # Case A: not yet broken out but within 3% below resistance (coiling)
                (not broke_out and (resistance - latest_close) / resistance * 100 <= 3)
                or
                # Case B: broke out but volume not yet confirmed or candle not ideal
                (broke_out and still_early and (not vol_confirmed or not good_candle))
            )
        )

        return {
            "Symbol":             symbol.replace(".NS",""),
            "Close":              round(latest_close, 2),
            # Base analysis
            "Base Weeks":         base_weeks,
            "Base Range %":       round(base_range_pct, 1),
            "Prior Advance %":    round(prior_advance, 1),
            "Resistance Level":   round(resistance, 2),
            # Breakout metrics
            "% Above Resistance": round(pct_above_resistance, 2) if broke_out else round((latest_close/resistance-1)*100, 2),
            "Vol Expansion":      round(vol_expansion, 2),
            "Candle Close Pos":   round(candle_close_pos, 2),
            # Context
            "RS vs Nifty 12W":    rs_12w,
            "Trend (EMA20>40)":   trend_ok,
            "Vol Drying":         volume_drying,
            "Above EMA40 Base":   round(pct_above_ema40 * 100, 0),
            "Weekly Volume":      int(latest_weekly_vol),
            # Pass flags
            "Pass Strict":        passes_strict,
            "Pass Watchlist":     near_breakout,
        }
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 5. TABLE STYLING
# ─────────────────────────────────────────────────────────────────────────────
def _c_vol(v):
    try:
        x = float(v)
        if x >= 3.0:   return "background-color:#1a4d35;color:#E8E6DF"
        elif x >= 2.0: return "background-color:#1d5c3e;color:#E8E6DF"
        elif x >= 1.5: return "background-color:#236b48;color:#E8E6DF"
        elif x >= 1.0: return "background-color:#2a3d32;color:#8A93A0"
        else:          return "background-color:#5c2222;color:#8A93A0"
    except: return ""

def _c_above_res(v):
    try:
        x = float(v)
        if x < 0:      return "background-color:#161D27;color:#8A93A0"   # below resistance
        elif x <= 3:   return "background-color:#1d5c3e;color:#E8E6DF"   # 0-3% above (ideal)
        elif x <= 7:   return "background-color:#3a6b2a;color:#E8E6DF"   # 3-7%
        elif x <= 10:  return "background-color:#7a6b1a;color:#E8E6DF"   # 7-10% (still ok)
        else:          return "background-color:#5c2222;color:#8A93A0"   # >10% chasing
    except: return ""

def _c_rs(v):
    try:
        x = float(v)
        if x >= 1.2:   return "background-color:#1a4d35;color:#E8E6DF"
        elif x >= 1.0: return "background-color:#236b48;color:#E8E6DF"
        elif x >= 0.9: return "background-color:#7a6b1a;color:#E8E6DF"
        else:          return "background-color:#5c2222;color:#8A93A0"
    except: return ""

def _c_range(v):
    try:
        x = float(v)
        if x <= 15:    return "background-color:#1d5c3e;color:#E8E6DF"   # very tight base
        elif x <= 25:  return "background-color:#3a6b2a;color:#E8E6DF"   # tight
        elif x <= 35:  return "background-color:#7a6b1a;color:#E8E6DF"   # moderate
        else:          return "background-color:#5c2222;color:#8A93A0"   # wide/sloppy
    except: return ""

def style_df(df: pd.DataFrame):
    if df.empty:
        return df
    s = df.style
    for col, fn in [
        ("Vol Expansion",      _c_vol),
        ("% Above Resistance", _c_above_res),
        ("RS vs Nifty 12W",    _c_rs),
        ("Base Range %",       _c_range),
    ]:
        if col in df.columns:
            s = s.map(fn, subset=[col])
    fmt = {
        "Close":              "{:.2f}",
        "Resistance Level":   "{:.2f}",
        "Base Range %":       "{:.1f}",
        "Prior Advance %":    "{:.1f}",
        "% Above Resistance": "{:.2f}",
        "Vol Expansion":      "{:.2f}",
        "Candle Close Pos":   "{:.2f}",
        "RS vs Nifty 12W":    "{:.2f}",
        "Above EMA40 Base":   "{:.0f}",
        "Weekly Volume":      "{:,.0f}",
    }
    return s.format({k:v for k,v in fmt.items() if k in df.columns}, na_rep="—")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Universe")
    uploaded = st.file_uploader(
        "Upload Nifty 500 CSV (Symbol column). If skipped, fetched live from NSE.",
        type=["csv"],
    )
    limit = st.slider("Stocks to scan", 50, 500, 500, step=50)

    st.header("Base Definition")
    base_weeks = st.slider(
        "Base length (weeks)", 18, 52, 26, 1,
        help="Minimum 18 weeks (~4.5 months). 26 = 6 months. 52 = 1 year."
    )
    base_range_pct = st.slider(
        "Max base range % (tight = lower)",
        10.0, 60.0, 35.0, 1.0,
        help="(High - Low) / Low over the base period. <25% = very tight VCP-style base. <35% = acceptable flat base."
    )
    prior_advance_pct = st.slider(
        "Prior advance required %",
        10.0, 80.0, 20.0, 5.0,
        help="Stock must have risen at least this much before the base. Proves it was in Stage 2, not breaking out of a downtrend."
    )

    st.header("Breakout Criteria")
    max_breakout_pct = st.slider(
        "Max % above resistance (don't chase)",
        2.0, 20.0, 10.0, 0.5,
        help="Stocks more than this % above the base resistance are already extended. 5-10% is ideal entry zone."
    )
    vol_expansion_x = st.slider(
        "Volume expansion (× base median)",
        1.0, 5.0, 1.5, 0.1,
        help="Breakout week volume vs median volume during the base. 1.5x = minimum. 2x+ = high conviction."
    )
    min_close_pos = st.slider(
        "Min candle close position",
        0.3, 0.9, 0.5, 0.05,
        help="0.5 = closed in upper half of the weekly range. 0.7+ = very strong close."
    )

    st.header("Liquidity")
    min_price       = st.number_input("Min price (₹)", value=50, step=10)
    min_weekly_vol  = st.number_input("Min weekly volume", value=200000, step=50000)

    run_btn = st.button("🔍 Run Scan", type="primary", use_container_width=True)

params = dict(
    base_weeks        = base_weeks,
    base_range_pct    = base_range_pct,
    prior_advance_pct = prior_advance_pct,
    max_breakout_pct  = max_breakout_pct,
    vol_expansion_x   = vol_expansion_x,
    min_close_position= min_close_pos,
    min_price         = min_price,
    min_weekly_volume = min_weekly_vol,
    nifty_weekly      = None,   # filled in at run time
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. RUN
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Fetching symbol list…"):
        symbols = symbols_from_upload(uploaded) if uploaded else fetch_nifty500()

    if not symbols:
        st.error(
            "Could not fetch Nifty 500 from NSE (they often block scripted requests). "
            "Download the CSV from https://www.niftyindices.com and upload it via the sidebar."
        )
        st.stop()

    symbols = symbols[:limit]
    progress = st.progress(0, text="Fetching Nifty 50 benchmark…")

    # Nifty 50 benchmark for RS
    nifty_close = None
    try:
        nr = yf.download("^NSEI", period="3y", interval="1wk",
                         progress=False, auto_adjust=True)
        nr = _last_closed_weekly(nr)
        c  = nr["Close"].squeeze()
        if isinstance(c, pd.Series) and len(c) >= 13:
            nifty_close = c
    except Exception:
        pass
    params["nifty_weekly"] = nifty_close

    progress.progress(10, text=f"Downloading 3 years of weekly data for {len(symbols)} stocks…")
    weekly = download_weekly(tuple(symbols))
    progress.progress(80, text="Evaluating Stage 2 breakout conditions…")

    results = []
    for sym in symbols:
        r = evaluate(sym, weekly.get(sym), params)
        if r:
            results.append(r)
    progress.progress(100, text="Done ✓")
    time.sleep(0.3)
    progress.empty()

    if not results:
        st.warning(
            "No stocks returned any data. NSE data may have been blocked. "
            "Try uploading the Nifty 500 CSV via the sidebar."
        )
        st.stop()

    df = pd.DataFrame(results)
    now_ist = pd.Timestamp.now(tz=IST).strftime("%d %b %Y, %H:%M IST")

    strict_df = (
        df[df["Pass Strict"]]
        .drop(columns=["Pass Strict","Pass Watchlist"])
        .sort_values("% Above Resistance", ascending=True)   # closest to resistance = best entry
        .reset_index(drop=True)
    )
    watch_df = (
        df[df["Pass Watchlist"] & ~df["Pass Strict"]]
        .drop(columns=["Pass Strict","Pass Watchlist"])
        .sort_values("% Above Resistance", ascending=True)
        .reset_index(drop=True)
    )

    st.markdown(f"""
    <div class="ticker-board">
      <div class="ticker-cell"><div class="label">Scanned</div>
        <div class="value">{len(results)}</div></div>
      <div class="ticker-cell"><div class="label">Breakouts (Strict)</div>
        <div class="value accent">{len(strict_df)}</div></div>
      <div class="ticker-cell"><div class="label">Watchlist</div>
        <div class="value bull">{len(watch_df)}</div></div>
      <div class="ticker-cell"><div class="label">Last Run (IST)</div>
        <div class="value sm">{now_ist}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── STRICT BREAKOUTS ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">FRESH STAGE 2 BREAKOUTS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        'Prior advance confirmed · 6-month+ base · broke above resistance this/last week · '
        'volume expansion · close in upper range · sorted by closeness to resistance (best entry first)'
        '</div>', unsafe_allow_html=True
    )
    if strict_df.empty:
        st.markdown("`No fresh breakouts right now. Check the Watchlist for stocks coiling near resistance.`")
    else:
        st.dataframe(style_df(strict_df), use_container_width=True, hide_index=True)

    # ── WATCHLIST ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">WATCHLIST — COILING NEAR BREAKOUT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        'Valid base + prior advance + within 3% below resistance — breakout imminent. '
        'Also: recent breakouts with volume/candle not fully confirmed yet.'
        '</div>', unsafe_allow_html=True
    )
    if watch_df.empty:
        st.markdown("`Nothing coiling near breakout right now.`")
    else:
        st.dataframe(style_df(watch_df), use_container_width=True, hide_index=True)

    # ── COLUMN GUIDE ─────────────────────────────────────────────────────
    with st.expander("📖 Column guide — what each number means"):
        st.markdown("""
| Column | What it tells you | Sweet spot |
|---|---|---|
| **Base Range %** | (High−Low)/Low across the base. Lower = tighter, better base | < 25% excellent, < 35% good |
| **Prior Advance %** | How much the stock ran up *before* forming the base | > 20% confirms Stage 2 entry |
| **Resistance Level** | The ceiling the stock just broke above (highest weekly high in base) | — |
| **% Above Resistance** | How far above the resistance the stock has already moved | 0–5% ideal, 5–10% still ok |
| **Vol Expansion** | Breakout week volume ÷ median base volume | > 1.5× minimum, > 2× high conviction |
| **Candle Close Pos** | Where weekly close sits in the weekly range (0=low, 1=high) | > 0.5 good, > 0.7 excellent |
| **RS vs Nifty 12W** | Stock 12W return ÷ Nifty 12W return | > 1.0 outperforming, prioritise > 1.1 |
| **Vol Drying** | True if volume was contracting into the end of the base (accumulation signal) | True preferred |
| **Above EMA40 Base** | % of base weeks where price was above the 40-week EMA | > 60% trend intact |
        """)

    # ── FULL DATA ────────────────────────────────────────────────────────
    with st.expander("Show all evaluated stocks"):
        display_cols = [c for c in df.columns if not c.startswith("Pass")]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download full results CSV", csv_bytes, "stage2_breakouts.csv", "text/csv")

else:
    st.markdown("""
    <div class="section-sub" style="margin-top:32px;">
    This scanner finds stocks that:<br><br>
    &nbsp;&nbsp;1. Had a prior bullish advance of 20%+ (confirmed Stage 2 entry)<br>
    &nbsp;&nbsp;2. Consolidated sideways for 6+ months in a tight base<br>
    &nbsp;&nbsp;3. Are NOW breaking above the top of that base on expanding volume<br>
    &nbsp;&nbsp;4. Have moved less than 10% from the breakout level (early entry, not chasing)<br><br>
    Configure the parameters in the sidebar and click <strong>Run Scan</strong>.
    </div>
    """, unsafe_allow_html=True)

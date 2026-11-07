i"""
Arpit FinTech - TradingView-style PAPER trading terminal
=========================================================

* 100% simulated. No broker connection, no real money, no real orders.
* Market data: Yahoo Finance through the `yfinance` package. It can be delayed
  and is NOT a real-time feed. The app labels stale / delayed data.
* Paper account (virtual Rs 1,00,000) is stored in SQLite (paper_trading.db).

Files needed: app.py + requirements.txt
Run locally:  streamlit run app.py
"""
from __future__ import annotations

import html
import math
import os
import re
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
APP_TITLE = "Arpit FinTech"
INITIAL_CASH = 100_000.0
DB_FILE_NAME = "paper_trading.db"
EPS = 1e-9
CACHE_TTL = 45  # seconds for chart data
QUOTE_TTL = 20  # seconds for latest price
FALLBACK_FX = {"USD": 88.0}  # used only when the live USD/INR rate cannot be fetched
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-=^&_]{0,19}$")

UP = "#26a69a"
DOWN = "#ef5350"
ACCENT = "#2962ff"
BG = "#0e1117"
PANEL = "#131722"
GRID = "#1e222d"

# category -> list of (yahoo symbol, display name, quote currency)
INSTRUMENTS: Dict[str, List[Tuple[str, str, str]]] = {
    "Crypto": [
        ("BTC-USD", "Bitcoin", "USD"),
        ("ETH-USD", "Ethereum", "USD"),
        ("SOL-USD", "Solana", "USD"),
        ("XRP-USD", "XRP", "USD"),
        ("DOGE-USD", "Dogecoin", "USD"),
        ("BNB-USD", "Binance Coin", "USD"),
        ("ADA-USD", "Cardano", "USD"),
    ],
    "Gold": [
        ("GC=F", "Gold Futures (COMEX)", "USD"),
        ("XAUUSD=X", "Gold Spot (XAU/USD)", "USD"),
        ("GLD", "SPDR Gold Shares ETF", "USD"),
        ("IAU", "iShares Gold Trust ETF", "USD"),
        ("GDX", "VanEck Gold Miners ETF", "USD"),
        ("GOLDBEES.NS", "Nippon India Gold BeES ETF", "INR"),
    ],
    "Indian Stocks": [
        ("RELIANCE.NS", "Reliance Industries", "INR"),
        ("HDFCBANK.NS", "HDFC Bank", "INR"),
        ("ICICIBANK.NS", "ICICI Bank", "INR"),
        ("TCS.NS", "TCS", "INR"),
        ("INFY.NS", "Infosys", "INR"),
        ("SBIN.NS", "SBI", "INR"),
        ("TATAMOTORS.NS", "Tata Motors", "INR"),
        ("ITC.NS", "ITC", "INR"),
        ("ADANIENT.NS", "Adani Enterprises", "INR"),
        ("WIPRO.NS", "Wipro", "INR"),
        ("^NSEI", "Nifty 50 (index)", "INR"),
        ("^NSEBANK", "Bank Nifty (index)", "INR"),
    ],
}
INSTRUMENT_INDEX: Dict[str, Dict[str, str]] = {
    sym: {"symbol": sym, "name": name, "category": cat, "currency": cur}
    for cat, items in INSTRUMENTS.items()
    for sym, name, cur in items
}

# timeframe label -> yfinance request. "minutes" is the bar length.
TIMEFRAMES: Dict[str, Dict[str, Any]] = {
    "1m": {"interval": "1m", "period": "5d", "minutes": 1, "resample": None},
    "5m": {"interval": "5m", "period": "1mo", "minutes": 5, "resample": None},
    "15m": {"interval": "15m", "period": "1mo", "minutes": 15, "resample": None},
    "30m": {"interval": "30m", "period": "1mo", "minutes": 30, "resample": None},
    "1H": {"interval": "60m", "period": "6mo", "minutes": 60, "resample": None},
    "4H": {"interval": "60m", "period": "1y", "minutes": 240, "resample": "4h"},
    "1D": {"interval": "1d", "period": "2y", "minutes": 1440, "resample": None},
    "1W": {"interval": "1wk", "period": "5y", "minutes": 10080, "resample": None},
    "1M": {"interval": "1mo", "period": "10y", "minutes": 43200, "resample": None},
}

PAGES = ["📈 Terminal", "💼 Portfolio", "🧾 History", "⚙️ Account"]

PLOT_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
}

REASON_LABELS = {
    "MANUAL_PARTIAL": "Manual partial close",
    "MANUAL_FULL": "Manual full close",
    "TAKE_PROFIT": "Take profit hit",
    "STOP_LOSS": "Stop loss hit",
}

# =============================================================================
# 2. SMALL HELPERS (formatting, time)
# =============================================================================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds")


def parse_ts(value: str) -> pd.Timestamp:
    """Parse an ISO string into a tz-aware UTC pandas Timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def to_ist_series(values: pd.Series) -> pd.Series:
    """Convert ISO strings to readable India time strings."""
    try:
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
        return parsed.dt.tz_convert("Asia/Kolkata").dt.strftime("%d-%b-%Y %H:%M:%S")
    except Exception:
        return values.astype(str)


def safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt_inr(value: Optional[float], decimals: int = 2) -> str:
    """Indian digit grouping, e.g. 1,00,000.00"""
    number = safe_float(value)
    if number is None:
        return "—"
    sign = "-" if number < 0 else ""
    text = f"{abs(number):.{decimals}f}"
    whole, _, frac = text.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts: List[str] = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return f"{sign}₹{whole}" + (f".{frac}" if decimals > 0 else "")


def fmt_pnl(value: Optional[float]) -> str:
    number = safe_float(value)
    if number is None:
        return "—"
    return ("+" if number >= 0 else "-") + fmt_inr(abs(number))


def fmt_price(value: Optional[float], currency: str = "INR") -> str:
    number = safe_float(value)
    if number is None:
        return "—"
    symbols = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
    prefix = symbols.get(currency, f"{currency} ")
    magnitude = abs(number)
    decimals = 2 if magnitude >= 1 else (4 if magnitude >= 0.01 else 6)
    return f"{prefix}{number:,.{decimals}f}"


def price_format(value: Optional[float]) -> str:
    number = safe_float(value) or 0.0
    magnitude = abs(number)
    if magnitude >= 100:
        return "%.2f"
    if magnitude >= 1:
        return "%.4f"
    return "%.6f"


def fmt_qty(value: Optional[float]) -> str:
    number = safe_float(value)
    if number is None:
        return "—"
    text = f"{number:,.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def fmt_volume(value: Optional[float]) -> str:
    number = safe_float(value) or 0.0
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= limit:
            return f"{number / limit:.2f}{suffix}"
    return f"{number:,.0f}"


def short_name(symbol: str) -> str:
    return (
        symbol.replace(".NS", "").replace(".BO", "").replace("-USD", "").replace("=X", "").replace("=F", "")
    )


def market_kind(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith(".NS") or s.endswith(".BO") or s in ("^NSEI", "^NSEBANK", "^BSESN"):
        return "india"
    if re.match(r"^[A-Z0-9]+-(USD|USDT|INR|EUR|GBP|BTC|ETH)$", s):
        return "crypto"
    return "other"


def static_currency(symbol: str) -> Optional[str]:
    s = symbol.upper()
    if market_kind(s) == "india" or s.endswith("INR=X") or s.endswith("-INR"):
        return "INR"
    return None


def default_quantity(price_inr_per_unit: float, cash: float) -> float:
    """Suggest a quantity worth about 10% of available cash."""
    if price_inr_per_unit <= 0 or cash <= 0:
        return 1.0
    units = (cash * 0.10) / price_inr_per_unit
    if units >= 1:
        return float(max(1, math.floor(units)))
    return float(max(round(units, 4), 0.0001))


# =============================================================================
# 3. TECHNICAL INDICATORS (pure pandas / numpy - safe on short data)
# =============================================================================


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    denominator = weights.sum()
    return series.rolling(window=period, min_periods=period).apply(
        lambda values: float(np.dot(values, weights) / denominator), raw=True
    )


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (as used by TradingView RSI/ATR): seeded with the simple average of the first `period` values."""
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    valid = np.flatnonzero(~np.isnan(values))
    if len(valid) == 0:
        return pd.Series(out, index=series.index)
    start = int(valid[0]) + period - 1
    if start >= len(values):
        return pd.Series(out, index=series.index)
    out[start] = float(np.mean(values[int(valid[0]): start + 1]))
    for i in range(start + 1, len(values)):
        value = values[i]
        out[i] = out[i - 1] if np.isnan(value) else (out[i - 1] * (period - 1) + value) / period
    return pd.Series(out, index=series.index)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(~((avg_loss == 0) & avg_gain.notna()), 100.0)
    return out


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast, slow = (fast, slow) if fast < slow else (slow, fast + 1 if fast == slow else fast)
    line = ema(series, fast) - ema(series, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, signal_line, line - signal_line


def bollinger(series: pd.Series, period: int = 20, mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std(ddof=0)
    return mid, mid + mult * std, mid - mult * std


def vwap(df: pd.DataFrame, anchored: bool = False) -> pd.Series:
    """Session VWAP (resets every day) for intraday bars; anchored VWAP otherwise."""
    volume = df["Volume"].astype(float)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    if anchored:
        keys: Any = np.zeros(len(df), dtype=int)
    else:
        keys = df.index.normalize()
    cum_pv = (typical * volume).groupby(keys).cumsum()
    cum_v = volume.groupby(keys).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3, smooth: int = 3
) -> Tuple[pd.Series, pd.Series]:
    lowest = low.rolling(window=k_period, min_periods=k_period).min()
    highest = high.rolling(window=k_period, min_periods=k_period).max()
    span = (highest - lowest).replace(0, np.nan)
    raw_k = 100.0 * (close - lowest) / span
    k = sma(raw_k, smooth) if smooth > 1 else raw_k
    d = sma(k, d_period)
    return k, d


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    parts = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return parts.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return rma(true_range(high, low, close), period)


def supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    """Returns (uptrend_line, downtrend_line); each is NaN where the other trend is active."""
    n = len(close)
    up_line = np.full(n, np.nan)
    down_line = np.full(n, np.nan)
    atr_values = atr(high, low, close, period).to_numpy(dtype=float)
    valid = np.where(~np.isnan(atr_values))[0]
    if n == 0 or len(valid) == 0:
        return pd.Series(up_line, index=close.index), pd.Series(down_line, index=close.index)
    start = int(valid[0])
    hl2 = ((high + low) / 2.0).to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    basic_upper = hl2 + multiplier * atr_values
    basic_lower = hl2 - multiplier * atr_values
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    uptrend = True
    for i in range(start, n):
        if i == start:
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
        else:
            if basic_upper[i] < final_upper[i - 1] or closes[i - 1] > final_upper[i - 1]:
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[i - 1]
            if basic_lower[i] > final_lower[i - 1] or closes[i - 1] < final_lower[i - 1]:
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[i - 1]
            if uptrend and closes[i] < final_lower[i]:
                uptrend = False
            elif (not uptrend) and closes[i] > final_upper[i]:
                uptrend = True
        if uptrend:
            up_line[i] = final_lower[i]
        else:
            down_line[i] = final_upper[i]
    return pd.Series(up_line, index=close.index), pd.Series(down_line, index=close.index)

def compute_indicators(df: pd.DataFrame, cfg: Dict[str, Dict[str, Any]], timeframe: str) -> Tuple[Dict[str, Any], List[str]]:
    """Calculate every enabled indicator. Never raises on short data - returns notes instead."""
    out: Dict[str, Any] = {}
    notes: List[str] = []
    n = len(df)
    close, high, low = df["Close"], df["High"], df["Low"]

    def enough(label: str, needed: int) -> bool:
        if n < needed:
            notes.append(f"{label} needs at least {needed} bars but only {n} are available - not drawn.")
            return False
        return True

    try:
        if cfg["sma"]["on"]:
            p = int(cfg["sma"]["period"])
            if enough(f"SMA({p})", p):
                out["sma"] = (f"SMA({p})", sma(close, p))
        if cfg["ema"]["on"]:
            p = int(cfg["ema"]["period"])
            if enough(f"EMA({p})", p):
                out["ema"] = (f"EMA({p})", ema(close, p))
        if cfg["wma"]["on"]:
            p = int(cfg["wma"]["period"])
            if enough(f"WMA({p})", p):
                out["wma"] = (f"WMA({p})", wma(close, p))
        if cfg["bb"]["on"]:
            p, m = int(cfg["bb"]["period"]), float(cfg["bb"]["mult"])
            if enough(f"Bollinger({p},{m:g})", p):
                out["bb"] = bollinger(close, p, m)
        if cfg["vwap"]["on"]:
            if (df["Volume"].fillna(0) > 0).any():
                anchored = TIMEFRAMES[timeframe]["minutes"] >= 1440
                out["vwap"] = ("VWAP (anchored)" if anchored else "VWAP (daily reset)", vwap(df, anchored))
            else:
                notes.append("VWAP needs volume data, which this instrument does not provide - not drawn.")
        if cfg["supertrend"]["on"]:
            p, m = int(cfg["supertrend"]["period"]), float(cfg["supertrend"]["mult"])
            if enough(f"Supertrend({p},{m:g})", p + 1):
                out["supertrend"] = supertrend(high, low, close, p, m)
        if cfg["rsi"]["on"]:
            p = int(cfg["rsi"]["period"])
            if enough(f"RSI({p})", p + 1):
                out["rsi"] = (f"RSI({p})", rsi(close, p))
        if cfg["macd"]["on"]:
            f, s, g = int(cfg["macd"]["fast"]), int(cfg["macd"]["slow"]), int(cfg["macd"]["signal"])
            if enough(f"MACD({f},{s},{g})", max(f, s) + g):
                out["macd"] = (f"MACD({f},{s},{g})",) + macd(close, f, s, g)
        if cfg["stoch"]["on"]:
            k, d, sm = int(cfg["stoch"]["k"]), int(cfg["stoch"]["d"]), int(cfg["stoch"]["smooth"])
            if enough(f"Stochastic({k},{d},{sm})", k + sm + d - 2):
                out["stoch"] = (f"Stochastic({k},{d},{sm})",) + stochastic(high, low, close, k, d, sm)
        if cfg["atr"]["on"]:
            p = int(cfg["atr"]["period"])
            if enough(f"ATR({p})", p + 1):
                out["atr"] = (f"ATR({p})", atr(high, low, close, p))
    except Exception as exc:  # defensive: an indicator problem must never crash the page
        notes.append(f"Indicator calculation problem: {exc}")
    return out, notes


# =============================================================================
# 4. MARKET DATA (yfinance, cached, always fails gracefully)
# =============================================================================


def empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV)


def normalize_ohlcv(raw: Any) -> pd.DataFrame:
    """Clean a yfinance frame: flat columns, numeric, no NaN rows, sorted, unique index."""
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return empty_ohlcv()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().title() for c in df.columns]
    needed = ["Open", "High", "Low", "Close"]
    if any(c not in df.columns for c in needed):
        return empty_ohlcv()
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df[OHLCV].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    if df.empty:
        return empty_ohlcv()
    df["Volume"] = df["Volume"].fillna(0.0)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def to_local_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the timezone but keep the exchange's wall-clock time (needed for chart gap handling)."""
    out = df.copy()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out


def last_bar_utc(df: pd.DataFrame) -> Optional[datetime]:
    if df.empty:
        return None
    stamp = pd.Timestamp(df.index[-1])
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def resample_ohlcv(df: pd.DataFrame, rule: str, offset: str = "0min") -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df.resample(rule, offset=offset).agg(agg)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_history(symbol: str, timeframe: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = TIMEFRAMES[timeframe]
    meta: Dict[str, Any] = {"ok": False, "error": None, "last_bar_utc": None, "derived": bool(cfg["resample"])}
    try:
        raw = yf.Ticker(symbol).history(period=cfg["period"], interval=cfg["interval"], auto_adjust=False, prepost=False)
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return empty_ohlcv(), meta
    df = normalize_ohlcv(raw)
    if df.empty:
        meta["error"] = "Yahoo Finance returned no data for this symbol / timeframe."
        return empty_ohlcv(), meta
    meta["last_bar_utc"] = last_bar_utc(df)
    df = to_local_naive(df)
    if cfg["resample"]:
        offset = "9h15min" if market_kind(symbol) == "india" else "0min"
        try:
            df = resample_ohlcv(df, cfg["resample"], offset)
        except Exception as exc:
            meta["error"] = f"Could not build 4-hour bars: {exc}"
            return empty_ohlcv(), meta
        if df.empty:
            meta["error"] = "Not enough data to build 4-hour bars."
            return empty_ohlcv(), meta
    meta["ok"] = True
    return df, meta


def _fetch_quote_raw(symbol: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"symbol": symbol, "price": None, "prev_close": None, "asof": None, "source": None, "error": None}
    try:
        ticker = yf.Ticker(symbol)
        intraday = normalize_ohlcv(ticker.history(period="1d", interval="1m", auto_adjust=False, prepost=False))
        if not intraday.empty:
            out["price"] = safe_float(intraday["Close"].iloc[-1])
            out["asof"] = last_bar_utc(intraday)
            out["source"] = "1m"
        daily = normalize_ohlcv(ticker.history(period="5d", interval="1d", auto_adjust=False, prepost=False))
        if not daily.empty:
            if out["price"] is None:
                out["price"] = safe_float(daily["Close"].iloc[-1])
                out["asof"] = last_bar_utc(daily)
                out["source"] = "1d"
            if len(daily) >= 2:
                out["prev_close"] = safe_float(daily["Close"].iloc[-2])
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    if out["price"] is not None and out["price"] <= 0:
        out["price"] = None
    return out


@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def load_quote(symbol: str) -> Dict[str, Any]:
    return _fetch_quote_raw(symbol)


def get_prices(symbols: List[str]) -> Dict[str, Optional[float]]:
    return {s: load_quote(s).get("price") for s in symbols}


def _fetch_snapshot_raw(symbol: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"price": None, "prev": None, "change_pct": None, "error": None}
    try:
        daily = normalize_ohlcv(yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=False, prepost=False))
        if not daily.empty:
            out["price"] = safe_float(daily["Close"].iloc[-1])
            if len(daily) >= 2:
                prev = safe_float(daily["Close"].iloc[-2])
                out["prev"] = prev
                if prev and out["price"] is not None:
                    out["change_pct"] = (out["price"] / prev - 1.0) * 100.0
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


@st.cache_data(ttl=60, show_spinner=False)
def load_snapshots(symbols: Tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if not symbols:
        return result
    with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as pool:
        for sym, snap in zip(symbols, pool.map(_fetch_snapshot_raw, symbols)):
            result[sym] = snap
    return result


@st.cache_data(ttl=30, show_spinner=False)
def load_trigger_bars(symbol: str, bucket: int) -> pd.DataFrame:
    """Recent price bars (UTC index) used to check take-profit / stop-loss / limit fills."""
    interval, period = {1: ("5m", "5d"), 2: ("5m", "1mo"), 3: ("1d", "2y")}.get(bucket, ("5m", "5d"))
    try:
        raw = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False, prepost=False)
    except Exception:
        return empty_ohlcv()
    df = normalize_ohlcv(raw)
    if df.empty:
        return df
    return to_utc_index(df)


def lookback_bucket(since: pd.Timestamp, now: Optional[datetime] = None) -> int:
    now = now or utc_now()
    age_days = (pd.Timestamp(now) - since).total_seconds() / 86400.0
    if age_days <= 3:
        return 1
    if age_days <= 27:
        return 2
    return 3


@st.cache_data(ttl=86400, show_spinner=False)
def detect_currency(symbol: str) -> str:
    known = INSTRUMENT_INDEX.get(symbol)
    if known:
        return known["currency"]
    static = static_currency(symbol)
    if static:
        return static
    try:
        value = yf.Ticker(symbol).fast_info["currency"]
        if isinstance(value, str) and len(value) == 3:
            return value.upper()
    except Exception:
        pass
    return "USD"


def get_fx(currency: str) -> Tuple[Optional[float], bool]:
    """Return (INR per 1 unit of currency, is_live). INR itself is always 1.0."""
    cur = (currency or "INR").upper()
    if cur == "INR":
        return 1.0, True
    price = safe_float(load_quote(f"{cur}INR=X").get("price"))
    if price and price > 0:
        return price, True
    if cur in FALLBACK_FX:
        return FALLBACK_FX[cur], False
    return None, False


def freshness_label(asof: Optional[datetime], source: Optional[str], interval_minutes: int = 1) -> str:
    if asof is None:
        return "⚪ Data time unknown"
    if source == "1d":
        return f"🟠 End-of-day data only (as of {asof.strftime('%d-%b-%Y')})"
    age_min = (utc_now() - asof).total_seconds() / 60.0
    if age_min <= max(interval_minutes * 2.5, 20):
        return f"🟢 Recent data ({max(age_min, 0):.0f} min old - may be delayed by the exchange)"
    if age_min < 180:
        return f"🟠 Delayed data - last update {age_min:.0f} min ago"
    if age_min < 3 * 1440:
        return f"🟠 Market probably closed - last update {age_min / 60:.1f} h ago"
    return f"🔴 Stale data - last update {age_min / 1440:.1f} days ago"


# =============================================================================
# 5. DATABASE (SQLite)
# =============================================================================
_DB_PATH_CACHE: Dict[str, str] = {}

SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS account (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        initial_cash REAL NOT NULL,
        cash REAL NOT NULL,created_at TEXT NOT NULL,
        resets INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        currency TEXT NOT NULL,
        direction TEXT NOT NULL CHECK (direction IN ('LONG','SHORT')),
        entry_price REAL NOT NULL,
        qty REAL NOT NULL,
        initial_qty REAL NOT NULL,
        fx REAL NOT NULL,
        target REAL,
        stop REAL,
        opened_at TEXT NOT NULL,
        check_from TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'OPEN',
        closed_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        currency TEXT NOT NULL,
        direction TEXT NOT NULL,
        order_type TEXT NOT NULL,
        qty REAL NOT NULL,
        price REAL NOT NULL,
        fill_price REAL,
        fx REAL NOT NULL,
        target REAL,
        stop REAL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL,
        filled_at TEXT,
        position_id INTEGER,
        note TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        position_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        name TEXT NOT NULL,
        currency TEXT NOT NULL,
        direction TEXT NOT NULL,
        side TEXT NOT NULL,
        entry_price REAL NOT NULL,
        exit_price REAL NOT NULL,
        qty REAL NOT NULL,
        fx REAL NOT NULL,
        pnl REAL NOT NULL,
        opened_at TEXT NOT NULL,
        closed_at TEXT NOT NULL,
        reason TEXT NOT NULL,
        status TEXT NOT NULL
    )""",
]


def get_db_path() -> str:
    override = os.environ.get("PAPER_TRADING_DB")
    if override:
        return override
    if "path" in _DB_PATH_CACHE:
        return _DB_PATH_CACHE["path"]
    try:
        base = Path(__file__).resolve().parent
    except NameError:
        base = Path.cwd()
    candidates = [base / DB_FILE_NAME, Path(tempfile.gettempdir()) / DB_FILE_NAME]
    for candidate in candidates:
        try:
            probe = sqlite3.connect(str(candidate), timeout=10)
            probe.execute("CREATE TABLE IF NOT EXISTS _probe (x INTEGER)")
            probe.execute("DROP TABLE _probe")
            probe.commit()
            probe.close()
            _DB_PATH_CACHE["path"] = str(candidate)
            return str(candidate)
        except Exception:
            continue
    _DB_PATH_CACHE["path"] = str(candidates[-1])
    return _DB_PATH_CACHE["path"]


@contextmanager
def db_session(write: bool = True):
    """Open a SQLite connection. Write sessions are one atomic transaction."""
    conn = sqlite3.connect(get_db_path(), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        if write:
            conn.execute("BEGIN IMMEDIATE")
        yield conn
        if write:
            conn.execute("COMMIT")
    except Exception:
        if write:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        raise
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    for statement in SCHEMA_STATEMENTS:
        conn.execute(statement)
    conn.execute(
        "INSERT OR IGNORE INTO account (id, initial_cash, cash, created_at) VALUES (1, ?, ?, ?)",
        (INITIAL_CASH, INITIAL_CASH, utc_now_iso()),
    )


def reset_account(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM trades")
    conn.execute("DELETE FROM orders")
    conn.execute("DELETE FROM positions")
    conn.execute(
        "UPDATE account SET initial_cash = ?, cash = ?, created_at = ?, resets = resets + 1 WHERE id = 1",
        (INITIAL_CASH, INITIAL_CASH, utc_now_iso()),
    )


def get_cash(conn: sqlite3.Connection) -> float:
    return float(conn.execute("SELECT cash FROM account WHERE id = 1").fetchone()[0])


# =============================================================================
# 6. TRADING ENGINE (pure functions on a SQLite connection - easy to test)
# =============================================================================


def calc_pnl(direction: str, entry: float, price: float, qty: float, fx: float = 1.0) -> float:
    """Profit/loss in INR. LONG gains when price rises, SHORT gains when price falls."""
    sign = 1.0 if direction == "LONG" else -1.0
    return sign * (price - entry) * qty * fx


def validate_levels(direction: str, entry: float, target: Optional[float], stop: Optional[float]) -> List[str]:
    errors: List[str] = []
    if target is not None:
        if direction == "LONG" and target <= entry:
            errors.append("For a LONG position the target price must be ABOVE the entry price.")
        if direction == "SHORT" and target >= entry:
            errors.append("For a SHORT position the target price must be BELOW the entry price.")
    if stop is not None:
        if direction == "LONG" and stop >= entry:
            errors.append("For a LONG position the stop-loss must be BELOW the entry price.")
        if direction == "SHORT" and stop <= entry:
            errors.append("For a SHORT position the stop-loss must be ABOVE the entry price.")
    return errors


def clean_level(value: Any) -> Optional[float]:
    number = safe_float(value)
    if number is None or number <= 0:
        return None
    return number


def validate_order_inputs(
    direction: str, qty: Any, price: Any, fx: Any, target: Optional[float], stop: Optional[float]
) -> List[str]:
    errors: List[str] = []
    if direction not in ("LONG", "SHORT"):
        errors.append("Direction must be LONG or SHORT.")
    q, p, f = safe_float(qty), safe_float(price), safe_float(fx)
    if q is None or q <= EPS:
        errors.append("Quantity must be greater than zero.")
    if p is None or p <= 0:
        errors.append("Price is not available or not valid.")
    if f is None or f <= 0:
        errors.append("Exchange rate is not available.")
    if errors:
        return errors
    return validate_levels(direction, p, target, stop)  # type: ignore[arg-type]


def _insert_position(
    conn: sqlite3.Connection,
    instr: Dict[str, str],
    direction: str,
    qty: float,
    price: float,
    fx: float,
    target: Optional[float],
    stop: Optional[float],
    opened_at: str,
    check_from: str,
) -> int:
    cost = price * qty * fx
    conn.execute("UPDATE account SET cash = cash - ? WHERE id = 1", (cost,))
    cursor = conn.execute(
        """INSERT INTO positions (symbol, name, category, currency, direction, entry_price, qty, initial_qty, fx,
                                  target, stop, opened_at, check_from, status)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, 'OPEN')""",
        (
            instr["symbol"], instr["name"], instr["category"], instr["currency"], direction,
            price, qty, qty, fx, target, stop, opened_at, check_from,
        ),
    )
    return int(cursor.lastrowid)


def open_position(
    conn: sqlite3.Connection,
    instr: Dict[str, str],
    direction: str,
    qty: float,
    price: float,
    fx: float,
    target: Optional[float] = None,
    stop: Optional[float] = None,
    now_iso: Optional[str] = None,
) -> Tuple[bool, str, Optional[int]]:
    """Simulated MARKET order. Cash is reduced by the full order value (no leverage)."""
    target, stop = clean_level(target), clean_level(stop)
    errors = validate_order_inputs(direction, qty, price, fx, target, stop)
    if errors:
        return False, " ".join(errors), None
    cost = float(price) * float(qty) * float(fx)
    cash = get_cash(conn)
    if cost > cash + EPS:
        return False, f"Insufficient balance. Order value {fmt_inr(cost)} is more than available {fmt_inr(cash)}.", None
    now_iso = now_iso or utc_now_iso()
    pid = _insert_position(conn, instr, direction, float(qty), float(price), float(fx), target, stop, now_iso, now_iso)
    conn.execute(
        """INSERT INTO orders (symbol, name, category, currency, direction, order_type, qty, price, fill_price, fx,
                               target, stop, created_at, status, filled_at, position_id, note)
           VALUES (?,?,?,?,?, 'MARKET', ?,?,?,?,?,?,?, 'FILLED', ?, ?, '')""",
        (instr["symbol"], instr["name"], instr["category"], instr["currency"], direction, qty, price, price, fx,
         target, stop, now_iso, now_iso, pid),
    )
    side = "BUY (LONG)" if direction == "LONG" else "SELL (SHORT)"
    return True, f"{side} {fmt_qty(qty)} {instr['symbol']} filled at {fmt_price(price, instr['currency'])}.", pid


def place_limit_order(
    conn: sqlite3.Connection,
    instr: Dict[str, str],
    direction: str,
    qty: float,
    limit_price: float,
    fx: float,
    target: Optional[float] = None,
    stop: Optional[float] = None,
    now_iso: Optional[str] = None,
) -> Tuple[bool, str, Optional[int]]:
    target, stop = clean_level(target), clean_level(stop)
    errors = validate_order_inputs(direction, qty, limit_price, fx, target, stop)
    if errors:
        return False, " ".join(errors), None
    cost = float(limit_price) * float(qty) * float(fx)
    cash = get_cash(conn)
    if cost > cash + EPS:
        return False, f"Insufficient balance. Order value {fmt_inr(cost)} is more than available {fmt_inr(cash)}.", None
    now_iso = now_iso or utc_now_iso()
    cursor = conn.execute(
        """INSERT INTO orders (symbol, name, category, currency, direction, order_type, qty, price, fx, target, stop,
                               created_at, status, note)
           VALUES (?,?,?,?,?, 'LIMIT', ?,?,?,?,?,?, 'PENDING', '')""",
        (instr["symbol"], instr["name"], instr["category"], instr["currency"], direction, qty, limit_price, fx,
         target, stop, now_iso),
    )
    side = "BUY (LONG)" if direction == "LONG" else "SELL (SHORT)"
    return True, f"Limit {side} order for {fmt_qty(qty)} {instr['symbol']} at {fmt_price(limit_price, instr['currency'])} is pending.", int(cursor.lastrowid)


def cancel_order(conn: sqlite3.Connection, order_id: int) -> Tuple[bool, str]:
    row = conn.execute("SELECT status FROM orders WHERE id = ?", (order_id,)).fetchone()
    if row is None or row["status"] != "PENDING":
        return False, "Only pending orders can be cancelled."
    conn.execute("UPDATE orders SET status = 'CANCELLED', note = 'Cancelled by user' WHERE id = ?", (order_id,))
    return True, "Order cancelled."


def fill_limit_order(
    conn: sqlite3.Connection, order: Dict[str, Any], fill_price: float, filled_at: str, check_from: str
) -> Tuple[bool, str, Optional[int]]:
    cost = fill_price * order["qty"] * order["fx"]
    cash = get_cash(conn)
    if cost > cash + EPS:
        conn.execute(
            "UPDATE orders SET status = 'REJECTED', note = ? WHERE id = ?",
            ("Insufficient balance when the limit price was reached", order["id"]),
        )
        return False, f"Limit order #{order['id']} rejected: insufficient balance at fill.", None
    instr = {k: order[k] for k in ("symbol", "name", "category", "currency")}
    pid = _insert_position(
        conn, instr, order["direction"], order["qty"], fill_price, order["fx"], order["target"], order["stop"],
        filled_at, check_from,
    )
    conn.execute(
        "UPDATE orders SET status = 'FILLED', fill_price = ?, filled_at = ?, position_id = ?, note = 'Limit price reached' WHERE id = ?",
        (fill_price, filled_at, pid, order["id"]),
    )
    return True, f"Limit order #{order['id']} filled: {order['symbol']} at {fmt_price(fill_price, order['currency'])}.", pid


def close_position(
    conn: sqlite3.Connection,
    position_id: int,
    qty: float,
    price: float,
    reason: str = "MANUAL_PARTIAL",
    when_iso: Optional[str] = None,
) -> Tuple[bool, str, float]:
    """Close part (or all) of an open position at `price`. Returns (ok, message, realized_pnl)."""
    row = conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
    if row is None or row["status"] != "OPEN":
        return False, "This position is not open.", 0.0
    pos = dict(row)
    q, p = safe_float(qty), safe_float(price)
    if q is None or q <= EPS:
        return False, "Quantity to close must be greater than zero.", 0.0
    if p is None or p <= 0:
        return False, "Current price is not available, so the position cannot be closed right now.", 0.0
    remaining = float(pos["qty"])
    if q > remaining + 1e-9:
        return False, f"Cannot close {fmt_qty(q)}; only {fmt_qty(remaining)} is open.", 0.0
    q = min(q, remaining)
    is_full = q >= remaining - 1e-9
    pnl = calc_pnl(pos["direction"], pos["entry_price"], p, q, pos["fx"])
    released = pos["entry_price"] * q * pos["fx"]
    when_iso = when_iso or utc_now_iso()
    conn.execute("UPDATE account SET cash = cash + ? WHERE id = 1", (released + pnl,))
    if is_full:
        conn.execute("UPDATE positions SET qty = 0, status = 'CLOSED', closed_at = ? WHERE id = ?", (when_iso, position_id))
        if reason == "MANUAL_PARTIAL":
            reason = "MANUAL_FULL"
    else:
        conn.execute("UPDATE positions SET qty = ? WHERE id = ?", (round(remaining - q, 10), position_id))
    side = "BUY" if pos["direction"] == "LONG" else "SELL"
    conn.execute(
        """INSERT INTO trades (position_id, symbol, name, currency, direction, side, entry_price, exit_price, qty, fx, pnl,
                               opened_at, closed_at, reason, status)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (position_id, pos["symbol"], pos["name"], pos["currency"], pos["direction"], side, pos["entry_price"], p, q,
         pos["fx"], pnl, pos["opened_at"], when_iso, reason, "CLOSED" if is_full else "PARTIAL"),
    )
    kind = "Closed" if is_full else "Partially closed"
    return True, f"{kind} {fmt_qty(q)} {pos['symbol']} at {fmt_price(p, pos['currency'])}. Realized P&L {fmt_pnl(pnl)}.", pnl


def update_levels(conn: sqlite3.Connection, position_id: int, target: Optional[float], stop: Optional[float]) -> Tuple[bool, str]:
    row = conn.execute("SELECT * FROM positions WHERE id = ? AND status = 'OPEN'", (position_id,)).fetchone()
    if row is None:
        return False, "This position is not open."
    target, stop = clean_level(target), clean_level(stop)
    errors = validate_levels(row["direction"], row["entry_price"], target, stop)
    if errors:
        return False, " ".join(errors)
    conn.execute("UPDATE positions SET target = ?, stop = ? WHERE id = ?", (target, stop, position_id))
    return True, "Target and stop-loss updated."

# ----- automatic triggers (take profit, stop loss, limit fills) --------------


def limit_fill_from_bar(direction: str, limit_price: float, o: float, h: float, l: float) -> Optional[float]:
    if direction == "LONG":
        if l <= limit_price:
            return min(limit_price, o)
    else:
        if h >= limit_price:
            return max(limit_price, o)
    return None


def exit_from_bar(direction: str, stop: Optional[float], target: Optional[float], o: float, h: float, l: float) -> Optional[Tuple[str, float]]:
    """Stop-loss is checked first: if both levels were touched in one bar we assume the worse outcome."""
    if direction == "LONG":
        if stop is not None and l <= stop:
            return "STOP_LOSS", min(stop, o)
        if target is not None and h >= target:
            return "TAKE_PROFIT", target
    else:
        if stop is not None and h >= stop:
            return "STOP_LOSS", max(stop, o)
        if target is not None and l <= target:
            return "TAKE_PROFIT", target
    return None


def exit_from_price(direction: str, stop: Optional[float], target: Optional[float], price: float) -> Optional[Tuple[str, float]]:
    if direction == "LONG":
        if stop is not None and price <= stop:
            return "STOP_LOSS", price
        if target is not None and price >= target:
            return "TAKE_PROFIT", target
    else:
        if stop is not None and price >= stop:
            return "STOP_LOSS", price
        if target is not None and price <= target:
            return "TAKE_PROFIT", target
    return None


def _bars_after(bars: Optional[pd.DataFrame], since: pd.Timestamp) -> List[Tuple[pd.Timestamp, float, float, float]]:
    if bars is None or bars.empty or bars.index.tz is None:
        return []
    sub = bars[bars.index > since]
    return list(zip(sub.index, sub["Open"].astype(float), sub["High"].astype(float), sub["Low"].astype(float)))


def process_triggers(
    conn: sqlite3.Connection,
    bars_fn: Callable[[str, pd.Timestamp], Optional[pd.DataFrame]],
    quote_fn: Callable[[str], Optional[float]],
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Fill pending limit orders and close positions whose take-profit / stop-loss was reached."""
    now_iso = (now or utc_now()).isoformat(timespec="seconds")
    events: List[Dict[str, Any]] = []

    pending = [dict(r) for r in conn.execute("SELECT * FROM orders WHERE status = 'PENDING' ORDER BY id").fetchall()]
    for order in pending:
        since = parse_ts(order["created_at"])
        fill: Optional[Tuple[float, str]] = None
        for ts, o, h, l in _bars_after(bars_fn(order["symbol"], since), since):
            price = limit_fill_from_bar(order["direction"], order["price"], o, h, l)
            if price is not None:
                fill = (price, ts.isoformat(timespec="seconds"))
                break
        if fill is None:
            quote = safe_float(quote_fn(order["symbol"]))
            if quote is not None:
                if (order["direction"] == "LONG" and quote <= order["price"]) or (
                    order["direction"] == "SHORT" and quote >= order["price"]
                ):
                    fill = (quote, now_iso)
        if fill is not None:
            ok, msg, _ = fill_limit_order(conn, order, fill[0], fill[1], fill[1])
            events.append({"kind": "success" if ok else "error", "message": msg})

    positions = [dict(r) for r in conn.execute("SELECT * FROM positions WHERE status = 'OPEN' ORDER BY id").fetchall()]
    for pos in positions:
        if pos["target"] is None and pos["stop"] is None:
            continue
        since = parse_ts(pos["check_from"])
        hit: Optional[Tuple[str, float, str]] = None
        for ts, o, h, l in _bars_after(bars_fn(pos["symbol"], since), since):
            result = exit_from_bar(pos["direction"], pos["stop"], pos["target"], o, h, l)
            if result is not None:
                hit = (result[0], result[1], ts.isoformat(timespec="seconds"))
                break
        if hit is None:
            quote = safe_float(quote_fn(pos["symbol"]))
            if quote is not None:
                result = exit_from_price(pos["direction"], pos["stop"], pos["target"], quote)
                if result is not None:
                    hit = (result[0], result[1], now_iso)
        if hit is not None:
            ok, msg, _ = close_position(conn, pos["id"], pos["qty"], hit[1], hit[0], hit[2])
            label = REASON_LABELS.get(hit[0], hit[0])
            events.append({"kind": "success" if ok else "error", "message": f"{label}: {msg}"})
    return events


# ----- account / portfolio maths ---------------------------------------------


def compute_account(conn: sqlite3.Connection, prices: Dict[str, Optional[float]]) -> Dict[str, Any]:
    acct = dict(conn.execute("SELECT * FROM account WHERE id = 1").fetchone())
    cash = float(acct["cash"])
    initial = float(acct["initial_cash"])
    realized = float(conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades").fetchone()[0])
    wins = int(conn.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0").fetchone()[0])
    losses = int(conn.execute("SELECT COUNT(*) FROM trades WHERE pnl < 0").fetchone()[0])
    n_trades = int(conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0])
    positions = [dict(r) for r in conn.execute("SELECT * FROM positions WHERE status = 'OPEN' ORDER BY id").fetchall()]
    live: List[Dict[str, Any]] = []
    invested = 0.0
    open_pnl = 0.0
    missing: List[str] = []
    for pos in positions:
        price = safe_float(prices.get(pos["symbol"]))
        if price is not None and price <= 0:
            price = None
        if price is None:
            missing.append(pos["symbol"])
        mark = price if price is not None else pos["entry_price"]
        pnl = calc_pnl(pos["direction"], pos["entry_price"], mark, pos["qty"], pos["fx"])
        cost = pos["entry_price"] * pos["qty"] * pos["fx"]
        invested += cost
        open_pnl += pnl
        live.append({**pos, "price": price, "unrealized": pnl, "invested": cost,
                     "return_pct": (pnl / cost * 100.0) if cost else 0.0})
    equity = cash + invested + open_pnl
    total_pnl = realized + open_pnl
    decided = wins + losses
    return {
        "cash": cash, "initial": initial, "realized": realized, "open_pnl": open_pnl, "total_pnl": total_pnl,
        "invested": invested, "equity": equity,
        "return_pct": (total_pnl / initial * 100.0) if initial else 0.0,
        "wins": wins, "losses": losses, "n_trades": n_trades,
        "win_rate": (wins / decided * 100.0) if decided else 0.0,
        "positions": live, "missing_prices": sorted(set(missing)),
    }


# =============================================================================
# 7. CHART BUILDING
# =============================================================================


def make_rangebreaks(symbol: str, timeframe: str) -> List[Dict[str, Any]]:
    kind = market_kind(symbol)
    minutes = TIMEFRAMES[timeframe]["minutes"]
    if kind == "crypto" or minutes >= 10080:
        return []
    breaks: List[Dict[str, Any]] = [dict(bounds=["sat", "mon"])]
    if kind == "india" and minutes <= 60:
        breaks.append(dict(pattern="hour", bounds=[15.5, 9.2]))
    return breaks


def build_chart(
    df: pd.DataFrame,
    symbol: str,
    chart_type: str,
    cfg: Dict[str, Dict[str, Any]],
    bars_shown: int,
    timeframe: str,
    levels: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Any, List[str]]:
    ind, notes = compute_indicators(df, cfg, timeframe)
    view = df.tail(int(bars_shown))
    count = len(view)

    def tail(series: pd.Series) -> pd.Series:
        return series.tail(count)

    show_volume = bool(cfg.get("volume", {}).get("on")) and bool((view["Volume"].fillna(0) > 0).any())
    panels = ["price"]
    if show_volume:
        panels.append("volume")
    for key in ("rsi", "macd", "stoch", "atr"):
        if key in ind:
            panels.append(key)
    weights = {"price": 6.0, "volume": 1.3, "rsi": 1.8, "macd": 2.0, "stoch": 1.8, "atr": 1.6}
    total = sum(weights[p] for p in panels)
    heights = [weights[p] / total for p in panels]
    row_of = {name: i + 1 for i, name in enumerate(panels)}

    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=heights)
    x = view.index

    # ---- main price series
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=x, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"], name="Price",
            increasing_line_color=UP, decreasing_line_color=DOWN, increasing_fillcolor=UP, decreasing_fillcolor=DOWN,
            showlegend=False), row=1, col=1)
    elif chart_type == "Bar (OHLC)":
        fig.add_trace(go.Ohlc(
            x=x, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"], name="Price",
            increasing_line_color=UP, decreasing_line_color=DOWN, showlegend=False), row=1, col=1)
    elif chart_type == "Area":
        baseline = float(view["Close"].min())
        fig.add_trace(go.Scatter(x=x, y=[baseline] * count, mode="lines", line=dict(width=0), hoverinfo="skip",
                                 showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=view["Close"], mode="lines", name="Close", fill="tonexty",
                                 fillcolor="rgba(41,98,255,0.18)", line=dict(color=ACCENT, width=2),
                                 showlegend=False), row=1, col=1)
    else:  # Line
        fig.add_trace(go.Scatter(x=x, y=view["Close"], mode="lines", name="Close",
                                 line=dict(color=ACCENT, width=2), showlegend=False), row=1, col=1)

    # ---- overlay indicators
    for key, color in (("sma", "#ffb300"), ("ema", "#00bcd4"), ("wma", "#e040fb"), ("vwap", "#ff7043")):
        if key in ind:
            label, series = ind[key]
            fig.add_trace(go.Scatter(x=x, y=tail(series), mode="lines", name=label,
                                     line=dict(color=color, width=1.4)), row=1, col=1)
    if "bb" in ind:
        mid, upper, lower = ind["bb"]
        fig.add_trace(go.Scatter(x=x, y=tail(upper), mode="lines", name="BB upper",
                                 line=dict(color="rgba(124,179,255,0.9)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=tail(lower), mode="lines", name="BB lower", fill="tonexty",
                                 fillcolor="rgba(41,98,255,0.08)",
                                 line=dict(color="rgba(124,179,255,0.9)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=tail(mid), mode="lines", name="BB mid",
                                 line=dict(color="rgba(124,179,255,0.6)", width=1, dash="dot")), row=1, col=1)
    if "supertrend" in ind:
        up_line, down_line = ind["supertrend"]
        fig.add_trace(go.Scatter(x=x, y=tail(up_line), mode="lines", name="Supertrend up",
                                 line=dict(color=UP, width=1.6)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=tail(down_line), mode="lines", name="Supertrend down",
                                 line=dict(color=DOWN, width=1.6)), row=1, col=1)

    # ---- open position levels
    for pos in levels or []:
        tag = f"#{pos['id']} {'LONG' if pos['direction'] == 'LONG' else 'SHORT'}"
        fig.add_hline(y=pos["entry_price"], line_dash="dash", line_color="#b2b5be", line_width=1, row=1, col=1,
                      annotation_text=f"{tag} entry", annotation_position="top left",
                      annotation_font_size=10, annotation_font_color="#b2b5be")
        if pos.get("target") is not None:
            fig.add_hline(y=pos["target"], line_dash="dot", line_color=UP, line_width=1, row=1, col=1,
                          annotation_text=f"{tag} TP", annotation_position="top left",
                          annotation_font_size=10, annotation_font_color=UP)
        if pos.get("stop") is not None:
            fig.add_hline(y=pos["stop"], line_dash="dot", line_color=DOWN, line_width=1, row=1, col=1,
                          annotation_text=f"{tag} SL", annotation_position="top left",
                          annotation_font_size=10, annotation_font_color=DOWN)

    # ---- volume
    if show_volume:
        colors = [UP if c >= o else DOWN for o, c in zip(view["Open"], view["Close"])]
        fig.add_trace(go.Bar(x=x, y=view["Volume"], name="Volume", marker_color=colors, opacity=0.7,
                             showlegend=False), row=row_of["volume"], col=1)
        fig.update_yaxes(title_text="Vol", row=row_of["volume"], col=1)

    # ---- oscillators
    if "rsi" in ind:
        label, series = ind["rsi"]
        r = row_of["rsi"]
        fig.add_trace(go.Scatter(x=x, y=tail(series), mode="lines", name=label,
                                 line=dict(color="#b388ff", width=1.4)), row=r, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=DOWN, line_width=0.8, row=r, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=UP, line_width=0.8, row=r, col=1)
        fig.update_yaxes(range=[0, 100], title_text="RSI", row=r, col=1)
    if "macd" in ind:
        label, line, signal_line, hist = ind["macd"]
        r = row_of["macd"]
        hist_view = tail(hist)
        fig.add_trace(go.Bar(x=x, y=hist_view, name="MACD hist", showlegend=False,
                             marker_color=[UP if (v == v and v >= 0) else DOWN for v in hist_view]), row=r, col=1)
        fig.add_trace(go.Scatter(x=x, y=tail(line), mode="lines", name=label,
                                 line=dict(color="#2962ff", width=1.3)), row=r, col=1)
        fig.add_trace(go.Scatter(x=x, y=tail(signal_line), mode="lines", name="Signal",
                                 line=dict(color="#ff9800", width=1.3)), row=r, col=1)
        fig.update_yaxes(title_text="MACD", row=r, col=1)
    if "stoch" in ind:
        label, k, d = ind["stoch"]
        r = row_of["stoch"]
        fig.add_trace(go.Scatter(x=x, y=tail(k), mode="lines", name="%K",
                                 line=dict(color="#2962ff", width=1.3)), row=r, col=1)
        fig.add_trace(go.Scatter(x=x, y=tail(d), mode="lines", name="%D",
                                 line=dict(color="#ff9800", width=1.3)), row=r, col=1)
        fig.add_hline(y=80, line_dash="dot", line_color=DOWN, line_width=0.8, row=r, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color=UP, line_width=0.8, row=r, col=1)
        fig.update_yaxes(range=[0, 100], title_text="Stoch", row=r, col=1)
    if "atr" in ind:
        label, series = ind["atr"]
        r = row_of["atr"]
        fig.add_trace(go.Scatter(x=x, y=tail(series), mode="lines", name=label,
                                 line=dict(color="#26c6da", width=1.4)), row=r, col=1)
        fig.update_yaxes(title_text="ATR", row=r, col=1)

    # ---- layout / theme
    height_px = int(430 + 150 * (len(panels) - 1))
    fig.update_layout(
        template="plotly_dark", height=height_px, margin=dict(l=6, r=6, t=26, b=6),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#b2b5be", size=11),
        hovermode="x unified", dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=10)),
        uirevision=f"{symbol}-{timeframe}-{chart_type}",
    )
    fig.update_xaxes(
        rangeslider_visible=False, showgrid=True, gridcolor=GRID, showspikes=True, spikemode="across",
        spikesnap="cursor", spikedash="dot", spikecolor="#787b86", spikethickness=1,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, side="right", showspikes=True, spikemode="across",
        spikedash="dot", spikecolor="#787b86", spikethickness=1, zeroline=False,
    )
    breaks = make_rangebreaks(symbol, timeframe)
    if breaks:
        fig.update_xaxes(rangebreaks=breaks)
    return fig, notes

# =============================================================================
# 8. STYLING
# =============================================================================
CSS = """
<style>
:root { --bg:#0e1117; --panel:#131722; --panel2:#1e222d; --border:#2a2e39; --text:#d1d4dc;
        --muted:#787b86; --up:#26a69a; --down:#ef5350; --accent:#2962ff; }
.stApp { background: var(--bg); color: var(--text); }
.stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp .stMarkdown, .stApp [data-testid="stCaptionContainer"] { color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
.block-container { padding-top: 1.1rem; padding-bottom: 3rem; max-width: 100%; }
[data-baseweb="input"] input, [data-baseweb="select"] > div, textarea, [data-baseweb="base-input"] {
    background: var(--panel2) !important; color: var(--text) !important; border-color: var(--border) !important; }
[data-baseweb="select"] span, [data-baseweb="select"] div { color: var(--text) !important; }
.stButton > button, .stDownloadButton > button {
    background: var(--panel2); color: var(--text); border: 1px solid var(--border); border-radius: 6px; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--accent); color: #fff; }
.st-key-buy_btn button { background: var(--up) !important; color: #fff !important; border: none !important; font-weight: 700; }
.st-key-sell_btn button { background: var(--down) !important; color: #fff !important; border: none !important; font-weight: 700; }
.st-key-buy_btn button:hover, .st-key-sell_btn button:hover { filter: brightness(1.12); }
[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: 0.78rem; }
[data-testid="stMetricValue"] { font-size: 1.2rem; }
[data-baseweb="tab-list"] { gap: 4px; }
[data-baseweb="tab"] { background: transparent; color: var(--muted); }
[data-baseweb="tab"][aria-selected="true"] { color: #fff; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--border); }
hr { border-color: var(--border); }
.brand { font-size: 1.25rem; font-weight: 800; letter-spacing: 0.2px; color: #fff; }
.brand small { display: block; font-size: 0.72rem; font-weight: 500; color: var(--muted); }
.sym-line { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px; margin-bottom: 2px; }
.sym-name { font-size: 1.35rem; font-weight: 700; color: #fff; }
.sym-tick { font-size: 0.85rem; color: var(--muted); }
.sym-price { font-size: 1.7rem; font-weight: 700; color: #fff; }
.up { color: var(--up) !important; } .down { color: var(--down) !important; }
.ohlc { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.85rem; padding: 6px 2px; color: var(--muted); }
.ohlc b { color: var(--text); font-weight: 600; }
.sim-banner { background: #1b2233; border: 1px solid #2a3350; border-radius: 6px; padding: 6px 10px;
              font-size: 0.78rem; color: #9db2ff; margin-bottom: 8px; }
@media (max-width: 640px) {
  .block-container { padding: 0.6rem 0.55rem 3rem; }
  .sym-price { font-size: 1.35rem; } .sym-name { font-size: 1.1rem; }
  [data-testid="stMetricValue"] { font-size: 1rem; }
}
#MainMenu { visibility: hidden; } footer { visibility: hidden; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# =============================================================================
# 9. UI HELPERS (flash messages, triggers, caches)
# =============================================================================


def flash(kind: str, message: str) -> None:
    queue = st.session_state.setdefault("flash_queue", [])
    queue.append((kind, message))


def show_flash() -> None:
    queue = st.session_state.get("flash_queue") or []
    st.session_state["flash_queue"] = []
    icons = {"success": "✅", "error": "❌", "info": "ℹ️"}
    for kind, message in queue:
        st.toast(message, icon=icons.get(kind, "ℹ️"))


def run_trigger_check() -> List[Dict[str, Any]]:
    """Check take-profit / stop-loss / pending limit orders. Safe to call every refresh."""
    try:
        with db_session(write=False) as conn:
            pend = [dict(r) for r in conn.execute("SELECT symbol, created_at FROM orders WHERE status = 'PENDING'")]
            pos = [dict(r) for r in conn.execute(
                "SELECT symbol, check_from FROM positions WHERE status = 'OPEN' AND (target IS NOT NULL OR stop IS NOT NULL)")]
        if not pend and not pos:
            return []
        # download first (network can be slow) so the database stays locked as briefly as possible
        for item in pend:
            since = parse_ts(item["created_at"])
            load_trigger_bars(item["symbol"], lookback_bucket(since))
            load_quote(item["symbol"])
        for item in pos:
            since = parse_ts(item["check_from"])
            load_trigger_bars(item["symbol"], lookback_bucket(since))
            load_quote(item["symbol"])

        def bars_fn(symbol: str, since: pd.Timestamp) -> pd.DataFrame:
            return load_trigger_bars(symbol, lookback_bucket(since))

        def quote_fn(symbol: str) -> Optional[float]:
            return load_quote(symbol).get("price")

        with db_session() as conn:
            return process_triggers(conn, bars_fn, quote_fn)
    except Exception as exc:
        st.session_state["trigger_error"] = f"{type(exc).__name__}: {exc}"
        return []


def check_triggers_and_rerun() -> None:
    events = run_trigger_check()
    if events:
        for event in events:
            flash(event["kind"], event["message"])
        st.rerun()


def get_live_account() -> Dict[str, Any]:
    with db_session(write=False) as conn:
        symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM positions WHERE status = 'OPEN'")]
    prices = get_prices(sorted(symbols)) if symbols else {}
    with db_session(write=False) as conn:
        return compute_account(conn, prices)


def reset_callback() -> None:
    with db_session() as conn:
        reset_account(conn)
    st.session_state["reset_ack"] = False
    st.session_state["reset_text"] = ""
    flash("success", "Account reset. Balance is back to ₹1,00,000.")


# =============================================================================
# 10. SIDEBAR
# =============================================================================

IND_DEFAULTS = {
    "sma": ("SMA", {"period": (20, 2, 500, 1)}),
    "ema": ("EMA", {"period": (50, 2, 500, 1)}),
    "wma": ("WMA", {"period": (20, 2, 500, 1)}),
    "bb": ("Bollinger Bands", {"period": (20, 2, 500, 1), "mult": (2.0, 0.5, 5.0, 0.5)}),
    "vwap": ("VWAP", {}),
    "supertrend": ("Supertrend", {"period": (10, 2, 200, 1), "mult": (3.0, 0.5, 10.0, 0.5)}),
    "rsi": ("RSI", {"period": (14, 2, 200, 1)}),
    "macd": ("MACD", {"fast": (12, 2, 100, 1), "slow": (26, 3, 200, 1), "signal": (9, 2, 100, 1)}),
    "stoch": ("Stochastic", {"k": (14, 2, 200, 1), "d": (3, 1, 50, 1), "smooth": (3, 1, 50, 1)}),
    "atr": ("ATR", {"period": (14, 2, 200, 1)}),
}


def render_indicator_controls() -> Dict[str, Dict[str, Any]]:
    cfg: Dict[str, Dict[str, Any]] = {}
    with st.expander("📊 Technical indicators", expanded=False):
        st.caption("Overlays draw on the price chart. RSI, MACD, Stochastic and ATR get their own panels.")
        cfg["volume"] = {"on": st.checkbox("Volume bars", value=True, key="ind_volume_on")}
        for key, (label, params) in IND_DEFAULTS.items():
            on = st.checkbox(label, value=False, key=f"ind_{key}_on")
            values: Dict[str, Any] = {"on": on}
            if params:
                columns = st.columns(len(params))
                for column, (pname, (default, lo, hi, step)) in zip(columns, params.items()):
                    with column:
                        values[pname] = st.number_input(
                            pname.capitalize() if len(pname) > 2 else f"%{pname.upper()}",
                            min_value=lo, max_value=hi, value=default, step=step,
                            key=f"ind_{key}_{pname}", disabled=not on,
                        )
            cfg[key] = values
    return cfg


def apply_pending_selection() -> None:
    """Apply a watchlist click BEFORE the sidebar widgets are created (Streamlit requires this order)."""
    pending = st.session_state.pop("pending_select", None)
    if pending:
        category, symbol = pending
        if category in INSTRUMENTS:
            st.session_state["category"] = category
            st.session_state[f"inst_{category}"] = symbol
            st.session_state["custom_symbol"] = ""


def render_sidebar() -> Dict[str, Any]:
    apply_pending_selection()
    with st.sidebar:
        st.markdown(f"<div class='brand'>📈 {html.escape(APP_TITLE)}<small>Paper trading terminal - simulated money only</small></div>",
                    unsafe_allow_html=True)
        page = st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        st.divider()
        category = st.selectbox("Market", list(INSTRUMENTS.keys()), key="category")
        options = INSTRUMENTS[category]
        labels = {sym: f"{name} ({short_name(sym)})" for sym, name, _ in options}
        choice = st.selectbox("Instrument", [o[0] for o in options], format_func=lambda s: labels.get(s, s),
                              key=f"inst_{category}")
        custom_raw = st.text_input("🔎 Search any ticker", key="custom_symbol",
                                   placeholder="AAPL, TSLA, EURUSD=X, MARUTI.NS",
                                   help="Use Yahoo Finance ticker symbols. Clear the box to go back to the list.")
        custom = custom_raw.strip().upper()
        if custom and not TICKER_RE.match(custom):
            st.warning("That ticker contains unsupported characters, so it was ignored.")
            custom = ""
        if custom:
            symbol = custom
            instrument = INSTRUMENT_INDEX.get(symbol) or {
                "symbol": symbol, "name": symbol, "category": "Custom", "currency": detect_currency(symbol)}
            st.caption(f"Using custom ticker **{html.escape(symbol)}**")
        else:
            instrument = INSTRUMENT_INDEX[choice]
        st.divider()
        with st.expander("🎛️ Chart settings", expanded=False):
            chart_type = st.radio("Chart type", ["Candlestick", "Line", "Bar (OHLC)", "Area"], key="chart_type")
            bars = st.slider("Bars to display", 50, 1500, 250, 50, key="bars_shown")
        indicators = render_indicator_controls()
        refresh_label = st.selectbox("🔄 Auto-refresh", ["Off", "15 s", "30 s", "60 s", "120 s"], index=2, key="refresh")
        refresh = None if refresh_label == "Off" else int(refresh_label.split()[0])
        st.divider()
        with db_session(write=False) as conn:
            cash = get_cash(conn)
            n_open = int(conn.execute("SELECT COUNT(*) FROM positions WHERE status = 'OPEN'").fetchone()[0])
        st.metric("Available balance", fmt_inr(cash))
        st.caption(f"{n_open} open position(s)")
    return {
        "page": page, "instrument": instrument, "category": category, "chart_type": chart_type,
        "bars": int(bars), "indicators": indicators, "refresh": refresh,
    }


# =============================================================================
# 11. TERMINAL PAGE
# =============================================================================


def show_chart(fig: Any, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, theme=None, config=PLOT_CONFIG, key=key)


def render_header(instr: Dict[str, str], quote: Dict[str, Any]) -> None:
    price = quote.get("price")
    prev = quote.get("prev_close")
    change_html = ""
    if price is not None and prev:
        pct = (price / prev - 1.0) * 100.0
        css = "up" if pct >= 0 else "down"
        change_html = f"<span class='{css}'>{pct:+.2f}%</span>"
    st.markdown(
        f"<div class='sym-line'><span class='sym-name'>{html.escape(instr['name'])}</span>"
        f"<span class='sym-tick'>{html.escape(instr['symbol'])} · {html.escape(instr['category'])} · quoted in {html.escape(instr['currency'])}</span></div>"
        f"<div class='sym-line'><span class='sym-price'>{fmt_price(price, instr['currency'])}</span>{change_html}</div>",
        unsafe_allow_html=True,
    )
    st.caption(freshness_label(quote.get("asof"), quote.get("source"), 1) +
               " · Source: Yahoo Finance via yfinance (not a real-time feed)")


def render_ohlc_strip(df: pd.DataFrame, currency: str) -> None:
    last = df.iloc[-1]
    prev_close = df["Close"].iloc[-2] if len(df) > 1 else None
    change = ""
    if prev_close:
        pct = (last["Close"] / prev_close - 1.0) * 100.0
        change = f"<span class='{'up' if pct >= 0 else 'down'}'>{pct:+.2f}%</span>"
    st.markdown(
        "<div class='ohlc'>"
        f"<span>Bar <b>{pd.Timestamp(df.index[-1]).strftime('%d-%b %H:%M')}</b></span>"
        f"<span>O <b>{fmt_price(last['Open'], currency)}</b></span>"
        f"<span>H <b>{fmt_price(last['High'], currency)}</b></span>"
        f"<span>L <b>{fmt_price(last['Low'], currency)}</b></span>"
        f"<span>C <b>{fmt_price(last['Close'], currency)}</b></span>"
        f"<span>Vol <b>{fmt_volume(last['Volume'])}</b></span>{change}</div>",
        unsafe_allow_html=True,
    )


def get_open_levels(symbol: str) -> List[Dict[str, Any]]:
    with db_session(write=False) as conn:
        rows = conn.execute(
            "SELECT id, direction, entry_price, target, stop FROM positions WHERE status = 'OPEN' AND symbol = ?",
            (symbol,),
        ).fetchall()
    return [dict(r) for r in rows]


def submit_order(
    instr: Dict[str, str], direction: str, order_type: str, qty: Optional[float], limit_price: Optional[float],
    target: Optional[float], stop: Optional[float], market_price: float, fx: float,
) -> None:
    qty_value = safe_float(qty)
    if qty_value is None or qty_value <= EPS:
        st.error("Quantity must be greater than zero.")
        return
    target, stop = clean_level(target), clean_level(stop)
    marketable = False
    if order_type == "Limit":
        lp = safe_float(limit_price)
        if lp is None or lp <= 0:
            st.error("Enter a valid limit price.")
            return
        marketable = (direction == "LONG" and market_price <= lp) or (direction == "SHORT" and market_price >= lp)
    with db_session() as conn:
        if order_type == "Market" or marketable:
            ok, msg, _ = open_position(conn, instr, direction, qty_value, market_price, fx, target, stop)
        else:
            ok, msg, _ = place_limit_order(conn, instr, direction, qty_value, float(limit_price), fx, target, stop)  # type: ignore[arg-type]
    if ok:
        flash("success", msg)
        st.rerun()
    else:
        st.error(msg)


def render_order_panel(instr: Dict[str, str], quote: Dict[str, Any]) -> None:
    st.markdown("##### 🧾 Order panel")
    price = quote.get("price")
    if price is None:
        st.warning("No price is available for this instrument right now, so trading is disabled. "
                   "Check the ticker or try again in a moment.")
        return
    fx, fx_live = get_fx(instr["currency"])
    if fx is None:
        st.warning(f"The {instr['currency']}→INR exchange rate is unavailable, so trading is disabled for now.")
        return
    with db_session(write=False) as conn:
        cash = get_cash(conn)
    symbol = instr["symbol"]
    c1, c2 = st.columns(2)
    c1.metric("Market price", fmt_price(price, instr["currency"]))
    c2.metric("Available", fmt_inr(cash))
    if instr["currency"] != "INR":
        st.caption(f"Converted at 1 {instr['currency']} = ₹{fx:,.2f} " +
                   ("(live rate, fixed when the position opens)" if fx_live else "(ESTIMATED fallback rate - live rate unavailable)"))
    order_type = st.radio("Order type", ["Market", "Limit"], horizontal=True, key="order_type")
    # defaults are remembered per instrument so auto-refresh never resets what you typed
    default_qty = st.session_state.setdefault(f"qty_default_{symbol}", default_quantity(price * fx, cash))
    default_limit = st.session_state.setdefault(f"lim_default_{symbol}", float(price))
    qty = st.number_input("Quantity", min_value=0.0, value=float(default_qty),
                          step=1.0 if default_qty >= 1 else 0.001, format="%.4f", key=f"qty_{symbol}")
    if order_type == "Limit":
        limit_price = st.number_input("Limit (entry) price", min_value=0.0, value=float(default_limit),
                                      format=price_format(default_limit), key=f"lim_{symbol}")
        entry = limit_price
        st.caption("A BUY limit fills when price falls to your limit or lower; a SELL limit fills when it rises to your limit or higher. "
                   "If the limit is already reachable it fills at the current price.")
    else:
        limit_price = None
        entry = price
        st.caption("Market order: entry price = current market price shown above.")
    target = st.number_input("Target profit price (optional)", min_value=0.0, value=None, placeholder="Leave empty for none",
                             format=price_format(price), key=f"tp_{symbol}")
    stop = st.number_input("Stop-loss price (optional)", min_value=0.0, value=None, placeholder="Leave empty for none",
                           format=price_format(price), key=f"sl_{symbol}")
    qty_value = safe_float(qty) or 0.0
    entry_value = safe_float(entry) or 0.0
    est_value = entry_value * qty_value * fx
    max_qty = (cash / (entry_value * fx)) if entry_value > 0 else 0.0
    m1, m2 = st.columns(2)
    m1.metric("Order value", fmt_inr(est_value))
    m2.metric("Max quantity", fmt_qty(math.floor(max_qty * 10000) / 10000))
    if est_value > cash + EPS:
        st.warning("Order value is higher than your available balance.")
    b1, b2 = st.columns(2)
    buy = b1.button("BUY / LONG", key="buy_btn", use_container_width=True)
    sell = b2.button("SELL / SHORT", key="sell_btn", use_container_width=True)
    st.caption("LONG: target above entry, stop-loss below. SHORT: target below entry, stop-loss above.")
    if buy or sell:
        direction = "LONG" if buy else "SHORT"
        submit_order(instr, direction, order_type, qty, limit_price, target, stop, float(price), float(fx))


def render_watchlist(category: str, active_symbol: str) -> None:
    st.markdown(f"##### 👁️ Watchlist · {category}")
    items = INSTRUMENTS[category]
    snaps = load_snapshots(tuple(sym for sym, _, _ in items))
    for sym, name, cur in items:
        snap = snaps.get(sym) or {}
        price, pct = snap.get("price"), snap.get("change_pct")

if price is None:
    continue
            label = f ⚪ {short_name(sym)} · unavailable

        else:
            dot = "🟢" if (pct or 0) >= 0 else "🔴"
            pct_text = f"{pct:+.2f}%" if pct is not None else "n/a"
            label = f"{dot} {short_name(sym)} · {fmt_price(price, cur)} · {pct_text}"
        if st.button(label, key=f"wl_{sym}", use_container_width=True,
                     type="primary" if sym == active_symbol else "secondary"):
            st.session_state["pending_select"] = (category, sym)
            st.rerun()
    st.caption("Watchlist prices are the latest daily bar (delayed). Tap a row to load its chart.")


def execute_close(pid: int, qty: float, symbol: str, full: bool) -> None:
    price = safe_float(load_quote(symbol).get("price"))
    if price is None:
        st.error("No current price is available, so this position cannot be closed right now.")
        return
    reason = "MANUAL_FULL" if full else "MANUAL_PARTIAL"
    with db_session() as conn:
        ok, msg, _ = close_position(conn, pid, qty, price, reason)
    if ok:
        flash("success", msg)
        st.rerun()
    else:
        st.error(msg)


def render_position_cards(live_positions: List[Dict[str, Any]], prefix: str) -> None:
    if not live_positions:
        st.info("No open positions. Place a BUY or SELL order from the Terminal page.")
        return
    for pos in live_positions:
        pid, symbol, cur = pos["id"], pos["symbol"], pos["currency"]
        with st.container(border=True):
            badge = "🟢 LONG (BUY)" if pos["direction"] == "LONG" else "🔴 SHORT (SELL)"
            partial = f" · {fmt_qty(pos['initial_qty'] - pos['qty'])} already closed" if pos["qty"] < pos["initial_qty"] - 1e-9 else ""
            st.markdown(f"**{html.escape(pos['name'])}** `{symbol}` · {badge} · Qty {fmt_qty(pos['qty'])}{partial}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Entry", fmt_price(pos["entry_price"], cur))
            m2.metric("Current", fmt_price(pos["price"], cur) if pos["price"] is not None else "n/a")
            m3.metric("Unrealized P&L", fmt_pnl(pos["unrealized"]), delta=f"{pos['return_pct']:+.2f}%")
            st.caption(f"Target: {fmt_price(pos['target'], cur) if pos['target'] else 'none'} · "
                       f"Stop-loss: {fmt_price(pos['stop'], cur) if pos['stop'] else 'none'} · "
                       f"Opened {to_ist_series(pd.Series([pos['opened_at']])).iloc[0]} IST")
            remaining = float(pos["qty"])
            half = round(remaining / 2.0, 6) if remaining > 0 else 0.0
            c1, c2, c3 = st.columns([2, 1.4, 1.4])
            close_qty = c1.number_input("Quantity to close", min_value=0.0, value=float(half), format="%.4f",
                                        step=1.0 if remaining >= 2 else 0.001,
                                        key=f"{prefix}_pq_{pid}_{remaining:.8f}")
            if c2.button("Partially Close", key=f"{prefix}_pc_{pid}", use_container_width=True):
                if safe_float(close_qty) is None or close_qty <= EPS:
                    st.error("Enter a quantity greater than zero.")
                elif close_qty > remaining + 1e-9:
                    st.error(f"Cannot close more than the open quantity ({fmt_qty(remaining)}).")
                else:
                    execute_close(pid, float(close_qty), symbol, full=False)
            if c3.button("Close Full Position", key=f"{prefix}_cf_{pid}", use_container_width=True):
                execute_close(pid, remaining, symbol, full=True)
            with st.expander("Edit target / stop-loss"):
                e1, e2 = st.columns(2)
                new_tp = e1.number_input("Target", min_value=0.0, value=float(pos["target"]) if pos["target"] else None,
                                         placeholder="none", format=price_format(pos["entry_price"]),
                                         key=f"{prefix}_tp_{pid}_{pos['target']}")
                new_sl = e2.number_input("Stop-loss", min_value=0.0, value=float(pos["stop"]) if pos["stop"] else None,
                                         placeholder="none", format=price_format(pos["entry_price"]),
                                         key=f"{prefix}_sl_{pid}_{pos['stop']}")
                if st.button("Save levels", key=f"{prefix}_sv_{pid}"):
                    with db_session() as conn:
                        ok, msg = update_levels(conn, pid, new_tp, new_sl)
                    if ok:
                        flash("success", msg)
                        st.rerun()
                    else:
                        st.error(msg)


def render_pending_orders(prefix: str) -> None:
    with db_session(write=False) as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM orders WHERE status = 'PENDING' ORDER BY id DESC")]
    if not rows:
        st.info("No pending limit orders.")
        return
    for od in rows:
        c1, c2 = st.columns([4, 1])
        side = "BUY (LONG)" if od["direction"] == "LONG" else "SELL (SHORT)"
        c1.markdown(f"**#{od['id']}** {side} {fmt_qty(od['qty'])} `{od['symbol']}` @ {fmt_price(od['price'], od['currency'])} "
                    f"· TP {fmt_price(od['target'], od['currency']) if od['target'] else '-'} "
                    f"· SL {fmt_price(od['stop'], od['currency']) if od['stop'] else '-'}")
        if c2.button("Cancel", key=f"{prefix}_co_{od['id']}", use_container_width=True):
            with db_session() as conn:
                ok, msg = cancel_order(conn, od["id"])
            flash("success" if ok else "error", msg)
            st.rerun()
    st.caption("Pending orders do not reserve cash; the balance is checked again when the order fills.")


def trades_dataframe(conn: sqlite3.Connection, limit: Optional[int] = None) -> pd.DataFrame:
    sql = "SELECT * FROM trades ORDER BY id DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = [dict(r) for r in conn.execute(sql).fetchall()]
    if not rows:
        return pd.DataFrame(columns=["Trade ID", "Date/Time (IST)", "Instrument", "Symbol", "Side", "Direction", "Entry",
                                     "Exit", "Quantity", "Realized P&L (₹)", "Return %", "Reason", "Status", "Position ID"])
    df = pd.DataFrame(rows)
    cost = df["entry_price"] * df["qty"] * df["fx"]
    return pd.DataFrame({
        "Trade ID": df["id"], "Date/Time (IST)": to_ist_series(df["closed_at"]), "Instrument": df["name"],
        "Symbol": df["symbol"], "Side": df["side"], "Direction": df["direction"], "Entry": df["entry_price"],
        "Exit": df["exit_price"], "Quantity": df["qty"], "Realized P&L (₹)": df["pnl"].round(2),
        "Return %": (df["pnl"] / cost * 100.0).round(2),
        "Reason": df["reason"].map(lambda r: REASON_LABELS.get(r, r)),
        "Status": df["status"].map({"PARTIAL": "Partial close", "CLOSED": "Closed"}),
        "Position ID": df["position_id"],
    })


TRADE_COLUMN_CONFIG = {
    "Entry": st.column_config.NumberColumn("Entry", format="%.4f"),
    "Exit": st.column_config.NumberColumn("Exit", format="%.4f"),
    "Quantity": st.column_config.NumberColumn("Quantity", format="%.6f"),
    "Realized P&L (₹)": st.column_config.NumberColumn("Realized P&L (₹)", format="₹%.2f"),
    "Return %": st.column_config.NumberColumn("Return %", format="%.2f%%"),
}


def render_pnl_dashboard(acct: Dict[str, Any], compact: bool = False) -> None:
    row1 = st.columns(4)
    row1[0].metric("Portfolio value", fmt_inr(acct["equity"]), delta=f"{acct['return_pct']:+.2f}%")
    row1[1].metric("Available balance", fmt_inr(acct["cash"]))
    row1[2].metric("Open P&L", fmt_pnl(acct["open_pnl"]))
    row1[3].metric("Realized P&L", fmt_pnl(acct["realized"]))
    if compact:
        return
    row2 = st.columns(4)
    row2[0].metric("Total P&L", fmt_pnl(acct["total_pnl"]), delta=f"{acct['return_pct']:+.2f}%")
    row2[1].metric("Invested (open positions)", fmt_inr(acct["invested"]))
    row2[2].metric("Winning / losing trades", f"{acct['wins']} / {acct['losses']}")
    row2[3].metric("Win rate", f"{acct['win_rate']:.1f}%")
    st.caption("Each (partial) close counts as one trade. Win rate = winning ÷ (winning + losing) trades. "
               "P&L uses the latest available delayed price; non-INR instruments use the exchange rate fixed when the position opened.")


def render_terminal(ctx: Dict[str, Any]) -> None:
    check_triggers_and_rerun()
    instr = ctx["instrument"]
    symbol, currency = instr["symbol"], instr["currency"]
    st.markdown("<div class='sim-banner'>🧪 SIMULATION ONLY - no real broker, no real money. Market data is from Yahoo Finance and may be delayed.</div>",
                unsafe_allow_html=True)
    quote = load_quote(symbol)
    render_header(instr, quote)
    left, right = st.columns([3.3, 1.35], gap="medium")
    with left:
        st.session_state.setdefault("timeframe", "15m")
        timeframe = st.radio("Timeframe", list(TIMEFRAMES.keys()), horizontal=True, key="timeframe",
                             label_visibility="collapsed")
        df, meta = load_history(symbol, timeframe)
        if df.empty:
            st.error(f"No chart data for **{symbol}** on the {timeframe} timeframe. {meta.get('error') or ''} "
                     "Try another timeframe (Yahoo limits how far back intraday data goes) or check the ticker symbol.")
        else:
            fig, notes = build_chart(df, symbol, ctx["chart_type"], ctx["indicators"], ctx["bars"], timeframe,
                                     get_open_levels(symbol))
            show_chart(fig, key="main_chart")
            render_ohlc_strip(df, currency)
            chart_status = freshness_label(meta.get("last_bar_utc"), "intraday" if TIMEFRAMES[timeframe]["minutes"] < 1440 else "bars",
                                           TIMEFRAMES[timeframe]["minutes"])
            derived = " 4H bars are built from 1-hour data." if meta.get("derived") else ""
            st.caption(f"Chart data: {chart_status}.{derived} Use the toolbar to zoom/pan/draw, and the ⤢ icon on the chart (where supported) for full screen.")
            for note in notes:
                st.caption(f"ℹ️ {note}")
    with right:
        tab_order, tab_watch = st.tabs(["Order", "Watchlist"])
        with tab_order:
            render_order_panel(instr, quote)
        with tab_watch:
            render_watchlist(ctx["category"], symbol)

    st.markdown("---")
    acct = get_live_account()
    render_pnl_dashboard(acct, compact=True)
    if acct["missing_prices"]:
        st.warning("Live price unavailable for: " + ", ".join(acct["missing_prices"]) + ". Their P&L is shown at entry price (0).")
    t_open, t_pending, t_hist = st.tabs([f"Open positions ({len(acct['positions'])})", "Pending orders", "Recent trades"])
    with t_open:
        render_position_cards(acct["positions"], "tm")
        st.caption("Take-profit / stop-loss are checked automatically only while this app is open and refreshing "
                   "(turn on Auto-refresh in the sidebar). When you reopen it, recent 5-minute price bars are checked back to when the position opened, if Yahoo still has them.")
    with t_pending:
        render_pending_orders("tm")
    with t_hist:
        with db_session(write=False) as conn:
            recent = trades_dataframe(conn, limit=20)
        if recent.empty:
            st.info("No closed trades yet.")
        else:
            st.dataframe(recent, hide_index=True, use_container_width=True, column_config=TRADE_COLUMN_CONFIG)
    err = st.session_state.get("trigger_error")
    if err:
        st.caption(f"⚠️ Last automatic-trigger check problem: {err}")


# =============================================================================
# 12. PORTFOLIO / HISTORY / ACCOUNT PAGES
# =============================================================================


def positions_dataframe(live_positions: List[Dict[str, Any]]) -> pd.DataFrame:
    columns = ["ID", "Instrument", "Symbol", "Direction", "Entry", "Current", "Quantity", "Target", "Stop-loss",
               "Unrealized P&L (₹)", "Return %", "Status", "Opened (IST)"]
    if not live_positions:
        return pd.DataFrame(columns=columns)
    rows = []
    for p in live_positions:
        status = "OPEN" if p["qty"] >= p["initial_qty"] - 1e-9 else f"PARTIAL ({fmt_qty(p['qty'])} of {fmt_qty(p['initial_qty'])} left)"
        rows.append({
            "ID": p["id"], "Instrument": p["name"], "Symbol": p["symbol"], "Direction": p["direction"],
            "Entry": p["entry_price"], "Current": p["price"] if p["price"] is not None else float("nan"),
            "Quantity": p["qty"], "Target": p["target"] if p["target"] else float("nan"),
            "Stop-loss": p["stop"] if p["stop"] else float("nan"),
            "Unrealized P&L (₹)": round(p["unrealized"], 2), "Return %": round(p["return_pct"], 2),
            "Status": status, "Opened (IST)": p["opened_at"],
        })
    df = pd.DataFrame(rows, columns=columns)
    df["Opened (IST)"] = to_ist_series(df["Opened (IST)"])
    return df


POSITION_COLUMN_CONFIG = {
    "Entry": st.column_config.NumberColumn("Entry", format="%.4f"),
    "Current": st.column_config.NumberColumn("Current", format="%.4f"),
    "Quantity": st.column_config.NumberColumn("Quantity", format="%.6f"),
    "Target": st.column_config.NumberColumn("Target", format="%.4f"),
    "Stop-loss": st.column_config.NumberColumn("Stop-loss", format="%.4f"),
    "Unrealized P&L (₹)": st.column_config.NumberColumn("Unrealized P&L (₹)", format="₹%.2f"),
    "Return %": st.column_config.NumberColumn("Return %", format="%.2f%%"),
}


def render_portfolio(ctx: Dict[str, Any]) -> None:
    check_triggers_and_rerun()
    st.subheader("Portfolio & live P&L")
    acct = get_live_account()
    render_pnl_dashboard(acct)
    if acct["missing_prices"]:
        st.warning("Live price unavailable for: " + ", ".join(acct["missing_prices"]) + ". Their P&L is shown at entry price (0).")
    df = positions_dataframe(acct["positions"])
    tab_table, tab_manage, tab_pending, tab_closed = st.tabs(["Positions table", "Manage / close", "Pending orders", "Closed positions"])
    with tab_table:
        if df.empty:
            st.info("No open positions yet.")
        else:
            f1, f2, f3, f4 = st.columns([2, 1.5, 2, 1.2])
            inst_options = ["All"] + sorted(df["Symbol"].unique().tolist())
            if st.session_state.get("pf_inst") not in inst_options:
                st.session_state["pf_inst"] = "All"
            inst = f1.selectbox("Instrument", inst_options, key="pf_inst")
            direction = f2.selectbox("Direction", ["All", "LONG", "SHORT"], key="pf_dir")sort_col = f3.selectbox("Sort by", ["ID", "Unrealized P&L (₹)", "Return %", "Instrument", "Entry", "Quantity"], key="pf_sort")
            ascending = f4.selectbox("Order", ["Descending", "Ascending"], key="pf_order") == "Ascending"
            view = df
            if inst != "All":
                view = view[view["Symbol"] == inst]
            if direction != "All":
                view = view[view["Direction"] == direction]
            view = view.sort_values(sort_col, ascending=ascending)
            st.dataframe(view, hide_index=True, use_container_width=True, column_config=POSITION_COLUMN_CONFIG)
            st.download_button("⬇️ Export open positions (CSV)", data=view.to_csv(index=False).encode("utf-8"),
                               file_name="open_positions.csv", mime="text/csv", key="dl_open_positions")
    with tab_manage:
        render_position_cards(acct["positions"], "pf")
    with tab_pending:
        render_pending_orders("pf")
    with tab_closed:
        closed = closed_positions_dataframe()
        if closed.empty:
            st.info("No fully closed positions yet.")
        else:
            st.dataframe(closed, hide_index=True, use_container_width=True, column_config=TRADE_COLUMN_CONFIG)


def closed_positions_dataframe() -> pd.DataFrame:
    columns = ["Position ID", "Instrument", "Symbol", "Direction", "Entry", "Average exit", "Quantity",
               "Realized P&L (₹)", "Return %", "Opened (IST)", "Closed (IST)", "Last exit reason"]
    with db_session(write=False) as conn:
        positions = [dict(r) for r in conn.execute("SELECT * FROM positions WHERE status = 'CLOSED' ORDER BY id DESC")]
        trades = [dict(r) for r in conn.execute("SELECT * FROM trades ORDER BY id")]
    if not positions:
        return pd.DataFrame(columns=columns)
    by_pos: Dict[int, List[Dict[str, Any]]] = {}
    for t in trades:
        by_pos.setdefault(t["position_id"], []).append(t)
    rows = []
    for p in positions:
        ts = by_pos.get(p["id"], [])
        qty_sum = sum(t["qty"] for t in ts)
        avg_exit = (sum(t["exit_price"] * t["qty"] for t in ts) / qty_sum) if qty_sum else float("nan")
        pnl = sum(t["pnl"] for t in ts)
        cost = p["entry_price"] * p["initial_qty"] * p["fx"]
        rows.append({
            "Position ID": p["id"], "Instrument": p["name"], "Symbol": p["symbol"], "Direction": p["direction"],
            "Entry": p["entry_price"], "Average exit": avg_exit, "Quantity": p["initial_qty"],
            "Realized P&L (₹)": round(pnl, 2), "Return %": round(pnl / cost * 100.0, 2) if cost else 0.0,
            "Opened (IST)": p["opened_at"], "Closed (IST)": p["closed_at"],
            "Last exit reason": REASON_LABELS.get(ts[-1]["reason"], ts[-1]["reason"]) if ts else "",
        })
    df = pd.DataFrame(rows, columns=columns)
    df["Opened (IST)"] = to_ist_series(df["Opened (IST)"])
    df["Closed (IST)"] = to_ist_series(df["Closed (IST)"])
    return df


def orders_dataframe(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = [dict(r) for r in conn.execute("SELECT * FROM orders ORDER BY id DESC").fetchall()]
    columns = ["Order ID", "Created (IST)", "Instrument", "Symbol", "Side", "Type", "Quantity", "Price", "Fill price",
               "Target", "Stop-loss", "Status", "Note"]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    return pd.DataFrame({
        "Order ID": df["id"], "Created (IST)": to_ist_series(df["created_at"]), "Instrument": df["name"],
        "Symbol": df["symbol"], "Side": df["direction"].map({"LONG": "BUY", "SHORT": "SELL"}), "Type": df["order_type"],
        "Quantity": df["qty"], "Price": df["price"], "Fill price": df["fill_price"], "Target": df["target"],
        "Stop-loss": df["stop"], "Status": df["status"], "Note": df["note"].fillna(""),
    })


def render_history(ctx: Dict[str, Any]) -> None:
    check_triggers_and_rerun()
    st.subheader("Trade history")
    with db_session(write=False) as conn:
        trades = trades_dataframe(conn)
        orders = orders_dataframe(conn)
    pnl_total = float(trades["Realized P&L (₹)"].sum()) if not trades.empty else 0.0
    wins = int((trades["Realized P&L (₹)"] > 0).sum()) if not trades.empty else 0
    losses = int((trades["Realized P&L (₹)"] < 0).sum()) if not trades.empty else 0
    c = st.columns(4)
    c[0].metric("Closed trades", len(trades))
    c[1].metric("Realized P&L", fmt_pnl(pnl_total))
    c[2].metric("Winning / losing", f"{wins} / {losses}")
    c[3].metric("Win rate", f"{(wins / (wins + losses) * 100.0) if (wins + losses) else 0.0:.1f}%")
    t_trades, t_orders, t_closed = st.tabs(["Closed trades (exits)", "All orders", "Closed positions"])
    with t_trades:
        if trades.empty:
            st.info("No closed trades yet.")
        else:
            st.caption("Side = the order that opened the position (BUY = long, SELL = short). Each partial close is its own row.")
            st.dataframe(trades, hide_index=True, use_container_width=True, column_config=TRADE_COLUMN_CONFIG)
            st.download_button("⬇️ Export trade history (CSV)", data=trades.to_csv(index=False).encode("utf-8"),
                               file_name="trade_history.csv", mime="text/csv", key="dl_trades")
    with t_orders:
        if orders.empty:
            st.info("No orders yet.")
        else:
            st.dataframe(orders, hide_index=True, use_container_width=True)
            st.download_button("⬇️ Export orders (CSV)", data=orders.to_csv(index=False).encode("utf-8"),
                               file_name="orders.csv", mime="text/csv", key="dl_orders")
    with t_closed:
        closed = closed_positions_dataframe()
        if closed.empty:
            st.info("No fully closed positions yet.")
        else:
            st.dataframe(closed, hide_index=True, use_container_width=True, column_config=TRADE_COLUMN_CONFIG)
            st.download_button("⬇️ Export closed positions (CSV)", data=closed.to_csv(index=False).encode("utf-8"),
                               file_name="closed_positions.csv", mime="text/csv", key="dl_closed")


def render_account(ctx: Dict[str, Any]) -> None:
    st.subheader("Account settings")
    acct = get_live_account()
    render_pnl_dashboard(acct)
    st.markdown("##### Account overview")
    st.write(f"Starting balance: **{fmt_inr(acct['initial'])}** · Open positions: **{len(acct['positions'])}** · "
             f"Closed trades: **{acct['n_trades']}**")
    st.info(f"Account data is stored in SQLite at `{get_db_path()}`. It survives app restarts on your own computer. "
            "On Streamlit Community Cloud the disk can be wiped when the app is rebuilt or rebooted, so export your history "
            "to CSV from the History page now and then. Everyone who opens the same app link shares this one paper account.")
    st.markdown("##### Reset account")
    st.warning("Resetting deletes ALL positions, orders and trade history and restores ₹1,00,000. This cannot be undone.")
    acknowledged = st.checkbox("I understand that everything will be deleted.", key="reset_ack")
    typed = st.text_input("Type RESET to confirm", key="reset_text")
    st.button("Reset account to ₹1,00,000", key="reset_btn", on_click=reset_callback,
              disabled=not (acknowledged and typed.strip().upper() == "RESET"))
    st.markdown("##### How this simulator works")
    st.markdown(
        "- Orders are **simulated only**. Nothing is sent to a broker.\n"
        "- A BUY/LONG or SELL/SHORT order uses cash equal to its full value (no leverage).\n"
        "- Prices come from Yahoo Finance and may be delayed or unavailable. Crypto is usually near real time; Indian stocks and futures can be delayed.\n"
        "- Prices in USD are converted to ₹ using the USD/INR rate at the moment the position opens.\n"
        "- If a stop-loss and take-profit are both touched inside one price bar, the stop-loss is assumed to hit first.\n"
        "- Index symbols (Nifty 50, Bank Nifty) are simulated as 1 unit = 1 index point."
    )


# =============================================================================
# 13. MAIN
# =============================================================================


def main() -> None:
    st.set_page_config(page_title=f"{APP_TITLE} - Paper Trading", page_icon="📈", layout="wide",
                       initial_sidebar_state="expanded")
    inject_css()
    with db_session() as conn:
        init_db(conn)
    ctx = render_sidebar()
    show_flash()
    refresh = ctx["refresh"]
    page = ctx["page"]
    if page == PAGES[0]:
        renderer: Callable[[Dict[str, Any]], None] = render_terminal
    elif page == PAGES[1]:
        renderer = render_portfolio
    elif page == PAGES[2]:
        renderer = render_history
    else:
        renderer = render_account
    if page == PAGES[3]:
        renderer(ctx)
    else:
        st.fragment(run_every=refresh)(renderer)(ctx)


if __name__ == "__main__":
    main()

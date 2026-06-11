"""
Shared data utilities for macro projects.
Downloads one ticker at a time with a small delay — mirrors how
MCP servers like yfinance-mcp avoid Yahoo Finance rate limits.

Priority:
  1. Load from ./data/prices.csv  (instant, no network)
  2. Download live one-by-one with throttle

Run download_data.py once to build the local cache.
"""

import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
try:
    import pandas_datareader.data as pdr
    _HAS_PDR = True
except ImportError:
    _HAS_PDR = False

DATA_DIR    = os.path.join(os.path.dirname(__file__), 'data')
PRICES_FILE = os.path.join(DATA_DIR, 'prices.csv')
RATES_FILE  = os.path.join(DATA_DIR, 'rates.csv')
INTER_DELAY = 1.5   # seconds between individual ticker downloads

_prices_cache = None


def _load_local_prices():
    global _prices_cache
    if _prices_cache is None and os.path.exists(PRICES_FILE):
        print('  Loading prices from local cache...')
        _prices_cache = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    return _prices_cache


def _make_session():
    s = requests.Session()
    s.headers.update({
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/124.0.0.0 Safari/537.36'
        ),
        'Accept': 'application/json,text/html,*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://finance.yahoo.com/',
    })
    return s


def _download_one(ticker, start=None, end=None, retries=3):
    """
    Download a single ticker. Tries yfinance first, falls back to Stooq
    (via pandas_datareader) if Yahoo Finance rate-limits us.
    """
    # --- attempt 1+: yfinance with browser session ---
    session = _make_session()
    kwargs  = dict(auto_adjust=True, progress=False, session=session)
    if start: kwargs['start'] = start
    if end:   kwargs['end']   = end

    for attempt in range(retries):
        try:
            df = yf.download(ticker, **kwargs)
            if df.empty:
                raise ValueError('empty result')
            close = df['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.name = ticker
            return close
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = 'rate' in err_str or 'too many' in err_str or '429' in err_str
            if attempt < retries - 1 and is_rate_limit:
                wait = 10 * (attempt + 1)
                print(f'    YF retry {attempt+1}/{retries} for {ticker} — wait {wait}s')
                time.sleep(wait)
                session = _make_session()
            else:
                # Fall through to Stooq
                break

    # --- fallback: Stooq via pandas_datareader ---
    if _HAS_PDR:
        try:
            stooq_ticker = ticker.replace('=X', '').upper()
            df = pdr.DataReader(stooq_ticker, 'stooq', start=start, end=end)
            if not df.empty:
                close = df['Close'].sort_index()
                close.name = ticker
                print(f'    ✓ Stooq fallback worked for {ticker}')
                return close
        except Exception as e2:
            print(f'    ✗ Stooq also failed for {ticker}: {e2}')

    print(f'    ✗ all sources failed for {ticker}')
    return None


def load_prices(tickers, start=None, end=None):
    """
    Load adjusted Close prices.
    `tickers` — list of ticker strings OR dict {label: ticker}.
    Reads from local CSV if available; downloads one-by-one otherwise.
    """
    if isinstance(tickers, dict):
        label_map   = tickers
        ticker_list = list(tickers.values())
    else:
        label_map   = {t: t for t in tickers}
        ticker_list = list(tickers)

    local = _load_local_prices()

    if local is not None:
        available = [t for t in ticker_list if t in local.columns]
        missing   = [t for t in ticker_list if t not in local.columns]
        parts     = [local[available]] if available else []

        if missing:
            print(f'  Downloading {len(missing)} missing tickers one-by-one...')
            live = _download_batch(missing, start=start, end=end)
            if not live.empty:
                parts.append(live)

        df = pd.concat(parts, axis=1) if parts else pd.DataFrame()
    else:
        print(f'  No local cache — downloading {len(ticker_list)} tickers...')
        df = _download_batch(ticker_list, start=start, end=end)

    if start: df = df.loc[start:]
    if end:   df = df.loc[:end]

    inv = {v: k for k, v in label_map.items()}
    df.columns = [inv.get(c, c) for c in df.columns]
    return df.dropna(how='all').ffill()


def _download_batch(ticker_list, start=None, end=None):
    """Download a list of tickers one at a time, return a DataFrame."""
    series = {}
    for i, ticker in enumerate(ticker_list):
        s = _download_one(ticker, start=start, end=end)
        if s is not None:
            series[ticker] = s
        if i < len(ticker_list) - 1:
            time.sleep(INTER_DELAY)
    return pd.DataFrame(series) if series else pd.DataFrame()


def load_rates(start=None, end=None):
    """Load FRED interest rates from local CSV."""
    if os.path.exists(RATES_FILE):
        df = pd.read_csv(RATES_FILE, index_col=0, parse_dates=True)
        if start: df = df.loc[start:]
        if end:   df = df.loc[:end]
        return df
    return None


# Backward-compat alias used by project scripts
def yf_download(tickers, start=None, end=None, **kwargs):
    if isinstance(tickers, list):
        return _download_batch(tickers, start=start, end=end)
    s = _download_one(str(tickers), start=start, end=end)
    return pd.DataFrame({str(tickers): s}) if s is not None else pd.DataFrame()

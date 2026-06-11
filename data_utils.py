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
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': 'application/json,text/html,*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://finance.yahoo.com/',
    })
    return s


def _download_one(ticker, start=None, end=None, retries=3):
    session = _make_session()
    kwargs = dict(auto_adjust=True, progress=False, session=session)
    if start:
        kwargs['start'] = start
    if end:
        kwargs['end'] = end

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
            err = str(e).lower()
            if attempt < retries - 1 and ('rate' in err or '429' in err):
                time.sleep(10 * (attempt + 1))
                session = _make_session()
            else:
                break

    # try stooq as fallback
    if _HAS_PDR:
        try:
            tk = ticker.replace('=X', '').upper()
            df = pdr.DataReader(tk, 'stooq', start=start, end=end)
            if not df.empty:
                close = df['Close'].sort_index()
                close.name = ticker
                return close
        except Exception:
            pass

    print(f'    could not download {ticker}')
    return None


def _download_batch(tickers, start=None, end=None):
    series = {}
    for i, ticker in enumerate(tickers):
        s = _download_one(ticker, start=start, end=end)
        if s is not None:
            series[ticker] = s
        if i < len(tickers) - 1:
            time.sleep(1.5)
    return pd.DataFrame(series) if series else pd.DataFrame()


def load_prices(tickers, start=None, end=None):
    if isinstance(tickers, dict):
        label_map = tickers
        ticker_list = list(tickers.values())
    else:
        label_map = {t: t for t in tickers}
        ticker_list = list(tickers)

    local = _load_local_prices()

    if local is not None:
        available = [t for t in ticker_list if t in local.columns]
        missing = [t for t in ticker_list if t not in local.columns]
        parts = [local[available]] if available else []

        if missing:
            print(f'  Downloading {len(missing)} missing tickers...')
            live = _download_batch(missing, start=start, end=end)
            if not live.empty:
                parts.append(live)

        df = pd.concat(parts, axis=1) if parts else pd.DataFrame()
    else:
        df = _download_batch(ticker_list, start=start, end=end)

    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]

    inv = {v: k for k, v in label_map.items()}
    df.columns = [inv.get(c, c) for c in df.columns]
    return df.dropna(how='all').ffill()


def load_rates(start=None, end=None):
    if not os.path.exists(RATES_FILE):
        return None
    df = pd.read_csv(RATES_FILE, index_col=0, parse_dates=True)
    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]
    return df


def yf_download(tickers, start=None, end=None, **kwargs):
    if isinstance(tickers, list):
        return _download_batch(tickers, start=start, end=end)
    s = _download_one(str(tickers), start=start, end=end)
    return pd.DataFrame({str(tickers): s}) if s is not None else pd.DataFrame()

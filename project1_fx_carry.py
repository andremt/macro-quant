"""
Project 1 — The Carry Machine
Ilmanen Ch.13: G10 FX carry strategy with momentum overlay and vol-weighting.

Data sources (all free):
  - FX spot rates:        yfinance  (e.g. EURUSD=X)
  - 3-month policy rates: FRED via pandas_datareader
    US:  DTB3
    G10: IRSTCI01{CC}M156N  (OECD overnight call rates, monthly)

Output: equity curve, drawdown, rolling Sharpe, per-currency contribution.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from data_utils import yf_download
import requests
import time
import io
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
START      = '2000-01-01'
END        = datetime.today().strftime('%Y-%m-%d')
LONG_N     = 3          # top N currencies to go long
LOOKBACK_VOL = 26       # weeks for realised vol (~6 months of weekly data)
LOOKBACK_MOM = 4        # weeks for momentum overlay (~1 month of weekly data)
REBAL      = 'ME'       # monthly rebalancing (pandas ME = month-end)

# G10 currencies — ticker, FRED rate series, whether to invert spot quote
G10 = {
    'EUR': ('EURUSD=X', 'IRSTCI01EZM156N', False),
    'GBP': ('GBPUSD=X', 'IRSTCI01GBM156N', False),
    'JPY': ('USDJPY=X', 'IRSTCI01JPM156N', True),   # invert: USD per JPY
    'AUD': ('AUDUSD=X', 'IRSTCI01AUM156N', False),
    'NZD': ('NZDUSD=X', 'IRSTCI01NZM156N', False),
    'CAD': ('USDCAD=X', 'IRSTCI01CAM156N', True),
    'CHF': ('USDCHF=X', 'IRSTCI01CHM156N', True),
    'NOK': ('USDNOK=X', 'IRSTCI01NOM156N', True),
    'SEK': ('USDSEK=X', 'IRSTCI01SEM156N', True),
}

# ── Data download ─────────────────────────────────────────────────────────────
def _make_yf_session():
    """Create a requests Session with browser headers to avoid rate limits."""
    import requests as _req
    s = _req.Session()
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36',
        'Accept': 'application/json,text/html,*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://finance.yahoo.com/',
    })
    return s

def _yf_download_retry(tickers, retries=4, delay=15, **kwargs):
    """yfinance download with retry and browser session on rate-limit errors."""
    session = _make_yf_session()
    for attempt in range(retries):
        try:
            df = yf.download(tickers, session=session, **kwargs)
            if df.empty and attempt < retries - 1:
                print(f'    Empty result, retrying in {delay}s (attempt {attempt+1}/{retries})...')
                time.sleep(delay)
                session = _make_yf_session()   # fresh session
                continue
            return df
        except Exception as e:
            if attempt < retries - 1:
                print(f'    Download error ({e}), retrying in {delay}s...')
                time.sleep(delay)
                session = _make_yf_session()
            else:
                raise
    return pd.DataFrame()

def _fred_csv(series_id, timeout=90):
    """Download a FRED series as CSV (no API key needed)."""
    # Try two endpoints
    urls = [
        f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}',
        f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&file_type=txt&api_key=abcdef01234567890abcdef01234567890ab',
    ]
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    url = urls[0]
    for attempt in range(4):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            text = r.text.strip()
            if not text or 'DATE' not in text.split('\n')[0].upper():
                raise ValueError('Unexpected response format')
            df = pd.read_csv(io.StringIO(text), index_col=0, parse_dates=True)
            df.columns = [series_id]
            # FRED returns '.' for missing values
            df = df.replace('.', np.nan)
            return df
        except Exception as e:
            wait = 10 * (attempt + 1)
            if attempt < 3:
                print(f'    FRED retry {attempt+1}/3 for {series_id} (wait {wait}s)...')
                time.sleep(wait)
            else:
                print(f'    FRED failed for {series_id}: {e}')
                return None

def get_spot(currencies):
    """Load weekly spot FX rates from local CSV cache, return as USD per ccy."""
    from data_utils import _load_local_prices
    local = _load_local_prices()
    spot = pd.DataFrame()
    for ccy, (ticker, _, invert) in currencies.items():
        if local is not None and ticker in local.columns:
            s = local[ticker].dropna()
            spot[ccy] = 1.0 / s if invert else s
        else:
            print(f'  Warning: {ticker} not in cache, skipping {ccy}')
    return spot.dropna(how='all')

def get_rates(currencies):
    """Load interest rates from local rates.csv, resample to weekly."""
    from data_utils import load_rates
    rates_df = load_rates()
    if rates_df is None:
        raise RuntimeError('data/rates.csv not found — run the data setup first')
    # Convert from % to decimal annual rate
    rates_df = rates_df.replace('.', np.nan).astype(float) / 100.0
    rates_df = rates_df.ffill()

    us_rate     = rates_df[['USD']] if 'USD' in rates_df.columns else pd.DataFrame()
    foreign_df  = rates_df.drop(columns=['USD'], errors='ignore')
    return us_rate, foreign_df

# ── Strategy ──────────────────────────────────────────────────────────────────
def compute_carry_signal(spot, us_rate, foreign_rates):
    """
    Carry signal = foreign short rate − US short rate (annualised).
    Rates are in decimal annual (e.g. 0.04 = 4%). Align to spot dates via ffill.
    """
    foreign_aligned = foreign_rates.reindex(spot.index, method='ffill')
    us_aligned      = us_rate.reindex(spot.index, method='ffill')

    carry = pd.DataFrame(index=spot.index)
    for ccy in spot.columns:
        if ccy in foreign_aligned.columns:
            carry[ccy] = foreign_aligned[ccy] - us_aligned['USD']
    return carry

def realised_vol(returns, window=LOOKBACK_VOL):
    # Weekly data → annualise with sqrt(52)
    return returns.rolling(window).std() * np.sqrt(52)

def momentum_signal(spot, window=LOOKBACK_MOM):
    """Return of each currency over the past `window` days."""
    returns = spot.pct_change()
    return returns.rolling(window).sum()

def backtest(spot, carry_signal, mom_signal, long_n=LONG_N):
    """
    Monthly rebalance: rank by carry, long top N short bottom N.
    Overlay: if momentum < 0, set that currency's weight to 0.
    Vol-weight: divide each weight by trailing realised vol.
    Uses monthly resampled returns (weekly data → month-end buckets).
    """
    weekly_ret = spot.pct_change()
    rvol       = realised_vol(weekly_ret)

    # Resample everything to month-end
    ret_m   = (1 + weekly_ret).resample(REBAL).prod() - 1
    carry_m = carry_signal.resample(REBAL).last()
    mom_m   = mom_signal.resample(REBAL).last()
    rvol_m  = rvol.resample(REBAL).last()

    # Common dates with enough history
    dates = carry_m.index

    port_returns = []
    for i in range(len(dates) - 1):
        date      = dates[i]
        next_date = dates[i + 1]

        c = carry_m.loc[date].dropna()
        if len(c) < 3:
            continue
        m = mom_m.loc[date].reindex(c.index).fillna(0)
        v = rvol_m.loc[date].reindex(c.index).replace(0, np.nan)

        # Rank carry: long top N, short bottom N
        ranked  = c.rank(ascending=False)
        weights = pd.Series(0.0, index=c.index)
        weights[ranked <= long_n]               =  1.0
        weights[ranked >= (len(c) - long_n + 1)] = -1.0

        # Momentum overlay: zero out positions with negative momentum
        weights[m < 0] = 0.0

        # Vol-weight then scale to unit gross exposure
        weights = weights / v.fillna(v.median())
        gross   = weights.abs().sum()
        if gross > 0:
            weights /= gross

        # Return for the NEXT month
        if next_date not in ret_m.index:
            continue
        monthly_ret = ret_m.loc[next_date]
        port_ret    = (weights * monthly_ret.reindex(weights.index).fillna(0)).sum()
        port_returns.append({'date': next_date, 'return': port_ret,
                             'weights': weights.to_dict()})

    returns_df = pd.DataFrame(port_returns).set_index('date')
    equity     = (1 + returns_df['return']).cumprod()
    return returns_df, equity

def performance_stats(returns):
    ann_ret = returns.mean() * 12
    ann_vol = returns.std() * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    cum     = (1 + returns).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    max_dd  = drawdown.min()
    return {'Ann. Return': f'{ann_ret:.1%}',
            'Ann. Vol':    f'{ann_vol:.1%}',
            'Sharpe':      f'{sharpe:.2f}',
            'Max DD':      f'{max_dd:.1%}'}

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_results(equity, returns_df, title='G10 FX Carry Strategy'):
    fig = plt.figure(figsize=(14, 10), facecolor='#F7F4ED')
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.2, 1.2], hspace=0.35)

    cum  = equity
    dd   = (cum - cum.cummax()) / cum.cummax()
    roll_sr = returns_df['return'].rolling(12).apply(
        lambda x: (x.mean() * 12) / (x.std() * np.sqrt(12)) if x.std() > 0 else 0)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cum.index, cum.values, color='#0E0E10', lw=2)
    ax1.fill_between(cum.index, 1, cum.values, alpha=0.08, color='#0E0E10')
    ax1.axhline(1, color='#AAAAAA', lw=0.8, ls='--')
    ax1.set_title(title, fontsize=16, fontweight='bold', loc='left', pad=10)
    ax1.set_ylabel('Growth of $1', fontsize=11)
    ax1.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax1.spines[['top', 'right']].set_visible(False)

    stats = performance_stats(returns_df['return'])
    txt   = '   '.join(f'{k}: {v}' for k, v in stats.items())
    ax1.text(0.01, 0.97, txt, transform=ax1.transAxes,
             fontsize=10, va='top', color='#3A3A3F',
             fontfamily='monospace')

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(dd.index, 0, dd.values, color='#FF3D8B', alpha=0.6)
    ax2.set_ylabel('Drawdown', fontsize=11)
    ax2.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax2.spines[['top', 'right']].set_visible(False)

    # 3. Rolling 12m Sharpe
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(roll_sr.index, roll_sr.values, color='#2B4BFF', lw=1.5)
    ax3.axhline(0, color='#AAAAAA', lw=0.8)
    ax3.set_ylabel('Rolling Sharpe (12m)', fontsize=11)
    ax3.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax3.spines[['top', 'right']].set_visible(False)

    plt.savefig('/Users/andretate/Desktop/macro_projects/p1_fx_carry.png',
                dpi=150, bbox_inches='tight', facecolor='#F7F4ED')
    plt.show()
    print('\nChart saved: macro_projects/p1_fx_carry.png')

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== Project 1: G10 FX Carry ===\n')

    print('Downloading spot rates...')
    spot = get_spot(G10)
    print(f'  {len(spot.columns)} currencies, {len(spot)} days')

    print('Downloading interest rates from FRED...')
    us_rate, foreign_rates = get_rates(G10)

    print('Computing signals...')
    carry = compute_carry_signal(spot, us_rate, foreign_rates)
    mom   = momentum_signal(spot)

    print('Backtesting...')
    returns_df, equity = backtest(spot, carry, mom)

    stats = performance_stats(returns_df['return'])
    print('\n── Performance ──────────────────')
    for k, v in stats.items():
        print(f'  {k:20s} {v}')

    print('\n── Latest carry rankings ────────')
    latest_carry = carry.iloc[-1].sort_values(ascending=False)
    for ccy, val in latest_carry.items():
        print(f'  {ccy}  {val:+.2%}')

    plot_results(equity, returns_df)

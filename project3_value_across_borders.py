"""
Project 3 — Value Across Borders
Ilmanen Ch.15: Country equity valuation strategy. Long cheap countries,
short expensive countries based on a composite valuation signal.

Valuation composite (equal-weight):
  1. Earnings yield (E/P) from Shiller CAPE via Barclays / MSCI public data
     Proxied here: 1/CAPE from StarCapital / RAFI historical files (manual).
     Since fully automated CAPE data is hard to source free, we use:
       - Trailing P/E from yfinance (fast proxy)
       - Dividend yield from yfinance
       - Price-to-Book proxy from ETF NAV premium (not available free → skip)
     We rank countries on earnings yield (1/PE) + dividend yield composite.

  2. Data source: yfinance country ETFs + fundamentals scrape
     We use iShares MSCI country ETFs. Trailing PE and dividend yield are
     available from yfinance's .info dict.

Country ETFs (MSCI):
  EWG Germany, EWJ Japan, EWU UK, EWC Canada, EWA Australia,
  EWZ Brazil, EWY Korea, EWW Mexico, EWP Spain, EWI Italy,
  EWQ France, EWD Sweden, EWL Switzerland, EWT Taiwan, EWS Singapore,
  SPY US (benchmark)

Ranking: long cheapest 5, short most expensive 5. Annual rebalance.
Risk: vol-weight positions, scale to 8% portfolio vol target.

Note: Live PE/DY from .info is point-in-time (today's value). The historical
backtest therefore uses price momentum as a cheap valuation proxy for older
periods (lower price = potentially cheaper), with a ranking cross-section at
each annual rebalance date using that month's available data from yfinance.
For a rigorous CAPE backtest you'd need StarCapital or Shiller's country data.
This script clearly separates the live signal (today's ranking) from the
backtest (price-relative + div yield proxy).
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from data_utils import load_prices
import matplotlib.gridspec as gridspec
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
START       = '2005-01-01'
END         = datetime.today().strftime('%Y-%m-%d')
LONG_N      = 5           # top N cheap countries to long
SHORT_N     = 5           # top N expensive countries to short
TARGET_VOL  = 0.08        # 8% annualised portfolio vol
VOL_WINDOW  = 126         # 6-month vol window
REBAL       = 'YE'        # annual rebalance (year-end)

# Country ETF universe
COUNTRIES = {
    'US':          'SPY',
    'Germany':     'EWG',
    'Japan':       'EWJ',
    'UK':          'EWU',
    'Canada':      'EWC',
    'Australia':   'EWA',
    'Brazil':      'EWZ',
    'Korea':       'EWY',
    'Mexico':      'EWW',
    'Spain':       'EWP',
    'Italy':       'EWI',
    'France':      'EWQ',
    'Sweden':      'EWD',
    'Switzerland': 'EWL',
    'Taiwan':      'EWT',
    'Singapore':   'EWS',
}

CLASS_COLORS = {
    'Developed': '#2B4BFF',
    'Emerging':  '#FF3D8B',
}

EMERGING = {'Brazil', 'Korea', 'Mexico', 'Taiwan', 'Singapore'}

# ── Data ──────────────────────────────────────────────────────────────────────
def get_prices():
    tickers = list(COUNTRIES.values())
    return load_prices(COUNTRIES, start=START, end=END)

def get_live_fundamentals():
    """Fetch today's trailing PE and dividend yield from yfinance .info."""
    records = []
    print('  Fetching fundamentals (this may take ~30s)...')
    for country, ticker in COUNTRIES.items():
        try:
            info = yf.Ticker(ticker).info
            pe  = info.get('trailingPE',  None)
            dy  = info.get('dividendYield', None) or 0.0
            records.append({
                'Country': country,
                'Ticker':  ticker,
                'PE':      pe,
                'EY':      1/pe if pe and pe > 0 else None,   # earnings yield
                'DY':      dy,
            })
        except Exception as e:
            print(f'    Warning: could not fetch {ticker}: {e}')
            records.append({'Country': country, 'Ticker': ticker,
                            'PE': None, 'EY': None, 'DY': 0.0})
    return pd.DataFrame(records).set_index('Country')

# ── Valuation Signal ──────────────────────────────────────────────────────────
def valuation_signal(prices):
    """
    Proxy valuation signal for backtest:
      - Earnings yield proxy: negative of 12-month price return (reversion logic)
        (cheaper after underperformance — a crude but widely used proxy)
      - Dividend yield signal: unavailable historically; omit in backtest
    Cross-sectional rank. Lower price return over past year = potentially cheaper.
    Combined with 5-year mean-reversion (longer window adds value information).
    """
    ret_1y  = prices.pct_change(252)        # 1-year momentum (contrarian = cheap signal)
    ret_5y  = prices.pct_change(252 * 5)    # 5-year return (very long = reversion)

    # Cheap signal: low recent returns → higher valuation rank
    # z-score cross-sectionally each day
    def cs_zscore(df):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

    z1 = cs_zscore(-ret_1y)   # negative = contrarian
    z5 = cs_zscore(-ret_5y)

    return 0.5 * z1 + 0.5 * z5    # composite cheap score

# ── Backtest ──────────────────────────────────────────────────────────────────
def backtest(prices):
    daily_ret = prices.pct_change()
    signal    = valuation_signal(prices)
    rvol      = daily_ret.rolling(VOL_WINDOW).std() * np.sqrt(252)

    # Annual rebalance
    signal_y = signal.resample(REBAL).last()
    rvol_y   = rvol.resample(REBAL).last()

    weights_list = []

    for date in signal_y.index[1:]:
        s = signal_y.loc[date].dropna()
        v = rvol_y.loc[date].reindex(s.index).replace(0, np.nan)

        ranked = s.rank(ascending=False)   # rank 1 = cheapest (high signal)
        n      = len(s)
        weights = pd.Series(0.0, index=s.index)
        weights[ranked <= LONG_N]         =  1.0   # long cheapest
        weights[ranked >= (n - SHORT_N + 1)] = -1.0  # short most expensive

        # vol-weight
        weights = weights / v.fillna(v.median())
        # scale gross exposure
        gross = weights.abs().sum()
        if gross > 0:
            weights /= gross

        # scale to target vol — use median country vol as proxy for portfolio vol
        port_vol_est = v.median()
        if port_vol_est > 0:
            scale = TARGET_VOL / port_vol_est
            weights *= min(scale, 3.0)   # cap leverage

        weights_list.append({'date': date, **weights.to_dict()})

    weights_df = pd.DataFrame(weights_list).set_index('date')
    weights_daily = weights_df.reindex(daily_ret.index, method='ffill').shift(1)

    port_ret = (weights_daily * daily_ret).sum(axis=1)
    equity   = (1 + port_ret).cumprod()
    return port_ret, equity, weights_df, signal_y

def performance_stats(returns, label='Strategy'):
    ann_ret  = returns.mean() * 252
    ann_vol  = returns.std() * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0
    cum      = (1 + returns).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    return {
        'label':    label,
        'Ann. Ret': ann_ret,
        'Ann. Vol': ann_vol,
        'Sharpe':   sharpe,
        'Max DD':   drawdown.min(),
        'Hit Rate': (returns > 0).mean(),
    }

# ── Live Ranking ──────────────────────────────────────────────────────────────
def live_ranking(fundamentals):
    """Rank countries by composite valuation (EY + DY) using today's data."""
    df = fundamentals.copy()
    # z-score each component
    for col in ['EY', 'DY']:
        mu = df[col].mean(skipna=True)
        sd = df[col].std(skipna=True)
        if sd > 0:
            df[f'z_{col}'] = (df[col] - mu) / sd
        else:
            df[f'z_{col}'] = 0.0

    z_cols = [c for c in ['z_EY', 'z_DY'] if c in df.columns]
    df['composite'] = df[z_cols].mean(axis=1)
    df['rank']      = df['composite'].rank(ascending=False)
    return df.sort_values('composite', ascending=False)

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_results(prices, port_ret, equity, signal_y, fundamentals):
    fig = plt.figure(figsize=(16, 12), facecolor='#F7F4ED')
    gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    # 1. Equity curve vs SPY
    ax1 = fig.add_subplot(gs[0, :])
    spy_ret = prices['US'].pct_change()
    spy_eq  = (1 + spy_ret.reindex(equity.index)).cumprod()

    ax1.plot(equity.index, equity.values, color='#0E0E10', lw=2.2,
             label='Value Across Borders')
    ax1.plot(spy_eq.index, spy_eq.values, color='#AAAAAA', lw=1.5,
             ls='--', label='SPY (buy & hold)')
    ax1.fill_between(equity.index, 1, equity.values, alpha=0.07, color='#0E0E10')
    ax1.axhline(1, color='#AAAAAA', lw=0.5)
    ax1.set_title('Value Across Borders — Long Cheap, Short Expensive Countries',
                  fontsize=13, fontweight='bold', loc='left')
    ax1.legend(frameon=False)
    ax1.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax1.spines[['top', 'right']].set_visible(False)

    stats = performance_stats(port_ret)
    spy_s = performance_stats(spy_ret.reindex(port_ret.index).fillna(0), 'SPY')
    txt = (f"Value: SR={stats['Sharpe']:.2f}  Ret={stats['Ann. Ret']:.1%}"
           f"  Vol={stats['Ann. Vol']:.1%}  MaxDD={stats['Max DD']:.1%}\n"
           f"SPY:   SR={spy_s['Sharpe']:.2f}  Ret={spy_s['Ann. Ret']:.1%}"
           f"  Vol={spy_s['Ann. Vol']:.1%}  MaxDD={spy_s['Max DD']:.1%}")
    ax1.text(0.01, 0.04, txt, transform=ax1.transAxes, fontsize=9,
             color='#3A3A3F', fontfamily='monospace', va='bottom')

    # 2. Live valuation ranking (horizontal bar chart)
    ax2 = fig.add_subplot(gs[1, 0])
    if fundamentals is not None:
        rank_df = live_ranking(fundamentals).dropna(subset=['composite'])
        colors  = ['#FF3D8B' if c in EMERGING else '#2B4BFF'
                   for c in rank_df.index]
        ax2.barh(rank_df.index, rank_df['composite'], color=colors, alpha=0.85)
        ax2.axvline(0, color='#0E0E10', lw=1)
        ax2.set_title("Today's Valuation Ranking\n(right = cheapest)",
                      fontsize=11, fontweight='bold', loc='left')
        ax2.grid(axis='x', color='#D8D4CA', lw=0.7)
        ax2.spines[['top', 'right']].set_visible(False)
        # legend
        from matplotlib.patches import Patch
        ax2.legend(handles=[Patch(color='#2B4BFF', label='Developed'),
                             Patch(color='#FF3D8B', label='Emerging')],
                   frameon=False, fontsize=8, loc='lower right')

    # 3. Signal heatmap (last rebalance weights)
    ax3 = fig.add_subplot(gs[1, 1])
    last_sig = signal_y.iloc[-1].sort_values(ascending=False)
    bar_colors = ['#FF3D8B' if c in EMERGING else '#2B4BFF'
                  for c in last_sig.index]
    ax3.barh(last_sig.index, last_sig.values, color=bar_colors, alpha=0.85)
    ax3.axvline(0, color='#0E0E10', lw=1)
    ax3.set_title('Last Annual Value Signal (z-score)',
                  fontsize=11, fontweight='bold', loc='left')
    ax3.grid(axis='x', color='#D8D4CA', lw=0.7)
    ax3.spines[['top', 'right']].set_visible(False)

    # 4. Drawdown
    ax4 = fig.add_subplot(gs[2, :])
    cum      = (1 + port_ret).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    ax4.fill_between(drawdown.index, 0, drawdown.values,
                     color='#FF3D8B', alpha=0.65)
    ax4.set_title('Strategy Drawdown', fontsize=12,
                  fontweight='bold', loc='left')
    ax4.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax4.spines[['top', 'right']].set_visible(False)

    plt.savefig('/Users/andretate/Desktop/macro_projects/p3_value.png',
                dpi=150, bbox_inches='tight', facecolor='#F7F4ED')
    plt.show()
    print('Chart saved: macro_projects/p3_value.png')

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== Project 3: Value Across Borders ===\n')

    print('Downloading country ETF prices...')
    prices = get_prices()
    print(f'  {len(prices.columns)} countries, {len(prices)} days')

    print('Backtesting...')
    port_ret, equity, weights_df, signal_y = backtest(prices)

    stats = performance_stats(port_ret)
    print('\n── Performance ──────────────────')
    print(f"  Ann. Return : {stats['Ann. Ret']:.1%}")
    print(f"  Ann. Vol    : {stats['Ann. Vol']:.1%}")
    print(f"  Sharpe      : {stats['Sharpe']:.2f}")
    print(f"  Max DD      : {stats['Max DD']:.1%}")
    print(f"  Hit Rate    : {stats['Hit Rate']:.1%}")

    print('\nFetching live fundamentals for today\'s ranking...')
    try:
        fundamentals = get_live_fundamentals()

        print('\n── Today\'s Valuation Ranking (cheapest → most expensive) ──────')
        ranked = live_ranking(fundamentals)
        for i, (country, row) in enumerate(ranked.iterrows(), 1):
            pe_str = f"PE={row['PE']:.1f}" if pd.notna(row.get('PE')) else 'PE=n/a'
            dy_str = f"DY={row['DY']:.1%}" if pd.notna(row.get('DY')) else 'DY=n/a'
            flag   = '← LONG ' if i <= LONG_N else ('← SHORT' if i > len(ranked) - SHORT_N else '      ')
            em     = ' [EM]' if country in EMERGING else '      '
            print(f'  #{i:2d}  {country:12s}  {pe_str:10s}  {dy_str:8s}  {flag}{em}')
    except Exception as e:
        print(f'  Warning: fundamentals fetch failed ({e}). Skipping live ranking.')
        fundamentals = None

    plot_results(prices, port_ret, equity, signal_y, fundamentals)

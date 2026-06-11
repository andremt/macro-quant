"""
Project 4 — Betting Against Beta (BAB)
Ilmanen Ch.11 / Frazzini & Pedersen (2014): Within the US equity market,
low-beta stocks deliver higher risk-adjusted returns than high-beta stocks.

Strategy:
  - Universe: S&P 500 large-cap proxy — use a basket of sector ETFs
    (XLK, XLV, XLF, XLE, XLI, XLY, XLP, XLU, XLB, XLRE, XLC) plus
    a broader set of mid/large-cap ETFs for more spread.
    For true stock-level BAB, we use a curated list of S&P 500 tickers
    split into quintiles by rolling beta.
  - Beta: rolling 252-day OLS beta vs SPY.
  - Portfolio: long low-beta quintile (Q1), leveraged to β=1.
               short high-beta quintile (Q5), de-leveraged to β=1.
  - Rebalance: monthly.
  - Sizing: equal-weight within each quintile.

Data: yfinance. Universe = 50-stock proxy for S&P 500 diversity
(10 representative tickers per GICS sector).

Note: A full S&P 500 backtest requires downloading ~500 tickers (slow).
This script uses a 60-stock sector-diversified proxy which captures
the core BAB effect robustly.
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
START      = '2010-01-01'
END        = datetime.today().strftime('%Y-%m-%d')
BETA_WIN   = 52       # 1-year rolling beta (52 weeks, since data is weekly)
VOL_WIN    = 60       # vol window for position sizing
QUINTILE_N = 2        # top/bottom N deciles (out of 10 buckets)
REBAL      = 'ME'     # month-end rebalance
TARGET_VOL = 0.10     # 10% annualised portfolio vol

# 60-stock S&P 500 proxy — 6 per sector, chosen for long history & liquidity
UNIVERSE = [
    # Tech
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'ORCL',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO',
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BRK-B',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'PSX', 'VLO',
    # Industrials
    'HON', 'UPS', 'CAT', 'DE', 'RTX', 'LMT',
    # Consumer Discretionary
    'AMZN', 'HD', 'MCD', 'NKE', 'TSLA', 'LOW',
    # Consumer Staples
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL',
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP',
    # Materials
    'LIN', 'APD', 'SHW', 'NEM', 'FCX', 'NUE',
    # Real Estate
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG',
]

SECTORS = {
    'AAPL':'Tech','MSFT':'Tech','NVDA':'Tech','GOOGL':'Tech','META':'Tech','ORCL':'Tech',
    'JNJ':'Health','UNH':'Health','PFE':'Health','ABBV':'Health','MRK':'Health','TMO':'Health',
    'JPM':'Finance','BAC':'Finance','WFC':'Finance','GS':'Finance','MS':'Finance','BRK-B':'Finance',
    'XOM':'Energy','CVX':'Energy','COP':'Energy','SLB':'Energy','PSX':'Energy','VLO':'Energy',
    'HON':'Industrl','UPS':'Industrl','CAT':'Industrl','DE':'Industrl','RTX':'Industrl','LMT':'Industrl',
    'AMZN':'Cons Disc','HD':'Cons Disc','MCD':'Cons Disc','NKE':'Cons Disc','TSLA':'Cons Disc','LOW':'Cons Disc',
    'PG':'Staples','KO':'Staples','PEP':'Staples','WMT':'Staples','COST':'Staples','CL':'Staples',
    'NEE':'Utilities','DUK':'Utilities','SO':'Utilities','D':'Utilities','EXC':'Utilities','AEP':'Utilities',
    'LIN':'Materials','APD':'Materials','SHW':'Materials','NEM':'Materials','FCX':'Materials','NUE':'Materials',
    'AMT':'Real Est','PLD':'Real Est','CCI':'Real Est','EQIX':'Real Est','PSA':'Real Est','SPG':'Real Est',
}

# ── Data ──────────────────────────────────────────────────────────────────────
def get_prices():
    print('  Downloading SPY...')
    spy_df = load_prices(['SPY'], start=START, end=END)
    spy = spy_df['SPY']
    spy.name = 'SPY'

    print(f'  Downloading {len(UNIVERSE)} stocks one-by-one (~{len(UNIVERSE)*2}s)...')
    stocks = load_prices(UNIVERSE, start=START, end=END)

    # Stock data from MCP is weekly — align SPY to the stock dates
    if len(stocks) < len(spy) * 0.3:
        # Stocks are weekly; snap SPY prices to the same dates
        spy = spy.reindex(stocks.index, method='ffill')

    # Drop stocks with insufficient history
    min_obs = BETA_WIN + 12
    stocks = stocks.loc[:, stocks.notna().sum() >= min_obs]
    stocks = stocks.ffill().dropna(how='all')

    return spy, stocks

# ── Rolling Beta ──────────────────────────────────────────────────────────────
def rolling_beta(stock_ret, market_ret, window=BETA_WIN):
    """
    OLS beta = Cov(r_i, r_m) / Var(r_m), computed with rolling window.
    """
    betas = pd.DataFrame(index=stock_ret.index, columns=stock_ret.columns, dtype=float)

    for col in stock_ret.columns:
        cov  = stock_ret[col].rolling(window).cov(market_ret)
        varm = market_ret.rolling(window).var()
        betas[col] = cov / varm.replace(0, np.nan)

    return betas

# ── BAB Portfolio ─────────────────────────────────────────────────────────────
def backtest(spy, stocks):
    """
    Monthly rebalance:
      1. Compute rolling beta for each stock vs SPY.
      2. Rank stocks into quintiles.
      3. Long Q1 (low-beta) scaled to β_L=1 / β_low_avg.
         Short Q5 (high-beta) scaled to β_S=1 / β_high_avg.
      4. Equal-weight within each quintile.
      5. Port return = long leg return - short leg return (zero-net-investment).
    """
    spy_ret    = spy.pct_change().dropna()
    stock_ret  = stocks.pct_change()

    betas = rolling_beta(stock_ret, spy_ret)

    # Monthly rebalance dates
    rebal_dates = betas.resample(REBAL).last().index
    betas_m     = betas.resample(REBAL).last()

    port_records = []

    for i, date in enumerate(rebal_dates[1:], 1):
        b = betas_m.loc[date].dropna()
        if len(b) < 10:
            continue

        # Clip extreme betas — floor at 0.1 to prevent leverage explosion
        b = b.clip(lower=max(b.quantile(0.05), 0.1), upper=b.quantile(0.95))

        # Quintile ranks
        n       = len(b)
        q_size  = n // 5
        ranked  = b.rank()

        low_beta  = b[ranked <= q_size]              # Q1 — low beta
        high_beta = b[ranked > n - q_size]           # Q5 — high beta

        avg_beta_l = low_beta.mean()
        avg_beta_h = high_beta.mean()

        if avg_beta_l <= 0 or avg_beta_h <= 0:
            continue

        # Cap leverage at 5x (i.e. min beta = 0.2) to keep risk realistic
        avg_beta_l = max(avg_beta_l, 0.2)
        avg_beta_h = max(avg_beta_h, 0.2)

        if len(low_beta) == 0 or len(high_beta) == 0:
            continue

        # Equal-weight legs, then lever/delever to β=1
        w_long  = pd.Series(1.0 / avg_beta_l / len(low_beta),  index=low_beta.index)
        w_short = pd.Series(-1.0 / avg_beta_h / len(high_beta), index=high_beta.index)

        weights = pd.concat([w_long, w_short])

        # Get return over next month
        next_idx  = rebal_dates[i+1] if i+1 < len(rebal_dates) else stock_ret.index[-1]
        period    = stock_ret.index[(stock_ret.index > date) & (stock_ret.index <= next_idx)]
        if len(period) == 0:
            continue

        period_ret = (1 + stock_ret.loc[period]).prod() - 1
        port_ret   = (weights * period_ret.reindex(weights.index).fillna(0)).sum()
        spy_period = (1 + spy_ret.loc[period]).prod() - 1

        port_records.append({
            'date':     date,
            'return':   port_ret,
            'spy_ret':  spy_period,
            'n_long':   len(low_beta),
            'n_short':  len(high_beta),
            'avg_beta_long':  avg_beta_l,
            'avg_beta_short': avg_beta_h,
        })

    results  = pd.DataFrame(port_records).set_index('date')
    equity   = (1 + results['return']).cumprod()
    spy_eq   = (1 + results['spy_ret']).cumprod()
    return results, equity, spy_eq, betas_m

def performance_stats(returns, freq=12, label='Strategy'):
    """Monthly return series."""
    ann_ret  = returns.mean() * freq
    ann_vol  = returns.std() * np.sqrt(freq)
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

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_results(results, equity, spy_eq, stocks, betas_m):
    fig = plt.figure(figsize=(16, 12), facecolor='#F7F4ED')
    gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    # 1. Equity curves
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity.index, equity.values, color='#0E0E10', lw=2.2,
             label='BAB Strategy')
    ax1.plot(spy_eq.index, spy_eq.values, color='#AAAAAA', lw=1.5,
             ls='--', label='SPY (buy & hold)')
    ax1.fill_between(equity.index, 1, equity.values, alpha=0.07, color='#0E0E10')
    ax1.axhline(1, color='#AAAAAA', lw=0.5)
    ax1.set_title('Betting Against Beta — Low-Beta vs High-Beta US Equities',
                  fontsize=13, fontweight='bold', loc='left')
    ax1.legend(frameon=False)
    ax1.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax1.spines[['top', 'right']].set_visible(False)

    bab_s = performance_stats(results['return'])
    spy_s = performance_stats(results['spy_ret'])
    txt = (f"BAB:  SR={bab_s['Sharpe']:.2f}  Ret={bab_s['Ann. Ret']:.1%}"
           f"  Vol={bab_s['Ann. Vol']:.1%}  MaxDD={bab_s['Max DD']:.1%}\n"
           f"SPY:  SR={spy_s['Sharpe']:.2f}  Ret={spy_s['Ann. Ret']:.1%}"
           f"  Vol={spy_s['Ann. Vol']:.1%}  MaxDD={spy_s['Max DD']:.1%}")
    ax1.text(0.01, 0.04, txt, transform=ax1.transAxes, fontsize=9,
             color='#3A3A3F', fontfamily='monospace', va='bottom')

    # 2. Rolling average beta of long vs short leg
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(results.index, results['avg_beta_long'],
             color='#2B4BFF', lw=1.8, label='Avg β — Long leg (Q1)')
    ax2.plot(results.index, results['avg_beta_short'],
             color='#FF3D8B', lw=1.8, label='Avg β — Short leg (Q5)')
    ax2.axhline(1.0, color='#0E0E10', lw=0.8, ls='--', alpha=0.5)
    ax2.set_title('Average Beta — Long vs Short Leg',
                  fontsize=11, fontweight='bold', loc='left')
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax2.spines[['top', 'right']].set_visible(False)

    # 3. Today's beta cross-section (bar chart, sorted)
    ax3 = fig.add_subplot(gs[1, 1])
    latest_betas = betas_m.iloc[-1].dropna().sort_values()
    # colour by sector
    sector_colors = {
        'Tech':'#0E0E10','Health':'#2B4BFF','Finance':'#FF3D8B',
        'Energy':'#FFD60A','Industrl':'#8B5CF6','Cons Disc':'#06D6A0',
        'Staples':'#FB8500','Utilities':'#118AB2','Materials':'#EF476F',
        'Real Est':'#AAAAAA',
    }
    bar_colors = [sector_colors.get(SECTORS.get(t, ''), '#AAAAAA')
                  for t in latest_betas.index]
    ax3.barh(latest_betas.index, latest_betas.values, color=bar_colors,
             height=0.7)
    ax3.axvline(1.0, color='#0E0E10', lw=1, ls='--')
    ax3.set_title("Today's Beta Cross-Section (sorted low→high)",
                  fontsize=11, fontweight='bold', loc='left')
    ax3.set_xlabel('Rolling 1-Year Beta vs SPY')
    ax3.tick_params(axis='y', labelsize=6)
    ax3.grid(axis='x', color='#D8D4CA', lw=0.7)
    ax3.spines[['top', 'right']].set_visible(False)

    # 4. Drawdown
    ax4 = fig.add_subplot(gs[2, :])
    cum      = (1 + results['return']).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    ax4.fill_between(drawdown.index, 0, drawdown.values,
                     color='#FF3D8B', alpha=0.65)
    ax4.set_title('BAB Strategy Drawdown', fontsize=12,
                  fontweight='bold', loc='left')
    ax4.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax4.spines[['top', 'right']].set_visible(False)

    plt.savefig('/Users/andretate/Desktop/macro_projects/p4_bab.png',
                dpi=150, bbox_inches='tight', facecolor='#F7F4ED')
    plt.show()
    print('Chart saved: macro_projects/p4_bab.png')

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== Project 4: Betting Against Beta ===\n')

    print('Downloading market data...')
    spy, stocks = get_prices()
    print(f'  Universe: {len(stocks.columns)} stocks, {len(stocks)} days')

    print('Computing rolling betas and backtesting...')
    results, equity, spy_eq, betas_m = backtest(spy, stocks)

    bab_s = performance_stats(results['return'])
    spy_s = performance_stats(results['spy_ret'])

    print('\n── BAB Performance ──────────────────')
    print(f"  Ann. Return : {bab_s['Ann. Ret']:.1%}")
    print(f"  Ann. Vol    : {bab_s['Ann. Vol']:.1%}")
    print(f"  Sharpe      : {bab_s['Sharpe']:.2f}")
    print(f"  Max DD      : {bab_s['Max DD']:.1%}")
    print(f"  Hit Rate    : {bab_s['Hit Rate']:.1%}")

    print('\n── SPY Performance ──────────────────')
    print(f"  Ann. Return : {spy_s['Ann. Ret']:.1%}")
    print(f"  Sharpe      : {spy_s['Sharpe']:.2f}")

    print('\n── Today\'s Beta Extremes ────────────')
    latest = betas_m.iloc[-1].dropna().sort_values()
    print(f"  Lowest beta  (LONG):  {latest.head(5).to_dict()}")
    print(f"  Highest beta (SHORT): {latest.tail(5).to_dict()}")

    print('\n── Average Leverage Applied ─────────')
    print(f"  Long leg leverage  (1/β_low) : {(1/results['avg_beta_long']).mean():.2f}x")
    print(f"  Short leg leverage (1/β_high): {(1/results['avg_beta_short']).mean():.2f}x")

    plot_results(results, equity, spy_eq, stocks, betas_m)

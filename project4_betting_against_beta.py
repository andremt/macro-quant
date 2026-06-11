# Betting Against Beta (BAB)
# Frazzini & Pedersen 2014 / Ilmanen ch.11
# long low-beta stocks leveraged to beta=1, short high-beta de-leveraged to beta=1
# 60-stock S&P 500 proxy, monthly rebalance

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import load_prices
import matplotlib.gridspec as gridspec
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.path.dirname(__file__)
START = '2010-01-01'
END = datetime.today().strftime('%Y-%m-%d')
BETA_WIN = 52
VOL_WIN = 60
QUINTILE_N = 2
REBAL = 'ME'
TARGET_VOL = 0.10

# 60-stock S&P 500 proxy, 6 per sector
UNIVERSE = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'ORCL',
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BRK-B',
    'XOM', 'CVX', 'COP', 'SLB', 'PSX', 'VLO',
    'HON', 'UPS', 'CAT', 'DE', 'RTX', 'LMT',
    'AMZN', 'HD', 'MCD', 'NKE', 'TSLA', 'LOW',
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL',
    'NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP',
    'LIN', 'APD', 'SHW', 'NEM', 'FCX', 'NUE',
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG',
]

SECTORS = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'GOOGL': 'Tech', 'META': 'Tech', 'ORCL': 'Tech',
    'JNJ': 'Health', 'UNH': 'Health', 'PFE': 'Health', 'ABBV': 'Health', 'MRK': 'Health', 'TMO': 'Health',
    'JPM': 'Finance', 'BAC': 'Finance', 'WFC': 'Finance', 'GS': 'Finance', 'MS': 'Finance', 'BRK-B': 'Finance',
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy',
    'HON': 'Industrl', 'UPS': 'Industrl', 'CAT': 'Industrl', 'DE': 'Industrl', 'RTX': 'Industrl', 'LMT': 'Industrl',
    'AMZN': 'Cons Disc', 'HD': 'Cons Disc', 'MCD': 'Cons Disc', 'NKE': 'Cons Disc', 'TSLA': 'Cons Disc', 'LOW': 'Cons Disc',
    'PG': 'Staples', 'KO': 'Staples', 'PEP': 'Staples', 'WMT': 'Staples', 'COST': 'Staples', 'CL': 'Staples',
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities', 'EXC': 'Utilities', 'AEP': 'Utilities',
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials', 'NEM': 'Materials', 'FCX': 'Materials', 'NUE': 'Materials',
    'AMT': 'Real Est', 'PLD': 'Real Est', 'CCI': 'Real Est', 'EQIX': 'Real Est', 'PSA': 'Real Est', 'SPG': 'Real Est',
}


def get_prices():
    print('  Downloading SPY...')
    spy_df = load_prices(['SPY'], start=START, end=END)
    spy = spy_df['SPY']

    print(f'  Downloading {len(UNIVERSE)} stocks...')
    stocks = load_prices(UNIVERSE, start=START, end=END)

    # stocks from cache are weekly, snap SPY to same dates
    if len(stocks) < len(spy) * 0.3:
        spy = spy.reindex(stocks.index, method='ffill')

    min_obs = BETA_WIN + 12
    stocks = stocks.loc[:, stocks.notna().sum() >= min_obs]
    stocks = stocks.ffill().dropna(how='all')

    return spy, stocks


def rolling_beta(stock_ret, market_ret, window=BETA_WIN):
    betas = pd.DataFrame(index=stock_ret.index, columns=stock_ret.columns, dtype=float)
    for col in stock_ret.columns:
        cov = stock_ret[col].rolling(window).cov(market_ret)
        varm = market_ret.rolling(window).var()
        betas[col] = cov / varm.replace(0, np.nan)
    return betas


def backtest(spy, stocks):
    spy_ret = spy.pct_change().dropna()
    stock_ret = stocks.pct_change()

    betas = rolling_beta(stock_ret, spy_ret)

    rebal_dates = betas.resample(REBAL).last().index
    betas_m = betas.resample(REBAL).last()

    port_records = []

    for i, date in enumerate(rebal_dates[1:], 1):
        b = betas_m.loc[date].dropna()
        if len(b) < 10:
            continue

        b = b.clip(lower=max(b.quantile(0.05), 0.1), upper=b.quantile(0.95))

        n = len(b)
        q_size = n // 5
        ranked = b.rank()

        low_beta = b[ranked <= q_size]
        high_beta = b[ranked > n - q_size]

        if len(low_beta) == 0 or len(high_beta) == 0:
            continue

        avg_beta_l = max(low_beta.mean(), 0.2)
        avg_beta_h = max(high_beta.mean(), 0.2)

        if avg_beta_l <= 0 or avg_beta_h <= 0:
            continue

        w_long = pd.Series(1.0 / avg_beta_l / len(low_beta), index=low_beta.index)
        w_short = pd.Series(-1.0 / avg_beta_h / len(high_beta), index=high_beta.index)
        weights = pd.concat([w_long, w_short])

        next_idx = rebal_dates[i+1] if i+1 < len(rebal_dates) else stock_ret.index[-1]
        period = stock_ret.index[(stock_ret.index > date) & (stock_ret.index <= next_idx)]
        if len(period) == 0:
            continue

        period_ret = (1 + stock_ret.loc[period]).prod() - 1
        port_ret = (weights * period_ret.reindex(weights.index).fillna(0)).sum()
        spy_period = (1 + spy_ret.loc[period]).prod() - 1

        port_records.append({
            'date': date,
            'return': port_ret,
            'spy_ret': spy_period,
            'n_long': len(low_beta),
            'n_short': len(high_beta),
            'avg_beta_long': avg_beta_l,
            'avg_beta_short': avg_beta_h,
        })

    results = pd.DataFrame(port_records).set_index('date')
    equity = (1 + results['return']).cumprod()
    spy_eq = (1 + results['spy_ret']).cumprod()
    return results, equity, spy_eq, betas_m


def perf(returns, freq=12, label=''):
    ann_ret = returns.mean() * freq
    ann_vol = returns.std() * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return {
        'label': label,
        'Ann. Ret': ann_ret,
        'Ann. Vol': ann_vol,
        'Sharpe': sharpe,
        'Max DD': dd.min(),
        'Hit Rate': (returns > 0).mean(),
    }


def plot(results, equity, spy_eq, stocks, betas_m):
    fig = plt.figure(figsize=(16, 12), facecolor='#F7F4ED')
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity.index, equity.values, color='#0E0E10', lw=2.2, label='BAB Strategy')
    ax1.plot(spy_eq.index, spy_eq.values, color='#AAAAAA', lw=1.5, ls='--', label='SPY buy & hold')
    ax1.fill_between(equity.index, 1, equity.values, alpha=0.07, color='#0E0E10')
    ax1.axhline(1, color='#AAAAAA', lw=0.5)
    ax1.set_title('Betting Against Beta — Low-Beta vs High-Beta US Equities',
                  fontsize=13, fontweight='bold', loc='left')
    ax1.legend(frameon=False)
    ax1.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax1.spines[['top', 'right']].set_visible(False)

    bab_s = perf(results['return'])
    spy_s = perf(results['spy_ret'])
    txt = (f"BAB:  SR={bab_s['Sharpe']:.2f}  Ret={bab_s['Ann. Ret']:.1%}"
           f"  Vol={bab_s['Ann. Vol']:.1%}  MaxDD={bab_s['Max DD']:.1%}\n"
           f"SPY:  SR={spy_s['Sharpe']:.2f}  Ret={spy_s['Ann. Ret']:.1%}"
           f"  Vol={spy_s['Ann. Vol']:.1%}  MaxDD={spy_s['Max DD']:.1%}")
    ax1.text(0.01, 0.04, txt, transform=ax1.transAxes, fontsize=9,
             color='#3A3A3F', fontfamily='monospace', va='bottom')

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(results.index, results['avg_beta_long'], color='#2B4BFF', lw=1.8, label='Avg beta, long (Q1)')
    ax2.plot(results.index, results['avg_beta_short'], color='#FF3D8B', lw=1.8, label='Avg beta, short (Q5)')
    ax2.axhline(1.0, color='#0E0E10', lw=0.8, ls='--', alpha=0.5)
    ax2.set_title('Average Beta — Long vs Short Leg', fontsize=11, fontweight='bold', loc='left')
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax2.spines[['top', 'right']].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 1])
    latest_betas = betas_m.iloc[-1].dropna().sort_values()
    sector_colors = {
        'Tech': '#0E0E10', 'Health': '#2B4BFF', 'Finance': '#FF3D8B',
        'Energy': '#FFD60A', 'Industrl': '#8B5CF6', 'Cons Disc': '#06D6A0',
        'Staples': '#FB8500', 'Utilities': '#118AB2', 'Materials': '#EF476F',
        'Real Est': '#AAAAAA',
    }
    bar_colors = [sector_colors.get(SECTORS.get(t, ''), '#AAAAAA') for t in latest_betas.index]
    ax3.barh(latest_betas.index, latest_betas.values, color=bar_colors, height=0.7)
    ax3.axvline(1.0, color='#0E0E10', lw=1, ls='--')
    ax3.set_title("Today's Beta Cross-Section (sorted low to high)", fontsize=11, fontweight='bold', loc='left')
    ax3.set_xlabel('Rolling 1-Year Beta vs SPY')
    ax3.tick_params(axis='y', labelsize=6)
    ax3.grid(axis='x', color='#D8D4CA', lw=0.7)
    ax3.spines[['top', 'right']].set_visible(False)

    ax4 = fig.add_subplot(gs[2, :])
    cum = (1 + results['return']).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    ax4.fill_between(dd.index, 0, dd.values, color='#FF3D8B', alpha=0.65)
    ax4.set_title('Drawdown', fontsize=12, fontweight='bold', loc='left')
    ax4.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax4.spines[['top', 'right']].set_visible(False)

    plt.savefig(os.path.join(OUT_DIR, 'p4_bab.png'), dpi=150, bbox_inches='tight', facecolor='#F7F4ED')
    plt.show()
    print('Chart saved: p4_bab.png')


if __name__ == '__main__':
    print('=== Project 4: Betting Against Beta ===\n')

    print('Downloading market data...')
    spy, stocks = get_prices()
    print(f'  {len(stocks.columns)} stocks, {len(stocks)} periods')

    print('Computing rolling betas and backtesting...')
    results, equity, spy_eq, betas_m = backtest(spy, stocks)

    bab_s = perf(results['return'])
    spy_s = perf(results['spy_ret'])

    print('\nBAB performance:')
    print(f"  Ann. Return : {bab_s['Ann. Ret']:.1%}")
    print(f"  Ann. Vol    : {bab_s['Ann. Vol']:.1%}")
    print(f"  Sharpe      : {bab_s['Sharpe']:.2f}")
    print(f"  Max DD      : {bab_s['Max DD']:.1%}")
    print(f"  Hit Rate    : {bab_s['Hit Rate']:.1%}")

    print('\nSPY (buy & hold):')
    print(f"  Ann. Return : {spy_s['Ann. Ret']:.1%}")
    print(f"  Sharpe      : {spy_s['Sharpe']:.2f}")

    print("\nToday's beta extremes:")
    latest = betas_m.iloc[-1].dropna().sort_values()
    print(f'  Lowest beta  (long leg):  {latest.head(5).to_dict()}')
    print(f'  Highest beta (short leg): {latest.tail(5).to_dict()}')

    plot(results, equity, spy_eq, stocks, betas_m)

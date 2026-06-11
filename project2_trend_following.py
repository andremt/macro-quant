# Multi-asset trend following (CTA-style)
# 12-1 momentum signal across equities, bonds, commodities, FX
# skip the most recent month to avoid short-term reversal
# ref: Ilmanen ch.14, Moskowitz et al. 2012 "Time Series Momentum"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from data_utils import load_prices
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.path.dirname(__file__)
START = '2005-01-01'
END = datetime.today().strftime('%Y-%m-%d')
MOM_LONG = 252
MOM_SKIP = 21
VOL_WIN = 60
TARGET_VOL = 0.10
REBAL = 'ME'

ASSETS = {
    'US Equity':  ('SPY',  'Equity'),
    'Intl Equity':('EFA',  'Equity'),
    'EM Equity':  ('EEM',  'Equity'),
    'US 20yr':    ('TLT',  'Bond'),
    'US 7-10yr':  ('IEF',  'Bond'),
    'Intl Govt':  ('IGOV', 'Bond'),
    'Gold':       ('GLD',  'Commodity'),
    'Oil':        ('USO',  'Commodity'),
    'Cmdty Bskt': ('DBC',  'Commodity'),
    'EUR':        ('FXE',  'FX'),
    'JPY':        ('FXY',  'FX'),
    'AUD':        ('FXA',  'FX'),
}

COLORS = {
    'Equity': '#0E0E10',
    'Bond': '#2B4BFF',
    'Commodity': '#FFD60A',
    'FX': '#FF3D8B',
}


def get_prices():
    ticker_map = {label: v[0] for label, v in ASSETS.items()}
    return load_prices(ticker_map, start=START, end=END)


def backtest(prices):
    daily_ret = prices.pct_change()

    long_ret = prices.pct_change(MOM_LONG)
    skip_ret = prices.pct_change(MOM_SKIP)
    signal = long_ret - skip_ret

    direction = np.sign(signal)
    rvol = daily_ret.rolling(VOL_WIN).std() * np.sqrt(252)

    n = prices.shape[1]
    target = TARGET_VOL / np.sqrt(n)
    raw_w = direction * (target / rvol.replace(0, np.nan))

    w_monthly = raw_w.resample(REBAL).last().reindex(daily_ret.index, method='ffill')
    port = (w_monthly.shift(1) * daily_ret).sum(axis=1)
    equity = (1 + port).cumprod()
    return port, equity, direction, raw_w


def perf(returns, label=''):
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return {'label': label, 'Ann. Ret': ann_ret, 'Ann. Vol': ann_vol,
            'Sharpe': sharpe, 'Max DD': dd.min(), 'Hit Rate': (returns > 0).mean()}


def plot(prices, port, equity, direction):
    fig = plt.figure(figsize=(16, 12), facecolor='#F7F4ED')
    gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.35)

    spy_ret = prices['US Equity'].pct_change()
    spy_eq = (1 + spy_ret.reindex(equity.index)).cumprod()

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity.index, equity.values, color='#0E0E10', lw=2.2, label='Trend Following')
    ax1.plot(spy_eq.index, spy_eq.values, color='#AAAAAA', lw=1.5, ls='--', label='SPY buy & hold')
    ax1.fill_between(equity.index, 1, equity.values, alpha=0.07, color='#0E0E10')
    ax1.axhline(1, color='#AAAAAA', lw=0.5)
    ax1.set_title('Multi-Asset Trend Following', fontsize=14, fontweight='bold', loc='left')
    ax1.legend(frameon=False)
    ax1.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax1.spines[['top', 'right']].set_visible(False)

    s = perf(port)
    ss = perf(spy_ret.reindex(port.index).fillna(0))
    ax1.text(0.01, 0.04,
             f"Trend: SR={s['Sharpe']:.2f}  Ret={s['Ann. Ret']:.1%}  MaxDD={s['Max DD']:.1%}\n"
             f"SPY:   SR={ss['Sharpe']:.2f}  Ret={ss['Ann. Ret']:.1%}  MaxDD={ss['Max DD']:.1%}",
             transform=ax1.transAxes, fontsize=9, color='#3A3A3F', fontfamily='monospace', va='bottom')

    ax2 = fig.add_subplot(gs[1, 0])
    latest = direction.iloc[-1]
    bar_colors = [COLORS[ASSETS[a][1]] for a in latest.index]
    ax2.barh(latest.index, latest.values, color=bar_colors)
    ax2.axvline(0, color='#0E0E10', lw=1)
    ax2.set_title('Current Signal (+1 long / -1 short)', fontsize=12, fontweight='bold', loc='left')
    ax2.grid(axis='x', color='#D8D4CA', lw=0.7)
    ax2.spines[['top', 'right']].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 1])
    for cls in COLORS:
        assets_in_class = [l for l, (t, c) in ASSETS.items() if c == cls and l in prices.columns]
        if not assets_in_class:
            continue
        cum = (1 + prices[assets_in_class].pct_change().mean(axis=1)).cumprod()
        ax3.plot(cum.index, cum.values, color=COLORS[cls], lw=1.8, label=cls)
    ax3.set_title('Asset Class Returns', fontsize=12, fontweight='bold', loc='left')
    ax3.legend(frameon=False, fontsize=9)
    ax3.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax3.spines[['top', 'right']].set_visible(False)

    ax4 = fig.add_subplot(gs[2, :])
    cum = (1 + port).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    ax4.fill_between(dd.index, 0, dd.values, color='#FF3D8B', alpha=0.65)
    ax4.set_title('Drawdown', fontsize=12, fontweight='bold', loc='left')
    ax4.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax4.spines[['top', 'right']].set_visible(False)

    plt.savefig(os.path.join(OUT_DIR, 'p2_trend.png'), dpi=150, bbox_inches='tight', facecolor='#F7F4ED')
    plt.show()
    print('Chart saved: p2_trend.png')


if __name__ == '__main__':
    print('=== Project 2: Multi-Asset Trend Following ===\n')

    print('Downloading prices...')
    prices = get_prices()
    print(f'  {len(prices.columns)} assets, {len(prices)} days')

    print('Backtesting...')
    port, equity, direction, weights = backtest(prices)

    s = perf(port)
    print(f"\nPerformance:")
    print(f"  Ann. Return : {s['Ann. Ret']:.1%}")
    print(f"  Ann. Vol    : {s['Ann. Vol']:.1%}")
    print(f"  Sharpe      : {s['Sharpe']:.2f}")
    print(f"  Max DD      : {s['Max DD']:.1%}")
    print(f"  Hit Rate    : {s['Hit Rate']:.1%}")

    print('\nCurrent positions:')
    latest = direction.iloc[-1].sort_values(ascending=False)
    for asset, sig in latest.items():
        arrow = 'LONG ' if sig > 0 else ('SHORT' if sig < 0 else 'FLAT ')
        print(f'  {arrow}  {asset}')

    plot(prices, port, equity, direction)

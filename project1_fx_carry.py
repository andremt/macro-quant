# G10 FX carry strategy
# borrow cheap (JPY, CHF), fund with high-yielders (NZD, AUD)
# momentum overlay to avoid catching falling knives
# ref: Ilmanen "Expected Returns" ch.13

import os
import io
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from data_utils import load_prices, load_rates
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.path.dirname(__file__)
START = '2000-01-01'
END = datetime.today().strftime('%Y-%m-%d')
LONG_N = 3
REBAL = 'ME'
VOL_WIN = 26   # ~6 months of weekly data
MOM_WIN = 4    # ~1 month

# (ticker, FRED series, invert to get USD per unit)
G10 = {
    'EUR': ('EURUSD=X', 'IRSTCI01EZM156N', False),
    'GBP': ('GBPUSD=X', 'IRSTCI01GBM156N', False),
    'JPY': ('USDJPY=X', 'IRSTCI01JPM156N', True),
    'AUD': ('AUDUSD=X', 'IRSTCI01AUM156N', False),
    'NZD': ('NZDUSD=X', 'IRSTCI01NZM156N', False),
    'CAD': ('USDCAD=X', 'IRSTCI01CAM156N', True),
    'CHF': ('USDCHF=X', 'IRSTCI01CHM156N', True),
    'NOK': ('USDNOK=X', 'IRSTCI01NOM156N', True),
    'SEK': ('USDSEK=X', 'IRSTCI01SEM156N', True),
}


def get_spot():
    from data_utils import _load_local_prices
    local = _load_local_prices()
    spot = pd.DataFrame()
    for ccy, (ticker, _, invert) in G10.items():
        if local is not None and ticker in local.columns:
            s = local[ticker].dropna()
            spot[ccy] = 1.0 / s if invert else s
    return spot.dropna(how='all')


def get_rates():
    rates = load_rates()
    if rates is None:
        raise RuntimeError('data/rates.csv not found')
    rates = rates.replace('.', np.nan).astype(float) / 100.0
    rates = rates.ffill()
    us = rates[['USD']] if 'USD' in rates.columns else pd.DataFrame()
    foreign = rates.drop(columns=['USD'], errors='ignore')
    return us, foreign


def carry_signal(spot, us_rate, foreign_rates):
    foreign = foreign_rates.reindex(spot.index, method='ffill')
    us = us_rate.reindex(spot.index, method='ffill')
    carry = pd.DataFrame(index=spot.index)
    for ccy in spot.columns:
        if ccy in foreign.columns:
            carry[ccy] = foreign[ccy] - us['USD']
    return carry


def backtest(spot, carry, mom_win=MOM_WIN, vol_win=VOL_WIN, long_n=LONG_N):
    weekly_ret = spot.pct_change()
    rvol = weekly_ret.rolling(vol_win).std() * np.sqrt(52)
    mom = weekly_ret.rolling(mom_win).sum()

    ret_m = (1 + weekly_ret).resample(REBAL).prod() - 1
    carry_m = carry.resample(REBAL).last()
    mom_m = mom.resample(REBAL).last()
    rvol_m = rvol.resample(REBAL).last()

    records = []
    dates = carry_m.index
    for i in range(len(dates) - 1):
        date = dates[i]
        next_date = dates[i + 1]

        c = carry_m.loc[date].dropna()
        if len(c) < 3:
            continue
        m = mom_m.loc[date].reindex(c.index).fillna(0)
        v = rvol_m.loc[date].reindex(c.index).replace(0, np.nan)

        ranked = c.rank(ascending=False)
        w = pd.Series(0.0, index=c.index)
        w[ranked <= long_n] = 1.0
        w[ranked >= len(c) - long_n + 1] = -1.0
        w[m < 0] = 0.0  # momentum overlay

        w = w / v.fillna(v.median())
        gross = w.abs().sum()
        if gross > 0:
            w /= gross

        if next_date not in ret_m.index:
            continue
        port_ret = (w * ret_m.loc[next_date].reindex(w.index).fillna(0)).sum()
        records.append({'date': next_date, 'return': port_ret, 'weights': w.to_dict()})

    results = pd.DataFrame(records).set_index('date')
    equity = (1 + results['return']).cumprod()
    return results, equity


def perf(returns):
    ann_ret = returns.mean() * 12
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return {
        'Ann. Return': f'{ann_ret:.1%}',
        'Ann. Vol': f'{ann_vol:.1%}',
        'Sharpe': f'{sharpe:.2f}',
        'Max DD': f'{dd.min():.1%}'
    }


def plot(equity, results):
    fig = plt.figure(figsize=(14, 10), facecolor='#F7F4ED')
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1.2, 1.2], hspace=0.35)

    cum = equity
    dd = (cum - cum.cummax()) / cum.cummax()
    roll_sr = results['return'].rolling(12).apply(
        lambda x: (x.mean() * 12) / (x.std() * np.sqrt(12)) if x.std() > 0 else 0
    )

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cum.index, cum.values, color='#0E0E10', lw=2)
    ax1.fill_between(cum.index, 1, cum.values, alpha=0.08, color='#0E0E10')
    ax1.axhline(1, color='#AAAAAA', lw=0.8, ls='--')
    ax1.set_title('G10 FX Carry Strategy', fontsize=16, fontweight='bold', loc='left', pad=10)
    ax1.set_ylabel('Growth of $1', fontsize=11)
    ax1.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax1.spines[['top', 'right']].set_visible(False)
    stats = perf(results['return'])
    ax1.text(0.01, 0.97, '   '.join(f'{k}: {v}' for k, v in stats.items()),
             transform=ax1.transAxes, fontsize=10, va='top', color='#3A3A3F', fontfamily='monospace')

    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(dd.index, 0, dd.values, color='#FF3D8B', alpha=0.6)
    ax2.set_ylabel('Drawdown', fontsize=11)
    ax2.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax2.spines[['top', 'right']].set_visible(False)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(roll_sr.index, roll_sr.values, color='#2B4BFF', lw=1.5)
    ax3.axhline(0, color='#AAAAAA', lw=0.8)
    ax3.set_ylabel('Rolling Sharpe (12m)', fontsize=11)
    ax3.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax3.spines[['top', 'right']].set_visible(False)

    plt.savefig(os.path.join(OUT_DIR, 'p1_fx_carry.png'), dpi=150, bbox_inches='tight', facecolor='#F7F4ED')
    plt.show()
    print('Chart saved: p1_fx_carry.png')


if __name__ == '__main__':
    print('=== Project 1: G10 FX Carry ===\n')

    print('Loading spot rates...')
    spot = get_spot()
    print(f'  {len(spot.columns)} currencies, {len(spot)} weeks')

    print('Loading interest rates...')
    us_rate, foreign_rates = get_rates()

    carry = carry_signal(spot, us_rate, foreign_rates)
    results, equity = backtest(spot, carry)

    stats = perf(results['return'])
    print('\nPerformance:')
    for k, v in stats.items():
        print(f'  {k}: {v}')

    print('\nCurrent carry rankings:')
    latest = carry.iloc[-1].sort_values(ascending=False)
    for ccy, val in latest.items():
        print(f'  {ccy}  {val:+.2%}')

    plot(equity, results)

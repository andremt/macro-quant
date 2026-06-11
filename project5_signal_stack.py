# Signal Stack: combining carry, trend, and value into one portfolio
# Ilmanen ch.19-20 — diversification across factors, not just assets
# equal-weight the three signals, then vol-target the composite
# ref: AQR "Value and Momentum Everywhere", Moskowitz et al. 2012

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import load_prices
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.path.dirname(__file__)
START = '2008-01-01'
END = datetime.today().strftime('%Y-%m-%d')
TREND_LONG = 252
TREND_SKIP = 21
VALUE_WIN = 252 * 5
VOL_WIN = 60
TARGET_VOL = 0.10
REBAL = 'ME'
FACTOR_WT = {'Carry': 1/3, 'Trend': 1/3, 'Value': 1/3}

ASSETS = {
    'US Equity':  ('SPY', 'Equity'),
    'DM Equity':  ('EFA', 'Equity'),
    'EM Equity':  ('EEM', 'Equity'),
    'US 20yr':    ('TLT', 'Bond'),
    'US 7-10yr':  ('IEF', 'Bond'),
    'Gold':       ('GLD', 'Commodity'),
    'Oil':        ('USO', 'Commodity'),
    'Cmdty Bskt': ('DBC', 'Commodity'),
    'EUR':        ('FXE', 'FX'),
    'JPY':        ('FXY', 'FX'),
    'AUD':        ('FXA', 'FX'),
    'Germany':    ('EWG', 'Country'),
    'Japan':      ('EWJ', 'Country'),
    'UK':         ('EWU', 'Country'),
    'Australia':  ('EWA', 'Country'),
    'Brazil':     ('EWZ', 'Country'),
    'Korea':      ('EWY', 'Country'),
    'Spain':      ('EWP', 'Country'),
    'Italy':      ('EWI', 'Country'),
    'Canada':     ('EWC', 'Country'),
}

CLASS_COLORS = {
    'Equity': '#0E0E10',
    'Bond': '#2B4BFF',
    'Commodity': '#FFD60A',
    'FX': '#FF3D8B',
    'Country': '#8B5CF6',
}


def get_prices():
    ticker_map = {label: v[0] for label, v in ASSETS.items()}
    return load_prices(ticker_map, start=START, end=END)


def get_benchmark():
    df = load_prices(['SPY', 'TLT'], start=START, end=END)
    bench = 0.6 * df['SPY'].pct_change() + 0.4 * df['TLT'].pct_change()
    return bench.dropna()


def cs_zscore(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def signal_carry(prices):
    # carry proxy: 12m total return minus 11m momentum (strips out price momentum)
    total_12m = prices.pct_change(TREND_LONG)
    mom_11m = prices.pct_change(TREND_LONG - TREND_SKIP)
    return cs_zscore(total_12m - mom_11m)


def signal_trend(prices):
    long_ret = prices.pct_change(TREND_LONG)
    skip_ret = prices.pct_change(TREND_SKIP)
    return cs_zscore(long_ret - skip_ret)


def signal_value(prices):
    # cheap = underperformed over 5 years, mean-reversion logic
    ret_5y = prices.pct_change(VALUE_WIN)
    return cs_zscore(-ret_5y)


def composite_signal(prices):
    s_carry = signal_carry(prices)
    s_trend = signal_trend(prices)
    s_value = signal_value(prices)
    composite = (FACTOR_WT['Carry'] * s_carry +
                 FACTOR_WT['Trend'] * s_trend +
                 FACTOR_WT['Value'] * s_value)
    return composite, s_carry, s_trend, s_value


def run_backtest(prices, signal):
    daily_ret = prices.pct_change()
    rvol = daily_ret.rolling(VOL_WIN).std() * np.sqrt(252)
    direction = np.sign(signal)
    n = prices.shape[1]
    target_per_asset = TARGET_VOL / np.sqrt(n)
    raw_weights = direction * (target_per_asset / rvol.replace(0, np.nan))
    weights_m = raw_weights.resample(REBAL).last().reindex(daily_ret.index, method='ffill')
    port_ret = (weights_m.shift(1) * daily_ret).sum(axis=1)
    equity = (1 + port_ret).cumprod()
    return port_ret, equity


def perf(returns, label=''):
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
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


def plot(prices, ret_comp, eq_comp, ret_carry, ret_trend, ret_value,
         s_carry, s_trend, s_value, s_composite, bench_ret):

    fig = plt.figure(figsize=(18, 14), facecolor='#F7F4ED')
    gs = gridspec.GridSpec(3, 3, hspace=0.48, wspace=0.38)

    bench_eq = (1 + bench_ret.reindex(eq_comp.index).fillna(0)).cumprod()
    eq_carry = (1 + ret_carry.reindex(eq_comp.index).fillna(0)).cumprod()
    eq_trend = (1 + ret_trend.reindex(eq_comp.index).fillna(0)).cumprod()
    eq_value = (1 + ret_value.reindex(eq_comp.index).fillna(0)).cumprod()

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(eq_comp.index, eq_comp.values, color='#0E0E10', lw=2.5, label='Signal Stack')
    ax1.plot(bench_eq.index, bench_eq.values, color='#AAAAAA', lw=1.5, ls='--', label='60/40 benchmark')
    ax1.plot(eq_carry.index, eq_carry.values, color='#FF3D8B', lw=1.2, alpha=0.6, label='Carry only')
    ax1.plot(eq_trend.index, eq_trend.values, color='#2B4BFF', lw=1.2, alpha=0.6, label='Trend only')
    ax1.plot(eq_value.index, eq_value.values, color='#FFD60A', lw=1.2, alpha=0.8, label='Value only')
    ax1.fill_between(eq_comp.index, 1, eq_comp.values, alpha=0.06, color='#0E0E10')
    ax1.axhline(1, color='#AAAAAA', lw=0.5)
    ax1.set_title('Signal Stack: Carry + Trend + Value (equal weight)', fontsize=13, fontweight='bold', loc='left')
    ax1.legend(frameon=False, fontsize=9, ncol=5)
    ax1.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax1.spines[['top', 'right']].set_visible(False)

    cs = perf(ret_comp, 'Stack')
    bs = perf(bench_ret.reindex(ret_comp.index).fillna(0), '60/40')
    txt = (f"Stack: SR={cs['Sharpe']:.2f}  Ret={cs['Ann. Ret']:.1%}  Vol={cs['Ann. Vol']:.1%}  MaxDD={cs['Max DD']:.1%}\n"
           f"60/40: SR={bs['Sharpe']:.2f}  Ret={bs['Ann. Ret']:.1%}  Vol={bs['Ann. Vol']:.1%}  MaxDD={bs['Max DD']:.1%}")
    ax1.text(0.01, 0.04, txt, transform=ax1.transAxes, fontsize=9, color='#3A3A3F', fontfamily='monospace', va='bottom')

    ax2 = fig.add_subplot(gs[1, 0])
    factor_rets = pd.DataFrame({
        'Carry': ret_carry, 'Trend': ret_trend, 'Value': ret_value, 'Stack': ret_comp,
    }).dropna()
    corr = factor_rets.resample('ME').sum().corr()
    im = ax2.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(len(corr.columns)))
    ax2.set_xticklabels(corr.columns, fontsize=9)
    ax2.set_yticks(range(len(corr.index)))
    ax2.set_yticklabels(corr.index, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            ax2.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=8, color='black')
    ax2.set_title('Factor Correlation Matrix\n(monthly returns)', fontsize=11, fontweight='bold', loc='left')
    plt.colorbar(im, ax=ax2, shrink=0.8)

    ax3 = fig.add_subplot(gs[1, 1:])
    latest = s_composite.iloc[-1].dropna().sort_values(ascending=False)
    bar_colors = [CLASS_COLORS.get(ASSETS[a][1], '#AAAAAA') for a in latest.index if a in ASSETS]
    ax3.barh(latest.index, latest.values, color=bar_colors, height=0.7)
    ax3.axvline(0, color='#0E0E10', lw=1)
    ax3.set_title('Live Composite Signal (today)\nPositive = long, negative = short',
                  fontsize=11, fontweight='bold', loc='left')
    ax3.grid(axis='x', color='#D8D4CA', lw=0.7)
    ax3.spines[['top', 'right']].set_visible(False)
    legend_patches = [Patch(color=c, label=l) for l, c in CLASS_COLORS.items()]
    ax3.legend(handles=legend_patches, frameon=False, fontsize=8, loc='lower right', ncol=2)

    ax4 = fig.add_subplot(gs[2, :2])
    roll_sr_stack = ret_comp.rolling(252).apply(
        lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0)
    roll_sr_bench = bench_ret.reindex(ret_comp.index).fillna(0).rolling(252).apply(
        lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0)
    ax4.plot(roll_sr_stack.index, roll_sr_stack.values, color='#0E0E10', lw=1.8, label='Signal Stack')
    ax4.plot(roll_sr_bench.index, roll_sr_bench.values, color='#AAAAAA', lw=1.2, ls='--', label='60/40')
    ax4.axhline(0, color='#AAAAAA', lw=0.8)
    ax4.set_title('Rolling 12m Sharpe', fontsize=11, fontweight='bold', loc='left')
    ax4.legend(frameon=False, fontsize=9)
    ax4.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax4.spines[['top', 'right']].set_visible(False)

    ax5 = fig.add_subplot(gs[2, 2])
    dd = (eq_comp - eq_comp.cummax()) / eq_comp.cummax()
    ax5.fill_between(dd.index, 0, dd.values, color='#FF3D8B', alpha=0.65)
    ax5.set_title('Drawdown', fontsize=11, fontweight='bold', loc='left')
    ax5.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax5.spines[['top', 'right']].set_visible(False)

    plt.savefig(os.path.join(OUT_DIR, 'p5_signal_stack.png'), dpi=150, bbox_inches='tight', facecolor='#F7F4ED')
    plt.show()
    print('Chart saved: p5_signal_stack.png')


def print_signals(s_carry, s_trend, s_value, s_composite):
    latest_carry = s_carry.iloc[-1]
    latest_trend = s_trend.iloc[-1]
    latest_value = s_value.iloc[-1]
    latest_comp = s_composite.iloc[-1]

    print('\nLive signals (sorted by composite z-score):')
    print(f"  {'Asset':14s}  {'Class':10s}  {'Carry':>7s}  {'Trend':>7s}  {'Value':>7s}  {'Total':>7s}  Dir")
    for asset in latest_comp.sort_values(ascending=False).index:
        if asset not in ASSETS:
            continue
        _, cls = ASSETS[asset]
        c = latest_carry.get(asset, float('nan'))
        t = latest_trend.get(asset, float('nan'))
        v = latest_value.get(asset, float('nan'))
        z = latest_comp.get(asset, float('nan'))
        direction = 'long ' if z > 0 else ('short' if z < 0 else 'flat ')
        print(f'  {asset:14s}  {cls:10s}  {c:+7.2f}  {t:+7.2f}  {v:+7.2f}  {z:+7.2f}  {direction}')


if __name__ == '__main__':
    print('=== Project 5: Signal Stack ===\n')

    print('Downloading universe...')
    prices = get_prices()
    print(f'  {len(prices.columns)} assets, {len(prices)} days')

    print('Downloading 60/40 benchmark...')
    bench_ret = get_benchmark()

    print('Computing signals...')
    s_composite, s_carry, s_trend, s_value = composite_signal(prices)

    print('Running backtests...')
    ret_comp, eq_comp = run_backtest(prices, s_composite)
    ret_carry, eq_carry = run_backtest(prices, s_carry)
    ret_trend, eq_trend = run_backtest(prices, s_trend)
    ret_value, eq_value = run_backtest(prices, s_value)

    cs = perf(ret_comp, 'Signal Stack')
    ks = perf(ret_carry, 'Carry')
    ts = perf(ret_trend, 'Trend')
    vs = perf(ret_value, 'Value')
    bs = perf(bench_ret.reindex(ret_comp.index).fillna(0), '60/40')

    print('\nPerformance:')
    print(f"  {'Strategy':18s}  {'Ann. Ret':>9s}  {'Sharpe':>7s}  {'Max DD':>8s}")
    for s in [cs, ks, ts, vs, bs]:
        print(f"  {s['label']:18s}  {s['Ann. Ret']:>9.1%}  {s['Sharpe']:>7.2f}  {s['Max DD']:>8.1%}")

    print('\nFactor correlations (monthly returns):')
    factor_rets = pd.DataFrame({
        'Carry': ret_carry, 'Trend': ret_trend, 'Value': ret_value, 'Stack': ret_comp,
    }).dropna()
    print(factor_rets.resample('ME').sum().corr().round(3).to_string())

    print_signals(s_carry, s_trend, s_value, s_composite)

    plot(prices, ret_comp, eq_comp, ret_carry, ret_trend, ret_value,
         s_carry, s_trend, s_value, s_composite, bench_ret)

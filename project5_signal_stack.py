"""
Project 5 — The Signal Stack: Multi-Factor Framework
Ilmanen Ch.19-20: Combining carry, trend, and value signals into a
single diversified portfolio. Each signal is computed independently,
z-scored cross-sectionally, then combined with equal or optimised weights.

This is the capstone project — it imports logic from projects 1-4 and
stacks everything into a unified backtest + live signal dashboard.

Universe: 20-asset multi-class (equities, bonds, commodities, FX, countries)
  Equities:    SPY, EFA, EEM
  Bonds:       TLT, IEF
  Commodities: GLD, USO, DBC
  FX ETFs:     FXE, FXY, FXA (EUR, JPY, AUD vs USD)
  Countries:   EWG, EWJ, EWU, EWA, EWZ, EWY, EWP, EWI

Signals (all normalised to z-scores):
  1. Carry     — trailing 12m dividend yield (proxy for funding advantage)
  2. Trend     — 12-1 month time-series momentum (TSMOM)
  3. Value     — negative of 5yr price return (reversion-based value)

Combination:
  Equal-weight (1/3 each) → composite signal → direction × vol-weight
  Portfolio vol target: 10% annualised.

Output:
  - Equity curves: composite + each factor leg + 60/40 benchmark
  - Factor correlation matrix (diversification benefit)
  - Live signal dashboard: current signal strength per asset per factor
  - Rolling Sharpe vs 60/40
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from data_utils import load_prices
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
START       = '2008-01-01'
END         = datetime.today().strftime('%Y-%m-%d')
TREND_LONG  = 252         # 12-month TSMOM
TREND_SKIP  = 21          # skip recent month
VALUE_WIN   = 252 * 5     # 5-year value window
VOL_WIN     = 60          # 60-day realised vol
TARGET_VOL  = 0.10        # 10% portfolio vol
REBAL       = 'ME'        # monthly rebalance
FACTOR_WT   = {'Carry': 1/3, 'Trend': 1/3, 'Value': 1/3}

# 20-asset universe
ASSETS = {
    # label: (ticker, class)
    'US Equity':   ('SPY',  'Equity'),
    'DM Equity':   ('EFA',  'Equity'),
    'EM Equity':   ('EEM',  'Equity'),
    'US 20yr':     ('TLT',  'Bond'),
    'US 7-10yr':   ('IEF',  'Bond'),
    'Gold':        ('GLD',  'Commodity'),
    'Oil':         ('USO',  'Commodity'),
    'Cmdty Bskt':  ('DBC',  'Commodity'),
    'EUR':         ('FXE',  'FX'),
    'JPY':         ('FXY',  'FX'),
    'AUD':         ('FXA',  'FX'),
    'Germany':     ('EWG',  'Country'),
    'Japan':       ('EWJ',  'Country'),
    'UK':          ('EWU',  'Country'),
    'Australia':   ('EWA',  'Country'),
    'Brazil':      ('EWZ',  'Country'),
    'Korea':       ('EWY',  'Country'),
    'Spain':       ('EWP',  'Country'),
    'Italy':       ('EWI',  'Country'),
    'Canada':      ('EWC',  'Country'),
}

CLASS_COLORS = {
    'Equity':    '#0E0E10',
    'Bond':      '#2B4BFF',
    'Commodity': '#FFD60A',
    'FX':        '#FF3D8B',
    'Country':   '#8B5CF6',
}

# ── Data ──────────────────────────────────────────────────────────────────────
def get_prices():
    ticker_map = {label: v[0] for label, v in ASSETS.items()}
    return load_prices(ticker_map, start=START, end=END)

def get_spy_for_benchmark():
    df    = load_prices(['SPY', 'TLT'], start=START, end=END)
    bench = 0.6 * df['SPY'].pct_change() + 0.4 * df['TLT'].pct_change()
    return bench.dropna()

# ── Signal Computation ────────────────────────────────────────────────────────
def cs_zscore(df):
    """Cross-sectional z-score: de-mean and scale across assets each day."""
    return df.sub(df.mean(axis=1), axis=0).div(
        df.std(axis=1).replace(0, np.nan), axis=0)

def signal_carry(prices):
    """
    Carry proxy: trailing 12-month total return minus 12-1 momentum.
    This approximates the dividend/yield advantage (higher carry = richer income).
    For FX ETFs this captures the interest rate differential embedded in the ETF.
    """
    total_12m = prices.pct_change(TREND_LONG)
    mom_11m   = prices.pct_change(TREND_LONG - TREND_SKIP)
    carry_proxy = total_12m - mom_11m    # strip out price momentum
    return cs_zscore(carry_proxy)

def signal_trend(prices):
    """12-1 month time-series momentum, cross-sectionally z-scored."""
    long_ret = prices.pct_change(TREND_LONG)
    skip_ret = prices.pct_change(TREND_SKIP)
    tsmom    = long_ret - skip_ret
    return cs_zscore(tsmom)

def signal_value(prices):
    """
    5-year reversal as value proxy. Negative of long-run return
    (cheap = underperformed for 5 years = mean-reversion signal).
    """
    ret_5y = prices.pct_change(VALUE_WIN)
    return cs_zscore(-ret_5y)

def composite_signal(prices):
    """Equal-weight combination of carry, trend, value."""
    s_carry = signal_carry(prices)
    s_trend = signal_trend(prices)
    s_value = signal_value(prices)

    composite = (FACTOR_WT['Carry'] * s_carry +
                 FACTOR_WT['Trend'] * s_trend +
                 FACTOR_WT['Value'] * s_value)
    return composite, s_carry, s_trend, s_value

# ── Backtest Engine ───────────────────────────────────────────────────────────
def run_backtest(prices, signal, label='Strategy'):
    """
    Generic backtest: signal → direction → vol-weight → monthly rebalance.
    """
    daily_ret = prices.pct_change()
    rvol      = daily_ret.rolling(VOL_WIN).std() * np.sqrt(252)

    # Direction = sign of signal
    direction = np.sign(signal)

    # Vol-weight
    n              = prices.shape[1]
    target_per_asset = TARGET_VOL / np.sqrt(n)
    raw_weights    = direction * (target_per_asset / rvol.replace(0, np.nan))

    # Monthly rebalance
    weights_m = raw_weights.resample(REBAL).last().reindex(
        daily_ret.index, method='ffill')

    port_ret = (weights_m.shift(1) * daily_ret).sum(axis=1)
    equity   = (1 + port_ret).cumprod()
    return port_ret, equity

def performance_stats(returns, label=''):
    ann_ret  = returns.mean() * 252
    ann_vol  = returns.std() * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0
    cum      = (1 + returns).cumprod()
    dd       = (cum - cum.cummax()) / cum.cummax()
    return {
        'label':    label,
        'Ann. Ret': ann_ret,
        'Ann. Vol': ann_vol,
        'Sharpe':   sharpe,
        'Max DD':   dd.min(),
        'Hit Rate': (returns > 0).mean(),
    }

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_results(prices, ret_composite, eq_composite,
                 ret_carry, ret_trend, ret_value,
                 s_carry, s_trend, s_value, s_composite,
                 bench_ret):

    fig = plt.figure(figsize=(18, 14), facecolor='#F7F4ED')
    gs  = gridspec.GridSpec(3, 3, hspace=0.48, wspace=0.38)

    bench_eq = (1 + bench_ret.reindex(eq_composite.index).fillna(0)).cumprod()
    eq_carry = (1 + ret_carry.reindex(eq_composite.index).fillna(0)).cumprod()
    eq_trend = (1 + ret_trend.reindex(eq_composite.index).fillna(0)).cumprod()
    eq_value = (1 + ret_value.reindex(eq_composite.index).fillna(0)).cumprod()

    # 1. Main equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(eq_composite.index, eq_composite.values,
             color='#0E0E10', lw=2.5, label='Signal Stack (composite)')
    ax1.plot(bench_eq.index, bench_eq.values,
             color='#AAAAAA', lw=1.5, ls='--', label='60/40 Benchmark')
    ax1.plot(eq_carry.index, eq_carry.values,
             color='#FF3D8B', lw=1.2, alpha=0.6, label='Carry only')
    ax1.plot(eq_trend.index, eq_trend.values,
             color='#2B4BFF', lw=1.2, alpha=0.6, label='Trend only')
    ax1.plot(eq_value.index, eq_value.values,
             color='#FFD60A', lw=1.2, alpha=0.8, label='Value only')
    ax1.fill_between(eq_composite.index, 1, eq_composite.values,
                     alpha=0.06, color='#0E0E10')
    ax1.axhline(1, color='#AAAAAA', lw=0.5)
    ax1.set_title('Signal Stack: Carry + Trend + Value (Equal-Weight Combination)',
                  fontsize=13, fontweight='bold', loc='left')
    ax1.legend(frameon=False, fontsize=9, ncol=5)
    ax1.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax1.spines[['top', 'right']].set_visible(False)

    cs = performance_stats(ret_composite, 'Stack')
    bs = performance_stats(bench_ret.reindex(ret_composite.index).fillna(0), '60/40')
    txt = (f"Stack: SR={cs['Sharpe']:.2f}  Ret={cs['Ann. Ret']:.1%}"
           f"  Vol={cs['Ann. Vol']:.1%}  MaxDD={cs['Max DD']:.1%}\n"
           f"60/40: SR={bs['Sharpe']:.2f}  Ret={bs['Ann. Ret']:.1%}"
           f"  Vol={bs['Ann. Vol']:.1%}  MaxDD={bs['Max DD']:.1%}")
    ax1.text(0.01, 0.04, txt, transform=ax1.transAxes, fontsize=9,
             color='#3A3A3F', fontfamily='monospace', va='bottom')

    # 2. Factor return correlation matrix
    ax2 = fig.add_subplot(gs[1, 0])
    factor_rets = pd.DataFrame({
        'Carry': ret_carry, 'Trend': ret_trend,
        'Value': ret_value, 'Stack': ret_composite,
    }).dropna()
    corr = factor_rets.resample('ME').sum().corr()
    im = ax2.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(len(corr.columns))); ax2.set_xticklabels(corr.columns, fontsize=9)
    ax2.set_yticks(range(len(corr.index)));   ax2.set_yticklabels(corr.index, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            ax2.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center',
                     fontsize=8, color='black')
    ax2.set_title('Factor Correlation Matrix\n(monthly returns)',
                  fontsize=11, fontweight='bold', loc='left')
    plt.colorbar(im, ax=ax2, shrink=0.8)

    # 3. Live signal heatmap — composite per asset
    ax3 = fig.add_subplot(gs[1, 1:])
    latest = s_composite.iloc[-1].dropna().sort_values(ascending=False)
    bar_colors = [CLASS_COLORS.get(ASSETS[a][1], '#AAAAAA')
                  for a in latest.index if a in ASSETS]
    ax3.barh(latest.index, latest.values, color=bar_colors, height=0.7)
    ax3.axvline(0, color='#0E0E10', lw=1)
    ax3.set_title('Live Composite Signal (today)\nPositive = long, negative = short',
                  fontsize=11, fontweight='bold', loc='left')
    ax3.grid(axis='x', color='#D8D4CA', lw=0.7)
    ax3.spines[['top', 'right']].set_visible(False)
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=l) for l, c in CLASS_COLORS.items()]
    ax3.legend(handles=legend_patches, frameon=False, fontsize=8,
               loc='lower right', ncol=2)

    # 4. Rolling 12m Sharpe vs 60/40
    ax4 = fig.add_subplot(gs[2, :2])
    roll_sr_stack = ret_composite.rolling(252).apply(
        lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0)
    roll_sr_bench = bench_ret.reindex(ret_composite.index).fillna(0).rolling(252).apply(
        lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0)
    ax4.plot(roll_sr_stack.index, roll_sr_stack.values, color='#0E0E10', lw=1.8,
             label='Signal Stack')
    ax4.plot(roll_sr_bench.index, roll_sr_bench.values, color='#AAAAAA', lw=1.2,
             ls='--', label='60/40')
    ax4.axhline(0, color='#AAAAAA', lw=0.8)
    ax4.set_title('Rolling 12m Sharpe — Stack vs 60/40',
                  fontsize=11, fontweight='bold', loc='left')
    ax4.legend(frameon=False, fontsize=9)
    ax4.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax4.spines[['top', 'right']].set_visible(False)

    # 5. Drawdown
    ax5 = fig.add_subplot(gs[2, 2])
    cum = eq_composite
    dd  = (cum - cum.cummax()) / cum.cummax()
    ax5.fill_between(dd.index, 0, dd.values, color='#FF3D8B', alpha=0.65)
    ax5.set_title('Stack Drawdown', fontsize=11, fontweight='bold', loc='left')
    ax5.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax5.spines[['top', 'right']].set_visible(False)

    plt.savefig('/Users/andretate/Desktop/macro_projects/p5_signal_stack.png',
                dpi=150, bbox_inches='tight', facecolor='#F7F4ED')
    plt.show()
    print('Chart saved: macro_projects/p5_signal_stack.png')

def print_live_dashboard(prices, s_carry, s_trend, s_value, s_composite):
    """Print a clean live signal table."""
    latest_carry = s_carry.iloc[-1]
    latest_trend = s_trend.iloc[-1]
    latest_value = s_value.iloc[-1]
    latest_comp  = s_composite.iloc[-1]

    print('\n── Live Signal Dashboard ────────────────────────────────────────')
    print(f"  {'Asset':14s}  {'Class':10s}  {'Carry':>7s}  {'Trend':>7s}  "
          f"{'Value':>7s}  {'Composite':>9s}  Direction")
    print('  ' + '─' * 75)
    for asset in latest_comp.sort_values(ascending=False).index:
        if asset not in ASSETS:
            continue
        _, cls = ASSETS[asset]
        c = latest_carry.get(asset, np.nan)
        t = latest_trend.get(asset, np.nan)
        v = latest_value.get(asset, np.nan)
        z = latest_comp.get(asset, np.nan)
        direction = '▲ LONG ' if z > 0 else ('▼ SHORT' if z < 0 else '── FLAT ')
        print(f"  {asset:14s}  {cls:10s}  {c:+7.2f}  {t:+7.2f}  "
              f"{v:+7.2f}  {z:+9.2f}  {direction}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== Project 5: Signal Stack — Multi-Factor Framework ===\n')

    print('Downloading 20-asset universe...')
    prices = get_prices()
    print(f'  {len(prices.columns)} assets, {len(prices)} days')

    print('Downloading 60/40 benchmark...')
    bench_ret = get_spy_for_benchmark()

    print('Computing signals...')
    s_composite, s_carry, s_trend, s_value = composite_signal(prices)

    print('Running backtests (composite + 3 factor legs)...')
    ret_comp,  eq_comp  = run_backtest(prices, s_composite, 'Stack')
    ret_carry, eq_carry = run_backtest(prices, s_carry,     'Carry')
    ret_trend, eq_trend = run_backtest(prices, s_trend,     'Trend')
    ret_value, eq_value = run_backtest(prices, s_value,     'Value')

    cs = performance_stats(ret_comp,  'Signal Stack')
    ks = performance_stats(ret_carry, 'Carry')
    ts = performance_stats(ret_trend, 'Trend')
    vs = performance_stats(ret_value, 'Value')
    bs = performance_stats(bench_ret.reindex(ret_comp.index).fillna(0), '60/40')

    print('\n── Performance Summary ──────────────────────────────────')
    print(f"  {'Strategy':18s}  {'Ann. Ret':>9s}  {'Sharpe':>7s}  {'Max DD':>8s}")
    print('  ' + '─' * 52)
    for s in [cs, ks, ts, vs, bs]:
        print(f"  {s['label']:18s}  {s['Ann. Ret']:>9.1%}  "
              f"{s['Sharpe']:>7.2f}  {s['Max DD']:>8.1%}")

    print('\n── Factor Correlations (monthly returns) ─────────────────')
    factor_rets = pd.DataFrame({
        'Carry': ret_carry, 'Trend': ret_trend,
        'Value': ret_value, 'Stack': ret_comp,
    }).dropna()
    print(factor_rets.resample('ME').sum().corr().round(3).to_string())

    print_live_dashboard(prices, s_carry, s_trend, s_value, s_composite)

    plot_results(prices, ret_comp, eq_comp,
                 ret_carry, ret_trend, ret_value,
                 s_carry, s_trend, s_value, s_composite,
                 bench_ret)

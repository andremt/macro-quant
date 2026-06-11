"""
Project 2 — Trend Following Across Asset Classes
Ilmanen Ch.14: Time-series momentum (CTA-style) across equities, bonds,
commodities, and FX. Volatility-weighted positions.

Data source: yfinance (ETF proxies — all free, no API key needed)
  Equities:    SPY (US), EFA (Developed Intl), EEM (Emerging)
  Bonds:       TLT (US 20yr), IEF (US 7-10yr), IGOV (Intl Govt)
  Commodities: GLD (Gold), USO (Oil), DBC (Commodity basket)
  FX:          FXE (EUR), FXY (JPY), FXA (AUD)

Signal: 12-month total return, skipping most recent month (12-1 momentum).
Position: long if signal > 0, short if < 0.
Sizing: inverse-volatility weighted. Scale to 10% annualised portfolio vol.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from data_utils import load_prices
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
START   = '2005-01-01'
END     = datetime.today().strftime('%Y-%m-%d')
FORMATION_LONG  = 252    # 12-month lookback
FORMATION_SKIP  = 21     # skip most recent month
VOL_WINDOW      = 60     # days for realised vol
TARGET_VOL      = 0.10   # 10% annualised portfolio vol
REBAL           = 'M'    # monthly rebalancing

ASSETS = {
    # label: (ticker, asset class)
    'US Equity':   ('SPY',  'Equity'),
    'Intl Equity': ('EFA',  'Equity'),
    'EM Equity':   ('EEM',  'Equity'),
    'US 20yr':     ('TLT',  'Bond'),
    'US 7-10yr':   ('IEF',  'Bond'),
    'Intl Govt':   ('IGOV', 'Bond'),
    'Gold':        ('GLD',  'Commodity'),
    'Oil':         ('USO',  'Commodity'),
    'Cmdty Bskt':  ('DBC',  'Commodity'),
    'EUR':         ('FXE',  'FX'),
    'JPY':         ('FXY',  'FX'),
    'AUD':         ('FXA',  'FX'),
}

CLASS_COLORS = {
    'Equity':    '#0E0E10',
    'Bond':      '#2B4BFF',
    'Commodity': '#FFD60A',
    'FX':        '#FF3D8B',
}

# ── Data ──────────────────────────────────────────────────────────────────────
def get_prices():
    ticker_map = {label: v[0] for label, v in ASSETS.items()}
    return load_prices(ticker_map, start=START, end=END)

# ── Signals ───────────────────────────────────────────────────────────────────
def momentum_signal(prices):
    """12-1 month time-series momentum."""
    long_ret = prices.pct_change(FORMATION_LONG)
    skip_ret = prices.pct_change(FORMATION_SKIP)
    signal   = long_ret - skip_ret      # subtract the most recent month
    return signal

def vol_scale(returns, window=VOL_WINDOW):
    """Annualised realised volatility."""
    return returns.rolling(window).std() * np.sqrt(252)

# ── Backtest ──────────────────────────────────────────────────────────────────
def backtest(prices):
    daily_ret = prices.pct_change()
    signal    = momentum_signal(prices)
    rvol      = vol_scale(daily_ret)

    # sign of signal = direction (long +1, short -1)
    direction = np.sign(signal)

    # vol-weight: position size = target_vol_per_asset / asset_vol
    n_assets         = prices.shape[1]
    target_per_asset = TARGET_VOL / np.sqrt(n_assets)
    raw_weights      = direction * (target_per_asset / rvol.replace(0, np.nan))

    # monthly rebalance: take weights at end of each month
    weights_m = raw_weights.resample(REBAL).last().reindex(
        daily_ret.index, method='ffill')

    port_daily = (weights_m.shift(1) * daily_ret).sum(axis=1)
    equity     = (1 + port_daily).cumprod()
    return port_daily, equity, direction, raw_weights

def performance_stats(returns, label='Strategy'):
    ann_ret  = returns.mean() * 252
    ann_vol  = returns.std() * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0
    cum      = (1 + returns).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    return {
        'label':      label,
        'Ann. Ret':   ann_ret,
        'Ann. Vol':   ann_vol,
        'Sharpe':     sharpe,
        'Max DD':     drawdown.min(),
        'Hit Rate':   (returns > 0).mean(),
    }

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_results(prices, daily_ret, equity, direction):
    fig = plt.figure(figsize=(16, 12), facecolor='#F7F4ED')
    gs  = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.35)

    # 1. Equity curve vs SPY
    ax1 = fig.add_subplot(gs[0, :])
    spy_ret = prices['US Equity'].pct_change()
    spy_eq  = (1 + spy_ret.reindex(equity.index)).cumprod()

    ax1.plot(equity.index, equity.values, color='#0E0E10', lw=2.2,
             label='Trend Following')
    ax1.plot(spy_eq.index, spy_eq.values, color='#AAAAAA', lw=1.5,
             ls='--', label='SPY (buy & hold)')
    ax1.fill_between(equity.index, 1, equity.values, alpha=0.07, color='#0E0E10')
    ax1.axhline(1, color='#AAAAAA', lw=0.5)
    ax1.set_title('Multi-Asset Trend Following vs Buy & Hold',
                  fontsize=14, fontweight='bold', loc='left')
    ax1.legend(frameon=False)
    ax1.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax1.spines[['top','right']].set_visible(False)

    stats = performance_stats(daily_ret)
    spy_s = performance_stats(spy_ret.reindex(daily_ret.index).fillna(0), 'SPY')
    stat_txt = (f"Trend: SR={stats['Sharpe']:.2f}  Ret={stats['Ann. Ret']:.1%}"
                f"  Vol={stats['Ann. Vol']:.1%}  MaxDD={stats['Max DD']:.1%}\n"
                f"SPY:   SR={spy_s['Sharpe']:.2f}  Ret={spy_s['Ann. Ret']:.1%}"
                f"  Vol={spy_s['Ann. Vol']:.1%}  MaxDD={spy_s['Max DD']:.1%}")
    ax1.text(0.01, 0.04, stat_txt, transform=ax1.transAxes,
             fontsize=9, color='#3A3A3F', fontfamily='monospace',
             va='bottom')

    # 2. Current signal heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    latest_dir = direction.iloc[-1]
    colors_bar = [CLASS_COLORS[ASSETS[a][1]] for a in latest_dir.index]
    bars = ax2.barh(latest_dir.index, latest_dir.values, color=colors_bar)
    ax2.axvline(0, color='#0E0E10', lw=1)
    ax2.set_title('Current Signal (Long=+1 / Short=-1)',
                  fontsize=12, fontweight='bold', loc='left')
    ax2.grid(axis='x', color='#D8D4CA', lw=0.7)
    ax2.spines[['top','right']].set_visible(False)

    # 3. Asset-class contribution
    ax3 = fig.add_subplot(gs[1, 1])
    class_ret = {}
    for label, (ticker, cls) in ASSETS.items():
        if label in prices.columns:
            class_ret.setdefault(cls, []).append(
                prices[label].pct_change())

    for cls, rets in class_ret.items():
        cum = (1 + pd.concat(rets, axis=1).mean(axis=1)).cumprod()
        ax3.plot(cum.index, cum.values, color=CLASS_COLORS[cls],
                 lw=1.8, label=cls)
    ax3.set_title('Asset Class Equity Curves', fontsize=12,
                  fontweight='bold', loc='left')
    ax3.legend(frameon=False, fontsize=9)
    ax3.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax3.spines[['top','right']].set_visible(False)

    # 4. Drawdown
    ax4 = fig.add_subplot(gs[2, :])
    cum      = (1 + daily_ret).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    ax4.fill_between(drawdown.index, 0, drawdown.values,
                     color='#FF3D8B', alpha=0.65)
    ax4.set_title('Strategy Drawdown', fontsize=12,
                  fontweight='bold', loc='left')
    ax4.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax4.spines[['top','right']].set_visible(False)

    plt.savefig('/Users/andretate/Desktop/macro_projects/p2_trend.png',
                dpi=150, bbox_inches='tight', facecolor='#F7F4ED')
    plt.show()
    print('Chart saved: macro_projects/p2_trend.png')

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== Project 2: Multi-Asset Trend Following ===\n')

    print('Downloading prices...')
    prices = get_prices()
    print(f'  {len(prices.columns)} assets, {len(prices)} days')

    print('Backtesting...')
    daily_ret, equity, direction, weights = backtest(prices)

    stats = performance_stats(daily_ret)
    print('\n── Performance ──────────────────')
    print(f"  Ann. Return : {stats['Ann. Ret']:.1%}")
    print(f"  Ann. Vol    : {stats['Ann. Vol']:.1%}")
    print(f"  Sharpe      : {stats['Sharpe']:.2f}")
    print(f"  Max DD      : {stats['Max DD']:.1%}")
    print(f"  Hit Rate    : {stats['Hit Rate']:.1%}")

    print('\n── Current positions ────────────')
    latest = direction.iloc[-1].sort_values(ascending=False)
    for asset, sig in latest.items():
        arrow = '▲ LONG ' if sig > 0 else ('▼ SHORT' if sig < 0 else '── FLAT ')
        print(f'  {arrow}  {asset}')

    plot_results(prices, daily_ret, equity, direction)

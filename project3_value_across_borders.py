# Country equity value strategy
# long cheap countries, short expensive ones
# using price returns as a value proxy (lower long-run return = potentially cheaper)
# live signal uses actual PE and dividend yield from yfinance
# ref: Ilmanen ch.15, Faber 2012

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from datetime import datetime
from data_utils import load_prices
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.path.dirname(__file__)
START = '2005-01-01'
END = datetime.today().strftime('%Y-%m-%d')
LONG_N = 5
SHORT_N = 5
TARGET_VOL = 0.08
VOL_WIN = 126
REBAL = 'YE'

COUNTRIES = {
    'US': 'SPY', 'Germany': 'EWG', 'Japan': 'EWJ', 'UK': 'EWU',
    'Canada': 'EWC', 'Australia': 'EWA', 'Brazil': 'EWZ', 'Korea': 'EWY',
    'Mexico': 'EWW', 'Spain': 'EWP', 'Italy': 'EWI', 'France': 'EWQ',
    'Sweden': 'EWD', 'Switzerland': 'EWL', 'Taiwan': 'EWT', 'Singapore': 'EWS',
}

EMERGING = {'Brazil', 'Korea', 'Mexico', 'Taiwan', 'Singapore'}


def get_prices():
    return load_prices(COUNTRIES, start=START, end=END)


def get_fundamentals():
    print('  Fetching fundamentals (this may take ~30s)...')
    records = []
    for country, ticker in COUNTRIES.items():
        try:
            info = yf.Ticker(ticker).info
            pe = info.get('trailingPE', None)
            dy = info.get('dividendYield', None) or 0.0
            records.append({
                'Country': country, 'Ticker': ticker,
                'PE': pe, 'EY': 1/pe if pe and pe > 0 else None, 'DY': dy
            })
        except Exception:
            records.append({'Country': country, 'Ticker': ticker, 'PE': None, 'EY': None, 'DY': 0.0})
    return pd.DataFrame(records).set_index('Country')


def valuation_signal(prices):
    # cheap = underperformed recently, mean-reversion logic
    def cs_z(df):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

    z1 = cs_z(-prices.pct_change(252))
    z5 = cs_z(-prices.pct_change(252 * 5))
    return 0.5 * z1 + 0.5 * z5


def backtest(prices):
    daily_ret = prices.pct_change()
    signal = valuation_signal(prices)
    rvol = daily_ret.rolling(VOL_WIN).std() * np.sqrt(252)

    signal_y = signal.resample(REBAL).last()
    rvol_y = rvol.resample(REBAL).last()

    weights_list = []
    for date in signal_y.index[1:]:
        s = signal_y.loc[date].dropna()
        v = rvol_y.loc[date].reindex(s.index).replace(0, np.nan)

        ranked = s.rank(ascending=False)
        n = len(s)
        w = pd.Series(0.0, index=s.index)
        w[ranked <= LONG_N] = 1.0
        w[ranked >= n - SHORT_N + 1] = -1.0

        w = w / v.fillna(v.median())
        gross = w.abs().sum()
        if gross > 0:
            w /= gross

        port_vol = v.median()
        if port_vol > 0:
            w *= min(TARGET_VOL / port_vol, 3.0)

        weights_list.append({'date': date, **w.to_dict()})

    weights_df = pd.DataFrame(weights_list).set_index('date')
    w_daily = weights_df.reindex(daily_ret.index, method='ffill').shift(1)
    port = (w_daily * daily_ret).sum(axis=1)
    equity = (1 + port).cumprod()
    return port, equity, weights_df, signal_y


def perf(returns, label=''):
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return {'label': label, 'Ann. Ret': ann_ret, 'Ann. Vol': ann_vol,
            'Sharpe': sharpe, 'Max DD': dd.min(), 'Hit Rate': (returns > 0).mean()}


def rank_countries(fundamentals):
    df = fundamentals.copy()
    for col in ['EY', 'DY']:
        mu, sd = df[col].mean(skipna=True), df[col].std(skipna=True)
        df[f'z_{col}'] = (df[col] - mu) / sd if sd > 0 else 0.0
    df['composite'] = df[[c for c in ['z_EY', 'z_DY'] if c in df.columns]].mean(axis=1)
    return df.sort_values('composite', ascending=False)


def plot(prices, port, equity, signal_y, fundamentals):
    fig = plt.figure(figsize=(16, 12), facecolor='#F7F4ED')
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    spy_ret = prices['US'].pct_change()
    spy_eq = (1 + spy_ret.reindex(equity.index)).cumprod()

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity.index, equity.values, color='#0E0E10', lw=2.2, label='Value Across Borders')
    ax1.plot(spy_eq.index, spy_eq.values, color='#AAAAAA', lw=1.5, ls='--', label='SPY buy & hold')
    ax1.fill_between(equity.index, 1, equity.values, alpha=0.07, color='#0E0E10')
    ax1.axhline(1, color='#AAAAAA', lw=0.5)
    ax1.set_title('Value Across Borders', fontsize=13, fontweight='bold', loc='left')
    ax1.legend(frameon=False)
    ax1.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax1.spines[['top', 'right']].set_visible(False)

    s = perf(port)
    ss = perf(spy_ret.reindex(port.index).fillna(0))
    ax1.text(0.01, 0.04,
             f"Value: SR={s['Sharpe']:.2f}  Ret={s['Ann. Ret']:.1%}  MaxDD={s['Max DD']:.1%}\n"
             f"SPY:   SR={ss['Sharpe']:.2f}  Ret={ss['Ann. Ret']:.1%}  MaxDD={ss['Max DD']:.1%}",
             transform=ax1.transAxes, fontsize=9, color='#3A3A3F', fontfamily='monospace', va='bottom')

    ax2 = fig.add_subplot(gs[1, 0])
    if fundamentals is not None:
        ranked = rank_countries(fundamentals).dropna(subset=['composite'])
        colors = ['#FF3D8B' if c in EMERGING else '#2B4BFF' for c in ranked.index]
        ax2.barh(ranked.index, ranked['composite'], color=colors, alpha=0.85)
        ax2.axvline(0, color='#0E0E10', lw=1)
        ax2.set_title("Today's Valuation Ranking\n(right = cheapest)", fontsize=11, fontweight='bold', loc='left')
        ax2.grid(axis='x', color='#D8D4CA', lw=0.7)
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.legend(handles=[Patch(color='#2B4BFF', label='Developed'), Patch(color='#FF3D8B', label='Emerging')],
                   frameon=False, fontsize=8, loc='lower right')

    ax3 = fig.add_subplot(gs[1, 1])
    last_sig = signal_y.iloc[-1].sort_values(ascending=False)
    colors = ['#FF3D8B' if c in EMERGING else '#2B4BFF' for c in last_sig.index]
    ax3.barh(last_sig.index, last_sig.values, color=colors, alpha=0.85)
    ax3.axvline(0, color='#0E0E10', lw=1)
    ax3.set_title('Last Annual Value Signal (z-score)', fontsize=11, fontweight='bold', loc='left')
    ax3.grid(axis='x', color='#D8D4CA', lw=0.7)
    ax3.spines[['top', 'right']].set_visible(False)

    ax4 = fig.add_subplot(gs[2, :])
    cum = (1 + port).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    ax4.fill_between(dd.index, 0, dd.values, color='#FF3D8B', alpha=0.65)
    ax4.set_title('Drawdown', fontsize=12, fontweight='bold', loc='left')
    ax4.grid(axis='y', color='#D8D4CA', lw=0.7)
    ax4.spines[['top', 'right']].set_visible(False)

    plt.savefig(os.path.join(OUT_DIR, 'p3_value.png'), dpi=150, bbox_inches='tight', facecolor='#F7F4ED')
    plt.show()
    print('Chart saved: p3_value.png')


if __name__ == '__main__':
    print('=== Project 3: Value Across Borders ===\n')

    print('Downloading country ETF prices...')
    prices = get_prices()
    print(f'  {len(prices.columns)} countries, {len(prices)} days')

    print('Backtesting...')
    port, equity, weights_df, signal_y = backtest(prices)

    s = perf(port)
    print(f"\nPerformance:")
    print(f"  Ann. Return : {s['Ann. Ret']:.1%}")
    print(f"  Ann. Vol    : {s['Ann. Vol']:.1%}")
    print(f"  Sharpe      : {s['Sharpe']:.2f}")
    print(f"  Max DD      : {s['Max DD']:.1%}")
    print(f"  Hit Rate    : {s['Hit Rate']:.1%}")

    print('\nFetching live fundamentals for today\'s ranking...')
    try:
        fundamentals = get_fundamentals()
        ranked = rank_countries(fundamentals)
        print('\nToday\'s Valuation Ranking (cheapest first):')
        for i, (country, row) in enumerate(ranked.iterrows(), 1):
            pe_str = f"PE={row['PE']:.1f}" if pd.notna(row.get('PE')) else 'PE=n/a'
            dy_str = f"DY={row['DY']:.1%}" if pd.notna(row.get('DY')) else 'DY=n/a'
            flag = 'LONG ' if i <= LONG_N else ('SHORT' if i > len(ranked) - SHORT_N else '     ')
            em = '[EM]' if country in EMERGING else '    '
            print(f'  #{i:2d}  {country:12s}  {pe_str:10s}  {dy_str:8s}  {flag} {em}')
    except Exception as e:
        print(f'  fundamentals fetch failed: {e}')
        fundamentals = None

    plot(prices, port, equity, signal_y, fundamentals)

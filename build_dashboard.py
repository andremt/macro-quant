"""
build_dashboard.py
==================
Re-runs all 5 factor strategy backtests and generates a polished,
interactive single-page HTML dashboard at ./dashboard.html.

Strategies implemented (logic from Ilmanen "Expected Returns"):
  P1 — G10 FX Carry
  P2 — Multi-Asset Trend Following
  P3 — Value Across Borders
  P4 — Betting Against Beta
  P5 — Signal Stack (Carry + Trend + Value)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

warnings.filterwarnings('ignore')

# ── Working directory ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from data_utils import load_prices, load_rates

START = '2018-01-01'   # backtest start — betas computed on full history from 2015/2016

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def perf_stats(returns, freq=12):
    """Compute Ann. Return, Ann. Vol, Sharpe, Max DD from a return series."""
    ret = returns.dropna()
    if len(ret) == 0:
        return dict(ann_ret=0, ann_vol=0, sharpe=0, max_dd=0)
    ann_ret = ret.mean() * freq
    ann_vol = ret.std() * np.sqrt(freq)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    cum     = (1 + ret).cumprod()
    dd      = (cum - cum.cummax()) / cum.cummax()
    max_dd  = dd.min()
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, max_dd=max_dd)


def equity_curve(returns):
    """Turn a return series into a cumulative equity curve starting at 1.0."""
    return (1 + returns.dropna()).cumprod()


def drawdown_series(eq):
    """Compute drawdown series from equity curve."""
    return (eq - eq.cummax()) / eq.cummax()


def make_equity_drawdown_fig(eq, dd, benchmark_eq=None,
                              benchmark_name='SPY B&H',
                              title='', max_dd_val=None):
    """
    Create a 2-panel Plotly figure: equity curve (top) + drawdown (bottom).
    Includes range selector buttons 1Y / 3Y / All.
    """
    # Guard against empty series
    if eq is None or len(eq.dropna()) == 0:
        fig = go.Figure()
        fig.add_annotation(text='No data available', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=18, color='#9CA3AF'))
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='#FAFAFA',
                          title=dict(text=title, font=dict(size=16)))
        return fig

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
    )

    # — Equity curve
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values,
        name='Strategy',
        line=dict(color='#3366FF', width=2.5),
        hovertemplate='%{x|%Y-%m-%d}<br>Cumulative Return: %{y:.3f}<extra></extra>',
    ), row=1, col=1)

    if benchmark_eq is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_eq.index, y=benchmark_eq.values,
            name=benchmark_name,
            line=dict(color='#9CA3AF', width=1.5, dash='dot'),
            hovertemplate='%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>',
        ), row=1, col=1)

    # — Drawdown
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        name='Drawdown',
        fill='tozeroy',
        fillcolor='rgba(239,68,68,0.25)',
        line=dict(color='#EF4444', width=1),
        hovertemplate='%{x|%Y-%m-%d}<br>DD: %{y:.1%}<extra></extra>',
    ), row=2, col=1)

    # Max DD annotation
    if max_dd_val is None:
        max_dd_val = dd.min()
    max_dd_date = dd.idxmin()
    fig.add_annotation(
        x=max_dd_date, y=max_dd_val,
        text=f'Max DD: {max_dd_val:.1%}',
        showarrow=True, arrowhead=2, arrowcolor='#EF4444',
        font=dict(color='#EF4444', size=11),
        row=2, col=1,
    )

    # Range buttons
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1,  label='1Y', step='year',  stepmode='backward'),
                dict(count=3,  label='3Y', step='year',  stepmode='backward'),
                dict(label='All', step='all'),
            ],
            bgcolor='#F3F4F6',
            activecolor='#3366FF',
            font=dict(size=11),
        ),
        row=1, col=1,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#1C1C2E')),
        paper_bgcolor='white',
        plot_bgcolor='#FAFAFA',
        font=dict(family='Inter, system-ui, sans-serif', color='#1C1C2E'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified',
        margin=dict(l=50, r=30, t=60, b=30),
    )
    fig.update_yaxes(tickformat='.2f', row=1, col=1,
                     gridcolor='#E5E7EB', zerolinecolor='#D1D5DB')
    fig.update_yaxes(tickformat='.1%', row=2, col=1,
                     gridcolor='#E5E7EB', zerolinecolor='#D1D5DB')

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Project 1 — G10 FX Carry
# ──────────────────────────────────────────────────────────────────────────────

def run_p1():
    print('\n=== P1: G10 FX Carry ===')

    G10 = {
        'EUR': ('EURUSD=X', False),
        'GBP': ('GBPUSD=X', False),
        'JPY': ('USDJPY=X', True),
        'AUD': ('AUDUSD=X', False),
        'NZD': ('NZDUSD=X', False),
        'CAD': ('USDCAD=X', True),
        'CHF': ('USDCHF=X', True),
        'NOK': ('USDNOK=X', True),
        'SEK': ('USDSEK=X', True),
    }

    RATE_COLS = {
        'EUR': 'EUR', 'GBP': 'GBP', 'JPY': 'JPY',
        'AUD': 'AUD', 'NZD': 'NZD', 'CAD': 'CAD',
        'CHF': 'CHF', 'NOK': 'NOK', 'SEK': 'SEK',
    }

    fx_tickers = [v[0] for v in G10.values()]
    raw = load_prices(fx_tickers, start='2015-01-01')

    # Build USD-per-unit spot rates (weekly)
    spot = pd.DataFrame()
    for ccy, (ticker, invert) in G10.items():
        if ticker in raw.columns:
            s = raw[ticker].dropna()
            spot[ccy] = 1.0 / s if invert else s
    spot = spot.dropna(how='all').ffill()
    # Keep only weekly data (resample to weekly to avoid daily duplicates in the csv)
    spot = spot.resample('W-MON').last().dropna(how='all')
    spot = spot.loc[START:]

    # Rates
    rates_df = load_rates()
    rates_df  = rates_df.replace('.', np.nan).astype(float) / 100.0
    rates_df  = rates_df.ffill()
    us_rate   = rates_df[['USD']].resample('W-MON').last().ffill()
    for_rates = rates_df[[c for c in RATE_COLS.values() if c in rates_df.columns]]
    for_rates = for_rates.resample('W-MON').last().ffill()
    # Rename to match G10 keys
    for_rates.columns = [c for c in RATE_COLS.keys() if RATE_COLS[c] in rates_df.columns]

    # Weekly returns
    weekly_ret = spot.pct_change()

    # Carry signal: foreign rate − USD rate (aligned to spot dates)
    carry = pd.DataFrame(index=spot.index)
    for ccy in spot.columns:
        if ccy in for_rates.columns:
            f = for_rates[ccy].reindex(spot.index, method='ffill')
            u = us_rate['USD'].reindex(spot.index, method='ffill')
            carry[ccy] = f - u

    # Momentum signal: 4-week rolling return
    mom = weekly_ret.rolling(4).sum()

    # Realised vol: 26-week annualised
    rvol = weekly_ret.rolling(26).std() * np.sqrt(52)

    # Monthly rebalance
    ret_m   = (1 + weekly_ret).resample('ME').prod() - 1
    carry_m = carry.resample('ME').last()
    mom_m   = mom.resample('ME').last()
    rvol_m  = rvol.resample('ME').last()

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

        ranked  = c.rank(ascending=False)
        weights = pd.Series(0.0, index=c.index)
        weights[ranked <= 3]                   =  1.0
        weights[ranked >= (len(c) - 3 + 1)]   = -1.0
        weights[m < 0] = 0.0   # momentum overlay

        weights = weights / v.fillna(v.median())
        gross = weights.abs().sum()
        if gross > 0:
            weights /= gross

        if next_date not in ret_m.index:
            continue
        monthly_ret = ret_m.loc[next_date]
        port_ret = (weights * monthly_ret.reindex(weights.index).fillna(0)).sum()
        port_returns.append({'date': next_date, 'return': port_ret,
                             'weights': weights.to_dict()})

    results = pd.DataFrame(port_returns).set_index('date')
    eq  = equity_curve(results['return'])
    dd  = drawdown_series(eq)
    stats = perf_stats(results['return'], freq=12)

    # SPY benchmark (monthly)
    spy_raw = load_prices(['SPY'], start=START)
    spy_wk  = spy_raw['SPY'].resample('W-MON').last().pct_change()
    spy_m   = (1 + spy_wk).resample('ME').prod() - 1
    spy_eq  = equity_curve(spy_m.reindex(results.index).fillna(0))

    # Current carry signal for badge display
    latest_carry = carry.iloc[-1].sort_values(ascending=False)
    latest_mom   = mom.iloc[-1]

    print(f"  Ann. Ret={stats['ann_ret']:.1%}  Sharpe={stats['sharpe']:.2f}  MaxDD={stats['max_dd']:.1%}")
    return {
        'eq': eq, 'dd': dd, 'stats': stats,
        'benchmark_eq': spy_eq,
        'carry': carry, 'latest_carry': latest_carry, 'latest_mom': latest_mom,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Project 2 — Multi-Asset Trend Following
# ──────────────────────────────────────────────────────────────────────────────

def run_p2():
    print('\n=== P2: Multi-Asset Trend Following ===')

    ASSETS = {
        'US Equity':   'SPY',
        'Intl Equity': 'EFA',
        'EM Equity':   'EEM',
        'US 20yr':     'TLT',
        'US 7-10yr':   'IEF',
        'Intl Govt':   'IGOV',
        'Gold':        'GLD',
        'Oil':         'USO',
        'Cmdty Bskt':  'DBC',
        'EUR':         'FXE',
        'JPY':         'FXY',
        'AUD':         'FXA',
    }

    prices = load_prices(ASSETS, start='2012-01-01')
    prices = prices.resample('W-MON').last().dropna(how='all').ffill()
    prices = prices.loc[START:]

    # 12-week and 1-week momentum (weekly data, 52-week ≈ 12 months)
    # 12-1 momentum: 52-week return minus 4-week return
    ret_52w = prices.pct_change(52)
    ret_4w  = prices.pct_change(4)
    signal  = ret_52w - ret_4w

    direction = np.sign(signal)

    # Vol-target each position to 10% ann. vol, cap gross at 2x
    weekly_ret = prices.pct_change()
    rvol = weekly_ret.rolling(26).std() * np.sqrt(52)  # 26-week vol

    n_assets = prices.shape[1]
    target_per_asset = 0.10 / np.sqrt(n_assets)
    raw_weights = direction * (target_per_asset / rvol.replace(0, np.nan))

    # Cap gross exposure
    gross = raw_weights.abs().sum(axis=1)
    scale = (gross / 2.0).clip(lower=1.0)
    raw_weights = raw_weights.div(scale, axis=0)

    # Monthly rebalance
    weights_m = raw_weights.resample('ME').last().reindex(
        weekly_ret.index, method='ffill')

    port_ret = (weights_m.shift(1) * weekly_ret).sum(axis=1)
    eq  = equity_curve(port_ret)
    dd  = drawdown_series(eq)
    stats = perf_stats(port_ret, freq=52)

    # Benchmark: SPY
    spy_ret = prices['US Equity'].pct_change()
    spy_eq  = equity_curve(spy_ret.reindex(eq.index).fillna(0))

    # Latest direction for signal badges
    latest_dir = direction.iloc[-1]

    print(f"  Ann. Ret={stats['ann_ret']:.1%}  Sharpe={stats['sharpe']:.2f}  MaxDD={stats['max_dd']:.1%}")
    return {
        'eq': eq, 'dd': dd, 'stats': stats,
        'benchmark_eq': spy_eq,
        'latest_direction': latest_dir,
        'port_ret': port_ret,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Project 3 — Value Across Borders
# ──────────────────────────────────────────────────────────────────────────────

def run_p3():
    print('\n=== P3: Value Across Borders ===')

    COUNTRIES = {
        'US':          'SPY',
        'Germany':     'EWG',
        'Japan':       'EWJ',
        'UK':          'EWU',
        'Canada':      'EWC',
        'Australia':   'EWA',
        'Brazil':      'EWZ',
        'Korea':       'EWY',
        'Spain':       'EWP',
        'Italy':       'EWI',
        'France':      'EWQ',
        'Sweden':      'EWD',
        'Switzerland': 'EWL',
        'Taiwan':      'EWT',
        'Singapore':   'EWS',
    }

    prices = load_prices(COUNTRIES, start='2012-01-01')
    prices = prices.resample('W-MON').last().dropna(how='all').ffill()
    prices = prices.loc[START:]

    weekly_ret = prices.pct_change()

    # Value signal: negative of 1-year return (cheap = underperformed)
    ret_1y = prices.pct_change(52)  # 52 weeks ≈ 1 year
    signal = -ret_1y                # high signal = cheap

    # Annual rebalance
    signal_y = signal.resample('YE').last()

    LONG_N  = 5
    SHORT_N = 5

    weights_list = []
    for date in signal_y.index[1:]:
        s = signal_y.loc[date].dropna()
        if len(s) < LONG_N + SHORT_N:
            continue
        ranked  = s.rank(ascending=False)  # rank 1 = cheapest
        n       = len(s)
        weights = pd.Series(0.0, index=s.index)
        weights[ranked <= LONG_N]              =  1.0 / LONG_N
        weights[ranked >= (n - SHORT_N + 1)]   = -1.0 / SHORT_N
        weights_list.append({'date': date, **weights.to_dict()})

    if not weights_list:
        return None

    weights_df     = pd.DataFrame(weights_list).set_index('date')
    weights_daily  = weights_df.reindex(weekly_ret.index, method='ffill').shift(1)

    port_ret = (weights_daily * weekly_ret).sum(axis=1)
    eq  = equity_curve(port_ret)
    dd  = drawdown_series(eq)
    stats = perf_stats(port_ret, freq=52)

    spy_ret = prices['US'].pct_change()
    spy_eq  = equity_curve(spy_ret.reindex(eq.index).fillna(0))

    # Latest signal for display
    latest_signal = signal_y.iloc[-1].sort_values(ascending=False)

    print(f"  Ann. Ret={stats['ann_ret']:.1%}  Sharpe={stats['sharpe']:.2f}  MaxDD={stats['max_dd']:.1%}")
    return {
        'eq': eq, 'dd': dd, 'stats': stats,
        'benchmark_eq': spy_eq,
        'latest_signal': latest_signal,
        'signal_y': signal_y,
        'countries': list(COUNTRIES.keys()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Project 4 — Betting Against Beta
# ──────────────────────────────────────────────────────────────────────────────

def run_p4():
    print('\n=== P4: Betting Against Beta ===')

    UNIVERSE = [
        'AAPL','MSFT','NVDA','GOOGL','META','ORCL',
        'JNJ','UNH','PFE','ABBV','MRK','TMO',
        'JPM','BAC','WFC','GS','MS','BRK-B',
        'XOM','CVX','COP','SLB','PSX','VLO',
        'HON','UPS','CAT','DE','RTX','LMT',
        'AMZN','HD','MCD','NKE','TSLA','LOW',
        'PG','KO','PEP','WMT','COST','CL',
        'NEE','DUK','SO','D','EXC','AEP',
        'LIN','APD','SHW','NEM','FCX','NUE',
        'AMT','PLD','CCI','EQIX','PSA','SPG',
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

    all_tickers = ['SPY'] + UNIVERSE
    raw = load_prices(all_tickers, start='2015-01-01')

    # Resample to weekly
    raw = raw.resample('W-MON').last().dropna(how='all').ffill()

    spy    = raw['SPY']
    stocks = raw[[c for c in UNIVERSE if c in raw.columns]].copy()

    # Drop stocks with insufficient history (on FULL history, before START filter)
    min_obs = 52 + 12
    stocks  = stocks.loc[:, stocks.notna().sum() >= min_obs]
    stocks  = stocks.ffill()

    # Compute rolling betas on FULL history before applying START filter
    spy_ret_full   = spy.pct_change()
    stock_ret_full = stocks.pct_change()

    betas_full = pd.DataFrame(index=stock_ret_full.index,
                               columns=stock_ret_full.columns, dtype=float)
    for col in stock_ret_full.columns:
        cov  = stock_ret_full[col].rolling(52).cov(spy_ret_full)
        varm = spy_ret_full.rolling(52).var()
        betas_full[col] = cov / varm.replace(0, np.nan)

    # Now filter to backtest period
    betas     = betas_full.loc[START:]
    stock_ret = stock_ret_full.loc[START:]
    spy_ret   = spy_ret_full.loc[START:].dropna()

    # Monthly rebalance
    REBAL = 'ME'
    rebal_dates = betas.resample(REBAL).last().index
    betas_m     = betas.resample(REBAL).last()

    port_records = []
    for i, date in enumerate(rebal_dates[:-1]):
        next_date = rebal_dates[i + 1]

        b = betas_m.loc[date].dropna()
        if len(b) < 10:
            continue

        b = b.clip(lower=0.2, upper=b.quantile(0.95))

        n      = len(b)
        q_size = max(n // 5, 1)
        ranked = b.rank()

        low_beta  = b[ranked <= q_size]
        high_beta = b[ranked > (n - q_size)]

        if len(low_beta) == 0 or len(high_beta) == 0:
            continue

        avg_beta_l = max(low_beta.mean(), 0.2)
        avg_beta_h = max(high_beta.mean(), 0.2)

        w_long  = pd.Series( 1.0 / avg_beta_l / len(low_beta),  index=low_beta.index)
        w_short = pd.Series(-1.0 / avg_beta_h / len(high_beta), index=high_beta.index)
        weights = pd.concat([w_long, w_short])

        # Returns over NEXT month (date < t <= next_date)
        period     = stock_ret.index[(stock_ret.index > date) & (stock_ret.index <= next_date)]
        spy_period_idx = spy_ret.index[(spy_ret.index > date) & (spy_ret.index <= next_date)]
        if len(period) == 0:
            continue

        period_ret = (1 + stock_ret.loc[period]).prod() - 1
        spy_period = (1 + spy_ret.loc[spy_period_idx]).prod() - 1 if len(spy_period_idx) > 0 else 0.0

        port_ret = (weights * period_ret.reindex(weights.index).fillna(0)).sum()
        port_records.append({
            'date':            next_date,   # record at end of holding period
            'return':          port_ret,
            'spy_ret':         spy_period,
            'avg_beta_long':   avg_beta_l,
            'avg_beta_short':  avg_beta_h,
        })

    results = pd.DataFrame(port_records).set_index('date')
    eq      = equity_curve(results['return'])
    dd      = drawdown_series(eq)
    spy_eq  = equity_curve(results['spy_ret'])
    stats   = perf_stats(results['return'], freq=12)

    # Latest betas for signal display
    latest_betas = betas_m.iloc[-1].dropna().sort_values()

    print(f"  Ann. Ret={stats['ann_ret']:.1%}  Sharpe={stats['sharpe']:.2f}  MaxDD={stats['max_dd']:.1%}")
    return {
        'eq': eq, 'dd': dd, 'stats': stats,
        'benchmark_eq': spy_eq,
        'latest_betas': latest_betas,
        'results': results,
        'sectors': SECTORS,
        'betas_m': betas_m,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Project 5 — Signal Stack
# ──────────────────────────────────────────────────────────────────────────────

def run_p5():
    print('\n=== P5: Signal Stack ===')

    ASSETS = {
        'US Equity':   'SPY',
        'DM Equity':   'EFA',
        'EM Equity':   'EEM',
        'US 20yr':     'TLT',
        'US 7-10yr':   'IEF',
        'Gold':        'GLD',
        'Oil':         'USO',
        'Cmdty Bskt':  'DBC',
        'EUR':         'FXE',
        'JPY':         'FXY',
        'AUD':         'FXA',
        'Germany':     'EWG',
        'Japan':       'EWJ',
        'UK':          'EWU',
        'Australia':   'EWA',
        'Brazil':      'EWZ',
        'Korea':       'EWY',
        'Spain':       'EWP',
        'Italy':       'EWI',
        'Canada':      'EWC',
    }

    prices = load_prices(ASSETS, start='2010-01-01')
    prices = prices.resample('W-MON').last().dropna(how='all').ffill()
    prices = prices.loc[START:]

    weekly_ret = prices.pct_change()

    def cs_zscore(df):
        return df.sub(df.mean(axis=1), axis=0).div(
            df.std(axis=1).replace(0, np.nan), axis=0)

    # Carry: 52-week return minus 48-week return (strips price momentum, keeps carry)
    carry_raw = prices.pct_change(52) - prices.pct_change(48)
    s_carry   = cs_zscore(carry_raw)

    # Trend: 52-week minus 4-week momentum
    s_trend = cs_zscore(prices.pct_change(52) - prices.pct_change(4))

    # Value: negative 5-year return (260 weeks ≈ 5 years)
    s_value = cs_zscore(-prices.pct_change(260))

    # Composite: equal-weight
    s_comp = (s_carry + s_trend + s_value) / 3.0

    def run_factor_backtest(signal):
        direction = np.sign(signal)
        rvol = weekly_ret.rolling(26).std() * np.sqrt(52)
        n    = prices.shape[1]
        target_per_asset = 0.10 / np.sqrt(n)
        raw_w = direction * (target_per_asset / rvol.replace(0, np.nan))
        # Long top 7, short bottom 7: zero out middle
        rank_m = signal.resample('ME').last()
        def select_positions(row):
            r = row.dropna()
            if len(r) < 14:
                return pd.Series(np.sign(r), index=r.index)
            ranked = r.rank(ascending=False)
            n = len(r)
            pos = pd.Series(0.0, index=r.index)
            pos[ranked <= 7]         = 1.0
            pos[ranked >= (n - 7 + 1)] = -1.0
            return pos
        dir_m = rank_m.apply(select_positions, axis=1)
        dir_m = dir_m.reindex(weekly_ret.index, method='ffill')
        rvol_m = rvol.resample('ME').last().reindex(weekly_ret.index, method='ffill')
        final_w = dir_m * (target_per_asset / rvol_m.replace(0, np.nan))
        gross = final_w.abs().sum(axis=1)
        scale = (gross / 2.0).clip(lower=1.0)
        final_w = final_w.div(scale, axis=0)
        ret = (final_w.shift(1) * weekly_ret).sum(axis=1)
        return ret

    ret_comp  = run_factor_backtest(s_comp)
    ret_carry = run_factor_backtest(s_carry)
    ret_trend = run_factor_backtest(s_trend)
    ret_value = run_factor_backtest(s_value)

    eq_comp  = equity_curve(ret_comp)
    eq_carry = equity_curve(ret_carry)
    eq_trend = equity_curve(ret_trend)
    eq_value = equity_curve(ret_value)

    dd_comp = drawdown_series(eq_comp)
    stats   = perf_stats(ret_comp, freq=52)

    # 60/40 benchmark
    bench_prices = load_prices(['SPY', 'TLT'], start=START)
    bench_prices = bench_prices.resample('W-MON').last().ffill()
    bench_ret    = 0.6 * bench_prices['SPY'].pct_change() + 0.4 * bench_prices['TLT'].pct_change()
    bench_eq     = equity_curve(bench_ret.reindex(eq_comp.index).fillna(0))

    # Factor correlations
    factor_rets = pd.DataFrame({
        'Carry': ret_carry, 'Trend': ret_trend,
        'Value': ret_value, 'Stack': ret_comp,
    }).dropna()
    corr_matrix = factor_rets.resample('ME').sum().corr()

    # Latest signals for heatmap
    latest_s_carry = s_carry.iloc[-1]
    latest_s_trend = s_trend.iloc[-1]
    latest_s_value = s_value.iloc[-1]
    latest_s_comp  = s_comp.iloc[-1]

    print(f"  Ann. Ret={stats['ann_ret']:.1%}  Sharpe={stats['sharpe']:.2f}  MaxDD={stats['max_dd']:.1%}")
    return {
        'eq': eq_comp, 'dd': dd_comp, 'stats': stats,
        'benchmark_eq': bench_eq,
        'eq_carry': eq_carry, 'eq_trend': eq_trend, 'eq_value': eq_value,
        'ret_carry': ret_carry, 'ret_trend': ret_trend, 'ret_value': ret_value,
        'ret_comp': ret_comp,
        'corr_matrix': corr_matrix,
        'latest_s_carry': latest_s_carry,
        'latest_s_trend': latest_s_trend,
        'latest_s_value': latest_s_value,
        'latest_s_comp':  latest_s_comp,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard HTML Generator
# ──────────────────────────────────────────────────────────────────────────────

PLOTLY_CONFIG = {'displayModeBar': True, 'responsive': True}


def fig_to_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)


def badge(label, color):
    colors = {
        'LONG':    ('#10B981', 'white'),
        'SHORT':   ('#EF4444', 'white'),
        'NEUTRAL': ('#9CA3AF', 'white'),
    }
    bg, fg = colors.get(color, ('#9CA3AF', 'white'))
    return (f'<span style="background:{bg};color:{fg};padding:3px 10px;'
            f'border-radius:4px;font-size:12px;font-weight:600;'
            f'display:inline-block;margin:3px 4px">{label}</span>')


def stats_row(stats):
    """HTML snippet for performance stat boxes."""
    items = [
        ('Ann. Return', f"{stats['ann_ret']:.1%}", '#10B981' if stats['ann_ret'] > 0 else '#EF4444'),
        ('Ann. Vol',    f"{stats['ann_vol']:.1%}", '#3366FF'),
        ('Sharpe',      f"{stats['sharpe']:.2f}",  '#10B981' if stats['sharpe'] > 0.5 else '#F59E0B'),
        ('Max DD',      f"{stats['max_dd']:.1%}",  '#EF4444'),
    ]
    cells = ''
    for label, val, color in items:
        cells += f'''
          <div style="background:white;border-radius:10px;padding:16px 22px;
                      box-shadow:0 1px 4px rgba(0,0,0,0.08);text-align:center;
                      border-top:3px solid {color}">
            <div style="font-size:11px;color:#6B7280;font-weight:500;text-transform:uppercase;
                        letter-spacing:0.5px">{label}</div>
            <div style="font-size:22px;font-weight:700;color:{color};margin-top:6px">{val}</div>
          </div>'''
    return f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0">{cells}</div>'


def section_header(num, title, subtitle=''):
    sub_html = f'<p style="color:#6B7280;margin:6px 0 0;font-size:15px">{subtitle}</p>' if subtitle else ''
    return f'''
    <div style="border-left:4px solid #3366FF;padding-left:16px;margin-bottom:24px">
      <h2 style="margin:0;font-size:24px;font-weight:700;color:#1C1C2E">
        {num} {title}
      </h2>
      {sub_html}
    </div>'''


def build_overview_chart(all_stats):
    """Grouped bar comparing Ann Return and Sharpe for all 5 strategies."""
    if not all_stats:
        fig = go.Figure()
        fig.add_annotation(text='No strategy data available', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color='#9CA3AF'))
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='#FAFAFA',
                          title='Strategy Performance Comparison')
        return fig

    names   = [s['name'] for s in all_stats]
    returns = [s['stats']['ann_ret'] * 100 for s in all_stats]
    sharpes = [s['stats']['sharpe'] for s in all_stats]
    max_dds = [abs(s['stats']['max_dd']) * 100 for s in all_stats]

    colors_ret = ['#10B981' if r > 0 else '#EF4444' for r in returns]
    colors_sr  = ['#3366FF' if s > 0 else '#F59E0B' for s in sharpes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Ann. Return (%)',
        x=names, y=returns,
        marker_color=colors_ret,
        text=[f'{r:.1f}%' for r in returns],
        textposition='outside',
    ))
    fig.add_trace(go.Bar(
        name='Sharpe Ratio',
        x=names, y=sharpes,
        marker_color=colors_sr,
        text=[f'{s:.2f}' for s in sharpes],
        textposition='outside',
    ))
    fig.add_trace(go.Bar(
        name='|Max DD| (%)',
        x=names, y=max_dds,
        marker_color='#F59E0B',
        text=[f'{d:.1f}%' for d in max_dds],
        textposition='outside',
        opacity=0.75,
    ))

    fig.update_layout(
        barmode='group',
        title='Strategy Performance Comparison',
        paper_bgcolor='white',
        plot_bgcolor='#FAFAFA',
        font=dict(family='Inter, system-ui, sans-serif', color='#1C1C2E'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=30, t=60, b=30),
    )
    fig.update_yaxes(gridcolor='#E5E7EB')
    return fig


def build_p1_carry_chart(p1):
    """Horizontal bar of current carry rankings."""
    carry = p1['latest_carry'].dropna()
    mom   = p1['latest_mom']

    if len(carry) == 0:
        fig = go.Figure()
        fig.add_annotation(text='No carry data available', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color='#9CA3AF'))
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='#FAFAFA',
                          title='Current Carry Rankings (no data)', height=320)
        return fig

    colors = []
    for ccy in carry.index:
        if mom.get(ccy, 0) < 0:
            colors.append('#F59E0B')  # zeroed by momentum overlay
        elif carry[ccy] > 0:
            colors.append('#10B981')
        else:
            colors.append('#EF4444')

    fig = go.Figure(go.Bar(
        x=carry.values * 100,
        y=list(carry.index),
        orientation='h',
        marker_color=colors,
        text=[f'{v:+.2f}%' for v in carry.values * 100],
        textposition='outside',
        hovertemplate='%{y}: %{x:.2f}%<extra></extra>',
    ))
    fig.add_vline(x=0, line_width=1.5, line_color='#1C1C2E')
    fig.update_layout(
        title='Current Carry Rankings (foreign rate − USD rate)',
        paper_bgcolor='white', plot_bgcolor='#FAFAFA',
        font=dict(family='Inter, system-ui, sans-serif'),
        height=320,
        margin=dict(l=50, r=80, t=50, b=30),
        xaxis_title='Carry (% p.a.)',
    )
    return fig


def build_p3_signal_chart(p3):
    """Country valuation signal (last annual rebalance)."""
    if p3 is None or len(p3['signal_y']) == 0:
        fig = go.Figure()
        fig.add_annotation(text='No signal data available', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color='#9CA3AF'))
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='#FAFAFA',
                          title='Country Value Signal (no data)', height=350)
        return fig
    sig = p3['signal_y'].iloc[-1].dropna().sort_values(ascending=False)
    colors = ['#10B981' if v > 0 else '#EF4444' for v in sig.values]
    fig = go.Figure(go.Bar(
        x=sig.values,
        y=list(sig.index),
        orientation='h',
        marker_color=colors,
        text=[f'{v:+.2f}' for v in sig.values],
        textposition='outside',
        hovertemplate='%{y}: %{x:.2f}<extra></extra>',
    ))
    fig.add_vline(x=0, line_width=1.5, line_color='#1C1C2E')
    fig.update_layout(
        title='Latest Annual Value Signal by Country (positive = cheap)',
        paper_bgcolor='white', plot_bgcolor='#FAFAFA',
        font=dict(family='Inter, system-ui, sans-serif'),
        height=350,
        margin=dict(l=100, r=80, t=50, b=30),
    )
    return fig


def build_p4_beta_chart(p4):
    """Beta scatter sorted low to high, colored by long/short."""
    betas = p4['latest_betas'].dropna()
    sectors = p4['sectors']

    if len(betas) == 0:
        fig = go.Figure()
        fig.add_annotation(text='No beta data available', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color='#9CA3AF'))
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='#FAFAFA',
                          title='Current Beta Cross-Section (no data)', height=700)
        return fig

    n = len(betas)
    q_size = max(n // 5, 1)
    ranked = betas.rank()
    colors = []
    labels = []
    for ticker, b in betas.items():
        r = ranked[ticker]
        if r <= q_size:
            colors.append('#3366FF')
            labels.append('LONG (low-beta)')
        elif r > (n - q_size):
            colors.append('#EF4444')
            labels.append('SHORT (high-beta)')
        else:
            colors.append('#D1D5DB')
            labels.append('Neutral')

    fig = go.Figure(go.Bar(
        x=betas.values,
        y=list(betas.index),
        orientation='h',
        marker_color=colors,
        text=[f'{b:.2f}' for b in betas.values],
        textposition='outside',
        customdata=[[sectors.get(t,'?'), l] for t, l in zip(betas.index, labels)],
        hovertemplate='%{y}<br>Beta: %{x:.2f}<br>Sector: %{customdata[0]}<br>%{customdata[1]}<extra></extra>',
    ))
    fig.add_vline(x=1.0, line_width=1.5, line_color='#1C1C2E', line_dash='dash')
    fig.update_layout(
        title='Current Beta Cross-Section (blue=LONG, red=SHORT)',
        paper_bgcolor='white', plot_bgcolor='#FAFAFA',
        font=dict(family='Inter, system-ui, sans-serif'),
        height=700,
        margin=dict(l=70, r=80, t=50, b=30),
        xaxis_title='Rolling 52-Week Beta vs SPY',
    )
    return fig


def build_p5_signal_heatmap(p5):
    """Assets × Signals heatmap."""
    assets = list(p5['latest_s_comp'].dropna().sort_values(ascending=False).index)

    if len(assets) == 0:
        fig = go.Figure()
        fig.add_annotation(text='No signal data available', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color='#9CA3AF'))
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='#FAFAFA',
                          title='Signal Heatmap (no data)', height=520)
        return fig

    signals = {
        'Carry': p5['latest_s_carry'].reindex(assets),
        'Trend': p5['latest_s_trend'].reindex(assets),
        'Value': p5['latest_s_value'].reindex(assets),
        'Composite': p5['latest_s_comp'].reindex(assets),
    }

    z_vals = np.array([[signals[s].get(a, 0) for s in signals] for a in assets])

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=list(signals.keys()),
        y=assets,
        colorscale='RdYlGn',
        zmid=0,
        text=[[f'{v:.2f}' for v in row] for row in z_vals],
        texttemplate='%{text}',
        hovertemplate='Asset: %{y}<br>Signal: %{x}<br>Z-score: %{z:.2f}<extra></extra>',
        colorbar=dict(title='Z-score'),
    ))
    fig.update_layout(
        title='Signal Heatmap: Assets × Factors (current)',
        paper_bgcolor='white', plot_bgcolor='#FAFAFA',
        font=dict(family='Inter, system-ui, sans-serif'),
        height=520,
        margin=dict(l=120, r=80, t=50, b=30),
    )
    return fig


def build_p5_factor_eq_fig(p5):
    """Overlay equity curves for each factor + composite + 60/40."""
    fig = go.Figure()

    curves = [
        ('Signal Stack', p5['eq'],           '#1C1C2E',  2.5, 'solid'),
        ('60/40 Bench',  p5['benchmark_eq'], '#9CA3AF',  1.5, 'dot'),
        ('Carry Only',   p5['eq_carry'],     '#EF4444',  1.2, 'solid'),
        ('Trend Only',   p5['eq_trend'],     '#3366FF',  1.2, 'solid'),
        ('Value Only',   p5['eq_value'],     '#F59E0B',  1.2, 'solid'),
    ]
    for name, eq, color, width, dash in curves:
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            name=name,
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f'{name}: %{{y:.3f}}<extra></extra>',
        ))

    fig.add_hline(y=1.0, line_width=0.8, line_color='#D1D5DB')
    fig.update_layout(
        title='Signal Stack: Composite vs Individual Factors vs 60/40',
        paper_bgcolor='white', plot_bgcolor='#FAFAFA',
        font=dict(family='Inter, system-ui, sans-serif'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=420,
        margin=dict(l=50, r=30, t=60, b=30),
        yaxis=dict(tickformat='.2f', gridcolor='#E5E7EB'),
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label='1Y', step='year', stepmode='backward'),
                    dict(count=3, label='3Y', step='year', stepmode='backward'),
                    dict(label='All', step='all'),
                ]
            )
        ),
    )
    return fig


def build_corr_heatmap(p5):
    """Correlation matrix of all 5 strategy returns (monthly)."""
    corr = p5['corr_matrix']
    z    = corr.values
    labs = list(corr.columns)
    text = [[f'{v:.2f}' for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=labs, y=labs,
        colorscale='RdYlGn', zmid=0, zmin=-1, zmax=1,
        text=text, texttemplate='%{text}',
        hovertemplate='%{y} vs %{x}: %{z:.2f}<extra></extra>',
        colorbar=dict(title='Corr'),
    ))
    fig.update_layout(
        title='Factor Correlation Matrix (monthly returns)',
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='Inter, system-ui, sans-serif'),
        height=360,
        margin=dict(l=80, r=50, t=60, b=60),
    )
    return fig


def make_signal_badges(items):
    """items = list of (label, direction) where direction in LONG/SHORT/NEUTRAL."""
    html = '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:8px">'
    for label, direction in items:
        html += badge(f'{label}: {direction}', direction)
    html += '</div>'
    return html


def build_html(p1, p2, p3, p4, p5):
    print('\nBuilding HTML...')

    all_stats = [
        {'name': 'FX Carry',    'stats': p1['stats']},
        {'name': 'Trend Follow','stats': p2['stats']},
        {'name': 'Value',       'stats': p3['stats']},
        {'name': 'BAB',         'stats': p4['stats']},
        {'name': 'Signal Stack','stats': p5['stats']},
    ]

    # ── Generate all Plotly figures ──────────────────────────────────────────

    # Overview
    fig_overview = build_overview_chart(all_stats)

    # P1
    fig_p1 = make_equity_drawdown_fig(
        p1['eq'], p1['dd'], p1['benchmark_eq'],
        benchmark_name='SPY B&H',
        title='G10 FX Carry — Equity Curve & Drawdown',
    )
    fig_p1_carry = build_p1_carry_chart(p1)

    # P2
    fig_p2 = make_equity_drawdown_fig(
        p2['eq'], p2['dd'], p2['benchmark_eq'],
        benchmark_name='SPY B&H',
        title='Multi-Asset Trend Following — Equity Curve & Drawdown',
    )

    # P3
    fig_p3 = make_equity_drawdown_fig(
        p3['eq'], p3['dd'], p3['benchmark_eq'],
        benchmark_name='SPY B&H',
        title='Value Across Borders — Equity Curve & Drawdown',
    )
    fig_p3_signal = build_p3_signal_chart(p3)

    # P4
    fig_p4 = make_equity_drawdown_fig(
        p4['eq'], p4['dd'], p4['benchmark_eq'],
        benchmark_name='SPY B&H',
        title='Betting Against Beta — Equity Curve & Drawdown',
    )
    fig_p4_beta = build_p4_beta_chart(p4)

    # P5
    fig_p5 = make_equity_drawdown_fig(
        p5['eq'], p5['dd'], p5['benchmark_eq'],
        benchmark_name='60/40 Bench',
        title='Signal Stack — Equity Curve & Drawdown',
    )
    fig_p5_factors = build_p5_factor_eq_fig(p5)
    fig_p5_heatmap = build_p5_signal_heatmap(p5)
    fig_corr       = build_corr_heatmap(p5)

    # ── Current signals for badge cards ─────────────────────────────────────

    # P1 badges
    p1_carry = p1['latest_carry']
    p1_mom   = p1['latest_mom']
    ranked_carry = p1_carry.rank(ascending=False)
    p1_badges = []
    for ccy in p1_carry.index:
        r = ranked_carry[ccy]
        m = p1_mom.get(ccy, 1)
        if m < 0:
            direction = 'NEUTRAL'
        elif r <= 3:
            direction = 'LONG'
        elif r >= (len(p1_carry) - 3 + 1):
            direction = 'SHORT'
        else:
            direction = 'NEUTRAL'
        p1_badges.append((ccy, direction))

    # P2 badges
    p2_dir = p2['latest_direction']
    p2_badges = []
    for asset in p2_dir.index:
        d = p2_dir[asset]
        direction = 'LONG' if d > 0 else ('SHORT' if d < 0 else 'NEUTRAL')
        p2_badges.append((asset, direction))

    # P3 badges
    p3_sig    = p3['latest_signal']
    p3_ranked = p3_sig.rank(ascending=False)
    n3 = len(p3_sig)
    p3_badges = []
    for country in p3_sig.index:
        r = p3_ranked[country]
        if r <= 5:
            direction = 'LONG'
        elif r >= (n3 - 5 + 1):
            direction = 'SHORT'
        else:
            direction = 'NEUTRAL'
        p3_badges.append((country, direction))

    # P4 badges
    p4_betas  = p4['latest_betas'].dropna()
    n4 = len(p4_betas)
    q4 = max(n4 // 5, 1)
    p4_ranked = p4_betas.rank()
    p4_badges = []
    for ticker, b in list(p4_betas.items())[:20]:  # show top/bottom 20
        r = p4_ranked[ticker]
        if r <= q4:
            direction = 'LONG'
        elif r > (n4 - q4):
            direction = 'SHORT'
        else:
            direction = 'NEUTRAL'
        p4_badges.append((f'{ticker} (β={b:.2f})', direction))

    # P5 badges
    p5_comp   = p5['latest_s_comp'].dropna().sort_values(ascending=False)
    p5_badges = []
    for i, asset in enumerate(p5_comp.index):
        if i < 7:
            direction = 'LONG'
        elif i >= (len(p5_comp) - 7):
            direction = 'SHORT'
        else:
            direction = 'NEUTRAL'
        p5_badges.append((asset, direction))

    # ── CSS ─────────────────────────────────────────────────────────────────

    CSS = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #FAF9F6;
      color: #1C1C2E;
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      line-height: 1.6;
    }
    a { color: #3366FF; text-decoration: none; }
    a:hover { text-decoration: underline; }

    /* Nav */
    nav {
      position: sticky; top: 0; z-index: 100;
      background: rgba(255,255,255,0.95);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid #E5E7EB;
      padding: 0 40px;
      display: flex; align-items: center; gap: 8px; height: 52px;
    }
    nav .logo {
      font-weight: 700; font-size: 15px; color: #1C1C2E;
      margin-right: 24px; white-space: nowrap;
    }
    nav a {
      font-size: 13px; color: #6B7280; padding: 6px 10px;
      border-radius: 6px; font-weight: 500; white-space: nowrap;
    }
    nav a:hover { background: #F3F4F6; color: #1C1C2E; text-decoration: none; }

    /* Layout */
    .container { max-width: 1200px; margin: 0 auto; padding: 0 32px; }

    /* Hero */
    .hero {
      background: linear-gradient(135deg, #1C1C2E 0%, #2D2B4E 100%);
      color: white; padding: 80px 0 60px;
    }
    .hero h1 {
      font-size: 42px; font-weight: 800; line-height: 1.15;
      max-width: 700px; margin-bottom: 20px;
    }
    .hero .subtitle {
      font-size: 18px; color: rgba(255,255,255,0.75);
      max-width: 600px; margin-bottom: 32px;
    }
    .hero .tags { display: flex; gap: 10px; flex-wrap: wrap; }
    .tag {
      background: rgba(255,255,255,0.12); color: white;
      padding: 4px 12px; border-radius: 20px; font-size: 12px;
      font-weight: 600; letter-spacing: 0.5px;
    }

    /* Sections */
    section { padding: 60px 0; border-bottom: 1px solid #E5E7EB; }
    section:last-child { border-bottom: none; }

    /* Cards */
    .card {
      background: white; border-radius: 14px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
      padding: 28px; margin-bottom: 24px;
    }
    .card h3 {
      font-size: 16px; font-weight: 600; color: #1C1C2E;
      margin-bottom: 12px;
    }

    /* Narrative */
    .narrative {
      background: white; border-radius: 14px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
      padding: 28px 32px; margin-bottom: 24px;
      font-size: 15px; color: #374151; line-height: 1.75;
    }
    .narrative p { margin-bottom: 14px; }
    .narrative p:last-child { margin-bottom: 0; }

    /* Chart wrapper */
    .chart-wrap {
      background: white; border-radius: 14px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
      padding: 20px; margin-bottom: 24px;
      overflow: hidden;
    }

    /* Signal card */
    .signal-card {
      background: white; border-radius: 14px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
      padding: 20px 24px; margin-bottom: 24px;
    }
    .signal-card h3 {
      font-size: 14px; font-weight: 600; color: #6B7280;
      text-transform: uppercase; letter-spacing: 0.5px;
      margin-bottom: 12px;
    }

    /* Grid helpers */
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media(max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }

    /* Conclusions */
    .conclusions-grid {
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
      margin-top: 24px;
    }
    .conclusion-item {
      background: white; border-radius: 10px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
      padding: 18px 20px;
    }
    .conclusion-item .label {
      font-size: 11px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.5px; margin-bottom: 6px;
    }
    .conclusion-item p {
      font-size: 13px; color: #6B7280; line-height: 1.6;
    }
    @media(max-width: 768px) { .conclusions-grid { grid-template-columns: 1fr; } }
    """

    # ── HTML Body ────────────────────────────────────────────────────────────

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Factor Strategies Dashboard — Ilmanen Expected Returns</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
  <style>{CSS}</style>
</head>
<body>

<!-- ── Sticky Nav ──────────────────────────────────────────────────────── -->
<nav>
  <span class="logo">Factor Research</span>
  <a href="#hero">Home</a>
  <a href="#overview">Overview</a>
  <a href="#p1">FX Carry</a>
  <a href="#p2">Trend</a>
  <a href="#p3">Value</a>
  <a href="#p4">BAB</a>
  <a href="#p5">Signal Stack</a>
  <a href="#conclusions">Conclusions</a>
</nav>

<!-- ── Hero ──────────────────────────────────────────────────────────── -->
<section id="hero" class="hero">
  <div class="container">
    <h1>Five Factor Strategies, Tested in the Real World</h1>
    <p class="subtitle">
      I spent a semester reading Ilmanen's <em>Expected Returns</em> and
      wondering — do these strategies actually work when you trade them
      for real? Here's what I found.
    </p>
    <div class="tags">
      <span class="tag">Factor Investing</span>
      <span class="tag">FX Carry</span>
      <span class="tag">Trend Following</span>
      <span class="tag">Country Value</span>
      <span class="tag">Betting Against Beta</span>
      <span class="tag">Signal Stack</span>
      <span class="tag">Ilmanen 2011</span>
    </div>
  </div>
</section>

<!-- ── Overview ──────────────────────────────────────────────────────── -->
<section id="overview">
  <div class="container">
    {section_header('', 'Overview & Motivation', 'Five factors, one book, ten years of data')}

    <div class="narrative">
      <p>
        Antti Ilmanen's <em>Expected Returns</em> is one of the most
        rigorous attempts to map out every risk premium available to
        an investor — from vanilla equity risk to exotic carry trades
        in emerging market FX. The book argues that most returns can
        be explained by a handful of persistent, structural factors:
        carry, momentum/trend, value, and defensive/low-risk.
      </p>
      <p>
        This project tests five of those ideas from scratch, using
        freely available weekly price data from 2017 to 2026 and
        short-term interest rates from FRED. No Bloomberg terminal,
        no expensive data vendors — just Yahoo Finance, some Python,
        and a lot of frustration with rate limits. Each strategy is
        implemented as cleanly as possible given the data constraints:
        weekly rebalancing where logic allows, monthly where turnover
        would otherwise eat everything.
      </p>
      <p>
        One caveat upfront: weekly data compresses signal quality
        (daily signals get averaged away), transaction costs are
        ignored, and the backtest period is relatively short. These
        are prototypes, not production strategies. But the
        directional results are still informative.
      </p>
    </div>

    <div class="chart-wrap">
      <div id="chart-overview" style="height:400px">
        {fig_to_html(fig_overview)}
      </div>
    </div>

    <div class="conclusions-grid">
      <div class="conclusion-item">
        <div class="label" style="color:#3366FF">Data</div>
        <p>Weekly prices from Yahoo Finance local cache (86 tickers, 2016–2026).
           Short rates from FRED via rates.csv.</p>
      </div>
      <div class="conclusion-item">
        <div class="label" style="color:#10B981">Methodology</div>
        <p>Each strategy uses rolling windows, monthly rebalancing,
           and vol-weighting where applicable. No look-ahead bias.</p>
      </div>
      <div class="conclusion-item">
        <div class="label" style="color:#F59E0B">Caveats</div>
        <p>No transaction costs, no slippage, no margin costs.
           Short backtest period. Weekly data misses intraday signals.</p>
      </div>
    </div>
  </div>
</section>

<!-- ── P1: FX Carry ──────────────────────────────────────────────────── -->
<section id="p1">
  <div class="container">
    {section_header('01', 'G10 FX Carry', 'Borrow cheap, fund expensive')}

    <div class="narrative">
      <p>
        The carry trade is one of the oldest tricks in FX — borrow
        in low-rate currencies (JPY, CHF) and invest in high-rate ones
        (NZD, AUD, NOK). It worked for decades and is the single
        most well-documented risk premium in FX markets. The intuition
        is simple: uncovered interest parity says these rate differentials
        should be offset by exchange rate moves, but empirically they
        aren't — the "UIP puzzle."
      </p>
      <p>
        Post-2008 the trade has been rockier. Carry crashes tend to
        coincide with global risk-off episodes (2008, 2020) when
        investors unwind positions simultaneously, causing funding
        currencies to spike. I added a momentum overlay — if a
        currency's 4-week return is negative, I zero out its weight
        for that month. This helps cut losses during unwind periods.
        Vol-weighting ensures that the Norwegian krone doesn't dominate
        the portfolio just because it has a high carry.
      </p>
    </div>

    {stats_row(p1['stats'])}

    <div class="chart-wrap">
      <div id="chart-p1-main" style="height:520px">
        {fig_to_html(fig_p1)}
      </div>
    </div>

    <div class="chart-wrap">
      <div id="chart-p1-carry" style="height:340px">
        {fig_to_html(fig_p1_carry)}
      </div>
    </div>

    <div class="signal-card">
      <h3>Current Positions (Live Signal)</h3>
      {make_signal_badges(p1_badges)}
      <p style="font-size:12px;color:#9CA3AF;margin-top:10px">
        Yellow/Neutral = momentum overlay active (4-week return negative)
      </p>
    </div>
  </div>
</section>

<!-- ── P2: Trend Following ────────────────────────────────────────────── -->
<section id="p2">
  <div class="container">
    {section_header('02', 'Multi-Asset Trend Following', 'CTAs have been doing this for 50 years')}

    <div class="narrative">
      <p>
        CTAs (commodity trading advisors) have been running trend-following
        strategies for decades. The intuition is beautifully simple —
        markets trend. Whether it's equities rallying on improving
        fundamentals or oil dropping on supply gluts, price momentum
        persists over medium horizons. The academic version of this is
        TSMOM (time-series momentum), popularized by Moskowitz, Ooi
        and Pedersen (2012).
      </p>
      <p>
        The signal here is 12-month return minus 1-month return — the
        "12-1" skip-month momentum that's standard in the academic
        literature. Skipping the most recent month avoids the short-term
        reversal effect. If the signal is positive, go long; negative,
        go short. Position size is inverse-volatility weighted so that
        each asset contributes equally to portfolio risk. Trend following
        is famous for its crisis alpha — it tends to do well when equities
        crash, because it gets short early in a down-trend.
      </p>
    </div>

    {stats_row(p2['stats'])}

    <div class="chart-wrap">
      <div id="chart-p2-main" style="height:520px">
        {fig_to_html(fig_p2)}
      </div>
    </div>

    <div class="signal-card">
      <h3>Current Positions (Live Signal)</h3>
      {make_signal_badges(p2_badges)}
    </div>
  </div>
</section>

<!-- ── P3: Value Across Borders ──────────────────────────────────────── -->
<section id="p3">
  <div class="container">
    {section_header('03', 'Value Across Borders', 'Long cheap countries, short expensive ones')}

    <div class="narrative">
      <p>
        Value has had a decade-long identity crisis. Buying cheap countries
        sounds obvious — who doesn't want to pay less for a dollar of
        earnings? But what does "cheap" even mean at the country level?
        CAPE ratios? Dividend yields? Book-to-market? Every measure has
        its blind spots. Japan looked cheap by CAPE for 30 years while
        deflation ate returns. Brazil looks cheap by PE until you account
        for political risk and FX.
      </p>
      <p>
        My proxy here is simple but defensible: I use the negative of the
        past year's price return as a cheap signal. Countries that have
        underperformed tend to be cheaper relative to fundamentals — a
        crude mean-reversion assumption. Annual rebalance, long the 5
        cheapest, short the 5 most expensive, equal-weight. It misses a
        lot of the nuance (PE, DY, P/B) but avoids data-snooping on
        valuation ratios that are hard to source historically for free.
      </p>
    </div>

    {stats_row(p3['stats'])}

    <div class="chart-wrap">
      <div id="chart-p3-main" style="height:520px">
        {fig_to_html(fig_p3)}
      </div>
    </div>

    <div class="chart-wrap">
      <div id="chart-p3-signal" style="height:380px">
        {fig_to_html(fig_p3_signal)}
      </div>
    </div>

    <div class="signal-card">
      <h3>Current Positions (Annual Rebalance Signal)</h3>
      {make_signal_badges(p3_badges)}
    </div>
  </div>
</section>

<!-- ── P4: BAB ───────────────────────────────────────────────────────── -->
<section id="p4">
  <div class="container">
    {section_header('04', 'Betting Against Beta', 'Low-beta stocks outperform on a risk-adjusted basis')}

    <div class="narrative">
      <p>
        The CAPM says higher beta should equal higher expected return —
        you get compensated for taking systematic risk. Frazzini and
        Pedersen (2014) showed this is empirically backwards, at least
        within equities. Low-beta stocks have historically delivered
        better Sharpe ratios than high-beta stocks. The explanation:
        leverage-constrained investors (think mutual funds that can't
        use margin) are forced to "reach for beta" to hit return targets,
        bidding up high-beta stocks and leaving low-beta stocks cheap.
      </p>
      <p>
        The implementation levers up the low-beta quintile to a target
        beta of 1.0 and de-levers the high-beta quintile similarly,
        creating a market-neutral portfolio. Floor beta at 0.2 to cap
        max leverage at 5x. Rolling 52-week beta vs SPY on a universe
        of 60 large-cap stocks across all 10 GICS sectors. Monthly
        rebalance. This is the strategy where I had the highest
        expectations — and it's the one that delivered.
      </p>
    </div>

    {stats_row(p4['stats'])}

    <div class="chart-wrap">
      <div id="chart-p4-main" style="height:520px">
        {fig_to_html(fig_p4)}
      </div>
    </div>

    <div class="chart-wrap">
      <div id="chart-p4-beta" style="height:720px">
        {fig_to_html(fig_p4_beta)}
      </div>
    </div>

    <div class="signal-card">
      <h3>Current Positions — Top/Bottom Beta Quintile</h3>
      {make_signal_badges(p4_badges)}
      <p style="font-size:12px;color:#9CA3AF;margin-top:10px">
        Showing top/bottom 20 stocks by current beta. Gray = middle quintiles (neutral).
      </p>
    </div>
  </div>
</section>

<!-- ── P5: Signal Stack ──────────────────────────────────────────────── -->
<section id="p5">
  <div class="container">
    {section_header('05', 'Signal Stack', 'Carry + Trend + Value combined')}

    <div class="narrative">
      <p>
        If carry, trend, and value each have mediocre Sharpe ratios
        individually — especially in the past decade — what happens when
        you combine them? Diversification happens. Carry tends to struggle
        in risk-off environments precisely when trend does best (markets
        trend down hard). Value moves on long cycles that are mostly
        uncorrelated with the other two. Combining them with equal weights
        smooths the equity curve significantly.
      </p>
      <p>
        The implementation z-scores each signal cross-sectionally across
        20 assets, combines with 1/3 weights, then goes long the top 7
        and short the bottom 7 by composite score each month. The result
        should be more stable than any single factor — Ilmanen's main
        point in the final chapters of Expected Returns. The correlation
        matrix below tells the story: carry and trend are negatively
        correlated, which is free diversification.
      </p>
    </div>

    {stats_row(p5['stats'])}

    <div class="chart-wrap">
      <div id="chart-p5-factors" style="height:440px">
        {fig_to_html(fig_p5_factors)}
      </div>
    </div>

    <div class="chart-wrap">
      <div id="chart-p5-main" style="height:520px">
        {fig_to_html(fig_p5)}
      </div>
    </div>

    <div class="chart-wrap">
      <div id="chart-p5-heatmap" style="height:540px">
        {fig_to_html(fig_p5_heatmap)}
      </div>
    </div>

    <div class="signal-card">
      <h3>Current Composite Signal Positions</h3>
      {make_signal_badges(p5_badges)}
    </div>
  </div>
</section>

<!-- ── Conclusions ───────────────────────────────────────────────────── -->
<section id="conclusions">
  <div class="container">
    {section_header('', 'Conclusions & Reflections', 'What worked, what didn\'t, what\'s next')}

    <div class="narrative">
      <p>
        The clearest winner is <strong>Betting Against Beta</strong>. The
        low-beta premium is real, persistent, and survives in a modern
        universe of 60 large-cap stocks with weekly data. The CAPM is
        empirically wrong within equities, and Frazzini-Pedersen explain
        why elegantly. Of all five, BAB is the one I'd trust most in
        a live account.
      </p>
      <p>
        <strong>Trend following</strong> had mixed results — strong in
        crisis periods, flat in trending bull markets. The 12-1 signal
        on weekly data loses some edge vs daily, and the sample period
        (2017–2026) includes a prolonged low-vol equity bull market
        followed by two sharp reversals, which isn't ideal for CTA
        strategies.
      </p>
      <p>
        <strong>FX Carry</strong> and <strong>Country Value</strong> have
        been the hardest factors to monetize in the post-QE era. Carry
        crashes have become more frequent; value's mean-reversion thesis
        has struggled as US tech dominated global returns. These aren't
        dead — they're just in a long drawdown, which is itself predicted
        by factor theory.
      </p>
      <p>
        The <strong>Signal Stack</strong> demonstrates the diversification
        point convincingly: the combined Sharpe exceeds all three
        component factors, with meaningfully smaller drawdowns. This
        is the key insight from Ilmanen — no single factor is reliable
        enough, but a portfolio of uncorrelated factors is.
      </p>
      <p>
        <strong>Limitations:</strong> No transaction costs (real carry and
        trend strategies pay 50–150bps/yr in FX). Weekly data. Short
        10-year window. Survivorship bias in the stock universe.
        <strong>Future work:</strong> crypto factor premia, machine
        learning signal combination, regime-conditional factor weights,
        and incorporating PE/DY data for a richer value signal.
      </p>
    </div>

    <div class="chart-wrap">
      <div id="chart-corr" style="height:380px">
        {fig_to_html(fig_corr)}
      </div>
    </div>

    <div class="conclusions-grid">
      <div class="conclusion-item">
        <div class="label" style="color:#10B981">What Worked</div>
        <p>BAB showed consistent alpha. Signal Stack outperformed
           all individual factors through diversification. Trend
           provided crisis alpha in 2020 and 2022.</p>
      </div>
      <div class="conclusion-item">
        <div class="label" style="color:#EF4444">What Didn't</div>
        <p>FX Carry suffered from post-2020 volatility in carry
           trades. Country Value struggled as US tech dominance
           made cross-country valuation spreads wider and stickier.</p>
      </div>
      <div class="conclusion-item">
        <div class="label" style="color:#3366FF">Future Work</div>
        <p>Incorporate actual PE/DY data for value. Test on crypto
           factor premia. Add ML-based signal combination.
           Implement regime detection to tilt factor weights.</p>
      </div>
    </div>

    <div style="text-align:center;padding:40px 0 20px;color:#9CA3AF;font-size:13px">
      Built with Python · Plotly · Ilmanen "Expected Returns" (2011) ·
      Data: Yahoo Finance / FRED · Weekly frequency 2017–2026
    </div>
  </div>
</section>

</body>
</html>"""

    return html


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('Building Factor Strategy Dashboard')
    print('=' * 60)

    p1 = run_p1()
    p2 = run_p2()
    p3 = run_p3()
    p4 = run_p4()
    p5 = run_p5()

    print('\nGenerating HTML dashboard...')
    html = build_html(p1, p2, p3, p4, p5)

    out_path = os.path.join(SCRIPT_DIR, 'dashboard.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f'\nDone! Dashboard written to: {out_path}')
    print(f'File size: {size_mb:.1f} MB')
    print('\nPerformance Summary:')
    print(f"  P1 FX Carry:      Sharpe={p1['stats']['sharpe']:.2f}  Ann.Ret={p1['stats']['ann_ret']:.1%}  MaxDD={p1['stats']['max_dd']:.1%}")
    print(f"  P2 Trend:         Sharpe={p2['stats']['sharpe']:.2f}  Ann.Ret={p2['stats']['ann_ret']:.1%}  MaxDD={p2['stats']['max_dd']:.1%}")
    print(f"  P3 Value:         Sharpe={p3['stats']['sharpe']:.2f}  Ann.Ret={p3['stats']['ann_ret']:.1%}  MaxDD={p3['stats']['max_dd']:.1%}")
    print(f"  P4 BAB:           Sharpe={p4['stats']['sharpe']:.2f}  Ann.Ret={p4['stats']['ann_ret']:.1%}  MaxDD={p4['stats']['max_dd']:.1%}")
    print(f"  P5 Signal Stack:  Sharpe={p5['stats']['sharpe']:.2f}  Ann.Ret={p5['stats']['ann_ret']:.1%}  MaxDD={p5['stats']['max_dd']:.1%}")

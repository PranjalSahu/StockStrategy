#!/usr/bin/env python3
"""
Dark-themed Momentum Trading Dashboard with "Top Picks Today"
- Top picks are derived from the latest backtest window for the best-performing strategy
- Top Picks now displayed horizontally in one line
"""

import os
import sys
import logging
from datetime import date, timedelta
from typing import List, Optional, Dict
from joblib import Parallel, delayed
from joblib import Memory
import pandas as pd
import numpy as np

# Create a persistent cache directory
memory = Memory("cache/backtest", verbose=0, compress=1)  # compress=1 for smaller files

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    import dash
    from dash import dcc, html, dash_table
    from dash.dependencies import Input, Output, State
    from dash.exceptions import PreventUpdate
except ImportError:
    print("Please install: pip install plotly dash")
    sys.exit(1)

from dash import dcc, html, dash_table, Input, Output, State, callback

pio.templates.default = 'plotly_dark'

DATA_DIR = "data"
REPORTS_DIR = "reports"
LOG_FILE = "momentum.log"
BENCHMARK_ETFS = ["SPY", "QQQ", "VTI"]

pd.options.display.width = 160
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)

# ===== Helpers =====

def today_str() -> str:
    return date.today().isoformat()

def read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df.sort_index()
    except Exception as e:
        logging.warning(f"Failed reading {path}: {e}")
        return None

def write_csv(df: pd.DataFrame, path: str):
    try:
        df.to_csv(path)
    except Exception as e:
        logging.error(f"Failed writing {path}: {e}")

def safe_latest_date(df: Optional[pd.DataFrame]) -> Optional[date]:
    if df is None or df.empty:
        return None
    try:
        return df.index.max().date()
    except Exception:
        return None

# ===== Download =====

def normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)
    rename_map = {"Adj Close": "Close", "adjclose": "Close", "close": "Close",
                  "open": "Open", "high": "High", "low": "Low", "volume": "Volume"}
    df = df.rename(columns=lambda c: rename_map.get(c, c))
    keep = ["Open", "High", "Low", "Close", "Volume"]
    existing = [c for c in keep if c in df.columns]
    return df[existing]

def sync_ticker(ticker: str, start: str, force_download: bool, skip_download: bool) -> Optional[pd.DataFrame]:
    if yf is None:
        logging.error("yfinance not installed")
        return None
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if skip_download:
        return read_csv_if_exists(path)
    existing = None if force_download else read_csv_if_exists(path)
    last_date = safe_latest_date(existing)
    start_str = (last_date + timedelta(days=1)).isoformat() if last_date else start
    end_str = today_str()
    if pd.to_datetime(start_str) >= pd.to_datetime(end_str) and existing is not None:
        return existing
    try:
        df_new = yf.download(ticker, start=start_str, end=end_str,
                             auto_adjust=True, progress=False, threads=False, actions=False)
    except Exception as e:
        logging.warning(f"[{ticker}] Download failed: {e}")
        return existing
    if df_new is None or df_new.empty:
        return existing
    df_new = normalize_yf_columns(df_new)
    df = pd.concat([existing, df_new]) if (existing is not None and not force_download) else df_new
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    write_csv(df, path)
    return df

# ===== Indicators =====

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)
    df["SMA20"] = close.rolling(20, min_periods=10).mean()
    df["SMA50"] = close.rolling(50, min_periods=10).mean()
    df["SMA200"] = close.rolling(200, min_periods=20).mean()
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()
    df["ROC10"] = close.pct_change(10, fill_method=None)
    df["ROC21"] = close.pct_change(21, fill_method=None)
    df["ROC63"] = close.pct_change(63, fill_method=None)
    df["ROC126"] = close.pct_change(126, fill_method=None)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))
    macd = df["EMA12"] - df["EMA26"]
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd
    df["MACDSignal"] = signal
    df["MACDHist"] = macd - signal
    df["BB_Middle"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Position"] = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"]).replace([np.inf, -np.inf], np.nan)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    df["Volume_SMA20"] = volume.rolling(20).mean()
    df["Volume_Ratio"] = volume / df["Volume_SMA20"].replace(0, np.nan)
    high52 = close.rolling(252, min_periods=60).max()
    low52 = close.rolling(252, min_periods=60).min()
    df["High52w"] = high52
    df["Low52w"] = low52
    df["PctTo52wHigh"] = (close / high52) - 1
    df["PctFrom52wLow"] = (close / low52) - 1
    df["Volatility20"] = close.pct_change(fill_method=None).rolling(20).std() * np.sqrt(252)
    return df

# ===== Scoring =====

def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

def compute_score(df: pd.DataFrame, strategy: str) -> pd.Series:
    strategies = {
        "momentum_pure": lambda: (
            0.40 * zscore(df["ROC63"]) + 0.30 * zscore(df["ROC21"]) + 0.20 * zscore(df["PctTo52wHigh"]) + 0.10 * zscore(df["ROC10"]) ),
        "momentum_trend": lambda: (
            0.35 * zscore(df["ROC63"]) + 0.25 * zscore(df["ROC21"]) + 0.20 * zscore(df["SMA50vs200"]) + 0.10 * zscore(df["PctTo52wHigh"]) + 0.10 * zscore(df["SMA20vs50"]) ),
        "swing_trader": lambda: (
            0.30 * zscore(df["ROC21"]) + 0.25 * zscore(df["MACDHist"]) + 0.20 * ((df["RSI14"] - 50) / 20) + 0.15 * zscore(df["BB_Position"]) + 0.10 * zscore(df["Volume_Ratio"]) ),
        "breakout": lambda: (
            0.35 * zscore(df["PctTo52wHigh"]) + 0.25 * zscore(df["ROC21"]) + 0.20 * zscore(df["Volume_Ratio"]) + 0.10 * zscore(df["BB_Width"]) + 0.10 * zscore(df["ROC10"]) ),
        "volatility_adjusted": lambda: (
            0.30 * zscore(df["ROC63"] / (df["Volatility20"] + 0.01)) + 0.25 * zscore(df["ROC21"] / (df["Volatility20"] + 0.01)) + 0.20 * zscore(df["MACDHist"]) + 0.15 * zscore(df["PctTo52wHigh"]) + 0.10 * zscore(df["SMA50vs200"]) ),
        "value_momentum": lambda: (
            0.30 * zscore(df["ROC63"]) + 0.25 * zscore(df["ROC21"]) + 0.20 * zscore(df["PctFrom52wLow"]) + 0.15 * (-zscore(df["BB_Position"])) + 0.10 * zscore(df["SMA50vs200"]) ),
        "mean_reversion": lambda: (
            0.35 * (-zscore(df["ROC10"])) + 0.25 * zscore(df["RSI14"] - 70) + 0.20 * (-zscore(df["BB_Position"])) + 0.20 * zscore(df["ROC63"]) ),
        "quality_momentum": lambda: (
            0.30 * zscore(df["ROC63"]) + 0.25 * zscore(df["ROC126"]) + 0.20 * (-zscore(df["Volatility20"])) + 0.15 * zscore(df["SMA50vs200"]) + 0.10 * zscore(df["PctTo52wHigh"]) ),
        "low_volatility": lambda: (
            0.40 * (-zscore(df["Volatility20"])) + 0.30 * (-zscore(df["ATR14"] / df["Close"])) + 0.20 * zscore(df["ROC126"]) + 0.10 * (-zscore(df["BB_Width"])) ),
        "trending_value": lambda: (
            0.30 * zscore(df["PctFrom52wLow"]) + 0.25 * zscore(df["SMA50vs200"]) + 0.20 * (-zscore(df["ROC10"])) + 0.15 * zscore(df["Volume_Ratio"]) + 0.10 * ((50 - df["RSI14"]) / 20) ),
        "volume_breakout": lambda: (
            0.35 * zscore(df["Volume_Ratio"]) + 0.30 * zscore(df["PctTo52wHigh"]) + 0.20 * zscore(df["ROC21"]) + 0.15 * zscore(df["BB_Position"]) ),
        "dividend_momentum": lambda: (
            0.35 * zscore(df["ROC63"]) + 0.25 * (-zscore(df["Volatility20"])) + 0.20 * zscore(df["SMA50vs200"]) + 0.20 * (-zscore(df["BB_Width"])) ),
        "contrarian": lambda: (
            0.35 * (-zscore(df["ROC21"])) + 0.25 * (-zscore(df["PctTo52wHigh"])) + 0.20 * ((30 - df["RSI14"]) / 20) + 0.20 * zscore(df["ROC126"]) ),
    }
    return strategies.get(strategy, strategies["momentum_pure"])()

def build_scoring_table(data_map: dict, strategy: str) -> pd.DataFrame:
    rows = []
    for t, df in data_map.items():
        try:
            last = df.dropna().iloc[-1]
            rows.append({
                "Ticker": t,
                "ROC10": last["ROC10"],
                "ROC21": last["ROC21"],
                "ROC63": last["ROC63"],
                "ROC126": last["ROC126"],
                "MACDHist": last["MACDHist"],
                "PctTo52wHigh": last["PctTo52wHigh"],
                "PctFrom52wLow": last["PctFrom52wLow"],
                "SMA50vs200": last["SMA50"] / last["SMA200"] - 1,
                "SMA20vs50": last["SMA20"] / last["SMA50"] - 1,
                "RSI14": last["RSI14"],
                "BB_Position": last["BB_Position"],
                "BB_Width": last["BB_Width"],
                "Volume_Ratio": last["Volume_Ratio"],
                "Volatility20": last["Volatility20"],
                "ATR14": last["ATR14"],
                "Close": last["Close"],
            })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Score"] = compute_score(df, strategy)
    return df.sort_values("Score", ascending=False).reset_index(drop=True)

# ===== Backtesting =====

def nearest_trading_date(df: pd.DataFrame, target_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    idx = df.index[df.index <= target_date]
    return idx.max() if len(idx) > 0 else None

def calculate_benchmark_returns(benchmark_data: dict, lookback_days: int, end_offset_days: int) -> Dict[str, float]:
    today = pd.Timestamp.today().normalize() - pd.Timedelta(days=end_offset_days)
    backtest_date = today - pd.Timedelta(days=lookback_days)
    benchmark_returns = {}
    for etf, df in benchmark_data.items():
        try:
            entry_date = nearest_trading_date(df, backtest_date)
            exit_date = nearest_trading_date(df, today)
            if entry_date is None or exit_date is None or entry_date >= exit_date:
                continue
            entry_price = df.loc[entry_date, "Close"]
            exit_price = df.loc[exit_date, "Close"]
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue
            ret = (exit_price / entry_price - 1) * 100
            benchmark_returns[etf] = ret
        except Exception:
            continue
    return benchmark_returns

def backtest_strategy(data_map: dict, benchmark_data: dict, strategy: str, top_n: int, lookback_days: int, end_offset_days: int) -> Optional[Dict]:
    today = pd.Timestamp.today().normalize() - pd.Timedelta(days=end_offset_days)
    backtest_date = today - pd.Timedelta(days=lookback_days)
    sliced_data = {}
    for t, df in data_map.items():
        bt_date = nearest_trading_date(df, backtest_date)
        if bt_date is None:
            continue
        sliced_df = df.loc[:bt_date].copy()
        if len(sliced_df) < 260:
            continue
        sliced_data[t] = sliced_df
    if len(sliced_data) < top_n:
        return None
    scores_bt = build_scoring_table(sliced_data, strategy)
    if scores_bt.empty or len(scores_bt) < top_n:
        return None
    top_bt = scores_bt.head(top_n)
    returns = []
    for _, row in top_bt.iterrows():
        tkr = row["Ticker"]
        df = data_map.get(tkr)
        if df is None:
            continue
        entry_date = nearest_trading_date(df, backtest_date)
        exit_date = nearest_trading_date(df, today)
        if entry_date is None or exit_date is None or entry_date >= exit_date:
            continue
        try:
            entry_price = df.loc[entry_date, "Close"]
            exit_price = df.loc[exit_date, "Close"]
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue
            returns.append((exit_price / entry_price - 1) * 100)
        except Exception:
            continue
    if not returns:
        return None
    benchmark_returns = calculate_benchmark_returns(benchmark_data, lookback_days, end_offset_days)
    result = {
        "strategy": strategy,
        "entry_date": backtest_date,
        "exit_date": today,
        "returns": returns,
        "top_tickers": list(top_bt['Ticker']),
        "mean_return": np.mean(returns),
        "median_return": np.median(returns),
        "count_total": len(returns),
        "count_gt_10": sum(1 for r in returns if r > 10),
        "count_gt_5": sum(1 for r in returns if r > 5),
        "count_gt_0": sum(1 for r in returns if r > 0),
        "pct_gt_10": sum(1 for r in returns if r > 10) / len(returns) * 100,
        "pct_gt_5": sum(1 for r in returns if r > 5) / len(returns) * 100,
        "pct_gt_0": sum(1 for r in returns if r > 0) / len(returns) * 100,
        "max_return": max(returns),
        "min_return": min(returns),
        "std_return": np.std(returns),
    }
    for etf, ret in benchmark_returns.items():
        result[f"{etf}_return"] = ret
    return result

# 2. Add this new helper function (insert before run_comprehensive_backtest)
def backtest_single_strategy_all_offsets(strategy: str, data_map: dict, benchmark_data: dict, 
                                          top_n: int, lookback_days: int, test_intervals: range) -> List[Dict]:
    """
    Helper function to backtest a single strategy across all time offsets.
    This function will be called in parallel for each strategy.
    """
    results = []
    for offset in test_intervals:
        try:
            r = backtest_strategy(data_map, benchmark_data, strategy, top_n, lookback_days, offset)
            if r:
                results.append(r)
        except Exception as e:
            logging.warning(f"Strategy {strategy}, offset {offset} failed: {e}")
            continue
    return results

def run_comprehensive_backtest_old(data_map: dict, benchmark_data: dict, strategies: List[str], top_n: int = 10, lookback_days: int = 30, months_back: int = 6) -> pd.DataFrame:
    results = []
    days_back = max(1, months_back) * 30
    test_intervals = range(0, days_back, 2)
    for strategy in strategies:
        for offset in test_intervals:
            try:
                r = backtest_strategy(data_map, benchmark_data, strategy, top_n, lookback_days, offset)
                if r:
                    results.append(r)
            except Exception:
                continue
    return pd.DataFrame(results)

# 3. Replace the entire run_comprehensive_backtest function with this:
@memory.cache
def run_comprehensive_backtest(data_map: dict, benchmark_data: dict, strategies: List[str], 
                                top_n: int = 10, lookback_days: int = 30, months_back: int = 6,
                                n_jobs: int = -1) -> pd.DataFrame:
    """
    Run comprehensive backtest with PARALLEL execution across strategies.
    
    Args:
        n_jobs: Number of parallel jobs. -1 means use all processors, -2 means all but one, etc.
    """
    days_back = max(1, months_back) * 30
    test_intervals = range(0, days_back, 2)
    
    logging.info(f"Starting parallel backtest: {len(strategies)} strategies × {len(test_intervals)} time windows")
    logging.info(f"Using n_jobs={n_jobs} for parallelization")
    
    # Parallel execution across strategies
    parallel_results = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(
        delayed(backtest_single_strategy_all_offsets)(
            strategy, data_map, benchmark_data, top_n, lookback_days, test_intervals
        )
        for strategy in strategies
    )
    
    # Flatten the list of lists into a single list of results
    results = []
    for strategy_results in parallel_results:
        results.extend(strategy_results)
    
    logging.info(f"Parallel backtest complete. Generated {len(results)} result rows.")
    
    return pd.DataFrame(results)

# ===== Dashboard with Horizontal Top Picks =====

def create_dashboard(backtest_results: pd.DataFrame, benchmark_columns: List[str], data_map: dict, benchmark_data: dict, strategies: List[str], default_top_n: int = 10, default_months_back: int = 6):
    if backtest_results.empty:
        return None
    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    def build_summaries(df: pd.DataFrame):
        agg = {
            'mean_return': 'mean', 'count_gt_10': 'sum', 'count_gt_5': 'sum', 'count_gt_0': 'sum', 'count_total': 'sum',
            'pct_gt_0': 'mean', 'pct_gt_5': 'mean', 'pct_gt_10': 'mean', 'max_return': 'max', 'min_return': 'min', 'std_return': 'mean'
        }
        for col in benchmark_columns:
            if col in df.columns:
                agg[col] = 'mean'
        strat = df.groupby('strategy').agg(agg).reset_index().sort_values('mean_return', ascending=False)
        bench = {}
        for etf in BENCHMARK_ETFS:
            c = f"{etf}_return"
            if c in df.columns:
                bench[etf] = df[c].mean()
        return strat, bench

    def compute_top_picks(df: pd.DataFrame):
        strat_summary, _ = build_summaries(df)
        if strat_summary.empty:
            return [], None
        
        # Get top 5 strategies instead of just 1
        top_5_strategies = strat_summary.head(5)['strategy'].tolist()
        
        all_picks = []
        latest_date = None
        
        for strategy in top_5_strategies:
            df_strat = df[df['strategy'] == strategy].sort_values('exit_date')
            if df_strat.empty:
                continue
            latest = df_strat.iloc[-1]
            if latest_date is None:
                latest_date = latest['exit_date']
            
            tickers = latest.get('top_tickers', [])
            if isinstance(tickers, str):
                try:
                    import json
                    tickers = json.loads(tickers)
                except Exception:
                    tickers = [tickers]
            
            # Add strategy name with each ticker for display
            for ticker in tickers[:10]:
                all_picks.append({'ticker': ticker, 'strategy': strategy})
        
        return all_picks, latest_date

    strat_summary, bench_summary = build_summaries(backtest_results)

    table_columns = [
        {'name': 'Strategy', 'id': 'strategy'},
        {'name': 'Avg Return %', 'id': 'mean_return'},
        {'name': 'Total Picks', 'id': 'count_total'},
        {'name': 'Wins (>0%)', 'id': 'count_gt_0'},
        {'name': 'Good (>5%)', 'id': 'count_gt_5'},
        {'name': 'Great (>10%)', 'id': 'count_gt_10'},
        {'name': '% Win Rate', 'id': 'pct_gt_0'},
        {'name': '% >5%', 'id': 'pct_gt_5'},
        {'name': '% >10%', 'id': 'pct_gt_10'},
    ]
    for etf in BENCHMARK_ETFS:
        col = f"{etf}_return"
        if col in strat_summary.columns:
            table_columns.append({'name': f'{etf} Avg %', 'id': col})
    table_columns += [
        {'name': 'Max Return %', 'id': 'max_return'},
        {'name': 'Min Return %', 'id': 'min_return'},
        {'name': 'Std Dev %', 'id': 'std_return'},
    ]

    store_initial = backtest_results.to_json(date_format='iso', orient='split')
    initial_top, initial_date = compute_top_picks(backtest_results)

    app.layout = html.Div(className='app', children=[
        html.Div(className='navbar', children=[
            html.Div(className='brand', children=[html.Div(className='brand-logo'), html.Span('Momentum Lab')]),
            html.Div(className='nav-actions', children=[html.Span(className='badge', children='Quant-powered Strategies'), html.Button('Re-run Backtest', id='rerun-btn', n_clicks=0, className='btn btn-primary')])
        ]),
        html.Div(className='container', children=[
            # Top Picks Today card — now horizontal
            # Update the Top Picks card in app.layout to group by strategy:
            html.Div(className='top-picks-card', children=[
                html.Div(className='top-picks-header', children=[
                    html.Div(className='top-picks-title', children='Top Picks Today - Top 5 Strategies'),
                    html.Div(className='top-picks-meta', children=(
                        f"From 5 best-performing strategies — as of {initial_date.strftime('%Y-%m-%d')}" if initial_date is not None else ''
                    ))
                ]),
                dcc.Tabs(
                    id='top-picks-tabs',
                    value=(sorted(set(pick['strategy'] for pick in initial_top), key=lambda s: [pick['strategy'] for pick in initial_top].index(s))[0] if initial_top else None),
                    className='top-picks-tabs',
                    children=[
                        dcc.Tab(
                            label=strategy,
                            value=strategy,
                            className='custom-tab',
                            selected_className='custom-tab--selected',
                            children=html.Div([
                                html.Div(strategy, style={'fontWeight': 'bold', 'fontSize': '11px', 'marginBottom': '5px', 'color': '#8b949e', 'textTransform': 'uppercase'}),
                                html.Div([
                                    html.Span(className='pick-chip', children=pick['ticker']) 
                                    for pick in initial_top if pick['strategy'] == strategy
                                ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px', 'marginBottom': '15px'})
                            ])
                        ) for strategy in sorted(set(pick['strategy'] for pick in initial_top), key=lambda s: [pick['strategy'] for pick in initial_top].index(s))
                    ]
                ),
            ]),

            html.Div(className='hero', children=[
                html.H1('Strategy Dashboard'),
                html.P('Backtested momentum & quality strategies vs SPY, QQQ, VTI — configurable Top N & period.'),
                html.Div(className='controls-card', children=[
                    html.Div(className='controls-row', children=[
                        html.Div(children=[html.Label('Backtest Months', htmlFor='input-months', className='label'), dcc.Input(id='input-months', type='number', min=1, max=24, value=default_months_back, className='input')]),
                        html.Div(children=[html.Label('Top N stocks', htmlFor='input-topn', className='label'), dcc.Input(id='input-topn', type='number', min=1, max=100, value=default_top_n, className='input')]),
                        dcc.Loading(id='loading-rerun', type='default', children=html.Div(id='run-status', className='status'))
                    ])
                ])
            ]),

            dcc.Store(id='store-results', data=store_initial),

            html.Div(className='card', children=[
                 html.H2('Strategy Performance Summary'),
                html.Div(id='strategy-table-container', children=[])
            ]),

            html.Div(className='card', children=[
                html.H2('Strategy Comparison Charts'),
                dcc.Tabs(id='chart-tabs', value='return-comparison', className='custom-tabs', children=[
                    dcc.Tab(label='Average Returns', value='return-comparison', className='custom-tab', selected_className='custom-tab--selected'),
                    dcc.Tab(label='vs Benchmarks', value='benchmark-comparison', className='custom-tab', selected_className='custom-tab--selected'),
                    dcc.Tab(label='Win Rates', value='win-rate-comparison', className='custom-tab', selected_className='custom-tab--selected'),
                    dcc.Tab(label='Return Distribution', value='return-distribution', className='custom-tab', selected_className='custom-tab--selected'),
                    dcc.Tab(label='Rolling Performance', value='rolling-performance', className='custom-tab', selected_className='custom-tab--selected'),
                ]),
                html.Div(id='chart-content', style={'marginTop': '20px'})
            ]),
            
            html.Div(className='card', children=[
                html.H2('Benchmark Performance Summary'),
                html.Div(className='flex gap-16 flex-center', children=[
                    html.Div(className='chip ' + ('success' if bench_summary.get('SPY',0)>0 else 'danger'), children=[html.Strong('SPY'), html.Span(f"{bench_summary.get('SPY',0):.2f}%")]),
                    html.Div(className='chip ' + ('success' if bench_summary.get('QQQ',0)>0 else 'danger'), children=[html.Strong('QQQ'), html.Span(f"{bench_summary.get('QQQ',0):.2f}%")]),
                    html.Div(className='chip ' + ('success' if bench_summary.get('VTI',0)>0 else 'danger'), children=[html.Strong('VTI'), html.Span(f"{bench_summary.get('VTI',0):.2f}%")]),
                ])
            ])
        ])
    ])

    # Rerun callback
    @app.callback([
        Output('store-results', 'data'),
        Output('run-status', 'children')
    ], [Input('rerun-btn', 'n_clicks')], [State('input-months', 'value'), State('input-topn', 'value')])
    def rerun(n, months, topn):
        if not n:
            raise PreventUpdate
        m = months if (months and months > 0) else default_months_back
        t = topn if (topn and topn > 0) else default_top_n
        new_df = run_comprehensive_backtest(data_map, benchmark_data, strategies, top_n=t, lookback_days=30, months_back=m, n_jobs=-1)
        if new_df.empty:
            return dash.no_update, 'No results. Check data and parameters.'
        return new_df.to_json(date_format='iso', orient='split'), f'Backtest complete — rows: {len(new_df)}, months={m}, top_n={t}'

    # Update visuals incl. Top Picks (now horizontal)
    @app.callback([
        Output('strategy-table', 'data'),
        Output('return-comparison', 'figure'),
        Output('benchmark-comparison', 'figure'),
        Output('win-rate-comparison', 'figure'),
        Output('return-distribution', 'figure'),
        Output('rolling-performance', 'figure'),
        Output('top-picks-tabs', 'children'),
        Output('top-picks-tabs', 'value')
    ], [Input('store-results', 'data')])
    def update(store_json):
        df = pd.read_json(store_json, orient='split')
        strat, bench = build_summaries(df)

        # Build top-picks tabs (one tab per top-5 strategy)
        picks, asof = compute_top_picks(df)
        strategies_in_picks = sorted(set(p['strategy'] for p in picks), key=lambda s: [p['strategy'] for p in picks].index(s))
        tabs_children = []
        for strategy in strategies_in_picks:
            strategy_picks = [p['ticker'] for p in picks if p['strategy'] == strategy]
            tab_content = html.Div([
                html.Div(strategy, style={'fontWeight': 'bold', 'fontSize': '11px', 'marginBottom': '5px', 'color': '#8b949e', 'textTransform': 'uppercase'}),
                html.Div([html.Span(className='pick-chip', children=ticker) for ticker in strategy_picks], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px', 'marginBottom': '15px'})
            ])
            tabs_children.append(dcc.Tab(label=strategy, value=strategy, className='top-picks-tab', selected_className='top-picks-tab--selected', children=tab_content))
        selected_tab = strategies_in_picks[0] if strategies_in_picks else None

        # Charts remain the same
        fig1 = go.Figure([go.Bar(x=strat['strategy'], y=strat['mean_return'],
                             marker_color=['#2ecc71' if x>0 else '#e74c3c' for x in strat['mean_return']],
                             text=strat['mean_return'].round(2), textposition='outside')])
        fig1.update_layout(title='Average Returns by Strategy', xaxis_title='Strategy', yaxis_title='Average Return (%)', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=420)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=strat['strategy'], y=strat['mean_return'], name='Strategy', marker_color='#58a6ff'))
        for etf in BENCHMARK_ETFS:
            if etf in bench:
                fig2.add_trace(go.Scatter(x=strat['strategy'], y=[bench[etf]]*len(strat), name=f'{etf} Benchmark', mode='lines', line=dict(width=3, dash='dash')))
        fig2.update_layout(title='Strategy Returns vs Benchmark ETFs', xaxis_title='Strategy', yaxis_title='Average Return (%)', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=420)

        fig3 = go.Figure([
            go.Bar(name='>0%', x=strat['strategy'], y=strat['pct_gt_0'], marker_color='#7ee787'),
            go.Bar(name='>5%', x=strat['strategy'], y=strat['pct_gt_5'], marker_color='#58a6ff'),
            go.Bar(name='>10%', x=strat['strategy'], y=strat['pct_gt_10'], marker_color='#e6cc00')
        ])
        fig3.update_layout(title='Win Rates by Threshold', xaxis_title='Strategy', yaxis_title='Percentage (%)', barmode='group', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=420)

        fig4 = go.Figure()
        for sname in df['strategy'].unique():
            sdata = df[df['strategy']==sname]
            all_returns = [r for returns in sdata['returns'] for r in returns]
            if len(all_returns) > 0:
                fig4.add_trace(go.Box(y=all_returns, name=sname))
        for etf in BENCHMARK_ETFS:
            col = f"{etf}_return"
            if col in df.columns:
                etf_returns = df[col].dropna()
                if len(etf_returns) > 0:
                    fig4.add_trace(go.Box(y=etf_returns, name=f'{etf} (Benchmark)', marker_color='orange', boxmean=True))
        fig4.update_layout(title='Return Distribution by Strategy (vs Benchmarks)', yaxis_title='Return (%)', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=520)

        fig5 = go.Figure()
        for sname in df['strategy'].unique():
            sdata = df[df['strategy']==sname].sort_values('entry_date')
            fig5.add_trace(go.Scatter(x=sdata['entry_date'], y=sdata['mean_return'], mode='lines+markers', name=sname))
        for etf in BENCHMARK_ETFS:
            col = f"{etf}_return"
            if col in df.columns:
                sorted_df = df.sort_values('entry_date')
                fig5.add_trace(go.Scatter(x=sorted_df['entry_date'], y=sorted_df[col], mode='lines', name=f'{etf} Benchmark', line=dict(width=3, dash='dot'), opacity=0.7))
        fig5.update_layout(title='Rolling 30-Day Performance (vs Benchmarks)', xaxis_title='Entry Date', yaxis_title='Mean Return (%)', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=520)

        return strat.round(2).to_dict('records'), fig1, fig2, fig3, fig4, fig5, tabs_children, selected_tab

    # new callback method
    @app.callback(
        Output('chart-content', 'children'),
        Input('chart-tabs', 'value'),
        Input('store-results', 'data')
    )
    def update_chart_content(selected_tab, store_json):
        if store_json is None:
            raise PreventUpdate
        
        df = pd.read_json(store_json, orient='split')
        strat, bench = build_summaries(df)

        # Reuse the same figure logic as before
        if selected_tab == 'return-comparison':
            fig = go.Figure([go.Bar(x=strat['strategy'], y=strat['mean_return'],
                                 marker_color=['#2ecc71' if x>0 else '#e74c3c' for x in strat['mean_return']],
                                 text=strat['mean_return'].round(2), textposition='outside')])
            fig.update_layout(title='Average Returns by Strategy', xaxis_title='Strategy', yaxis_title='Average Return (%)',
                              template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500)

        elif selected_tab == 'benchmark-comparison':
            fig = go.Figure()
            fig.add_trace(go.Bar(x=strat['strategy'], y=strat['mean_return'], name='Strategy', marker_color='#58a6ff'))
            for etf in BENCHMARK_ETFS:
                if etf in bench:
                    fig.add_trace(go.Scatter(x=strat['strategy'], y=[bench[etf]]*len(strat), name=f'{etf} Benchmark',
                                             mode='lines', line=dict(width=3, dash='dash')))
            fig.update_layout(title='Strategy Returns vs Benchmark ETFs', xaxis_title='Strategy', yaxis_title='Average Return (%)',
                              template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500)

        elif selected_tab == 'win-rate-comparison':
            fig = go.Figure([
                go.Bar(name='>0%', x=strat['strategy'], y=strat['pct_gt_0'], marker_color='#7ee787'),
                go.Bar(name='>5%', x=strat['strategy'], y=strat['pct_gt_5'], marker_color='#58a6ff'),
                go.Bar(name='>10%', x=strat['strategy'], y=strat['pct_gt_10'], marker_color='#e6cc00')
            ])
            fig.update_layout(title='Win Rates by Threshold', xaxis_title='Strategy', yaxis_title='Percentage (%)', barmode='group',
                              template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500)

        elif selected_tab == 'return-distribution':
            fig = go.Figure()
            for sname in df['strategy'].unique():
                sdata = df[df['strategy']==sname]
                all_returns = [r for returns in sdata['returns'] for r in returns]
                if len(all_returns) > 0:
                    fig.add_trace(go.Box(y=all_returns, name=sname))
            for etf in BENCHMARK_ETFS:
                col = f"{etf}_return"
                if col in df.columns:
                    etf_returns = df[col].dropna().tolist()
                    if etf_returns:
                        fig.add_trace(go.Box(y=etf_returns, name=f'{etf} (Benchmark)', marker_color='orange', boxmean=True))
            fig.update_layout(title='Return Distribution by Strategy (vs Benchmarks)', yaxis_title='Return (%)',
                              template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=550)

        elif selected_tab == 'rolling-performance':
            fig = go.Figure()
            for sname in df['strategy'].unique():
                sdata = df[df['strategy']==sname].sort_values('entry_date')
                fig.add_trace(go.Scatter(x=sdata['entry_date'], y=sdata['mean_return'], mode='lines+markers', name=sname))
            for etf in BENCHMARK_ETFS:
                col = f"{etf}_return"
                if col in df.columns:
                    sorted_df = df.sort_values('entry_date')
                    fig.add_trace(go.Scatter(x=sorted_df['entry_date'], y=sorted_df[col], mode='lines',
                                             name=f'{etf} Benchmark', line=dict(width=3, dash='dot'), opacity=0.7))
            fig.update_layout(title='Rolling 30-Day Performance (vs Benchmarks)', xaxis_title='Entry Date', yaxis_title='Mean Return (%)',
                              template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=550)

        return dcc.Graph(figure=fig)

    @app.callback(
        Output('strategy-table-container', 'children'),
        Input('store-results', 'data')
    )
    def update_strategy_table(store_json):
        if store_json is None:
            raise PreventUpdate
        
        df = pd.read_json(store_json, orient='split')
        strat, _ = build_summaries(df)
        strat = strat.round(2)
        
        # Sort by average return descending (same as before)
        strat = strat.sort_values('mean_return', ascending=False)
        
        # Build HTML table
        headers = [
            html.Th(col['name'], style={
                'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#0f172a',
                'color': '#e6edf3', 'fontWeight': 'bold'
            }) for col in table_columns
        ]
        
        rows = []
        for _, row in strat.iterrows():
            cells = []
            for col in table_columns:
                col_id = col['id']
                value = row[col_id]
                # Apply conditional background for mean_return column
                cell_style = {
                    'textAlign': 'center',
                    'padding': '10px',
                    'fontSize': '12px',
                    'backgroundColor': 'transparent',
                    'color': '#e6edf3'
                }
                if col_id == 'mean_return':
                    if value > 5:
                        cell_style['backgroundColor'] = 'rgba(46,204,113,0.2)'
                    elif value < 0:
                        cell_style['backgroundColor'] = 'rgba(231,76,60,0.2)'
                cells.append(html.Td(f"{value:.2f}" if isinstance(value, float) else str(value), style=cell_style))
            rows.append(html.Tr(cells))
        
        table = html.Table([
            html.Thead(html.Tr(headers)),
            html.Tbody(rows)
        ], style={'width': '100%', 'borderCollapse': 'collapse'})    
        return table

    # @app.callback(
    #     [Output('strategy-table', 'data')],
    #     Input('store-results', 'data')
    # )
    # def update_dashboard(store_json):
    #     if store_json is None:
    #         raise PreventUpdate
        
    #     df = pd.read_json(store_json, orient='split')
    #     strat, bench = build_summaries(df)

    #     # Update table
    #     table_data = strat.round(2).to_dict('records')

    #     # Update top picks (your existing logic)
    #     picks, asof = compute_top_picks(df)
    #     top_children = []
    #     for strategy in sorted(set(p['strategy'] for p in picks)):
    #         strategy_picks = [p['ticker'] for p in picks if p['strategy'] == strategy]
    #         top_children.append(
    #             html.Div([
    #                 html.Div(strategy.replace('_', ' ').title(), style={'fontWeight': 'bold', 'marginBottom': '8px', 'color': '#58a6ff'}),
    #                 html.Div([html.Span(className='pick-chip', children=t) for t in strategy_picks],
    #                         style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px'})
    #             ], style={'marginBottom': '20px'})
    #         )

    #     # For tabbed charts: return current chart (or let separate callback handle it)
    #     # Here we just return table and picks
    #     return table_data
    
    return app

# ===== Main =====

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Dark Momentum Dashboard with Horizontal Top Picks')
    p.add_argument('--start', default='2022-01-01')
    p.add_argument('--top', type=int, default=10)
    p.add_argument('--aflag', action='store_true')
    p.add_argument('--skip-download', action='store_true')
    p.add_argument('--port', type=int, default=8050)
    p.add_argument('--months', type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    tickers = [
        "AAPL", "ADP", "ADBE", "AMD", "ABNB", "ALNY", "AEP", "AMGN", "ADI", "AMAT",
        "AMZN", "APP", "ARM", "ASML", "AXON", "AVGO", "BKNG", "CHTR", "CMCSA", "COST",
        "CSCO", "CDNS", "COPART", "CRWD", "DDOG", "DOCU", "DXCM", "EA", "EXC", "FAST",
        "FER", "FISV", "GILD", "GOOG", "GOOGL", "HBAN", "HON", "IDXX", "INTC", "INTU",
        "ISRG", "KDP", "KLAC", "LRCX", "LIND", "MRNA", "META", "MARV", "MRVL", "MSFT",
        "MU", "MLB", "NTES", "NFLX", "NVDA", "NXPI", "ORLY", "ODFL", "PANW", "PDD",
        "PAYX", "PYPL", "PEP", "QCOM", "REGN", "ROST", "SBUX", "SE", "SNPS", "SWKS",
        "SIRI", "TMUS", "TSLA", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XLNX", "XEL",
        "ZS", "JPM", "JNJ", "V", "UNH", "XOM", "MA", "PG", "HD", "BAC", "CVX", "VZ",
        "KO", "ABT", "WMT", "CRM", "TMO", "ACN", "MDT", "NEE", "DHR", "LLY", "MCD",
        "NKE", "QCOM", "LIN", "UPS", "IBM", "CAT", "SCHW", "AXP", "NOW", "PM", "ORCL",
        "GE", "T", "BLK", "SPGI", "C", "PLD", "MMM", "LMT", "USB", "BDX", "TGT", "DE",
        "RTX", "MS", "EL", "SYK", "AMT", "LOW", "GILD", "MO", "CCI", "ISRG", "CL",
        "ZTS", "CSX", "DUK", "SO", "APD", "ETN", "BDX", "ITW", "BSX", "HUM", "BK",
        "MMC", "BIIB", "SLB", "PNC", "TJX", "SHW", "EMR", "OKE", "EXC", "ECL", "MCO",
        "TT", "EW", "HCA", "CME", "SYF", "MTB", "CNC", "TFC", "ES", "CF", "KEY", "STT",
        "AON", "KMB", "ALL", "SRE", "ELV", "MSI", "NOC", "DOW", "ADM", "F", "AIG",
        "CNP", "WBA", "CARR", "ROK", "CPRT", "AWK", "A", "HPQ", "KHC", "AEE", 
        "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA", "BRK.B",
        "LLY", "JPM", "WMT", "V", "ORCL", "MA", "XOM", "JNJ", "PLTR", "BAC", "ABBV",
        "NFLX", "COST", "AMD", "HD", "PG", "GE", "MU", "CSCO", "CVX", "KO", "WFC",
        "UNH", "MS", "IBM", "CAT", "GS", "AXP", "MRK", "PM", "RTX", "CRM", "APP",
        "TMUS", "MCD", "LRCX", "C", "ABT", "TMO", "AMAT", "ISRG", "DIS", "LIN",
        "PEP", "INTU", "QCOM", "GEV", "SCHW", "AMGN", "BKNG", "T", "TJX", "BA",
        "INTC", "VZ", "UBER", "APH", "KLAC", "ACN", "NEE", "ANET", "DHR", "TXN",
        "SPGI", "NOW", "COF", "GILD", "ADBE", "PFE", "BSX", "UNP", "LOW", "ADI",
        "SYK", "PGR", "WELL", "DE", "ETN", "MDT", "HON", "CB", "BX",
        "PLD", "CRWD", "VRTX", "KKR", "COP", "NEM", "CEG", "LMT", "PH", "BMY",
        "HCA", "CMCSA", "ADP", "MCK", "DASH", "CVS", "CME", "MO", "SO", "SBUX",
        "GD", "ICE", "MCO", "MMC", "DUK", "SNPS", "WM", "NKE", "TT", "APO",
        "CDNS", "UPS", "USB", "DELL", "HWM", "MMM", "CRH", "MAR", "PNC", "NOC",
        "ABNB", "REGN", "BK", "AMT", "RCL", "SHW", "ORLY", "GM", "CTAS", "GLW",
        "EMR", "AON", "ELV", "MNST", "ECL", "EQIX", "JCI", "FCX", "WMB", "WBD",
        "CMI", "MDLZ", "FDX", "TEL", "CSX", "HLT", "AJG", "RSG", "COR", "NSC",
        "TRV", "TFC", "CL", "COIN", "PWR", "ADSK", "MSI", "CVNA", "STX", "SPG",
        "AEP", "WDC", "KMI", "FTNT", "ROST", "PCAR", "AFL", "SRE", "AZO", "WDAY",
        "SLB", "EOG", "NXPI", "NDAQ", "LHX", "BDX", "PYPL", "ZTS", "VST", "ALL",
        "IDXX", "APD", "MET", "DLR", "F", "URI", "PSX", "O", "EA", "D", "CMG",
        "VLO", "EW", "MPC", "CAH", "CBRE", "GWW", "ROP", "FAST", "AME", "DDOG",
        "AIG", "AMP", "AXON", "DAL", "TTWO", "OKE", "PSA", "MPWR", "CTVA", "CARR",
        "ROK", "BKR", "LVS", "MSCI", "EXC", "XEL", "TGT", "YUM", "DHI", "FANG",
        "ETR", "PAYX", "CCL", "CTSH", "FICO", "PEG", "PRU", "KR", "TRGP", "OXY",
        "GRMN", "A", "HIG", "VMC", "EL", "MLM", "IQV", "CCI", "EBAY", "GEHC",
        "KDP", "CPRT", "NUE", "WAB", "VTR", "HSY", "ARES", "UAL", "STT", "FISV",
        "ED", "RMD", "KEYS", "SYY", "MCHP", "ACGL", "EXPE", "FIS", "PCG", "WEC",
        "OTIS", "FIX", "XYL", "EQT", "LYV", "KMB", "ODFL", "KVUE", "HPE", "RJF",
        "IR", "WTW", "FITB", "MTB", "HUM", "TER", "SYF", "NRG", "VRSK", "VICI",
        "DG", "IBKR", "ROL", "MTD", "FSLR", "CSGP", "KHC", "HBAN", "EME", "ADM",
        "EXR", "BRO", "DOV", "ATO", "TSCO", "EFX", "AEE", "ULTA", "CHTR", "CBOE",
        "DTE", "WRB", "TPR", "NTRS", "BR", "DXCM", "LEN", "CINF", "AVB", "BIIB",
        "FE", "PPL", "CFG", "AWK", "STLD", "VLTO", "OMC", "STE", "JBL", "ES",
        "GIS", "CNP", "LULU", "TDY", "RF", "STZ", "HUBB", "IRM", "DLTR", "EQR",
        "LDOS", "HAL", "PPG", "PHM", "EIX", "KEY", "VRSN", "WAT", "TROW", "DVN",
        "WSM", "L", "ON", "DRI", "RL", "NTAP", "CPAY", "LUV", "HPQ", "CMS",
        "LH", "PTC", "IP", "TSN", "CHD", "SBAC", "TPL", "EXPD", "PODD", "NVR",
        "WST", "SW", "PFG", "INCY", "NI", "TYL", "CNC", "CTRA", "CHRW", "DGX",
        "GPN", "AMCR", "TRMB", "JBHT", "PKG", "MKC", "SNA", "IT", "SMCI", "TTD",
        "CDW", "ZBH", "FTV", "Q", "GPC", "LII", "PNR", "BG", "ALB", "DD", "TKO",
        "GDDY", "WY", "IFF", "GEN", "ESS", "LNT", "EVRG", "INVH", "HOLX", "APTV",
        "DOW", "COO", "MAA", "J", "TXT", "FOXA", "FOX", "FFIV", "PSKY", "ERIE",
        "DECK", "BBY", "DPZ", "VTRS", "EG", "UHS", "AVY", "BALL", "SOLV", "ALLE",
        "HII", "KIM", "LYB", "NDSN", "IEX", "JKHY", "MAS", "HRL", "WYNN", "AKAM",
        "MRNA", "REG", "HST", "BEN", "ZBRA", "BF.B", "CF", "AIZ", "IVZ", "UDR",
        "CLX", "EPAM", "CPT", "SWK", "GL", "BLDR", "HAS", "ALGN", "DOC", "DAY",
        "RVTY", "BXP", "FDS", "PNW", "SJM", "NCLH", "MGM", "CRL", "AES", "BAX",
        "NWSA", "SWKS", "AOS", "TECH", "TAP", "HSIC", "FRT", "PAYC", "APA", "POOL",
        "MOH", "ARE", "CPB", "GNRC", "CAG", "DVA", "MTCH", "MOS", "LW", "NWS"
    ]
    data_map, benchmark_data = {}, {}
    for etf in BENCHMARK_ETFS:
        df = sync_ticker(etf, args.start, args.aflag, args.skip_download)
        if df is not None and not df.empty:
            benchmark_data[etf] = df
    for t in tickers:
        df = sync_ticker(t, args.start, args.aflag, args.skip_download)
        if df is None or df.empty:
            continue
        df = compute_indicators(df)
        data_map[t] = df
    if len(data_map) == 0:
        print('No tickers loaded.')
        return
    strategies = [
        "momentum_pure","momentum_trend","swing_trader","breakout",
        "volatility_adjusted","value_momentum","quality_momentum",
        "mean_reversion","low_volatility","trending_value","volume_breakout","dividend_momentum","contrarian"
    ]
    backtest_results = run_comprehensive_backtest(data_map, benchmark_data, strategies, top_n=args.top, lookback_days=30, months_back=args.months, n_jobs=-1)
    if backtest_results.empty:
        print('No backtest results.')
        return
    output_path = os.path.join(REPORTS_DIR, f"backtest_top_picks_{today_str()}.csv")
    backtest_results.to_csv(output_path, index=False)
    benchmark_columns = [f"{etf}_return" for etf in BENCHMARK_ETFS]
    app = create_dashboard(backtest_results, benchmark_columns, data_map, benchmark_data, strategies, default_top_n=args.top, default_months_back=args.months)
    if app:
        app.run(debug=False, port=args.port)

if __name__ == '__main__':
    main()

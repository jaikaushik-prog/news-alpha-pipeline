"""
Event study module.

Implements intraday event study methodology for budget speech analysis.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats

from ..utils.logging import get_logger
from ..utils.stats_utils import test_car_significance, newey_west_se

logger = get_logger(__name__)


def calculate_abnormal_returns(
    sector_returns: pd.DataFrame,
    market_returns: pd.Series,
    method: str = 'market_adjusted'
) -> pd.DataFrame:
    """
    Calculate abnormal returns using various methods.
    
    Parameters
    ----------
    sector_returns : pd.DataFrame
        Sector returns (timestamps x sectors)
    market_returns : pd.Series
        Market return series
    method : str
        Method: 'market_adjusted', 'constant_mean', or 'market_model'
        
    Returns
    -------
    pd.DataFrame
        Abnormal returns
    """
    if method == 'market_adjusted':
        # Simple market-adjusted: AR = R - Rm
        ar = sector_returns.sub(market_returns, axis=0)
        
    elif method == 'constant_mean':
        # Mean-adjusted: AR = R - mean(R)
        ar = sector_returns.sub(sector_returns.mean(), axis=0)
        
    elif method == 'market_model':
        # Market model: AR = R - (alpha + beta * Rm)
        ar = sector_returns.copy()
        
        for sector in sector_returns.columns:
            y = sector_returns[sector].dropna()
            x = market_returns.reindex(y.index).dropna()
            
            if len(x) < 30:
                ar[sector] = y - x
                continue
            
            # Estimate OLS
            X = np.column_stack([np.ones(len(x)), x.values])
            try:
                coeffs = np.linalg.lstsq(X, y.values, rcond=None)[0]
                predicted = coeffs[0] + coeffs[1] * market_returns
                ar[sector] = sector_returns[sector] - predicted
            except:
                ar[sector] = sector_returns[sector] - market_returns
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return ar


def calculate_car(
    abnormal_returns: pd.DataFrame,
    event_time: pd.Timestamp,
    horizons: List[int] = None
) -> Dict[str, pd.Series]:
    """
    Calculate Cumulative Abnormal Returns at various horizons.
    
    Parameters
    ----------
    abnormal_returns : pd.DataFrame
        Abnormal returns (timestamps x sectors)
    event_time : pd.Timestamp
        Event timestamp
    horizons : list, optional
        Horizons in minutes (default: [5, 15, 30, 60])
        
    Returns
    -------
    dict
        CAR for each horizon
    """
    if horizons is None:
        horizons = [5, 15, 30, 60]
    
    from datetime import timedelta
    from ..utils.time_utils import get_bar_start
    
    event_bar = get_bar_start(event_time)
    
    cars = {}
    
    for horizon in horizons:
        end_time = event_bar + timedelta(minutes=horizon)
        
        # Get AR in window
        mask = (abnormal_returns.index >= event_bar) & (abnormal_returns.index <= end_time)
        window_ar = abnormal_returns[mask]
        
        # Cumulative sum
        cars[f'car_{horizon}m'] = window_ar.sum()
    
    return cars


def event_study_single(
    abnormal_returns: pd.DataFrame,
    event_time: pd.Timestamp,
    sector: str,
    pre_window: int = 60,
    post_window: int = 120,
    bar_size_minutes: int = 5
) -> Dict:
    """
    Conduct event study for a single event.
    
    Parameters
    ----------
    abnormal_returns : pd.DataFrame
        Abnormal returns
    event_time : pd.Timestamp
        Event timestamp
    sector : str
        Sector name
    pre_window : int
        Pre-event window in minutes
    post_window : int
        Post-event window in minutes
    bar_size_minutes : int
        Size of each bar in minutes
        
    Returns
    -------
    dict
        Event study results
    """
    from datetime import timedelta
    from ..utils.time_utils import get_bar_start
    
    if sector not in abnormal_returns.columns:
        return {}
    
    ar = abnormal_returns[sector]
    event_bar = get_bar_start(event_time)
    
    # Define windows
    pre_start = event_bar - timedelta(minutes=pre_window)
    post_end = event_bar + timedelta(minutes=post_window)
    
    # Pre-event CAR
    pre_mask = (ar.index >= pre_start) & (ar.index < event_bar)
    pre_car = ar[pre_mask].sum()
    
    # Post-event CAR at various horizons
    results = {
        'sector': sector,
        'event_time': event_time,
        'pre_car': pre_car,
    }
    
    for horizon in [5, 15, 30, 60, 120]:
        if horizon <= post_window:
            horizon_end = event_bar + timedelta(minutes=horizon)
            post_mask = (ar.index >= event_bar) & (ar.index <= horizon_end)
            results[f'car_{horizon}m'] = ar[post_mask].sum()
    
    # Volatility around event
    full_mask = (ar.index >= pre_start) & (ar.index <= post_end)
    results['event_volatility'] = ar[full_mask].std()
    
    # Max/min AR
    post_full_mask = (ar.index >= event_bar) & (ar.index <= post_end)
    post_ar = ar[post_full_mask]
    if len(post_ar) > 0:
        results['max_ar'] = post_ar.max()
        results['min_ar'] = post_ar.min()
        results['max_ar_bar'] = post_ar.idxmax()
        results['min_ar_bar'] = post_ar.idxmin()
    
    return results


def event_study_batch(
    panel: pd.DataFrame,
    abnormal_returns_by_date: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Conduct event studies for all events in panel.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel with sector, event_time, budget_date
    abnormal_returns_by_date : dict
        Date -> abnormal returns DataFrame
        
    Returns
    -------
    pd.DataFrame
        Event study results
    """
    results = []
    
    for _, event in panel.iterrows():
        date_str = event.get('budget_date')
        sector = event.get('sector')
        event_time = event.get('event_timestamp') or event.get('event_time')
        
        if date_str not in abnormal_returns_by_date:
            continue
        
        ar = abnormal_returns_by_date[date_str]
        
        result = event_study_single(ar, event_time, sector)
        
        if result:
            result['fiscal_year'] = event.get('fiscal_year')
            results.append(result)
    
    return pd.DataFrame(results)


def test_car_significance_by_sector(
    event_study_results: pd.DataFrame,
    car_column: str = 'car_60m'
) -> pd.DataFrame:
    """
    Test significance of CARs by sector.
    
    Parameters
    ----------
    event_study_results : pd.DataFrame
        Results from event_study_batch
    car_column : str
        CAR column to test
        
    Returns
    -------
    pd.DataFrame
        Significance tests by sector
    """
    tests = []
    
    for sector, group in event_study_results.groupby('sector'):
        cars = group[car_column].dropna().values
        
        if len(cars) < 2:
            continue
        
        result = test_car_significance(cars)
        result['sector'] = sector
        tests.append(result)
    
    return pd.DataFrame(tests)


def calculate_average_car_pattern(
    event_study_results: pd.DataFrame,
    sectors: List[str] = None
) -> pd.DataFrame:
    """
    Calculate average CAR pattern across events.
    
    Parameters
    ----------
    event_study_results : pd.DataFrame
        Event study results
    sectors : list, optional
        Sectors to include
        
    Returns
    -------
    pd.DataFrame
        Average CAR at each horizon
    """
    car_cols = [c for c in event_study_results.columns if c.startswith('car_')]
    
    if sectors:
        data = event_study_results[event_study_results['sector'].isin(sectors)]
    else:
        data = event_study_results
    
    # Calculate mean and standard error
    means = data[car_cols].mean()
    stds = data[car_cols].std()
    counts = data[car_cols].count()
    
    pattern = pd.DataFrame({
        'mean_car': means,
        'std': stds,
        'n': counts,
        'se': stds / np.sqrt(counts),
        't_stat': means / (stds / np.sqrt(counts)),
        'p_value': 2 * (1 - stats.t.cdf(np.abs(means / (stds / np.sqrt(counts))), counts - 1))
    })
    
    # Parse horizon from column names
    pattern['horizon_minutes'] = pattern.index.str.extract(r'car_(\d+)m')[0].astype(int)
    pattern = pattern.sort_values('horizon_minutes')
    
    return pattern


def calculate_volume_weighted_attention(
    attention: float,
    volume: pd.Series,
    event_time: pd.Timestamp,
    lookback_bars: int = 12
) -> float:
    """
    Calculate volume-weighted attention score.
    
    Parameters
    ----------
    attention : float
        Raw attention score
    volume : pd.Series
        Volume series with datetime index
    event_time : pd.Timestamp
        Event timestamp
    lookback_bars : int
        Lookback period for average volume (default: 12 bars = 60 min)
        
    Returns
    -------
    float
        Volume-weighted attention
    """
    if volume.empty:
        return attention
    
    # Get volume at event time
    post_event_vol = volume[volume.index >= event_time]
    if len(post_event_vol) == 0:
        return attention
    
    current_vol = post_event_vol.iloc[0]
    
    # Calculate rolling average
    rolling_avg = volume.rolling(window=lookback_bars, min_periods=1).mean()
    avg_vol = rolling_avg[rolling_avg.index < event_time].iloc[-1] if len(rolling_avg[rolling_avg.index < event_time]) > 0 else current_vol
    
    # Relative volume
    rvol = current_vol / avg_vol if avg_vol > 0 else 1.0
    
    return attention * rvol


def analyze_strategies(
    event_results: pd.DataFrame,
    car_column: str = 'car_60m',
    sentiment_column: str = 'sentiment'
) -> pd.DataFrame:
    """
    Compare Momentum vs Contrarian strategies.
    
    Parameters
    ----------
    event_results : pd.DataFrame
        Event study results with CAR and sentiment
    car_column : str
        CAR column to use for PnL
    sentiment_column : str
        Sentiment column to determine direction
        
    Returns
    -------
    pd.DataFrame
        Strategy performance comparison
    """
    if car_column not in event_results.columns or sentiment_column not in event_results.columns:
        logger.warning(f"Missing {car_column} or {sentiment_column} for strategy analysis")
        return pd.DataFrame()
    
    df = event_results.copy()
    
    # Filter out rows with no sentiment (neutral)
    df = df[df[sentiment_column] != 0]
    
    if df.empty:
        logger.warning("No events with non-zero sentiment for strategy analysis")
        return pd.DataFrame()
    
    # Momentum: Buy if Sentiment > 0, Short if < 0
    df['momentum_pnl'] = np.sign(df[sentiment_column]) * df[car_column]
    
    # Contrarian: Opposite of Momentum
    df['contrarian_pnl'] = -1 * df['momentum_pnl']
    
    strategies = {
        'Momentum (Follow)': df['momentum_pnl'],
        'Contrarian (Fade)': df['contrarian_pnl']
    }
    
    results = []
    
    for name, pnl in strategies.items():
        total_ret = pnl.sum()
        mean_ret = pnl.mean()
        win_rate = (pnl > 0).mean()
        sharpe = (mean_ret / pnl.std()) * np.sqrt(len(pnl)) if pnl.std() > 0 else 0
        
        results.append({
            'strategy': name,
            'total_return': total_ret,
            'avg_trade': mean_ret,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'n_trades': len(pnl)
        })
        
        logger.info(f"{name}: Total={total_ret*100:.3f}%, WinRate={win_rate*100:.1f}%, Sharpe={sharpe:.3f}")
    
    return pd.DataFrame(results)

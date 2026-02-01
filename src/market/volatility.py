"""
Volatility calculation module.

Calculates realized volatility and related metrics at stock and sector level.
"""

from typing import Optional, Union
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


def realized_volatility(
    returns: pd.Series,
    window: int = 12,
    annualize: bool = False,
    annualization_factor: float = 137.48
) -> pd.Series:
    """
    Calculate rolling realized volatility.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    window : int
        Rolling window size (in bars)
    annualize : bool
        Whether to annualize
    annualization_factor : float
        sqrt(bars per year) for annualization
        
    Returns
    -------
    pd.Series
        Rolling volatility
    """
    vol = returns.rolling(window=window, min_periods=max(1, window // 2)).std()
    
    if annualize:
        vol = vol * annualization_factor
    
    return vol


def intraday_volatility(
    df: pd.DataFrame,
    method: str = 'garman_klass'
) -> pd.Series:
    """
    Calculate intraday volatility using OHLC data.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    method : str
        Volatility estimator: 'parkinson', 'garman_klass', or 'rogers_satchell'
        
    Returns
    -------
    pd.Series
        Volatility estimate
    """
    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']
    
    if method == 'parkinson':
        # Parkinson volatility
        vol = np.sqrt((np.log(h / l) ** 2) / (4 * np.log(2)))
        
    elif method == 'garman_klass':
        # Garman-Klass volatility
        vol = np.sqrt(
            0.5 * np.log(h / l) ** 2 - 
            (2 * np.log(2) - 1) * np.log(c / o) ** 2
        )
        
    elif method == 'rogers_satchell':
        # Rogers-Satchell volatility
        vol = np.sqrt(
            np.log(h / c) * np.log(h / o) + 
            np.log(l / c) * np.log(l / o)
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return vol


def volatility_ratio(
    returns: pd.Series,
    short_window: int = 12,
    long_window: int = 60
) -> pd.Series:
    """
    Calculate volatility ratio (short-term / long-term).
    
    High ratio indicates volatility spike.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    short_window : int
        Short-term window
    long_window : int
        Long-term window
        
    Returns
    -------
    pd.Series
        Volatility ratio
    """
    short_vol = realized_volatility(returns, window=short_window)
    long_vol = realized_volatility(returns, window=long_window)
    
    return short_vol / long_vol.replace(0, np.nan)


def volatility_change(
    returns: pd.Series,
    window: int = 12,
    pct_change_periods: int = 1
) -> pd.Series:
    """
    Calculate change in volatility.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    window : int
        Rolling window for volatility
    pct_change_periods : int
        Periods for percent change
        
    Returns
    -------
    pd.Series
        Volatility percent change
    """
    vol = realized_volatility(returns, window=window)
    return vol.pct_change(periods=pct_change_periods)


def daily_aggregated_volatility(
    df: pd.DataFrame,
    return_col: str = 'return'
) -> pd.DataFrame:
    """
    Aggregate 5-min volatility to daily level.
    
    Parameters
    ----------
    df : pd.DataFrame
        5-minute data with returns
    return_col : str
        Name of return column
        
    Returns
    -------
    pd.DataFrame
        Daily volatility metrics
    """
    # Group by date
    df = df.copy()
    df['trading_date'] = df.index.date
    
    daily = df.groupby('trading_date').agg({
        return_col: [
            ('realized_vol', lambda x: x.std() * np.sqrt(75)),  # Annualize within day
            ('open_vol', lambda x: x.head(12).std()),  # First hour
            ('close_vol', lambda x: x.tail(12).std()),  # Last hour
            ('max_abs_return', lambda x: x.abs().max()),
        ]
    })
    
    # Flatten column names
    daily.columns = [f"{col[1]}" for col in daily.columns]
    
    return daily


def sector_volatility(
    sector_returns: pd.DataFrame,
    window: int = 12
) -> pd.DataFrame:
    """
    Calculate volatility for all sectors.
    
    Parameters
    ----------
    sector_returns : pd.DataFrame
        Sector returns (timestamps x sectors)
    window : int
        Rolling window
        
    Returns
    -------
    pd.DataFrame
        Sector volatilities
    """
    vol = sector_returns.apply(
        lambda x: realized_volatility(x, window=window)
    )
    
    return vol


def pre_post_event_volatility(
    returns: pd.Series,
    event_idx: int,
    pre_window: int = 12,
    post_window: int = 12
) -> dict:
    """
    Calculate volatility before and after an event.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    event_idx : int
        Index of the event
    pre_window : int
        Bars before event
    post_window : int
        Bars after event
        
    Returns
    -------
    dict
        Pre and post volatility
    """
    pre_start = max(0, event_idx - pre_window)
    pre_returns = returns.iloc[pre_start:event_idx]
    
    post_end = min(len(returns), event_idx + post_window)
    post_returns = returns.iloc[event_idx:post_end]
    
    return {
        'pre_volatility': pre_returns.std() if len(pre_returns) > 1 else np.nan,
        'post_volatility': post_returns.std() if len(post_returns) > 1 else np.nan,
        'volatility_change': (
            (post_returns.std() / pre_returns.std() - 1) 
            if len(pre_returns) > 1 and len(post_returns) > 1 and pre_returns.std() > 0
            else np.nan
        )
    }

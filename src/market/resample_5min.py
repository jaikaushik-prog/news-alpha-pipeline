"""
5-minute bar resampling and cleaning module.

Validates, cleans, and aligns 5-minute OHLCV data.
"""

from typing import Dict, List, Optional, Tuple
from datetime import date, datetime, time
import pandas as pd
import numpy as np

from ..utils.logging import get_logger
from ..utils.time_utils import (
    IST, MARKET_OPEN, MARKET_CLOSE, BAR_SIZE_MINUTES,
    get_trading_bars, is_market_hours
)

logger = get_logger(__name__)


def validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and fix OHLC data integrity.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
        
    Returns
    -------
    pd.DataFrame
        Validated data with fixes applied
    """
    df = df.copy()
    
    # Ensure high >= low
    invalid_hl = df['high'] < df['low']
    if invalid_hl.any():
        logger.warning(f"Found {invalid_hl.sum()} bars with high < low, swapping")
        df.loc[invalid_hl, ['high', 'low']] = df.loc[invalid_hl, ['low', 'high']].values
    
    # Ensure open and close are within high-low range
    df['open'] = df['open'].clip(lower=df['low'], upper=df['high'])
    df['close'] = df['close'].clip(lower=df['low'], upper=df['high'])
    
    # Replace zero/negative prices with NaN
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        invalid = df[col] <= 0
        if invalid.any():
            logger.warning(f"Found {invalid.sum()} zero/negative {col} values")
            df.loc[invalid, col] = np.nan
    
    # Replace negative volume with 0
    if 'volume' in df.columns:
        df['volume'] = df['volume'].clip(lower=0)
    
    return df


def filter_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data to market hours only.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with datetime index
        
    Returns
    -------
    pd.DataFrame
        Filtered data
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Index is not DatetimeIndex, attempting conversion")
        df.index = pd.to_datetime(df.index)
    
    # Extract time
    times = df.index.time
    
    # Filter to market hours (09:15 - 15:25 for 5-min bars)
    # Last bar starts at 15:25, captures 15:25-15:30
    mask = (times >= MARKET_OPEN) & (times <= time(15, 25))
    
    filtered = df[mask]
    
    if len(filtered) < len(df):
        logger.info(f"Filtered {len(df) - len(filtered)} bars outside market hours")
    
    return filtered


def fill_missing_bars(
    df: pd.DataFrame,
    trading_date: date,
    method: str = 'ffill'
) -> pd.DataFrame:
    """
    Fill missing 5-minute bars for a trading day.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data for a single day
    trading_date : date
        The trading date
    method : str
        Fill method: 'ffill', 'bfill', or 'interpolate'
        
    Returns
    -------
    pd.DataFrame
        Data with missing bars filled
    """
    # Generate expected timestamps
    expected = get_trading_bars(trading_date)
    
    # Reindex to expected
    df = df.reindex(expected)
    
    # Fill missing values
    if method == 'ffill':
        df = df.ffill()
    elif method == 'bfill':
        df = df.bfill()
    elif method == 'interpolate':
        df = df.interpolate(method='time')
    
    # Fill any remaining NaNs at start with bfill
    df = df.bfill()
    
    return df


def calculate_returns(
    df: pd.DataFrame,
    price_col: str = 'close',
    log_returns: bool = False
) -> pd.Series:
    """
    Calculate returns from price data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Price data
    price_col : str
        Column to use for returns
    log_returns : bool
        If True, calculate log returns
        
    Returns
    -------
    pd.Series
        Returns series
    """
    prices = df[price_col]
    
    if log_returns:
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    
    return returns


def clean_stock_data(
    df: pd.DataFrame,
    fill_missing: bool = True,
    validate: bool = True
) -> pd.DataFrame:
    """
    Full cleaning pipeline for a single stock.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data
    fill_missing : bool
        Whether to fill missing bars
    validate : bool
        Whether to validate OHLC integrity
        
    Returns
    -------
    pd.DataFrame
        Cleaned data
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Validate OHLC
    if validate:
        df = validate_ohlc(df)
    
    # Filter to trading hours
    df = filter_trading_hours(df)
    
    # Sort by index
    df = df.sort_index()
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Add returns
    df['return'] = calculate_returns(df)
    
    return df


def create_common_index(
    data: Dict[str, pd.DataFrame],
    min_coverage: float = 0.8
) -> Tuple[pd.DatetimeIndex, Dict[str, pd.DataFrame]]:
    """
    Create a common datetime index across all stocks.
    
    Parameters
    ----------
    data : dict
        Dictionary of symbol -> DataFrame
    min_coverage : float
        Minimum coverage required to include a timestamp
        
    Returns
    -------
    tuple
        (common_index, aligned_data)
    """
    if not data:
        return pd.DatetimeIndex([]), {}
    
    # Get all unique timestamps
    all_timestamps = set()
    for df in data.values():
        all_timestamps.update(df.index.tolist())
    
    all_timestamps = sorted(all_timestamps)
    
    # Count coverage at each timestamp
    coverage = pd.Series(0, index=all_timestamps)
    for df in data.values():
        for ts in df.index:
            if ts in coverage.index:
                coverage[ts] += 1
    
    # Filter to timestamps with sufficient coverage
    threshold = min_coverage * len(data)
    common_index = coverage[coverage >= threshold].index
    
    logger.info(f"Created common index with {len(common_index)} timestamps "
                f"({len(all_timestamps) - len(common_index)} excluded)")
    
    # Reindex all DataFrames
    aligned = {}
    for symbol, df in data.items():
        aligned[symbol] = df.reindex(common_index)
    
    return pd.DatetimeIndex(common_index), aligned


def resample_to_5min(
    df: pd.DataFrame,
    freq: str = '5min'
) -> pd.DataFrame:
    """
    Resample data to 5-minute bars (if not already).
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    freq : str
        Target frequency
        
    Returns
    -------
    pd.DataFrame
        Resampled data
    """
    # Already at 5-min, just aggregate any duplicates
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Only include columns that exist
    agg_dict = {k: v for k, v in ohlc_dict.items() if k in df.columns}
    
    return df.resample(freq).agg(agg_dict).dropna()

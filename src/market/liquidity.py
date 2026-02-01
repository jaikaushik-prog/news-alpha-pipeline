"""
Liquidity metrics module.

Calculates various liquidity proxies for intraday data.
"""

from typing import Optional
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


def amihud_illiquidity(
    returns: pd.Series,
    volume: pd.Series,
    window: int = 20,
    scale: float = 1e6
) -> pd.Series:
    """
    Calculate Amihud illiquidity measure.
    
    Amihud = |return| / (volume * scale)
    
    Higher values indicate lower liquidity.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    volume : pd.Series
        Volume series  
    window : int
        Rolling window for averaging
    scale : float
        Scaling factor for volume
        
    Returns
    -------
    pd.Series
        Rolling Amihud illiquidity
    """
    # Avoid division by zero
    volume_safe = volume.replace(0, np.nan)
    
    # Calculate bar-level illiquidity
    illiq = np.abs(returns) / (volume_safe / scale)
    
    # Rolling average
    return illiq.rolling(window=window, min_periods=max(1, window // 2)).mean()


def roll_spread(
    returns: pd.Series,
    window: int = 60
) -> pd.Series:
    """
    Calculate Roll implied spread from return autocovariance.
    
    Roll = 2 * sqrt(-cov(r_t, r_{t-1}))
    
    Based on the idea that bid-ask bounce causes negative autocorrelation.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    window : int
        Rolling window
        
    Returns
    -------
    pd.Series
        Rolling implied spread
    """
    def calc_roll(x):
        if len(x) < 2:
            return np.nan
        # Autocovariance at lag 1
        cov = np.cov(x[:-1], x[1:])[0, 1]
        if cov >= 0:
            return 0  # No negative autocorrelation
        return 2 * np.sqrt(-cov)
    
    return returns.rolling(window=window).apply(calc_roll, raw=True)


def volume_anomaly(
    volume: pd.Series,
    baseline_window: int = 60,
    current_window: int = 1
) -> pd.Series:
    """
    Calculate volume anomaly (current vs historical).
    
    Value > 1 indicates above-average volume.
    
    Parameters
    ----------
    volume : pd.Series
        Volume series
    baseline_window : int
        Window for baseline calculation
    current_window : int
        Window for current volume (1 = single bar)
        
    Returns
    -------
    pd.Series
        Volume ratio
    """
    baseline = volume.rolling(window=baseline_window).mean()
    
    if current_window > 1:
        current = volume.rolling(window=current_window).mean()
    else:
        current = volume
    
    return current / baseline.replace(0, np.nan)


def effective_spread_proxy(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> pd.Series:
    """
    Calculate effective spread proxy from OHLC data.
    
    Uses high-low range as a proxy for bid-ask spread.
    
    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
        
    Returns
    -------
    pd.Series
        Spread proxy (as fraction of price)
    """
    return (high - low) / close


def kyle_lambda(
    returns: pd.Series,
    volume: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Calculate Kyle's lambda (price impact coefficient).
    
    Lambda = Cov(r, sign(V)) / Var(sign(V))
    
    Approximated using rolling regression.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    volume : pd.Series
        Signed volume (positive for buys, negative for sells)
        If unsigned, we use volume * sign(return) as proxy
    window : int
        Rolling window
        
    Returns
    -------
    pd.Series
        Rolling Kyle's lambda
    """
    # Use return sign as order flow proxy
    signed_volume = volume * np.sign(returns)
    
    def calc_lambda(data):
        r = data[:len(data)//2]
        v = data[len(data)//2:]
        if len(r) < 2:
            return np.nan
        try:
            cov = np.cov(r, v)[0, 1]
            var = np.var(v)
            if var == 0:
                return np.nan
            return cov / var
        except:
            return np.nan
    
    # Combine for rolling application
    combined = pd.concat([returns, signed_volume], axis=0)
    
    # Simple implementation using correlation as proxy
    correlation = returns.rolling(window=window).corr(signed_volume)
    vol_r = returns.rolling(window=window).std()
    vol_v = signed_volume.rolling(window=window).std()
    
    return correlation * vol_r / vol_v.replace(0, np.nan)


def turnover_ratio(
    volume: pd.Series,
    shares_outstanding: float = 1e9
) -> pd.Series:
    """
    Calculate turnover ratio.
    
    Parameters
    ----------
    volume : pd.Series
        Volume series
    shares_outstanding : float
        Total shares outstanding (default 1B for normalization)
        
    Returns
    -------
    pd.Series
        Turnover ratio
    """
    return volume / shares_outstanding


def calculate_all_liquidity_metrics(
    df: pd.DataFrame,
    windows: dict = None
) -> pd.DataFrame:
    """
    Calculate all liquidity metrics for a stock.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with returns
    windows : dict, optional
        Custom window sizes
        
    Returns
    -------
    pd.DataFrame
        DataFrame with liquidity metrics added
    """
    if windows is None:
        windows = {
            'amihud': 20,
            'roll': 60,
            'volume_anomaly': 60
        }
    
    result = df.copy()
    
    # Amihud illiquidity
    if 'return' in df.columns and 'volume' in df.columns:
        result['amihud'] = amihud_illiquidity(
            df['return'], df['volume'], 
            window=windows.get('amihud', 20)
        )
    
    # Roll spread
    if 'return' in df.columns:
        result['roll_spread'] = roll_spread(
            df['return'],
            window=windows.get('roll', 60)
        )
    
    # Volume anomaly
    if 'volume' in df.columns:
        result['volume_anomaly'] = volume_anomaly(
            df['volume'],
            baseline_window=windows.get('volume_anomaly', 60)
        )
    
    # Effective spread proxy
    if all(col in df.columns for col in ['high', 'low', 'close']):
        result['spread_proxy'] = effective_spread_proxy(
            df['high'], df['low'], df['close']
        )
    
    return result

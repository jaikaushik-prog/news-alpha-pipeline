"""
Event windows module.

Builds pre/post event windows around sector mentions.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..utils.logging import get_logger
from ..utils.time_utils import get_bar_start, get_event_window_bars, IST

logger = get_logger(__name__)


# Default event windows (in minutes)
DEFAULT_PRE_WINDOWS = [-60, -30, -15, -5]
DEFAULT_POST_WINDOWS = [5, 15, 30, 60, 120]


def create_event_window(
    event_time: datetime,
    pre_windows: List[int] = None,
    post_windows: List[int] = None
) -> Dict[str, datetime]:
    """
    Create event window timestamps.
    
    Parameters
    ----------
    event_time : datetime
        The event timestamp (e.g., sector mention time)
    pre_windows : list, optional
        Pre-event window minutes (negative)
    post_windows : list, optional
        Post-event window minutes (positive)
        
    Returns
    -------
    dict
        Dictionary of window label -> timestamp
    """
    if pre_windows is None:
        pre_windows = DEFAULT_PRE_WINDOWS
    if post_windows is None:
        post_windows = DEFAULT_POST_WINDOWS
    
    event_bar = get_bar_start(event_time)
    
    windows = {'event': event_bar}
    
    for mins in pre_windows:
        label = f't{mins}' if mins >= 0 else f't{mins}'
        windows[label] = event_bar + timedelta(minutes=mins)
    
    for mins in post_windows:
        label = f't+{mins}'
        windows[label] = event_bar + timedelta(minutes=mins)
    
    return windows


def extract_window_returns(
    returns: pd.Series,
    event_time: datetime,
    pre_windows: List[int] = None,
    post_windows: List[int] = None
) -> Dict[str, float]:
    """
    Extract returns at event window points.
    
    Parameters
    ----------
    returns : pd.Series
        Return series with datetime index
    event_time : datetime
        Event timestamp
    pre_windows : list, optional
        Pre-event windows
    post_windows : list, optional
        Post-event windows
        
    Returns
    -------
    dict
        Returns at each window point
    """
    windows = create_event_window(event_time, pre_windows, post_windows)
    
    result = {}
    
    for label, ts in windows.items():
        # Find nearest timestamp
        if ts in returns.index:
            result[f'ret_{label}'] = returns.loc[ts]
        else:
            # Find nearest
            idx = returns.index.get_indexer([ts], method='nearest')[0]
            if 0 <= idx < len(returns):
                result[f'ret_{label}'] = returns.iloc[idx]
            else:
                result[f'ret_{label}'] = np.nan
    
    return result


def calculate_cumulative_returns(
    returns: pd.Series,
    event_time: datetime,
    horizons: List[int] = None
) -> Dict[str, float]:
    """
    Calculate cumulative returns at various horizons after event.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    event_time : datetime
        Event timestamp
    horizons : list, optional
        Horizons in minutes (default: [5, 15, 30, 60])
        
    Returns
    -------
    dict
        Cumulative returns at each horizon
    """
    if horizons is None:
        horizons = [5, 15, 30, 60]
    
    event_bar = get_bar_start(event_time)
    
    result = {}
    
    for horizon in horizons:
        end_time = event_bar + timedelta(minutes=horizon)
        
        # Get returns in window
        mask = (returns.index >= event_bar) & (returns.index <= end_time)
        window_returns = returns[mask]
        
        if len(window_returns) > 0:
            # Cumulative return (product of 1+r)
            cum_ret = (1 + window_returns).prod() - 1
            result[f'cum_ret_{horizon}m'] = cum_ret
        else:
            result[f'cum_ret_{horizon}m'] = np.nan
    
    return result


def build_event_panel(
    mentions_df: pd.DataFrame,
    sector_returns: pd.DataFrame,
    pre_windows: List[int] = None,
    post_windows: List[int] = None,
    horizons: List[int] = None
) -> pd.DataFrame:
    """
    Build panel dataset of events with returns.
    
    Parameters
    ----------
    mentions_df : pd.DataFrame
        First mentions data with sector and timestamp
    sector_returns : pd.DataFrame
        Sector returns (timestamps x sectors)
    pre_windows : list, optional
        Pre-event windows
    post_windows : list, optional
        Post-event windows
    horizons : list, optional
        Cumulative return horizons
        
    Returns
    -------
    pd.DataFrame
        Event panel with returns
    """
    if mentions_df.empty:
        return pd.DataFrame()
    
    panels = []
    
    for _, mention in mentions_df.iterrows():
        sector = mention['sector']
        event_time = mention.get('estimated_timestamp')
        
        if event_time is None or pd.isna(event_time):
            continue
        
        # Get sector returns
        if sector not in sector_returns.columns:
            continue
        
        returns = sector_returns[sector]
        
        # Build event record
        record = {
            'sector': sector,
            'event_time': event_time,
            'sentence_position': mention.get('sentence_position'),
            'cumulative_attention': mention.get('cumulative_attention'),
        }
        
        # Add window returns
        window_returns = extract_window_returns(
            returns, event_time, pre_windows, post_windows
        )
        record.update(window_returns)
        
        # Add cumulative returns
        cum_returns = calculate_cumulative_returns(returns, event_time, horizons)
        record.update(cum_returns)
        
        panels.append(record)
    
    df = pd.DataFrame(panels)
    
    logger.info(f"Built event panel with {len(df)} events")
    
    return df


def add_pre_event_metrics(
    panel: pd.DataFrame,
    sector_returns: pd.DataFrame,
    sector_volatility: pd.DataFrame,
    lookback_bars: int = 12
) -> pd.DataFrame:
    """
    Add pre-event metrics to event panel.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel
    sector_returns : pd.DataFrame
        Sector returns
    sector_volatility : pd.DataFrame
        Sector volatility (or will be calculated)
    lookback_bars : int
        Lookback period for metrics
        
    Returns
    -------
    pd.DataFrame
        Panel with pre-event metrics added
    """
    result = panel.copy()
    
    pre_vol = []
    pre_ret = []
    
    for _, row in result.iterrows():
        sector = row['sector']
        event_time = row['event_time']
        
        if sector not in sector_returns.columns:
            pre_vol.append(np.nan)
            pre_ret.append(np.nan)
            continue
        
        returns = sector_returns[sector]
        
        # Find event bar index
        event_bar = get_bar_start(event_time)
        
        # Get pre-event window
        mask = returns.index < event_bar
        pre_returns = returns[mask].tail(lookback_bars)
        
        if len(pre_returns) > 1:
            pre_vol.append(pre_returns.std())
            pre_ret.append(pre_returns.sum())
        else:
            pre_vol.append(np.nan)
            pre_ret.append(np.nan)
    
    result['pre_volatility'] = pre_vol
    result['pre_return'] = pre_ret
    
    return result


def add_market_controls(
    panel: pd.DataFrame,
    market_returns: pd.Series,
    horizons: List[int] = None
) -> pd.DataFrame:
    """
    Add market return controls to event panel.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel
    market_returns : pd.Series
        Market (NIFTY 500) returns
    horizons : list, optional
        Horizons for market returns
        
    Returns
    -------
    pd.DataFrame
        Panel with market controls
    """
    if horizons is None:
        horizons = [5, 15, 30, 60]
    
    result = panel.copy()
    
    for horizon in horizons:
        mkt_rets = []
        
        for _, row in result.iterrows():
            event_time = row['event_time']
            cum_ret = calculate_cumulative_returns(
                market_returns, event_time, [horizon]
            ).get(f'cum_ret_{horizon}m', np.nan)
            mkt_rets.append(cum_ret)
        
        result[f'mkt_ret_{horizon}m'] = mkt_rets
    
    return result

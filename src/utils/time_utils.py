"""
Time utilities for Budget Speech Impact Analysis.

Handles:
- IST timezone operations
- Trading calendar functions
- Intraday time alignment
"""

from datetime import datetime, date, time, timedelta
from typing import List, Optional, Tuple, Union
import pytz
import pandas as pd
import numpy as np


# IST Timezone
IST = pytz.timezone("Asia/Kolkata")

# Trading hours
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# 5-minute bar configuration
BAR_SIZE_MINUTES = 5
BARS_PER_DAY = 75  # (15:30 - 09:15) / 5 minutes


def to_ist(dt: Union[datetime, pd.Timestamp]) -> datetime:
    """
    Convert datetime to IST timezone.
    
    Parameters
    ----------
    dt : datetime or pd.Timestamp
        Input datetime (timezone-aware or naive)
        
    Returns
    -------
    datetime
        IST-localized datetime
    """
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    
    if dt.tzinfo is None:
        return IST.localize(dt)
    else:
        return dt.astimezone(IST)


def is_market_hours(dt: datetime) -> bool:
    """
    Check if datetime falls within market hours.
    
    Parameters
    ----------
    dt : datetime
        Datetime to check
        
    Returns
    -------
    bool
        True if within market hours (09:15 - 15:30 IST)
    """
    dt_ist = to_ist(dt)
    t = dt_ist.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


def get_bar_start(dt: datetime) -> datetime:
    """
    Get the start time of the 5-minute bar containing this datetime.
    
    Parameters
    ----------
    dt : datetime
        Any datetime
        
    Returns
    -------
    datetime
        Start of the 5-minute bar
    """
    dt_ist = to_ist(dt)
    minutes = dt_ist.minute
    bar_start_minute = (minutes // BAR_SIZE_MINUTES) * BAR_SIZE_MINUTES
    
    return dt_ist.replace(minute=bar_start_minute, second=0, microsecond=0)


def get_trading_bars(trading_date: date) -> pd.DatetimeIndex:
    """
    Generate all 5-minute bar timestamps for a trading day.
    
    Parameters
    ----------
    trading_date : date
        The trading date
        
    Returns
    -------
    pd.DatetimeIndex
        All 75 bar timestamps (09:15, 09:20, ..., 15:25)
    """
    start = IST.localize(datetime.combine(trading_date, MARKET_OPEN))
    end = IST.localize(datetime.combine(trading_date, time(15, 25)))  # Last bar starts at 15:25
    
    return pd.date_range(start=start, end=end, freq=f"{BAR_SIZE_MINUTES}min")


def get_event_window_bars(
    event_time: datetime,
    pre_minutes: List[int],
    post_minutes: List[int]
) -> Tuple[List[datetime], List[datetime]]:
    """
    Get bar timestamps for pre and post event windows.
    
    Parameters
    ----------
    event_time : datetime
        The event timestamp (e.g., sector mention time)
    pre_minutes : list of int
        Negative minutes for pre-event window (e.g., [-30, -15, -5])
    post_minutes : list of int
        Positive minutes for post-event window (e.g., [5, 15, 30, 60])
        
    Returns
    -------
    tuple
        (pre_event_bars, post_event_bars) as lists of datetimes
    """
    event_bar = get_bar_start(event_time)
    
    pre_bars = [
        event_bar + timedelta(minutes=m)
        for m in sorted(pre_minutes)
    ]
    
    post_bars = [
        event_bar + timedelta(minutes=m)
        for m in sorted(post_minutes)
    ]
    
    return pre_bars, post_bars


def minutes_since_open(dt: datetime) -> float:
    """
    Calculate minutes elapsed since market open.
    
    Parameters
    ----------
    dt : datetime
        Any datetime
        
    Returns
    -------
    float
        Minutes since 09:15 IST (can be negative if before open)
    """
    dt_ist = to_ist(dt)
    open_dt = dt_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    
    delta = dt_ist - open_dt
    return delta.total_seconds() / 60


def speech_time_to_bar(
    speech_start: datetime,
    sentence_position: int,
    total_sentences: int,
    speech_duration_minutes: float
) -> datetime:
    """
    Estimate the bar timestamp for a sentence based on its position.
    
    Assumes linear progression through the speech.
    
    Parameters
    ----------
    speech_start : datetime
        When the speech started
    sentence_position : int
        0-indexed position of the sentence
    total_sentences : int
        Total number of sentences in the speech
    speech_duration_minutes : float
        Total speech duration in minutes
        
    Returns
    -------
    datetime
        Estimated bar timestamp for this sentence
    """
    if total_sentences <= 1:
        progress = 0
    else:
        progress = sentence_position / (total_sentences - 1)
    
    elapsed_minutes = progress * speech_duration_minutes
    sentence_time = speech_start + timedelta(minutes=elapsed_minutes)
    
    return get_bar_start(sentence_time)


def get_budget_date_bars(
    budget_date: date,
    speech_start: time,
    speech_end: time
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Get pre-speech, during-speech, and post-speech bars for a budget day.
    
    Parameters
    ----------
    budget_date : date
        The budget presentation date
    speech_start : time
        Speech start time
    speech_end : time
        Speech end time
        
    Returns
    -------
    tuple
        (pre_speech_bars, speech_bars, post_speech_bars)
    """
    all_bars = get_trading_bars(budget_date)
    
    speech_start_dt = IST.localize(datetime.combine(budget_date, speech_start))
    speech_end_dt = IST.localize(datetime.combine(budget_date, speech_end))
    
    pre_speech = all_bars[all_bars < speech_start_dt]
    during_speech = all_bars[(all_bars >= speech_start_dt) & (all_bars <= speech_end_dt)]
    post_speech = all_bars[all_bars > speech_end_dt]
    
    return pre_speech, during_speech, post_speech


def create_aligned_datetime_index(
    start_date: date,
    end_date: date,
    include_weekends: bool = False
) -> pd.DatetimeIndex:
    """
    Create a DatetimeIndex of all trading bar timestamps between dates.
    
    Parameters
    ----------
    start_date : date
        Start date
    end_date : date
        End date
    include_weekends : bool
        Whether to include weekends (default False)
        
    Returns
    -------
    pd.DatetimeIndex
        All trading bar timestamps
    """
    all_bars = []
    current = start_date
    
    while current <= end_date:
        # Skip weekends unless requested
        if not include_weekends and current.weekday() >= 5:
            current += timedelta(days=1)
            continue
        
        day_bars = get_trading_bars(current)
        all_bars.extend(day_bars)
        current += timedelta(days=1)
    
    return pd.DatetimeIndex(all_bars)

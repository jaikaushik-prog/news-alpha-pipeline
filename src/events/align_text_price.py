"""
Text-price alignment module.

Merges speech events with price reactions.
"""

from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from ..utils.logging import get_logger
from ..ingestion.load_speeches import get_speech_metadata, get_speech_duration_minutes
from ..utils.time_utils import IST

logger = get_logger(__name__)


def align_speech_to_market(
    sentences_df: pd.DataFrame,
    fiscal_year: str,
    speech_metadata: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Align speech sentences to market timestamps.
    
    Parameters
    ----------
    sentences_df : pd.DataFrame
        Tokenized sentences
    fiscal_year : str
        Fiscal year for this speech
    speech_metadata : dict, optional
        Speech timing metadata
        
    Returns
    -------
    pd.DataFrame
        Sentences with aligned timestamps
    """
    if speech_metadata is None:
        speech_metadata = get_speech_metadata(fiscal_year)
    
    if speech_metadata is None:
        logger.warning(f"No metadata found for {fiscal_year}")
        return sentences_df
    
    result = sentences_df.copy()
    
    # Parse speech start time
    date_str = speech_metadata['date']
    start_str = speech_metadata.get('speech_start', '11:00')
    
    speech_start = datetime.strptime(
        f"{date_str} {start_str}",
        "%Y-%m-%d %H:%M"
    )
    speech_start = IST.localize(speech_start)
    
    # Get speech duration
    duration = get_speech_duration_minutes(fiscal_year)
    
    # Calculate timestamps based on position
    total_sentences = len(result)
    
    timestamps = []
    for i in range(total_sentences):
        if total_sentences > 1:
            progress = i / (total_sentences - 1)
        else:
            progress = 0
        
        from datetime import timedelta
        elapsed = progress * duration
        ts = speech_start + timedelta(minutes=elapsed)
        timestamps.append(ts)
    
    result['aligned_timestamp'] = timestamps
    result['fiscal_year'] = fiscal_year
    result['budget_date'] = speech_metadata['date']
    
    return result


def create_sector_event_record(
    mention: Dict,
    fiscal_year: str,
    sector_returns: pd.Series,
    sector_volatility: Optional[pd.Series] = None,
    market_returns: Optional[pd.Series] = None
) -> Dict:
    """
    Create a complete event record for a sector mention.
    
    Parameters
    ----------
    mention : dict
        First mention information
    fiscal_year : str
        Fiscal year
    sector_returns : pd.Series
        Sector return series
    sector_volatility : pd.Series, optional
        Sector volatility series
    market_returns : pd.Series, optional
        Market return series
        
    Returns
    -------
    dict
        Complete event record
    """
    from .event_windows import calculate_cumulative_returns, extract_window_returns
    from ..market.volatility import pre_post_event_volatility
    
    event_time = mention.get('estimated_timestamp')
    
    record = {
        'fiscal_year': fiscal_year,
        'sector': mention['sector'],
        'event_timestamp': event_time,
        'sentence_position': mention.get('sentence_position'),
        'cumulative_attention': mention.get('cumulative_attention'),
        'mention_probability': mention.get('probability'),
        'sentiment': mention.get('avg_sentiment', 0.0),
    }
    
    if event_time is None:
        return record
    
    # Add return windows
    if sector_returns is not None and len(sector_returns) > 0:
        window_rets = extract_window_returns(sector_returns, event_time)
        record.update(window_rets)
        
        cum_rets = calculate_cumulative_returns(sector_returns, event_time)
        record.update(cum_rets)
    
    # Add volatility
    if sector_volatility is not None and len(sector_volatility) > 0:
        # Find nearest volatility values
        vol_idx = sector_volatility.index.get_indexer([event_time], method='nearest')[0]
        if 0 <= vol_idx < len(sector_volatility):
            record['event_volatility'] = sector_volatility.iloc[vol_idx]
    
    # Add pre/post volatility from returns
    if sector_returns is not None and len(sector_returns) > 0:
        event_idx = sector_returns.index.get_indexer([event_time], method='nearest')[0]
        if 0 <= event_idx < len(sector_returns):
            vol_metrics = pre_post_event_volatility(sector_returns, event_idx)
            record.update(vol_metrics)
    
    # Add market returns
    if market_returns is not None and len(market_returns) > 0:
        mkt_cum = calculate_cumulative_returns(market_returns, event_time)
        record.update({f'mkt_{k}': v for k, v in mkt_cum.items()})
    
    return record


def build_aligned_panel(
    sentences_by_year: Dict[str, pd.DataFrame],
    mentions_by_year: Dict[str, pd.DataFrame],
    sector_returns_by_date: Dict[str, pd.DataFrame],
    market_returns: Optional[pd.Series] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Build complete aligned panel across all years.
    
    Parameters
    ----------
    sentences_by_year : dict
        Fiscal year -> sentences DataFrame
    mentions_by_year : dict
        Fiscal year -> mentions DataFrame  
    sector_returns_by_date : dict
        Date string -> sector returns DataFrame
    market_returns : pd.Series, optional
        Overall market returns
    output_path : Path, optional
        Where to save output
        
    Returns
    -------
    pd.DataFrame
        Complete aligned panel
    """
    all_records = []
    
    for fiscal_year, mentions_df in mentions_by_year.items():
        if mentions_df.empty:
            continue
        
        # Get corresponding sentences
        sentences_df = sentences_by_year.get(fiscal_year, pd.DataFrame())
        
        # Get budget date
        from ..ingestion.load_speeches import get_speech_metadata
        metadata = get_speech_metadata(fiscal_year)
        
        if metadata is None:
            continue
        
        budget_date = metadata['date']
        
        # Get sector returns for this date
        sector_returns = sector_returns_by_date.get(budget_date, pd.DataFrame())
        
        for _, mention in mentions_df.iterrows():
            sector = mention['sector']
            
            # Get sector's returns
            if sector in sector_returns.columns:
                sector_ret = sector_returns[sector]
            else:
                sector_ret = None
            
            # Create record
            record = create_sector_event_record(
                mention.to_dict(),
                fiscal_year,
                sector_ret,
                market_returns=market_returns
            )
            
            all_records.append(record)
    
    panel = pd.DataFrame(all_records)
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(output_path)
        logger.info(f"Saved aligned panel to {output_path}")
    
    logger.info(f"Built aligned panel with {len(panel)} events across {len(mentions_by_year)} years")
    
    return panel


def calculate_abnormal_returns(
    panel: pd.DataFrame,
    return_cols: List[str] = None,
    market_cols: List[str] = None
) -> pd.DataFrame:
    """
    Calculate abnormal returns (sector - market).
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel with sector and market returns
    return_cols : list, optional
        Sector return columns
    market_cols : list, optional
        Corresponding market return columns
        
    Returns
    -------
    pd.DataFrame
        Panel with abnormal return columns added
    """
    result = panel.copy()
    
    if return_cols is None:
        return_cols = [c for c in panel.columns if c.startswith('cum_ret_')]
    
    if market_cols is None:
        market_cols = [f'mkt_{c}' for c in return_cols]
    
    for ret_col, mkt_col in zip(return_cols, market_cols):
        if ret_col in result.columns and mkt_col in result.columns:
            ar_col = ret_col.replace('cum_ret_', 'ar_')
            result[ar_col] = result[ret_col] - result[mkt_col]
    
    return result

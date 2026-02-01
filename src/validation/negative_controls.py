"""
Negative controls module.

Implements negative control tests for causal inference.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..utils.logging import get_logger

logger = get_logger(__name__)


def non_mentioned_sectors_test(
    panel: pd.DataFrame,
    all_sectors: List[str],
    sector_returns_by_date: Dict[str, pd.DataFrame],
    mention_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Test whether non-mentioned sectors show no systematic reaction.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel (mentioned sectors)
    all_sectors : list
        All sector names
    sector_returns_by_date : dict
        Date -> sector returns
    mention_threshold : float
        Threshold for "mentioned"
        
    Returns
    -------
    pd.DataFrame
        Results for non-mentioned sectors
    """
    from ..events.event_windows import calculate_cumulative_returns
    from ..utils.time_utils import IST
    
    results = []
    
    # Get mentioned sectors per date
    mentioned_by_date = (
        panel.groupby('budget_date')['sector']
        .apply(set)
        .to_dict()
    )
    
    for date_str, sector_returns in sector_returns_by_date.items():
        mentioned = mentioned_by_date.get(date_str, set())
        non_mentioned = set(all_sectors) - mentioned
        
        for sector in non_mentioned:
            if sector not in sector_returns.columns:
                continue
            
            returns = sector_returns[sector]
            
            # Assume standard budget time
            date = datetime.strptime(date_str, '%Y-%m-%d')
            event_time = date.replace(hour=11, minute=0)
            event_time = IST.localize(event_time)
            
            # Calculate CARs
            cars = calculate_cumulative_returns(returns, event_time)
            
            result = {
                'budget_date': date_str,
                'sector': sector,
                'is_mentioned': False,
                **cars
            }
            results.append(result)
    
    df = pd.DataFrame(results)
    
    logger.info(f"Non-mentioned sectors test: {len(df)} observations")
    
    return df


def compare_mentioned_vs_non_mentioned(
    mentioned_results: pd.DataFrame,
    non_mentioned_results: pd.DataFrame,
    car_column: str = 'cum_ret_60m'
) -> Dict:
    """
    Compare reactions between mentioned and non-mentioned sectors.
    
    Parameters
    ----------
    mentioned_results : pd.DataFrame
        Results for mentioned sectors
    non_mentioned_results : pd.DataFrame
        Results for non-mentioned sectors
    car_column : str
        CAR column to compare
        
    Returns
    -------
    dict
        Comparison statistics
    """
    from scipy import stats
    
    mentioned = mentioned_results[car_column].dropna()
    non_mentioned = non_mentioned_results[car_column].dropna()
    
    # Non-mentioned should be close to zero (null)
    null_test = stats.ttest_1samp(non_mentioned, 0)
    
    # Difference test
    diff_test = stats.ttest_ind(mentioned, non_mentioned)
    
    comparison = {
        'mentioned_mean': mentioned.mean(),
        'mentioned_std': mentioned.std(),
        'mentioned_n': len(mentioned),
        'non_mentioned_mean': non_mentioned.mean(),
        'non_mentioned_std': non_mentioned.std(),
        'non_mentioned_n': len(non_mentioned),
        'null_t_stat': null_test.statistic,
        'null_p_value': null_test.pvalue,
        'diff_t_stat': diff_test.statistic,
        'diff_p_value': diff_test.pvalue,
        'passes_negative_control': null_test.pvalue > 0.05  # Non-rejected null
    }
    
    logger.info(f"Negative control: non-mentioned mean={non_mentioned.mean():.4f}, "
                f"null p={null_test.pvalue:.4f}")
    
    return comparison


def pre_trend_test(
    panel: pd.DataFrame,
    sector_returns_by_date: Dict[str, pd.DataFrame],
    pre_windows: List[int] = None
) -> pd.DataFrame:
    """
    Test for pre-trends before budget events.
    
    If significant pre-trends exist, it suggests anticipation
    or confounding effects.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel
    sector_returns_by_date : dict
        Date -> sector returns
    pre_windows : list
        Pre-event windows to test (negative minutes)
        
    Returns
    -------
    pd.DataFrame
        Pre-trend test results by sector
    """
    from scipy import stats
    from ..events.event_windows import extract_window_returns
    
    if pre_windows is None:
        pre_windows = [-60, -30, -15, -5]
    
    results = []
    
    for sector, group in panel.groupby('sector'):
        pre_returns = {f'pre_{abs(w)}m': [] for w in pre_windows}
        
        for _, event in group.iterrows():
            date_str = event.get('budget_date')
            event_time = event.get('event_timestamp') or event.get('event_time')
            
            if date_str not in sector_returns_by_date:
                continue
            
            returns_df = sector_returns_by_date[date_str]
            
            if sector not in returns_df.columns:
                continue
            
            returns = returns_df[sector]
            window_rets = extract_window_returns(returns, event_time, pre_windows, [])
            
            for w in pre_windows:
                key = f'ret_t{w}'
                if key in window_rets:
                    pre_returns[f'pre_{abs(w)}m'].append(window_rets[key])
        
        # Test each pre-window
        sector_result = {'sector': sector}
        
        for window_name, values in pre_returns.items():
            values = np.array(values)
            values = values[~np.isnan(values)]
            
            if len(values) >= 2:
                # Test if significantly different from zero
                t_stat, p_value = stats.ttest_1samp(values, 0)
                sector_result[f'{window_name}_mean'] = values.mean()
                sector_result[f'{window_name}_t'] = t_stat
                sector_result[f'{window_name}_p'] = p_value
        
        results.append(sector_result)
    
    df = pd.DataFrame(results)
    
    # Check for any significant pre-trends
    p_cols = [c for c in df.columns if c.endswith('_p')]
    if p_cols:
        df['any_pre_trend_sig'] = (df[p_cols] < 0.05).any(axis=1)
    
    logger.info(f"Pre-trend test: {df['any_pre_trend_sig'].sum()}/{len(df)} "
                f"sectors with significant pre-trends" if 'any_pre_trend_sig' in df.columns else "")
    
    return df


def cross_sector_spillover_test(
    panel: pd.DataFrame,
    sector_returns_by_date: Dict[str, pd.DataFrame],
    related_sectors: Dict[str, List[str]] = None
) -> pd.DataFrame:
    """
    Test for cross-sector spillover effects.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel
    sector_returns_by_date : dict
        Date -> sector returns
    related_sectors : dict, optional
        Mapping of sector -> related sectors
        
    Returns
    -------
    pd.DataFrame
        Spillover test results
    """
    from ..events.event_windows import calculate_cumulative_returns
    
    # Default related sectors (based on supply chain/thematic links)
    if related_sectors is None:
        related_sectors = {
            'infrastructure': ['cement', 'metals_mining'],
            'banking_nbfc': ['realty'],
            'energy_power': ['metals_mining'],
            'auto': ['metals_mining', 'energy_power'],
            'agriculture': ['fmcg'],
        }
    
    results = []
    
    for _, event in panel.iterrows():
        sector = event['sector']
        date_str = event.get('budget_date')
        event_time = event.get('event_timestamp') or event.get('event_time')
        
        if sector not in related_sectors:
            continue
        
        if date_str not in sector_returns_by_date:
            continue
        
        related = related_sectors[sector]
        returns_df = sector_returns_by_date[date_str]
        
        for related_sector in related:
            if related_sector not in returns_df.columns:
                continue
            
            returns = returns_df[related_sector]
            cars = calculate_cumulative_returns(returns, event_time)
            
            result = {
                'mentioned_sector': sector,
                'related_sector': related_sector,
                'budget_date': date_str,
                'event_time': event_time,
                **cars
            }
            results.append(result)
    
    df = pd.DataFrame(results)
    
    logger.info(f"Cross-sector spillover test: {len(df)} observations")
    
    return df

"""
Hazard/survival model module.

Models time-to-reaction for sector price response.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


def calculate_reaction_time(
    returns: pd.Series,
    event_time: pd.Timestamp,
    threshold: float = 0.001,
    max_bars: int = 24,
    direction: str = 'abs'
) -> Dict:
    """
    Calculate time to significant reaction.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    event_time : pd.Timestamp
        Event timestamp
    threshold : float
        Return threshold for significant reaction
    max_bars : int
        Maximum bars to search
    direction : str
        'abs' (any), 'pos', or 'neg'
        
    Returns
    -------
    dict
        Reaction time info
    """
    from datetime import timedelta
    from ..utils.time_utils import get_bar_start
    
    event_bar = get_bar_start(event_time)
    
    # Get post-event returns
    mask = returns.index >= event_bar
    post_returns = returns[mask].head(max_bars)
    
    if len(post_returns) == 0:
        return {'time_to_reaction': None, 'censored': True}
    
    # Find first significant reaction
    if direction == 'abs':
        significant = np.abs(post_returns) >= threshold
    elif direction == 'pos':
        significant = post_returns >= threshold
    elif direction == 'neg':
        significant = post_returns <= -threshold
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    if not significant.any():
        return {
            'time_to_reaction': max_bars * 5,  # Censored at max time
            'censored': True,
            'n_bars_searched': len(post_returns)
        }
    
    # Time to first reaction
    first_reaction_idx = significant.idxmax()
    reaction_bar = list(post_returns.index).index(first_reaction_idx)
    
    return {
        'time_to_reaction': reaction_bar * 5,  # In minutes
        'reaction_bar': reaction_bar,
        'reaction_timestamp': first_reaction_idx,
        'reaction_return': post_returns.loc[first_reaction_idx],
        'censored': False
    }


def prepare_survival_data(
    panel: pd.DataFrame,
    sector_returns_by_date: Dict[str, pd.DataFrame],
    threshold: float = 0.001,
    max_bars: int = 24
) -> pd.DataFrame:
    """
    Prepare data for survival analysis.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel
    sector_returns_by_date : dict
        Date -> sector returns
    threshold : float
        Reaction threshold
    max_bars : int
        Maximum bars to search
        
    Returns
    -------
    pd.DataFrame
        Survival data with duration and event indicator
    """
    survival_data = []
    
    for _, event in panel.iterrows():
        date_str = event.get('budget_date')
        sector = event.get('sector')
        event_time = event.get('event_timestamp') or event.get('event_time')
        
        if date_str not in sector_returns_by_date:
            continue
        
        returns_df = sector_returns_by_date[date_str]
        
        if sector not in returns_df.columns:
            continue
        
        returns = returns_df[sector]
        
        reaction = calculate_reaction_time(
            returns, event_time, threshold, max_bars
        )
        
        record = {
            'sector': sector,
            'fiscal_year': event.get('fiscal_year'),
            'event_time': event_time,
            'duration': reaction.get('time_to_reaction', max_bars * 5),
            'event': 0 if reaction.get('censored', True) else 1,
            'cumulative_attention': event.get('cumulative_attention'),
            'importance_weight': event.get('importance_weight'),
        }
        
        survival_data.append(record)
    
    df = pd.DataFrame(survival_data)
    
    logger.info(f"Prepared survival data: {len(df)} events, "
                f"{df['event'].mean():.1%} uncensored")
    
    return df


def fit_cox_model(
    survival_data: pd.DataFrame,
    duration_col: str = 'duration',
    event_col: str = 'event',
    covariates: List[str] = None
) -> Dict:
    """
    Fit Cox proportional hazards model.
    
    Parameters
    ----------
    survival_data : pd.DataFrame
        Survival data
    duration_col : str
        Duration column
    event_col : str
        Event indicator column
    covariates : list, optional
        Covariate columns
        
    Returns
    -------
    dict
        Cox model results
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        logger.warning("lifelines not installed, cannot fit Cox model")
        return {'error': 'lifelines not installed'}
    
    if covariates is None:
        covariates = ['cumulative_attention', 'importance_weight']
    
    # Filter to available covariates
    available = [c for c in covariates if c in survival_data.columns]
    
    # Prepare data
    cols = [duration_col, event_col] + available
    data = survival_data[cols].dropna()
    
    if len(data) < 10:
        return {'error': 'Insufficient data for Cox model'}
    
    # Fit model
    cph = CoxPHFitter()
    cph.fit(data, duration_col=duration_col, event_col=event_col)
    
    results = {
        'summary': cph.summary,
        'hazard_ratios': np.exp(cph.params_),
        'concordance': cph.concordance_index_,
        'log_likelihood': cph.log_likelihood_ratio_test(),
        'n_events': data[event_col].sum(),
        'n_observations': len(data)
    }
    
    logger.info(f"Cox model concordance: {results['concordance']:.3f}")
    
    return results


def kaplan_meier_by_group(
    survival_data: pd.DataFrame,
    group_col: str = 'sector',
    duration_col: str = 'duration',
    event_col: str = 'event'
) -> Dict[str, pd.DataFrame]:
    """
    Estimate Kaplan-Meier survival curves by group.
    
    Parameters
    ----------
    survival_data : pd.DataFrame
        Survival data
    group_col : str
        Grouping column
    duration_col : str
        Duration column
    event_col : str
        Event indicator column
        
    Returns
    -------
    dict
        Group name -> survival curve DataFrame
    """
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        logger.warning("lifelines not installed")
        return {}
    
    kmf = KaplanMeierFitter()
    curves = {}
    
    for group, data in survival_data.groupby(group_col):
        if len(data) < 3:
            continue
        
        kmf.fit(
            data[duration_col],
            event_observed=data[event_col],
            label=group
        )
        
        curves[group] = pd.DataFrame({
            'survival_prob': kmf.survival_function_[group],
            'conf_low': kmf.confidence_interval_survival_function_.iloc[:, 0],
            'conf_high': kmf.confidence_interval_survival_function_.iloc[:, 1]
        })
    
    return curves


def compare_survival_log_rank(
    survival_data: pd.DataFrame,
    group_col: str = 'sector',
    duration_col: str = 'duration',
    event_col: str = 'event'
) -> Dict:
    """
    Compare survival curves using log-rank test.
    
    Parameters
    ----------
    survival_data : pd.DataFrame
        Survival data
    group_col : str
        Grouping column
    duration_col : str
        Duration column
    event_col : str
        Event indicator column
        
    Returns
    -------
    dict
        Log-rank test results
    """
    try:
        from lifelines.statistics import multivariate_logrank_test
    except ImportError:
        logger.warning("lifelines not installed")
        return {}
    
    result = multivariate_logrank_test(
        survival_data[duration_col],
        survival_data[group_col],
        survival_data[event_col]
    )
    
    return {
        'test_statistic': result.test_statistic,
        'p_value': result.p_value,
        'degrees_of_freedom': result.degrees_freedom,
        'n_groups': survival_data[group_col].nunique()
    }

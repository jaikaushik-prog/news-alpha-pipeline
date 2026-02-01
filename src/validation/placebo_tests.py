"""
Placebo tests module.

Implements placebo/falsification tests for robustness validation.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..utils.logging import get_logger
from ..utils.time_utils import IST

logger = get_logger(__name__)


def generate_placebo_dates(
    actual_dates: List[datetime],
    n_placebos: int = 10,
    offset_range: Tuple[int, int] = (-30, -7),
    random_seed: int = 42
) -> List[datetime]:
    """
    Generate placebo (fake) budget dates.
    
    Parameters
    ----------
    actual_dates : list
        Actual budget dates
    n_placebos : int
        Number of placebo dates per actual date
    offset_range : tuple
        Range of days offset from actual date
    random_seed : int
        Random seed
        
    Returns
    -------
    list
        Placebo dates
    """
    np.random.seed(random_seed)
    
    placebo_dates = []
    
    for actual in actual_dates:
        for _ in range(n_placebos):
            # Random offset within range
            offset = np.random.randint(offset_range[0], offset_range[1])
            placebo = actual + timedelta(days=offset)
            
            # Avoid weekends
            while placebo.weekday() >= 5:
                offset -= 1
                placebo = actual + timedelta(days=offset)
            
            placebo_dates.append(placebo)
    
    return placebo_dates


def run_placebo_event_study(
    placebo_dates: List[datetime],
    sector_returns_by_date: Dict[str, pd.DataFrame],
    sectors: List[str],
    horizons: List[int] = None
) -> pd.DataFrame:
    """
    Run event study on placebo dates.
    
    Parameters
    ----------
    placebo_dates : list
        Placebo event dates
    sector_returns_by_date : dict
        Date string -> sector returns
    sectors : list
        Sectors to analyze
    horizons : list
        CAR horizons
        
    Returns
    -------
    pd.DataFrame
        Placebo event study results
    """
    from ..events.event_windows import calculate_cumulative_returns
    from ..utils.time_utils import get_bar_start
    
    if horizons is None:
        horizons = [5, 15, 30, 60]
    
    results = []
    
    for placebo_date in placebo_dates:
        # Get matching date string
        date_str = placebo_date.strftime('%Y-%m-%d')
        
        if date_str not in sector_returns_by_date:
            continue
        
        sector_returns = sector_returns_by_date[date_str]
        
        for sector in sectors:
            if sector not in sector_returns.columns:
                continue
            
            returns = sector_returns[sector]
            
            # Use 11:00 as placeholder event time (like actual budgets)
            event_time = placebo_date.replace(hour=11, minute=0)
            event_time = IST.localize(event_time)
            
            # Calculate CARs
            cars = calculate_cumulative_returns(returns, event_time, horizons)
            
            result = {
                'placebo_date': placebo_date,
                'sector': sector,
                'is_placebo': True,
                **cars
            }
            
            results.append(result)
    
    df = pd.DataFrame(results)
    
    logger.info(f"Ran placebo event study on {len(placebo_dates)} dates, "
                f"{len(df)} sector-events")
    
    return df


def compare_actual_vs_placebo(
    actual_results: pd.DataFrame,
    placebo_results: pd.DataFrame,
    car_column: str = 'cum_ret_60m'
) -> Dict:
    """
    Compare actual event results to placebo results.
    
    Parameters
    ----------
    actual_results : pd.DataFrame
        Results from actual budget events
    placebo_results : pd.DataFrame
        Results from placebo events
    car_column : str
        CAR column to compare
        
    Returns
    -------
    dict
        Comparison statistics
    """
    from scipy import stats
    
    actual_cars = actual_results[car_column].dropna()
    placebo_cars = placebo_results[car_column].dropna()
    
    # T-test
    t_stat, t_pvalue = stats.ttest_ind(actual_cars, placebo_cars)
    
    # Mann-Whitney U test
    u_stat, u_pvalue = stats.mannwhitneyu(actual_cars, placebo_cars, alternative='two-sided')
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.ks_2samp(actual_cars, placebo_cars)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(actual_cars) - 1) * actual_cars.std() ** 2 + 
         (len(placebo_cars) - 1) * placebo_cars.std() ** 2) /
        (len(actual_cars) + len(placebo_cars) - 2)
    )
    cohens_d = (actual_cars.mean() - placebo_cars.mean()) / pooled_std
    
    comparison = {
        'actual_mean': actual_cars.mean(),
        'placebo_mean': placebo_cars.mean(),
        'actual_std': actual_cars.std(),
        'placebo_std': placebo_cars.std(),
        'actual_n': len(actual_cars),
        'placebo_n': len(placebo_cars),
        't_statistic': t_stat,
        't_pvalue': t_pvalue,
        'u_statistic': u_stat,
        'u_pvalue': u_pvalue,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'cohens_d': cohens_d
    }
    
    logger.info(f"Actual vs Placebo: t-stat={t_stat:.2f}, p={t_pvalue:.4f}, "
                f"Cohen's d={cohens_d:.2f}")
    
    return comparison


def generate_random_sector_assignment(
    panel: pd.DataFrame,
    n_iterations: int = 100,
    random_seed: int = 42
) -> List[pd.DataFrame]:
    """
    Generate random sector assignments for permutation test.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel
    n_iterations : int
        Number of random permutations
    random_seed : int
        Random seed
        
    Returns
    -------
    list
        List of DataFrames with permuted sectors
    """
    np.random.seed(random_seed)
    
    permuted_panels = []
    sectors = panel['sector'].unique()
    
    for _ in range(n_iterations):
        permuted = panel.copy()
        permuted['sector'] = np.random.permutation(panel['sector'].values)
        permuted_panels.append(permuted)
    
    return permuted_panels


def run_permutation_test(
    panel: pd.DataFrame,
    test_statistic_func: callable,
    n_iterations: int = 100,
    random_seed: int = 42
) -> Tuple[float, float, np.ndarray]:
    """
    Run permutation test for sector assignment.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel
    test_statistic_func : callable
        Function to compute test statistic from panel
    n_iterations : int
        Number of permutations
    random_seed : int
        Random seed
        
    Returns
    -------
    tuple
        (actual_statistic, p_value, null_distribution)
    """
    # Actual test statistic
    actual_stat = test_statistic_func(panel)
    
    # Permutations
    permuted_panels = generate_random_sector_assignment(panel, n_iterations, random_seed)
    
    null_distribution = np.array([
        test_statistic_func(p) for p in permuted_panels
    ])
    
    # P-value (two-sided)
    p_value = np.mean(np.abs(null_distribution) >= np.abs(actual_stat))
    
    logger.info(f"Permutation test: actual={actual_stat:.4f}, "
                f"null_mean={null_distribution.mean():.4f}, p={p_value:.4f}")
    
    return actual_stat, p_value, null_distribution

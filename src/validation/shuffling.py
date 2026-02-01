"""
Shuffling and bootstrap validation module.

Implements random shuffling tests for robustness.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


def shuffle_timestamps(
    sentences_df: pd.DataFrame,
    n_iterations: int = 100,
    random_seed: int = 42
) -> List[pd.DataFrame]:
    """
    Shuffle sentence timestamps while keeping text fixed.
    
    Tests whether the timing of mentions matters.
    
    Parameters
    ----------
    sentences_df : pd.DataFrame
        Sentences with timestamps
    n_iterations : int
        Number of shuffles
    random_seed : int
        Random seed
        
    Returns
    -------
    list
        Shuffled DataFrames
    """
    np.random.seed(random_seed)
    
    shuffled = []
    
    for _ in range(n_iterations):
        df = sentences_df.copy()
        if 'estimated_timestamp' in df.columns:
            df['estimated_timestamp'] = np.random.permutation(df['estimated_timestamp'].values)
        shuffled.append(df)
    
    return shuffled


def shuffle_sector_probabilities(
    sentences_df: pd.DataFrame,
    n_iterations: int = 100,
    random_seed: int = 42
) -> List[pd.DataFrame]:
    """
    Shuffle sector probability assignments.
    
    Tests whether sector classification matters.
    
    Parameters
    ----------
    sentences_df : pd.DataFrame
        Sentences with sector probabilities
    n_iterations : int
        Number of shuffles
    random_seed : int
        Random seed
        
    Returns
    -------
    list
        Shuffled DataFrames
    """
    np.random.seed(random_seed)
    
    # Find probability columns
    prob_cols = [c for c in sentences_df.columns if c.startswith('prob_')]
    
    shuffled = []
    
    for _ in range(n_iterations):
        df = sentences_df.copy()
        
        # Shuffle all probability columns together (maintain row structure)
        indices = np.random.permutation(len(df))
        for col in prob_cols:
            df[col] = df[col].values[indices]
        
        shuffled.append(df)
    
    return shuffled


def bootstrap_panel(
    panel: pd.DataFrame,
    n_iterations: int = 1000,
    random_seed: int = 42,
    stratify_by: str = None
) -> List[pd.DataFrame]:
    """
    Bootstrap sample the event panel.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel
    n_iterations : int
        Number of bootstrap samples
    random_seed : int
        Random seed
    stratify_by : str, optional
        Column to stratify sampling by
        
    Returns
    -------
    list
        Bootstrap samples
    """
    np.random.seed(random_seed)
    
    n = len(panel)
    samples = []
    
    for _ in range(n_iterations):
        if stratify_by and stratify_by in panel.columns:
            # Stratified bootstrap
            indices = []
            for _, group in panel.groupby(stratify_by):
                group_indices = np.random.choice(
                    group.index, size=len(group), replace=True
                )
                indices.extend(group_indices)
            sample = panel.loc[indices].reset_index(drop=True)
        else:
            # Simple bootstrap
            indices = np.random.choice(n, size=n, replace=True)
            sample = panel.iloc[indices].reset_index(drop=True)
        
        samples.append(sample)
    
    return samples


def bootstrap_confidence_interval(
    panel: pd.DataFrame,
    statistic_func: callable,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel
    statistic_func : callable
        Function to compute statistic from panel
    n_iterations : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level
    random_seed : int
        Random seed
        
    Returns
    -------
    dict
        Point estimate and confidence interval
    """
    # Point estimate
    point_estimate = statistic_func(panel)
    
    # Bootstrap samples
    samples = bootstrap_panel(panel, n_iterations, random_seed)
    
    # Calculate statistic for each sample
    bootstrap_stats = np.array([statistic_func(s) for s in samples])
    
    # Confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    result = {
        'point_estimate': point_estimate,
        'lower_ci': lower,
        'upper_ci': upper,
        'confidence_level': confidence_level,
        'bootstrap_mean': bootstrap_stats.mean(),
        'bootstrap_std': bootstrap_stats.std(),
        'bootstrap_bias': bootstrap_stats.mean() - point_estimate,
        'n_iterations': n_iterations
    }
    
    logger.info(f"Bootstrap CI: {point_estimate:.4f} [{lower:.4f}, {upper:.4f}]")
    
    return result


def leave_one_year_out(
    panel: pd.DataFrame,
    test_func: callable,
    year_col: str = 'fiscal_year'
) -> pd.DataFrame:
    """
    Leave-one-year-out cross-validation.
    
    Tests stability across different budget years.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel
    test_func : callable
        Function to run on each fold
    year_col : str
        Year column
        
    Returns
    -------
    pd.DataFrame
        Results for each fold
    """
    years = panel[year_col].unique()
    results = []
    
    for held_out_year in years:
        # Train on all but one year
        train = panel[panel[year_col] != held_out_year]
        test = panel[panel[year_col] == held_out_year]
        
        fold_result = test_func(train, test)
        fold_result['held_out_year'] = held_out_year
        fold_result['train_n'] = len(train)
        fold_result['test_n'] = len(test)
        
        results.append(fold_result)
    
    return pd.DataFrame(results)


def jackknife_estimate(
    panel: pd.DataFrame,
    statistic_func: callable
) -> Dict:
    """
    Jackknife estimate with bias correction.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel
    statistic_func : callable
        Function to compute statistic
        
    Returns
    -------
    dict
        Jackknife estimate and standard error
    """
    n = len(panel)
    
    # Full estimate
    full_estimate = statistic_func(panel)
    
    # Leave-one-out estimates
    loo_estimates = []
    
    for i in range(n):
        loo_panel = panel.drop(panel.index[i])
        loo_estimates.append(statistic_func(loo_panel))
    
    loo_estimates = np.array(loo_estimates)
    loo_mean = loo_estimates.mean()
    
    # Jackknife estimate (bias-corrected)
    jackknife_estimate = n * full_estimate - (n - 1) * loo_mean
    
    # Standard error
    jackknife_var = ((n - 1) / n) * np.sum((loo_estimates - loo_mean) ** 2)
    jackknife_se = np.sqrt(jackknife_var)
    
    result = {
        'full_estimate': full_estimate,
        'jackknife_estimate': jackknife_estimate,
        'jackknife_se': jackknife_se,
        'bias': jackknife_estimate - full_estimate,
        'n': n
    }
    
    return result

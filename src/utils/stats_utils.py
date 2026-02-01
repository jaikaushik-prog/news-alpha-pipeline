"""
Statistical utilities for Budget Speech Impact Analysis.

Includes:
- HAC standard errors
- Event study statistics
- Hypothesis testing
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_1samp, ttest_ind, wilcoxon, mannwhitneyu


def calculate_returns(prices: pd.Series, log_returns: bool = False) -> pd.Series:
    """
    Calculate returns from price series.
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    log_returns : bool
        If True, calculate log returns; else simple returns
        
    Returns
    -------
    pd.Series
        Return series
    """
    if log_returns:
        return np.log(prices / prices.shift(1))
    else:
        return prices.pct_change()


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
        Whether to annualize the volatility
    annualization_factor : float
        Annualization factor (sqrt of bars per year)
        
    Returns
    -------
    pd.Series
        Rolling volatility
    """
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * annualization_factor
        
    return vol


def return_dispersion(returns_df: pd.DataFrame) -> pd.Series:
    """
    Calculate cross-sectional return dispersion.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame with stock returns (columns) over time (rows)
        
    Returns
    -------
    pd.Series
        Cross-sectional standard deviation at each time point
    """
    return returns_df.std(axis=1)


def calculate_car(
    returns: pd.Series,
    event_idx: int,
    pre_window: int = -60,
    post_window: int = 60,
    estimation_window: int = 240
) -> Dict[str, float]:
    """
    Calculate Cumulative Abnormal Returns around an event.
    
    Parameters
    ----------
    returns : pd.Series
        Return series with datetime index
    event_idx : int
        Index position of the event
    pre_window : int
        Pre-event window (negative number)
    post_window : int
        Post-event window (positive number)
    estimation_window : int
        Number of periods for expected return estimation
        
    Returns
    -------
    dict
        Dictionary with CAR values at different horizons
    """
    # Estimation period
    est_start = event_idx - estimation_window - abs(pre_window)
    est_end = event_idx + pre_window
    
    if est_start < 0:
        est_start = 0
    
    # Expected return (mean over estimation period)
    expected_return = returns.iloc[est_start:est_end].mean()
    
    # Event window returns
    event_start = event_idx + pre_window
    event_end = event_idx + post_window
    
    event_returns = returns.iloc[event_start:event_end + 1]
    abnormal_returns = event_returns - expected_return
    
    # Calculate CAR at different horizons
    car_results = {}
    for horizon in [5, 15, 30, 60]:
        if horizon <= post_window:
            car_results[f'car_{horizon}m'] = abnormal_returns.iloc[
                abs(pre_window):abs(pre_window) + horizon // 5 + 1
            ].sum()
    
    return car_results


def test_car_significance(
    car_values: np.ndarray,
    null_hypothesis: float = 0.0
) -> Dict[str, float]:
    """
    Test whether CAR is statistically significant.
    
    Parameters
    ----------
    car_values : np.ndarray
        Array of CAR values across events
    null_hypothesis : float
        Null hypothesis value (usually 0)
        
    Returns
    -------
    dict
        t-statistic, p-value, mean, std
    """
    car_values = car_values[~np.isnan(car_values)]
    
    if len(car_values) < 2:
        return {
            't_stat': np.nan,
            'p_value': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'n': len(car_values)
        }
    
    t_stat, p_value = ttest_1samp(car_values, null_hypothesis)
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'mean': np.mean(car_values),
        'std': np.std(car_values, ddof=1),
        'n': len(car_values)
    }


def newey_west_se(
    residuals: np.ndarray,
    X: np.ndarray,
    lags: Optional[int] = None
) -> np.ndarray:
    """
    Calculate Newey-West HAC standard errors.
    
    Parameters
    ----------
    residuals : np.ndarray
        Regression residuals
    X : np.ndarray
        Design matrix
    lags : int, optional
        Number of lags (default: floor(n^0.25))
        
    Returns
    -------
    np.ndarray
        HAC standard errors
    """
    n = len(residuals)
    k = X.shape[1]
    
    if lags is None:
        lags = int(np.floor(n ** 0.25))
    
    # Meat of the sandwich
    u = residuals.reshape(-1, 1) * X
    S = u.T @ u / n
    
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        Gamma = (u[lag:].T @ u[:-lag]) / n
        S += weight * (Gamma + Gamma.T)
    
    # Bread of the sandwich
    XTX_inv = np.linalg.inv(X.T @ X / n)
    
    # Full sandwich
    V = XTX_inv @ S @ XTX_inv / n
    
    return np.sqrt(np.diag(V))


def bootstrap_ci(
    data: np.ndarray,
    statistic_func: callable = np.mean,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    statistic_func : callable
        Function to compute statistic
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (0-1)
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (point_estimate, lower_ci, upper_ci)
    """
    np.random.seed(random_seed)
    
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return statistic_func(data), lower, upper


def compare_distributions(
    group1: np.ndarray,
    group2: np.ndarray,
    test: str = 'ttest'
) -> Dict[str, float]:
    """
    Compare two distributions with appropriate tests.
    
    Parameters
    ----------
    group1 : np.ndarray
        First group
    group2 : np.ndarray
        Second group
    test : str
        Test type: 'ttest', 'mannwhitney', 'wilcoxon'
        
    Returns
    -------
    dict
        Test results including statistic and p-value
    """
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]
    
    results = {
        'n1': len(group1),
        'n2': len(group2),
        'mean1': np.mean(group1),
        'mean2': np.mean(group2),
        'diff': np.mean(group1) - np.mean(group2)
    }
    
    if test == 'ttest':
        stat, pval = ttest_ind(group1, group2)
        results['t_stat'] = stat
        results['p_value'] = pval
    elif test == 'mannwhitney':
        stat, pval = mannwhitneyu(group1, group2, alternative='two-sided')
        results['u_stat'] = stat
        results['p_value'] = pval
    elif test == 'wilcoxon':
        # For paired samples
        stat, pval = wilcoxon(group1, group2)
        results['w_stat'] = stat
        results['p_value'] = pval
    
    return results


def amihud_illiquidity(
    returns: pd.Series,
    volume: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Calculate Amihud illiquidity measure.
    
    Amihud = |return| / volume
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    volume : pd.Series
        Volume series
    window : int
        Rolling window for averaging
        
    Returns
    -------
    pd.Series
        Rolling Amihud illiquidity
    """
    # Avoid division by zero
    volume_safe = volume.replace(0, np.nan)
    
    illiquidity = np.abs(returns) / volume_safe
    
    return illiquidity.rolling(window=window).mean()


def roll_spread(returns: pd.Series, window: int = 60) -> pd.Series:
    """
    Calculate Roll implied spread from return autocovariance.
    
    Roll = 2 * sqrt(-cov(r_t, r_{t-1}))
    
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
        cov = np.cov(x[:-1], x[1:])[0, 1]
        if cov >= 0:
            return 0  # No spread implied
        return 2 * np.sqrt(-cov)
    
    return returns.rolling(window=window).apply(calc_roll, raw=True)

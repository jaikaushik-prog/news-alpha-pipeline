"""
Panel regression module.

Implements panel data models for budget speech impact analysis.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats

from ..utils.logging import get_logger
from ..utils.stats_utils import newey_west_se

logger = get_logger(__name__)


def prepare_regression_data(
    panel: pd.DataFrame,
    dependent_var: str = 'car_60m',
    independent_vars: List[str] = None,
    controls: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare data for regression.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel
    dependent_var : str
        Dependent variable column
    independent_vars : list
        Independent variable columns
    controls : list
        Control variable columns
        
    Returns
    -------
    tuple
        (X, y, info) where info contains metadata
    """
    if independent_vars is None:
        independent_vars = ['cumulative_attention', 'importance_weight']
    
    if controls is None:
        controls = ['pre_volatility', 'pre_return']
    
    # Combine all regressors
    all_vars = independent_vars + controls
    
    # Filter to available columns
    available_vars = [v for v in all_vars if v in panel.columns]
    
    if dependent_var not in panel.columns:
        raise ValueError(f"Dependent variable {dependent_var} not in panel")
    
    # Drop missing values
    data = panel[[dependent_var] + available_vars].dropna()
    
    y = data[dependent_var]
    X = data[available_vars]
    
    # Add constant
    X = pd.concat([pd.Series(1, index=X.index, name='const'), X], axis=1)
    
    info = {
        'n_obs': len(data),
        'n_vars': len(available_vars),
        'dependent_var': dependent_var,
        'independent_vars': available_vars,
        'missing_vars': [v for v in all_vars if v not in panel.columns]
    }
    
    logger.info(f"Prepared regression data: {info['n_obs']} obs, {info['n_vars']} vars")
    
    return X, y, info


def ols_regression(
    X: pd.DataFrame,
    y: pd.Series,
    hac_se: bool = True,
    hac_lags: int = None
) -> Dict:
    """
    Run OLS regression with optional HAC standard errors.
    
    Parameters
    ----------
    X : pd.DataFrame
        Regressors (including constant)
    y : pd.Series
        Dependent variable
    hac_se : bool
        Whether to use HAC standard errors
    hac_lags : int, optional
        Number of lags for HAC
        
    Returns
    -------
    dict
        Regression results
    """
    # OLS estimation
    X_arr = X.values
    y_arr = y.values
    
    try:
        coeffs = np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
    except:
        return {'error': 'OLS estimation failed'}
    
    # Residuals
    y_pred = X_arr @ coeffs
    residuals = y_arr - y_pred
    
    # Standard errors
    n = len(y)
    k = X.shape[1]
    
    if hac_se:
        se = newey_west_se(residuals, X_arr, lags=hac_lags)
    else:
        # OLS standard errors
        sigma2 = np.sum(residuals ** 2) / (n - k)
        XTX_inv = np.linalg.inv(X_arr.T @ X_arr)
        se = np.sqrt(sigma2 * np.diag(XTX_inv))
    
    # t-statistics and p-values
    t_stats = coeffs / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
    
    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    
    # F-statistic
    f_stat = (r_squared / (k - 1)) / ((1 - r_squared) / (n - k))
    f_pvalue = 1 - stats.f.cdf(f_stat, k - 1, n - k)
    
    results = {
        'coefficients': pd.Series(coeffs, index=X.columns),
        'std_errors': pd.Series(se, index=X.columns),
        't_statistics': pd.Series(t_stats, index=X.columns),
        'p_values': pd.Series(p_values, index=X.columns),
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'f_statistic': f_stat,
        'f_pvalue': f_pvalue,
        'n_obs': n,
        'n_vars': k,
        'residuals': residuals,
        'fitted_values': y_pred,
        'hac_se': hac_se
    }
    
    return results


def fixed_effects_regression(
    panel: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    entity_col: str = 'sector',
    time_col: str = 'fiscal_year',
    entity_fe: bool = True,
    time_fe: bool = True
) -> Dict:
    """
    Run fixed effects panel regression.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Panel data
    dependent_var : str
        Dependent variable
    independent_vars : list
        Independent variables
    entity_col : str
        Entity (panel) identifier
    time_col : str
        Time period identifier
    entity_fe : bool
        Include entity fixed effects
    time_fe : bool
        Include time fixed effects
        
    Returns
    -------
    dict
        Regression results
    """
    # Create dummies
    data = panel.copy()
    
    # Available independent vars
    available = [v for v in independent_vars if v in data.columns]
    
    # Create FE dummies
    dummies = []
    
    if entity_fe and entity_col in data.columns:
        entity_dummies = pd.get_dummies(data[entity_col], prefix='fe_entity', drop_first=True)
        dummies.append(entity_dummies)
    
    if time_fe and time_col in data.columns:
        time_dummies = pd.get_dummies(data[time_col], prefix='fe_time', drop_first=True)
        dummies.append(time_dummies)
    
    # Combine X
    X_parts = [data[available]]
    X_parts.extend(dummies)
    
    X = pd.concat(X_parts, axis=1)
    X = pd.concat([pd.Series(1, index=X.index, name='const'), X], axis=1)
    
    y = data[dependent_var]
    
    # Drop missing
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    # Run OLS
    results = ols_regression(X, y, hac_se=True)
    
    # Add FE info
    results['entity_fe'] = entity_fe
    results['time_fe'] = time_fe
    results['independent_vars'] = available
    
    # Filter to main coefficients
    main_coefs = ['const'] + available
    results['main_coefficients'] = results['coefficients'][results['coefficients'].index.isin(main_coefs)]
    results['main_pvalues'] = results['p_values'][results['p_values'].index.isin(main_coefs)]
    
    return results


def format_regression_table(
    results: Dict,
    main_only: bool = True,
    stars: bool = True
) -> pd.DataFrame:
    """
    Format regression results as publication table.
    
    Parameters
    ----------
    results : dict
        Regression results
    main_only : bool
        Only show main coefficients (not FE dummies)
    stars : bool
        Add significance stars
        
    Returns
    -------
    pd.DataFrame
        Formatted table
    """
    if main_only and 'main_coefficients' in results:
        coefs = results['main_coefficients']
        pvals = results['main_pvalues']
        se = results['std_errors'][coefs.index]
        tstat = results['t_statistics'][coefs.index]
    else:
        coefs = results['coefficients']
        pvals = results['p_values']
        se = results['std_errors']
        tstat = results['t_statistics']
    
    # Add stars
    def add_stars(coef, pval):
        stars_str = ''
        if pval < 0.01:
            stars_str = '***'
        elif pval < 0.05:
            stars_str = '**'
        elif pval < 0.10:
            stars_str = '*'
        return f"{coef:.4f}{stars_str}"
    
    if stars:
        coef_str = [add_stars(c, p) for c, p in zip(coefs, pvals)]
    else:
        coef_str = [f"{c:.4f}" for c in coefs]
    
    table = pd.DataFrame({
        'Coefficient': coef_str,
        'Std. Error': [f"({s:.4f})" for s in se],
        't-statistic': [f"{t:.2f}" for t in tstat],
        'p-value': [f"{p:.3f}" for p in pvals]
    }, index=coefs.index)
    
    # Add summary stats
    summary = pd.DataFrame({
        'Coefficient': [
            f"{results['r_squared']:.3f}",
            f"{results['adj_r_squared']:.3f}",
            f"{results['n_obs']}"
        ],
        'Std. Error': ['', '', ''],
        't-statistic': ['', '', ''],
        'p-value': ['', '', '']
    }, index=['R-squared', 'Adj. R-squared', 'N'])
    
    table = pd.concat([table, summary])
    
    return table

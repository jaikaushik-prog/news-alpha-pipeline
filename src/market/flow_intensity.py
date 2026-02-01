"""
FII/DII Flow Intensity Module.

Enhanced institutional flow analysis with directional conviction signals.
Superior to raw flows because it combines volume context with volatility.

Key Features:
- Flow Intensity = |Net Flow| / 20-day Avg Volume
- Conviction signals combining flow direction with IV
- Smart money classification
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


def load_fii_dii_data(
    data_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load FII/DII daily data from CSV.
    
    Data source: NSE India - Participant-wise Open Interest
    Available at: https://www.nseindia.com/reports/fiidii
    
    Parameters
    ----------
    data_dir : Path
        Directory containing FII/DII CSVs
    start_date : str, optional
        Start date filter
    end_date : str, optional
        End date filter
        
    Returns
    -------
    pd.DataFrame
        FII/DII flow data
    """
    try:
        fii_file = data_dir / 'fii_dii_data.csv'
        
        if not fii_file.exists():
            logger.warning(f"FII/DII data not found at {fii_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(fii_file, parse_dates=['Date'])
        df = df.sort_values('Date')
        
        if start_date:
            df = df[df['Date'] >= start_date]
        if end_date:
            df = df[df['Date'] <= end_date]
        
        logger.info(f"Loaded FII/DII data: {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error loading FII/DII data: {e}")
        return pd.DataFrame()


def calculate_flow_intensity(
    fii_dii_df: pd.DataFrame,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Calculate Flow Intensity metric.
    
    Flow_Intensity = |Net Flow| / 20-day Avg Volume
    
    Higher intensity = Stronger conviction in position
    
    Parameters
    ----------
    fii_dii_df : pd.DataFrame
        FII/DII data with columns: Date, FII_Buy, FII_Sell, DII_Buy, DII_Sell
    lookback : int
        Days for rolling average
        
    Returns
    -------
    pd.DataFrame
        Data with flow intensity metrics
    """
    df = fii_dii_df.copy()
    
    # Calculate net flows
    df['fii_net'] = df.get('FII_Buy', 0) - df.get('FII_Sell', 0)
    df['dii_net'] = df.get('DII_Buy', 0) - df.get('DII_Sell', 0)
    df['total_net'] = df['fii_net'] + df['dii_net']
    
    # Calculate total volume
    df['fii_volume'] = df.get('FII_Buy', 0) + df.get('FII_Sell', 0)
    df['dii_volume'] = df.get('DII_Buy', 0) + df.get('DII_Sell', 0)
    df['total_volume'] = df['fii_volume'] + df['dii_volume']
    
    # Rolling averages
    df['avg_volume_20d'] = df['total_volume'].rolling(lookback, min_periods=5).mean()
    
    # Flow intensity metrics
    df['fii_intensity'] = df['fii_net'].abs() / df['avg_volume_20d'].replace(0, 1)
    df['dii_intensity'] = df['dii_net'].abs() / df['avg_volume_20d'].replace(0, 1)
    df['total_intensity'] = df['total_net'].abs() / df['avg_volume_20d'].replace(0, 1)
    
    # Directional intensity (signed)
    df['fii_signed_intensity'] = np.sign(df['fii_net']) * df['fii_intensity']
    df['dii_signed_intensity'] = np.sign(df['dii_net']) * df['dii_intensity']
    
    return df


def classify_flow_regime(
    fii_net: float,
    dii_net: float,
    iv_level: Optional[float] = None,
    iv_change: Optional[float] = None
) -> Dict[str, str]:
    """
    Classify institutional positioning regime.
    
    Combines flow direction with IV context for smart money signals.
    
    Parameters
    ----------
    fii_net : float
        FII net flow (positive = buying)
    dii_net : float
        DII net flow (positive = buying)
    iv_level : float, optional
        Current IV level
    iv_change : float, optional
        IV change vs previous day
        
    Returns
    -------
    dict
        Regime classification
    """
    regime = {}
    
    # FII classification
    if fii_net > 0:
        fii_stance = 'buying'
    elif fii_net < 0:
        fii_stance = 'selling'
    else:
        fii_stance = 'neutral'
    
    # DII classification
    if dii_net > 0:
        dii_stance = 'buying'
    elif dii_net < 0:
        dii_stance = 'selling'
    else:
        dii_stance = 'neutral'
    
    regime['fii_stance'] = fii_stance
    regime['dii_stance'] = dii_stance
    
    # Combined regime
    if fii_stance == 'buying' and dii_stance == 'buying':
        regime['combined_regime'] = 'strong_risk_on'
    elif fii_stance == 'selling' and dii_stance == 'selling':
        regime['combined_regime'] = 'strong_risk_off'
    elif fii_stance == 'buying' and dii_stance == 'selling':
        regime['combined_regime'] = 'fii_led_rally'  # FIIs bullish, DIIs taking profits
    elif fii_stance == 'selling' and dii_stance == 'buying':
        regime['combined_regime'] = 'dii_support'  # DIIs absorbing FII selling
    else:
        regime['combined_regime'] = 'mixed'
    
    # IV-adjusted classification (if available)
    if iv_level is not None and iv_change is not None:
        if fii_stance == 'buying' and iv_change < 0:
            regime['smart_money_signal'] = 'confident_risk_on'
        elif fii_stance == 'selling' and iv_change > 0:
            regime['smart_money_signal'] = 'defensive_exit'
        elif fii_stance == 'buying' and iv_change > 0:
            regime['smart_money_signal'] = 'hedged_accumulation'
        elif fii_stance == 'selling' and iv_change < 0:
            regime['smart_money_signal'] = 'orderly_distribution'
        else:
            regime['smart_money_signal'] = 'unclear'
    
    return regime


def get_pre_budget_flow_signal(
    fii_dii_df: pd.DataFrame,
    budget_date: str,
    lookback_days: int = 5
) -> Dict[str, float]:
    """
    Get FII/DII flow signals before budget.
    
    Parameters
    ----------
    fii_dii_df : pd.DataFrame
        FII/DII data
    budget_date : str
        Budget date in YYYY-MM-DD format
    lookback_days : int
        Days to analyze before budget
        
    Returns
    -------
    dict
        Pre-budget flow signals
    """
    budget_dt = pd.to_datetime(budget_date)
    
    # Get pre-budget window
    pre_budget = fii_dii_df[
        (fii_dii_df['Date'] < budget_dt) & 
        (fii_dii_df['Date'] >= budget_dt - timedelta(days=lookback_days + 5))
    ].tail(lookback_days)
    
    if pre_budget.empty:
        return {}
    
    # Calculate intensity if not present
    if 'fii_intensity' not in pre_budget.columns:
        pre_budget = calculate_flow_intensity(pre_budget)
    
    # Aggregate signals
    signals = {
        'fii_net_sum': pre_budget['fii_net'].sum(),
        'dii_net_sum': pre_budget['dii_net'].sum(),
        'fii_avg_intensity': pre_budget['fii_intensity'].mean(),
        'dii_avg_intensity': pre_budget['dii_intensity'].mean(),
        'fii_trend': 'buying' if pre_budget['fii_net'].sum() > 0 else 'selling',
        'dii_trend': 'buying' if pre_budget['dii_net'].sum() > 0 else 'selling',
        'days_analyzed': len(pre_budget)
    }
    
    # Get regime for last day
    last_day = pre_budget.iloc[-1]
    regime = classify_flow_regime(last_day['fii_net'], last_day['dii_net'])
    signals.update(regime)
    
    return signals


def create_mock_fii_dii_data(
    start_date: str = '2024-01-01',
    end_date: str = '2025-02-01'
) -> pd.DataFrame:
    """
    Create mock FII/DII data for demonstration.
    
    Parameters
    ----------
    start_date : str
        Start date
    end_date : str
        End date
        
    Returns
    -------
    pd.DataFrame
        Mock FII/DII data
    """
    np.random.seed(42)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    n = len(dates)
    
    # Generate realistic flow patterns
    # FII: More volatile, trend-following
    fii_buy = np.random.lognormal(mean=8, sigma=0.5, size=n) * 1000  # Crores
    fii_sell = np.random.lognormal(mean=8, sigma=0.5, size=n) * 1000
    
    # DII: More stable, sometimes contrarian
    dii_buy = np.random.lognormal(mean=7.5, sigma=0.3, size=n) * 1000
    dii_sell = np.random.lognormal(mean=7.5, sigma=0.3, size=n) * 1000
    
    df = pd.DataFrame({
        'Date': dates,
        'FII_Buy': fii_buy,
        'FII_Sell': fii_sell,
        'DII_Buy': dii_buy,
        'DII_Sell': dii_sell
    })
    
    logger.info(f"Created mock FII/DII data: {len(df)} records")
    return df

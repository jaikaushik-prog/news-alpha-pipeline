"""
Option Implied Volatility Module.

Fetches India VIX and option chain data as pre-event sentiment indicators.
Supports loading from local CSV files downloaded from NSE.
"""

from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Project root for locating CSV files
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_india_vix_csv(year: int = None) -> pd.DataFrame:
    """
    Load India VIX data from local CSV files.
    
    CSV files should be named like:
    - hist_india_vix_-01-01-2023-to-31-12-2023.csv
    - hist_india_vix_-01-01-2024-to-30-12-2024.csv
    - hist_india_vix_-01-01-2025-to-31-12-2025 (1).csv
    
    Parameters
    ----------
    year : int, optional
        Specific year to load. If None, loads all available years.
        
    Returns
    -------
    pd.DataFrame
        India VIX data with columns: Date, Open, High, Low, Close
    """
    vix_files = list(PROJECT_ROOT.glob('hist_india_vix_*.csv'))
    
    if not vix_files:
        logger.warning(f"No India VIX CSV files found in {PROJECT_ROOT}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(vix_files)} India VIX CSV file(s)")
    
    all_data = []
    
    for vix_file in vix_files:
        try:
            # Read CSV - handle spaces in column names
            df = pd.read_csv(vix_file)
            
            # Clean column names (remove leading/trailing spaces)
            df.columns = df.columns.str.strip()
            
            # Parse date - format is DD-MMM-YYYY
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y', dayfirst=True)
            
            # Filter by year if specified
            if year is not None:
                df = df[df['Date'].dt.year == year]
            
            if not df.empty:
                all_data.append(df)
                logger.info(f"Loaded {len(df)} rows from {vix_file.name}")
                
        except Exception as e:
            logger.warning(f"Error reading {vix_file.name}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=['Date']).sort_values('Date')
    
    logger.info(f"Total India VIX data: {len(combined)} rows")
    return combined


def fetch_india_vix(
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Fetch India VIX historical data from NSE.
    
    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    use_cache : bool
        Whether to use cached data
    cache_dir : Path, optional
        Directory for cache files
        
    Returns
    -------
    pd.DataFrame
        India VIX data with columns: Date, Open, High, Low, Close
    """
    try:
        from nsepy import get_history
        from nsepy.symbols import get_symbol_list
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Fetch VIX data
        vix_data = get_history(
            symbol="INDIAVIX",
            start=start,
            end=end,
            index=True
        )
        
        if vix_data.empty:
            logger.warning("No India VIX data returned from nsepy")
            return pd.DataFrame()
        
        vix_data = vix_data.reset_index()
        logger.info(f"Fetched India VIX data: {len(vix_data)} rows")
        return vix_data
        
    except ImportError:
        logger.warning("nsepy not installed. Install with: pip install nsepy")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching India VIX: {e}")
        return pd.DataFrame()


def fetch_india_vix_nselib(
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Alternative: Fetch India VIX using nselib package.
    
    Parameters
    ----------
    start_date : str
        Start date in DD-MM-YYYY format
    end_date : str
        End date in DD-MM-YYYY format
        
    Returns
    -------
    pd.DataFrame
        India VIX data
    """
    try:
        from nselib import capital_market
        
        vix_data = capital_market.India_VIX_data(
            from_date=start_date,
            to_date=end_date
        )
        
        if vix_data is not None and not vix_data.empty:
            logger.info(f"Fetched India VIX via nselib: {len(vix_data)} rows")
            return vix_data
        
        return pd.DataFrame()
        
    except ImportError:
        logger.warning("nselib not installed. Install with: pip install nselib")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching India VIX via nselib: {e}")
        return pd.DataFrame()


def calculate_vix_signals(
    vix_data: pd.DataFrame,
    event_date: str,
    lookback_days: int = 5
) -> Dict[str, float]:
    """
    Calculate VIX-based sentiment signals.
    
    Parameters
    ----------
    vix_data : pd.DataFrame
        India VIX historical data
    event_date : str
        Budget date in YYYY-MM-DD format
    lookback_days : int
        Days to look back for average
        
    Returns
    -------
    dict
        VIX signals including:
        - vix_pre_event: VIX level day before event
        - vix_avg_lookback: Average VIX over lookback period
        - vix_zscore: How unusual is current VIX vs recent history
        - vix_regime: 'low', 'normal', or 'high' fear
    """
    if vix_data.empty:
        return {}
    
    # Ensure date column
    date_col = 'Date' if 'Date' in vix_data.columns else 'date'
    if date_col not in vix_data.columns:
        logger.warning("No date column found in VIX data")
        return {}
    
    vix_data = vix_data.copy()
    vix_data[date_col] = pd.to_datetime(vix_data[date_col])
    vix_data = vix_data.sort_values(date_col)
    
    event_dt = pd.to_datetime(event_date)
    
    # Get pre-event VIX (day before budget)
    pre_event_data = vix_data[vix_data[date_col] < event_dt]
    
    if pre_event_data.empty:
        return {}
    
    close_col = 'Close' if 'Close' in vix_data.columns else 'close'
    
    vix_pre_event = pre_event_data[close_col].iloc[-1]
    
    # Calculate lookback average
    lookback_data = pre_event_data.tail(lookback_days)
    vix_avg = lookback_data[close_col].mean()
    vix_std = lookback_data[close_col].std()
    
    # Z-score
    vix_zscore = (vix_pre_event - vix_avg) / vix_std if vix_std > 0 else 0
    
    # Regime classification
    if vix_zscore > 1.5:
        regime = 'high_fear'
    elif vix_zscore < -1.0:
        regime = 'low_fear'
    else:
        regime = 'normal'
    
    signals = {
        'vix_pre_event': vix_pre_event,
        'vix_avg_lookback': vix_avg,
        'vix_std_lookback': vix_std,
        'vix_zscore': vix_zscore,
        'vix_regime': regime,
        'vix_change_pct': ((vix_pre_event - vix_avg) / vix_avg) * 100 if vix_avg > 0 else 0
    }
    
    logger.info(f"VIX signals for {event_date}: VIX={vix_pre_event:.2f}, Regime={regime}")
    
    return signals


def get_pre_budget_vix_signal(
    budget_date: str,
    lookback_days: int = 10
) -> Dict[str, float]:
    """
    Convenience function to get VIX sentiment before a budget.
    
    Checks local CSV files first, then falls back to API.
    
    Parameters
    ----------
    budget_date : str
        Budget date in YYYY-MM-DD format
    lookback_days : int
        Days to look back
        
    Returns
    -------
    dict
        VIX-based sentiment signals
    """
    event_dt = datetime.strptime(budget_date, '%Y-%m-%d')
    year = event_dt.year
    
    # TRY LOCAL CSV FIRST (preferred - uses real downloaded data)
    logger.info(f"Checking local CSV files for India VIX data (year: {year})...")
    vix_data = load_india_vix_csv(year)
    
    # If local data exists for this year, use it
    if not vix_data.empty:
        logger.info(f"✓ Using local India VIX CSV data ({len(vix_data)} rows)")
        return calculate_vix_signals(vix_data, budget_date, lookback_days)
    
    # FALLBACK: Try API (nsepy)
    logger.info("Local CSV not found, trying nsepy API...")
    start_date = (event_dt - timedelta(days=lookback_days + 10)).strftime('%Y-%m-%d')
    end_date = budget_date
    vix_data = fetch_india_vix(start_date, end_date)
    
    # FALLBACK 2: nselib with different date format
    if vix_data.empty:
        logger.info("nsepy failed, trying nselib...")
        start_nselib = (event_dt - timedelta(days=lookback_days + 10)).strftime('%d-%m-%Y')
        end_nselib = event_dt.strftime('%d-%m-%Y')
        vix_data = fetch_india_vix_nselib(start_nselib, end_nselib)
    
    if vix_data.empty:
        logger.warning(f"Could not fetch VIX data for {budget_date}")
        return {}
    
    return calculate_vix_signals(vix_data, budget_date, lookback_days)


def interpret_vix_for_strategy(vix_signals: Dict[str, float]) -> str:
    """
    Interpret VIX signals for trading strategy.
    
    Parameters
    ----------
    vix_signals : dict
        VIX signal dictionary
        
    Returns
    -------
    str
        Strategy recommendation
    """
    if not vix_signals:
        return "No VIX data available"
    
    regime = vix_signals.get('vix_regime', 'normal')
    zscore = vix_signals.get('vix_zscore', 0)
    
    if regime == 'high_fear':
        return (
            f"HIGH FEAR (VIX z-score: {zscore:.2f}). "
            "Market is nervous pre-Budget. Positive surprises may trigger sharp rallies. "
            "Consider momentum strategy with tight stops."
        )
    elif regime == 'low_fear':
        return (
            f"LOW FEAR (VIX z-score: {zscore:.2f}). "
            "Market is complacent. Unexpected announcements may cause volatility. "
            "Consider hedged positions or wait-and-see approach."
        )
    else:
        return (
            f"NORMAL (VIX z-score: {zscore:.2f}). "
            "Market uncertainty is typical. Standard event-driven strategy applicable."
        )

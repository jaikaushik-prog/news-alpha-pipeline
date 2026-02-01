"""
Market data loading module for Budget Speech Impact Analysis.

Loads 5-minute OHLCV data for NIFTY 500 stocks from CSV files.
Data is read-only - no modifications to raw files.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import date, datetime
import pandas as pd
import numpy as np
import yaml

from ..utils.logging import get_logger
from ..utils.time_utils import IST

logger = get_logger(__name__)


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    Load configuration from YAML files.
    
    Parameters
    ----------
    config_path : Path, optional
        Path to config directory
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parents[2] / "config"
    
    config = {}
    
    for yaml_file in ['paths.yaml', 'sectors.yaml', 'event_dates.yaml', 'model_params.yaml']:
        file_path = config_path / yaml_file
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                config[yaml_file.replace('.yaml', '')] = yaml.safe_load(f)
    
    return config


def get_data_dir() -> Path:
    """
    Get the base data directory.
    
    Returns
    -------
    Path
        Data directory path
    """
    # Default location - same level as src
    return Path(__file__).parents[2]


def get_available_stocks() -> List[str]:
    """
    Get list of available stock symbols from CSV files.
    
    Returns
    -------
    list
        List of stock symbols (without .csv extension)
    """
    data_dir = get_data_dir()
    csv_files = list(data_dir.glob("*.csv"))
    
    # Filter out non-stock files
    exclude = ['portfolio_pairs', 'stat_arb_pairs', 'stock_sector_map']
    
    symbols = []
    for f in csv_files:
        symbol = f.stem
        if symbol not in exclude and not symbol.endswith('-checkpoint'):
            symbols.append(symbol)
    
    logger.info(f"Found {len(symbols)} stock data files")
    return sorted(symbols)


def load_single_stock(
    symbol: str,
    data_dir: Optional[Path] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Load 5-minute OHLCV data for a single stock.
    
    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., 'RELIANCE')
    data_dir : Path, optional
        Data directory path
    start_date : date, optional
        Filter data from this date
    end_date : date, optional
        Filter data to this date
    parse_dates : bool
        Whether to parse date column to datetime
        
    Returns
    -------
    pd.DataFrame
        OHLCV data with columns: date, open, high, low, close, volume
    """
    if data_dir is None:
        data_dir = get_data_dir()
    
    filepath = data_dir / f"{symbol}.csv"
    
    if not filepath.exists():
        logger.warning(f"Data file not found: {filepath}")
        return pd.DataFrame()
    
    # Read CSV
    df = pd.read_csv(filepath)
    
    # Parse dates
    if parse_dates and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
    
    # Filter by date range
    if start_date is not None:
        start_dt = pd.Timestamp(start_date).tz_localize(IST)
        df = df[df.index >= start_dt]
    
    if end_date is not None:
        end_dt = pd.Timestamp(end_date).tz_localize(IST) + pd.Timedelta(days=1)
        df = df[df.index < end_dt]
    
    logger.debug(f"Loaded {len(df)} bars for {symbol}")
    
    return df


def load_multiple_stocks(
    symbols: List[str],
    data_dir: Optional[Path] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load data for multiple stocks.
    
    Parameters
    ----------
    symbols : list of str
        Stock symbols to load
    data_dir : Path, optional
        Data directory path
    start_date : date, optional
        Filter data from this date
    end_date : date, optional
        Filter data to this date
        
    Returns
    -------
    dict
        Dictionary mapping symbol -> DataFrame
    """
    data = {}
    
    for symbol in symbols:
        df = load_single_stock(symbol, data_dir, start_date, end_date)
        if not df.empty:
            data[symbol] = df
    
    logger.info(f"Loaded data for {len(data)}/{len(symbols)} stocks")
    
    return data


def load_sector_stocks(
    sector: str,
    config: Optional[dict] = None
) -> List[str]:
    """
    Get list of stocks in a sector.
    
    Parameters
    ----------
    sector : str
        Sector name (e.g., 'banking_nbfc')
    config : dict, optional
        Configuration dictionary
        
    Returns
    -------
    list
        Stock symbols in the sector
    """
    if config is None:
        config = load_config()
    
    sectors_config = config.get('sectors', {}).get('sectors', {})
    
    if sector not in sectors_config:
        logger.warning(f"Unknown sector: {sector}")
        return []
    
    return sectors_config[sector].get('stocks', [])


def get_all_sectors() -> List[str]:
    """
    Get list of all sector names.
    
    Returns
    -------
    list
        Sector names
    """
    config = load_config()
    return list(config.get('sectors', {}).get('sectors', {}).keys())


def validate_data_quality(df: pd.DataFrame, symbol: str = "") -> Dict:
    """
    Validate data quality for a stock.
    
    Parameters
    ----------
    df : pd.DataFrame
        Stock data
    symbol : str
        Stock symbol (for logging)
        
    Returns
    -------
    dict
        Quality metrics
    """
    quality = {
        'symbol': symbol,
        'total_rows': len(df),
        'date_range': (df.index.min(), df.index.max()) if len(df) > 0 else (None, None),
        'missing_values': df.isnull().sum().to_dict(),
        'zero_volume_pct': (df['volume'] == 0).mean() * 100 if 'volume' in df.columns else None,
        'negative_prices': (df[['open', 'high', 'low', 'close']] < 0).any().any() if len(df) > 0 else False,
        'high_low_valid': (df['high'] >= df['low']).all() if len(df) > 0 else True,
        'open_in_range': ((df['open'] >= df['low']) & (df['open'] <= df['high'])).all() if len(df) > 0 else True,
        'close_in_range': ((df['close'] >= df['low']) & (df['close'] <= df['high'])).all() if len(df) > 0 else True,
    }
    
    # Check for duplicate timestamps
    quality['duplicate_timestamps'] = df.index.duplicated().sum()
    
    # Check for gaps (missing bars)
    if len(df) > 0:
        expected_bars = (df.index.max() - df.index.min()).total_seconds() / 300  # 5-min bars
        quality['coverage_pct'] = (len(df) / expected_bars * 100) if expected_bars > 0 else 100
    else:
        quality['coverage_pct'] = 0
    
    return quality


def get_trading_dates(
    df: pd.DataFrame,
    return_as: str = 'list'
) -> Union[List[date], pd.DatetimeIndex]:
    """
    Get unique trading dates from data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Stock data with datetime index
    return_as : str
        'list' or 'index'
        
    Returns
    -------
    list or pd.DatetimeIndex
        Unique trading dates
    """
    dates = df.index.normalize().unique()
    
    if return_as == 'list':
        return [d.date() for d in dates]
    return dates


def get_budget_day_data(
    symbol: str,
    budget_date: date,
    data_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Get intraday data for a specific budget day.
    
    Parameters
    ----------
    symbol : str
        Stock symbol
    budget_date : date
        Budget presentation date
    data_dir : Path, optional
        Data directory
        
    Returns
    -------
    pd.DataFrame
        Intraday OHLCV data for the budget day
    """
    return load_single_stock(
        symbol,
        data_dir=data_dir,
        start_date=budget_date,
        end_date=budget_date
    )

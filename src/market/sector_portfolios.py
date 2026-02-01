"""
Sector portfolio construction module.

Builds sector-level portfolios with equal and volume weighting.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

from ..utils.logging import get_logger
from ..ingestion.load_market_data import load_multiple_stocks, load_config

logger = get_logger(__name__)


def load_sector_mapping() -> Dict[str, List[str]]:
    """
    Load sector to stocks mapping from config.
    
    Returns
    -------
    dict
        Sector name -> list of stock symbols
    """
    config = load_config()
    sectors_config = config.get('sectors', {}).get('sectors', {})
    
    mapping = {}
    for sector, info in sectors_config.items():
        mapping[sector] = info.get('stocks', [])
    
    return mapping


def build_return_matrix(
    data: Dict[str, pd.DataFrame],
    return_col: str = 'return'
) -> pd.DataFrame:
    """
    Build a matrix of returns across all stocks.
    
    Parameters
    ----------
    data : dict
        Dictionary of symbol -> DataFrame with returns
    return_col : str
        Name of return column
        
    Returns
    -------
    pd.DataFrame
        Matrix with timestamps as index, symbols as columns
    """
    returns = {}
    
    for symbol, df in data.items():
        if return_col in df.columns:
            returns[symbol] = df[return_col]
    
    return pd.DataFrame(returns)


def build_volume_matrix(
    data: Dict[str, pd.DataFrame],
    volume_col: str = 'volume'
) -> pd.DataFrame:
    """
    Build a matrix of volumes across all stocks.
    
    Parameters
    ----------
    data : dict
        Dictionary of symbol -> DataFrame
    volume_col : str
        Name of volume column
        
    Returns
    -------
    pd.DataFrame
        Matrix with timestamps as index, symbols as columns
    """
    volumes = {}
    
    for symbol, df in data.items():
        if volume_col in df.columns:
            volumes[symbol] = df[volume_col]
    
    return pd.DataFrame(volumes)


def calculate_sector_return_equal_weight(
    return_matrix: pd.DataFrame,
    sector_stocks: List[str]
) -> pd.Series:
    """
    Calculate equal-weighted sector return.
    
    Parameters
    ----------
    return_matrix : pd.DataFrame
        Returns matrix (timestamps x symbols)
    sector_stocks : list
        Stocks in the sector
        
    Returns
    -------
    pd.Series
        Sector return series
    """
    # Filter to sector stocks
    available = [s for s in sector_stocks if s in return_matrix.columns]
    
    if not available:
        return pd.Series(index=return_matrix.index, dtype=float)
    
    sector_returns = return_matrix[available]
    
    # Equal-weighted mean
    return sector_returns.mean(axis=1)


def calculate_sector_return_volume_weight(
    return_matrix: pd.DataFrame,
    volume_matrix: pd.DataFrame,
    sector_stocks: List[str]
) -> pd.Series:
    """
    Calculate volume-weighted sector return.
    
    Parameters
    ----------
    return_matrix : pd.DataFrame
        Returns matrix
    volume_matrix : pd.DataFrame
        Volumes matrix
    sector_stocks : list
        Stocks in the sector
        
    Returns
    -------
    pd.Series
        Sector return series
    """
    # Filter to sector stocks
    available = [s for s in sector_stocks if s in return_matrix.columns and s in volume_matrix.columns]
    
    if not available:
        return pd.Series(index=return_matrix.index, dtype=float)
    
    sector_returns = return_matrix[available]
    sector_volumes = volume_matrix[available]
    
    # Normalize volumes to weights
    weights = sector_volumes.div(sector_volumes.sum(axis=1), axis=0)
    
    # Volume-weighted return
    weighted_return = (sector_returns * weights).sum(axis=1)
    
    return weighted_return


def build_sector_portfolios(
    data: Dict[str, pd.DataFrame],
    weighting: str = 'equal'
) -> pd.DataFrame:
    """
    Build sector portfolio returns.
    
    Parameters
    ----------
    data : dict
        Dictionary of symbol -> DataFrame
    weighting : str
        'equal' or 'volume'
        
    Returns
    -------
    pd.DataFrame
        Sector returns (timestamps x sectors)
    """
    # Build matrices
    return_matrix = build_return_matrix(data)
    volume_matrix = build_volume_matrix(data) if weighting == 'volume' else None
    
    # Get sector mapping
    sector_mapping = load_sector_mapping()
    
    # Calculate sector returns
    sector_returns = {}
    
    for sector, stocks in sector_mapping.items():
        if weighting == 'volume' and volume_matrix is not None:
            sector_returns[sector] = calculate_sector_return_volume_weight(
                return_matrix, volume_matrix, stocks
            )
        else:
            sector_returns[sector] = calculate_sector_return_equal_weight(
                return_matrix, stocks
            )
        
        # Log coverage
        available = [s for s in stocks if s in return_matrix.columns]
        logger.debug(f"Sector {sector}: {len(available)}/{len(stocks)} stocks available")
    
    df = pd.DataFrame(sector_returns)
    
    logger.info(f"Built sector portfolios: {len(df)} timestamps, {len(df.columns)} sectors")
    
    return df


def calculate_sector_volume(
    data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Calculate aggregate sector volume.
    
    Parameters
    ----------
    data : dict
        Dictionary of symbol -> DataFrame
        
    Returns
    -------
    pd.DataFrame
        Sector volumes (timestamps x sectors)
    """
    volume_matrix = build_volume_matrix(data)
    sector_mapping = load_sector_mapping()
    
    sector_volumes = {}
    
    for sector, stocks in sector_mapping.items():
        available = [s for s in stocks if s in volume_matrix.columns]
        if available:
            sector_volumes[sector] = volume_matrix[available].sum(axis=1)
    
    return pd.DataFrame(sector_volumes)


def calculate_sector_dispersion(
    data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Calculate within-sector return dispersion.
    
    Parameters
    ----------
    data : dict
        Dictionary of symbol -> DataFrame
        
    Returns
    -------
    pd.DataFrame
        Sector dispersions (timestamps x sectors)
    """
    return_matrix = build_return_matrix(data)
    sector_mapping = load_sector_mapping()
    
    dispersions = {}
    
    for sector, stocks in sector_mapping.items():
        available = [s for s in stocks if s in return_matrix.columns]
        if len(available) >= 2:
            sector_returns = return_matrix[available]
            dispersions[sector] = sector_returns.std(axis=1)
    
    return pd.DataFrame(dispersions)


def calculate_market_portfolio(
    data: Dict[str, pd.DataFrame],
    weighting: str = 'equal'
) -> pd.Series:
    """
    Calculate overall market portfolio return.
    
    Parameters
    ----------
    data : dict
        Dictionary of symbol -> DataFrame
    weighting : str
        'equal' or 'volume'
        
    Returns
    -------
    pd.Series
        Market return series
    """
    return_matrix = build_return_matrix(data)
    
    if weighting == 'volume':
        volume_matrix = build_volume_matrix(data)
        weights = volume_matrix.div(volume_matrix.sum(axis=1), axis=0)
        return (return_matrix * weights).sum(axis=1)
    else:
        return return_matrix.mean(axis=1)


def build_all_sector_metrics(
    data: Dict[str, pd.DataFrame],
    output_dir: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """
    Build all sector-level metrics.
    
    Parameters
    ----------
    data : dict
        Dictionary of symbol -> DataFrame
    output_dir : Path, optional
        Directory to save outputs
        
    Returns
    -------
    dict
        Dictionary of metric name -> DataFrame
    """
    metrics = {}
    
    # Sector returns (equal weighted)
    metrics['sector_returns_equal'] = build_sector_portfolios(data, weighting='equal')
    
    # Sector returns (volume weighted)
    metrics['sector_returns_volume'] = build_sector_portfolios(data, weighting='volume')
    
    # Sector volume
    metrics['sector_volume'] = calculate_sector_volume(data)
    
    # Sector dispersion
    metrics['sector_dispersion'] = calculate_sector_dispersion(data)
    
    # Market portfolio
    metrics['market_return'] = calculate_market_portfolio(data)
    
    # Save if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in metrics.items():
            if isinstance(df, pd.Series):
                df = df.to_frame(name=name)
            df.to_parquet(output_dir / f"{name}.parquet")
            logger.info(f"Saved {name} to {output_dir}")
    
    return metrics

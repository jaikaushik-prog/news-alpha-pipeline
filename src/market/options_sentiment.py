"""
Options Market Sentiment Module.

Extracts options-implied sentiment signals as pre-budget indicators:
- Put-Call Ratio (PCR) - Fear vs Greed
- Open Interest changes - Position build-up
- Implied Volatility - Event fear
- IV Skew - Tail risk

These are SUPERIOR to social media sentiment because they reflect
real capital-at-risk positions from institutional traders.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Index options for sector sentiment
SECTOR_OPTIONS = {
    'banking_nbfc': 'BANKNIFTY',
    'financials': 'FINNIFTY',
    'it_technology': 'NIFTYIT',
    'market': 'NIFTY'
}


def fetch_option_chain_nse(
    symbol: str = 'NIFTY',
    expiry: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch option chain data from NSE.
    
    Parameters
    ----------
    symbol : str
        Index symbol (NIFTY, BANKNIFTY, FINNIFTY, NIFTYIT)
    expiry : str, optional
        Expiry date (if None, uses nearest expiry)
        
    Returns
    -------
    pd.DataFrame
        Option chain with strikes, OI, IV, etc.
    """
    try:
        from nselib import derivatives
        
        # Get option chain
        chain = derivatives.nse_live_option_chain(symbol)
        
        if chain is not None and not chain.empty:
            logger.info(f"Fetched {symbol} option chain: {len(chain)} strikes")
            return chain
        
        return pd.DataFrame()
        
    except ImportError:
        logger.warning("nselib not installed. Install with: pip install nselib")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching option chain: {e}")
        return pd.DataFrame()


def calculate_pcr(option_chain: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate Put-Call Ratio from option chain.
    
    Parameters
    ----------
    option_chain : pd.DataFrame
        Option chain data
        
    Returns
    -------
    dict
        PCR metrics
    """
    if option_chain.empty:
        return {}
    
    try:
        # Identify put and call columns
        put_oi = option_chain.get('PE_openInterest', option_chain.get('putOI', pd.Series([0]))).sum()
        call_oi = option_chain.get('CE_openInterest', option_chain.get('callOI', pd.Series([0]))).sum()
        
        put_vol = option_chain.get('PE_totalTradedVolume', option_chain.get('putVolume', pd.Series([0]))).sum()
        call_vol = option_chain.get('CE_totalTradedVolume', option_chain.get('callVolume', pd.Series([0]))).sum()
        
        pcr_oi = put_oi / call_oi if call_oi > 0 else 0
        pcr_vol = put_vol / call_vol if call_vol > 0 else 0
        
        return {
            'pcr_oi': pcr_oi,
            'pcr_volume': pcr_vol,
            'total_put_oi': put_oi,
            'total_call_oi': call_oi,
            'pcr_interpretation': _interpret_pcr(pcr_oi)
        }
        
    except Exception as e:
        logger.error(f"Error calculating PCR: {e}")
        return {}


def _interpret_pcr(pcr: float) -> str:
    """Interpret PCR value."""
    if pcr > 1.2:
        return 'extreme_fear'
    elif pcr > 1.0:
        return 'bearish'
    elif pcr > 0.7:
        return 'neutral'
    elif pcr > 0.5:
        return 'bullish'
    else:
        return 'extreme_greed'


def calculate_atm_iv(
    option_chain: pd.DataFrame,
    spot_price: float
) -> Dict[str, float]:
    """
    Calculate At-The-Money Implied Volatility.
    
    Parameters
    ----------
    option_chain : pd.DataFrame
        Option chain data
    spot_price : float
        Current spot price
        
    Returns
    -------
    dict
        IV metrics
    """
    if option_chain.empty:
        return {}
    
    try:
        # Find ATM strike (closest to spot)
        strikes = option_chain.get('strikePrice', option_chain.get('strike', pd.Series()))
        if strikes.empty:
            return {}
        
        atm_strike = strikes.iloc[(strikes - spot_price).abs().argmin()]
        
        atm_row = option_chain[
            option_chain.get('strikePrice', option_chain.get('strike')) == atm_strike
        ].iloc[0]
        
        call_iv = atm_row.get('CE_impliedVolatility', atm_row.get('callIV', 0))
        put_iv = atm_row.get('PE_impliedVolatility', atm_row.get('putIV', 0))
        
        atm_iv = (call_iv + put_iv) / 2 if (call_iv > 0 and put_iv > 0) else max(call_iv, put_iv)
        
        return {
            'atm_iv': atm_iv,
            'atm_strike': atm_strike,
            'call_iv': call_iv,
            'put_iv': put_iv,
            'iv_skew': put_iv - call_iv  # Positive = more put demand
        }
        
    except Exception as e:
        logger.error(f"Error calculating ATM IV: {e}")
        return {}


def calculate_oi_change(
    current_chain: pd.DataFrame,
    previous_chain: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate Open Interest changes.
    
    Parameters
    ----------
    current_chain : pd.DataFrame
        Current option chain
    previous_chain : pd.DataFrame
        Previous day's option chain
        
    Returns
    -------
    dict
        OI change metrics
    """
    if current_chain.empty or previous_chain.empty:
        return {}
    
    try:
        current_put_oi = current_chain.get('PE_openInterest', pd.Series([0])).sum()
        current_call_oi = current_chain.get('CE_openInterest', pd.Series([0])).sum()
        
        prev_put_oi = previous_chain.get('PE_openInterest', pd.Series([0])).sum()
        prev_call_oi = previous_chain.get('CE_openInterest', pd.Series([0])).sum()
        
        put_oi_change = current_put_oi - prev_put_oi
        call_oi_change = current_call_oi - prev_call_oi
        
        return {
            'put_oi_change': put_oi_change,
            'call_oi_change': call_oi_change,
            'net_oi_change': call_oi_change - put_oi_change,
            'put_oi_change_pct': (put_oi_change / prev_put_oi * 100) if prev_put_oi > 0 else 0,
            'call_oi_change_pct': (call_oi_change / prev_call_oi * 100) if prev_call_oi > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error calculating OI change: {e}")
        return {}


def get_options_sentiment(
    symbol: str = 'NIFTY',
    spot_price: Optional[float] = None
) -> Dict[str, float]:
    """
    Get comprehensive options-derived sentiment.
    
    Parameters
    ----------
    symbol : str
        Index symbol
    spot_price : float, optional
        Current spot price (fetched if not provided)
        
    Returns
    -------
    dict
        Options sentiment signals
    """
    chain = fetch_option_chain_nse(symbol)
    
    if chain.empty:
        logger.warning(f"Could not fetch {symbol} option chain")
        return {}
    
    # Calculate all metrics
    pcr = calculate_pcr(chain)
    
    # Estimate spot if not provided
    if spot_price is None:
        try:
            strikes = chain.get('strikePrice', chain.get('strike', pd.Series()))
            spot_price = strikes.median()  # Rough estimate
        except:
            spot_price = 20000  # Default for NIFTY
    
    iv = calculate_atm_iv(chain, spot_price)
    
    sentiment = {
        'symbol': symbol,
        'timestamp': datetime.now(),
        **pcr,
        **iv
    }
    
    # Calculate composite score
    sentiment['options_sentiment_score'] = _calculate_composite_score(sentiment)
    
    return sentiment


def _calculate_composite_score(metrics: Dict) -> float:
    """
    Calculate composite options sentiment score.
    
    Score interpretation:
    - Positive = Bullish (low PCR, low IV, greed)
    - Negative = Bearish (high PCR, high IV, fear)
    """
    score = 0.0
    
    pcr = metrics.get('pcr_oi', 1.0)
    iv = metrics.get('atm_iv', 15.0)
    skew = metrics.get('iv_skew', 0.0)
    
    # PCR contribution (-1 to +1)
    # Low PCR = bullish, High PCR = bearish
    pcr_score = (1.0 - pcr) if pcr <= 1.0 else -(pcr - 1.0)
    pcr_score = max(-1, min(1, pcr_score))
    
    # IV contribution (-1 to +1)
    # Normalize: 10-30 range
    iv_score = -(iv - 20) / 10  # High IV = bearish
    iv_score = max(-1, min(1, iv_score))
    
    # Skew contribution (-1 to +1)
    skew_score = -skew / 5  # Positive skew (put demand) = bearish
    skew_score = max(-1, min(1, skew_score))
    
    # Weighted composite
    score = (0.5 * pcr_score) + (0.3 * iv_score) + (0.2 * skew_score)
    
    return round(score, 3)


def get_sector_options_sentiment() -> pd.DataFrame:
    """
    Get options sentiment for all tracked sectors.
    
    Returns
    -------
    pd.DataFrame
        Sector-wise options sentiment
    """
    results = []
    
    for sector, symbol in SECTOR_OPTIONS.items():
        logger.info(f"Fetching {symbol} options for {sector}...")
        sentiment = get_options_sentiment(symbol)
        
        if sentiment:
            sentiment['sector'] = sector
            results.append(sentiment)
    
    if results:
        return pd.DataFrame(results)
    
    return pd.DataFrame()


def create_mock_options_sentiment(budget_date: str = None) -> pd.DataFrame:
    """
    Create mock options sentiment data for demonstration.
    
    Parameters
    ----------
    budget_date : str
        Budget date for context
        
    Returns
    -------
    pd.DataFrame
        Mock options sentiment
    """
    np.random.seed(42)
    
    sectors = list(SECTOR_OPTIONS.keys())
    
    mock_data = []
    for sector in sectors:
        # Generate realistic pre-budget levels
        # Pre-budget typically sees elevated IV and PCR
        mock_data.append({
            'sector': sector,
            'symbol': SECTOR_OPTIONS[sector],
            'pcr_oi': np.random.uniform(0.8, 1.3),
            'pcr_volume': np.random.uniform(0.7, 1.4),
            'pcr_interpretation': np.random.choice(['neutral', 'bearish', 'bullish']),
            'atm_iv': np.random.uniform(15, 28),  # Elevated pre-event
            'iv_skew': np.random.uniform(-2, 4),  # Slightly positive (put demand)
            'put_oi_change_pct': np.random.uniform(-5, 15),
            'call_oi_change_pct': np.random.uniform(-3, 10),
            'options_sentiment_score': np.random.uniform(-0.5, 0.5)
        })
    
    df = pd.DataFrame(mock_data)
    logger.info(f"Created mock options sentiment for {len(sectors)} sectors")
    return df

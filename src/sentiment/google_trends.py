"""
Google Trends Module.

Uses search interest as curiosity/anxiety signal for pre-budget sentiment.
Superior retail proxy without social media noise.

No API key required - uses pytrends library.

Key Features:
- Sector-specific keyword tracking
- Rising searches = attention spike = volatility risk
- Comparison across sectors for relative interest
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Budget-related search terms by sector
BUDGET_SEARCH_TERMS = {
    'general': [
        'union budget 2025',
        'budget expectations',
        'nirmala sitharaman budget'
    ],
    'tax': [
        'income tax budget',
        'income tax exemption',
        'tax slab budget',
        'capital gains tax budget'
    ],
    'infrastructure': [
        'infrastructure budget',
        'railway budget',
        'highway budget',
        'metro budget allocation'
    ],
    'defence': [
        'defence budget india',
        'defence spending budget',
        'HAL budget'
    ],
    'agriculture': [
        'agriculture budget',
        'MSP budget',
        'farmer budget',
        'fertilizer subsidy'
    ],
    'realty': [
        'housing budget',
        'real estate budget',
        'affordable housing scheme'
    ],
    'healthcare': [
        'healthcare budget',
        'pharma budget',
        'ayushman bharat budget'
    ],
    'education': [
        'education budget',
        'skill development budget'
    ],
    'ev_auto': [
        'electric vehicle budget',
        'EV subsidy budget',
        'automobile budget'
    ],
    'energy': [
        'solar budget',
        'green energy budget',
        'renewable energy budget'
    ]
}


def fetch_google_trends(
    keywords: List[str],
    timeframe: str = 'today 1-m',
    geo: str = 'IN'
) -> pd.DataFrame:
    """
    Fetch Google Trends data for keywords.
    
    Parameters
    ----------
    keywords : list
        List of search terms (max 5 per request)
    timeframe : str
        Time period: 'today 1-m', 'today 3-m', 'today 12-m'
    geo : str
        Geographic region code
        
    Returns
    -------
    pd.DataFrame
        Trends data with interest over time
    """
    try:
        from pytrends.request import TrendReq
        
        pytrends = TrendReq(hl='en-IN', tz=330)  # IST offset
        
        # Limit to 5 keywords (Google Trends limit)
        keywords = keywords[:5]
        
        pytrends.build_payload(
            kw_list=keywords,
            cat=0,
            timeframe=timeframe,
            geo=geo
        )
        
        df = pytrends.interest_over_time()
        
        if df.empty:
            logger.warning(f"No trends data for: {keywords}")
            return pd.DataFrame()
        
        # Remove 'isPartial' column if present
        if 'isPartial' in df.columns:
            df = df.drop('isPartial', axis=1)
        
        logger.info(f"Fetched trends for {len(keywords)} keywords: {len(df)} data points")
        return df
        
    except ImportError:
        logger.warning("pytrends not installed. Install with: pip install pytrends")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching Google Trends: {e}")
        return pd.DataFrame()


def calculate_attention_spike(
    trends_df: pd.DataFrame,
    baseline_days: int = 14
) -> Dict[str, float]:
    """
    Calculate attention spike relative to baseline.
    
    Parameters
    ----------
    trends_df : pd.DataFrame
        Google Trends data
    baseline_days : int
        Days for baseline calculation
        
    Returns
    -------
    dict
        Attention spike metrics per keyword
    """
    if trends_df.empty:
        return {}
    
    spikes = {}
    
    for col in trends_df.columns:
        series = trends_df[col]
        
        # Baseline: average of first N days
        baseline = series.head(baseline_days).mean()
        
        # Recent: last 3 days
        recent = series.tail(3).mean()
        
        # Current: latest value
        current = series.iloc[-1]
        
        # Spike calculation
        spike_vs_baseline = ((recent - baseline) / baseline * 100) if baseline > 0 else 0
        
        spikes[col] = {
            'current_interest': current,
            'recent_avg': recent,
            'baseline_avg': baseline,
            'spike_pct': spike_vs_baseline,
            'is_spiking': spike_vs_baseline > 50  # >50% increase
        }
    
    return spikes


def get_sector_attention(sector: str) -> Dict[str, float]:
    """
    Get Google Trends attention for a specific sector.
    
    Parameters
    ----------
    sector : str
        Sector name (must be in BUDGET_SEARCH_TERMS)
        
    Returns
    -------
    dict
        Aggregated attention metrics
    """
    keywords = BUDGET_SEARCH_TERMS.get(sector, [])
    
    if not keywords:
        logger.warning(f"No keywords defined for sector: {sector}")
        return {}
    
    trends_df = fetch_google_trends(keywords[:5])
    
    if trends_df.empty:
        return {'sector': sector, 'attention_score': 0}
    
    spikes = calculate_attention_spike(trends_df)
    
    # Aggregate across keywords
    avg_spike = np.mean([v['spike_pct'] for v in spikes.values()])
    any_spiking = any(v['is_spiking'] for v in spikes.values())
    max_current = max(v['current_interest'] for v in spikes.values())
    
    return {
        'sector': sector,
        'avg_spike_pct': avg_spike,
        'any_spiking': any_spiking,
        'max_current_interest': max_current,
        'attention_score': min(100, avg_spike) / 100,  # Normalize to 0-1
        'keywords_analyzed': len(spikes)
    }


def get_all_sector_attention() -> pd.DataFrame:
    """
    Get Google Trends attention for all sectors.
    
    Returns
    -------
    pd.DataFrame
        Sector-wise attention metrics
    """
    results = []
    
    for sector in BUDGET_SEARCH_TERMS.keys():
        logger.info(f"Fetching trends for {sector}...")
        attention = get_sector_attention(sector)
        results.append(attention)
    
    return pd.DataFrame(results)


def get_pre_budget_attention_signal(
    budget_date: str,
    days_before: int = 7
) -> Dict[str, float]:
    """
    Get aggregated attention signal before budget.
    
    Parameters
    ----------
    budget_date : str
        Budget date
    days_before : int
        Days to analyze
        
    Returns
    -------
    dict
        Pre-budget attention signal
    """
    # Use 1-month timeframe to capture pre-budget period
    all_keywords = []
    for keywords in BUDGET_SEARCH_TERMS.values():
        all_keywords.extend(keywords[:2])  # Top 2 per sector
    
    # Deduplicate and limit
    all_keywords = list(set(all_keywords))[:5]
    
    trends_df = fetch_google_trends(all_keywords, timeframe='today 1-m')
    
    if trends_df.empty:
        return {'attention_level': 'unknown'}
    
    spikes = calculate_attention_spike(trends_df)
    
    avg_spike = np.mean([v['spike_pct'] for v in spikes.values()])
    
    # Classify attention level
    if avg_spike > 100:
        attention_level = 'extreme_high'
    elif avg_spike > 50:
        attention_level = 'elevated'
    elif avg_spike > 20:
        attention_level = 'moderate'
    else:
        attention_level = 'normal'
    
    return {
        'avg_spike_pct': avg_spike,
        'attention_level': attention_level,
        'interpretation': _interpret_attention(attention_level),
        'keywords_spiking': sum(1 for v in spikes.values() if v['is_spiking'])
    }


def _interpret_attention(level: str) -> str:
    """Interpret attention level for trading."""
    interpretations = {
        'extreme_high': 'Maximum retail attention. High event volatility likely. Wide spreads expected.',
        'elevated': 'Significant public interest. Increased volatility around announcements.',
        'moderate': 'Normal pre-budget attention. Standard event-day moves expected.',
        'normal': 'Low attention. Market may be under-positioned for surprises.',
        'unknown': 'Unable to assess attention level.'
    }
    return interpretations.get(level, 'Unknown')


def create_mock_trends_data(sectors: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create mock Google Trends data for demonstration.
    
    Parameters
    ----------
    sectors : list, optional
        Sectors to include
        
    Returns
    -------
    pd.DataFrame
        Mock attention data
    """
    np.random.seed(42)
    
    if sectors is None:
        sectors = list(BUDGET_SEARCH_TERMS.keys())
    
    mock_data = []
    for sector in sectors:
        mock_data.append({
            'sector': sector,
            'avg_spike_pct': np.random.uniform(-10, 80),
            'any_spiking': np.random.choice([True, False]),
            'max_current_interest': np.random.randint(20, 100),
            'attention_score': np.random.uniform(0, 0.8),
            'keywords_analyzed': len(BUDGET_SEARCH_TERMS.get(sector, []))
        })
    
    logger.info(f"Created mock trends data for {len(sectors)} sectors")
    return pd.DataFrame(mock_data)

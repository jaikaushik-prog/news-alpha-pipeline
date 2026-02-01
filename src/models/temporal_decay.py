"""
Temporal Decay - Layer 6

Information half-life modeling for news items.

Different news types decay at different rates:
- Breaking policy: 4 hours
- Earnings: 1 day
- Macro: 3 days
- Commentary: 2 hours

Formula:
    Effective_Sentiment = Σ S_i × exp(−Δt / HalfLife_i)
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Half-life configurations (in hours)
HALF_LIFE_CONFIG = {
    'policy': 8,           # Policy announcements: 8 hours
    'earnings': 24,        # Earnings news: 1 day
    'macro': 72,           # Macro indicators: 3 days
    'breaking': 4,         # Breaking news: 4 hours
    'commentary': 2,       # Opinion/commentary: 2 hours
    'general': 12,         # General news: 12 hours
    'default': 6           # Default: 6 hours
}


@dataclass
class DecayedSentiment:
    """Sentiment with temporal decay applied."""
    original_sentiment: float
    decayed_sentiment: float
    half_life_hours: float
    age_hours: float
    decay_factor: float


def get_half_life(category: str) -> float:
    """
    Get half-life in hours for news category.
    
    Parameters
    ----------
    category : str
        News category
        
    Returns
    -------
    float
        Half-life in hours
    """
    return HALF_LIFE_CONFIG.get(category, HALF_LIFE_CONFIG['default'])


def calculate_decay_factor(
    age_hours: float,
    half_life_hours: float
) -> float:
    """
    Calculate exponential decay factor.
    
    Formula: decay = exp(-ln(2) * age / half_life)
    
    Parameters
    ----------
    age_hours : float
        Age of news in hours
    half_life_hours : float
        Half-life in hours
        
    Returns
    -------
    float
        Decay factor between 0 and 1
    """
    if half_life_hours <= 0:
        return 0.0
    
    decay = np.exp(-np.log(2) * age_hours / half_life_hours)
    return float(np.clip(decay, 0, 1))


def apply_temporal_decay(
    sentiment: float,
    timestamp: datetime,
    category: str = 'general',
    reference_time: Optional[datetime] = None
) -> DecayedSentiment:
    """
    Apply temporal decay to sentiment.
    
    Parameters
    ----------
    sentiment : float
        Original sentiment score
    timestamp : datetime
        Timestamp of news item
    category : str
        News category
    reference_time : datetime, optional
        Reference time (default: now)
        
    Returns
    -------
    DecayedSentiment
        Sentiment with decay applied
    """
    if reference_time is None:
        reference_time = datetime.now()
    
    # Calculate age in hours
    age_delta = reference_time - timestamp
    age_hours = age_delta.total_seconds() / 3600
    
    # Get half-life for category
    half_life = get_half_life(category)
    
    # Calculate decay
    decay_factor = calculate_decay_factor(age_hours, half_life)
    
    decayed = sentiment * decay_factor
    
    return DecayedSentiment(
        original_sentiment=sentiment,
        decayed_sentiment=decayed,
        half_life_hours=half_life,
        age_hours=age_hours,
        decay_factor=decay_factor
    )


def aggregate_decayed_sentiment(
    news_items: List[Dict],
    reference_time: Optional[datetime] = None
) -> float:
    """
    Aggregate sentiment with temporal decay.
    
    Formula: Effective_Sentiment = Σ S_i × exp(−Δt / τ_i)
    
    Parameters
    ----------
    news_items : list
        List of news items with 'sentiment', 'timestamp', 'category'
    reference_time : datetime, optional
        Reference time for decay calculation
        
    Returns
    -------
    float
        Aggregated decayed sentiment
    """
    if not news_items:
        return 0.0
    
    if reference_time is None:
        reference_time = datetime.now()
    
    total_sentiment = 0.0
    total_weight = 0.0
    
    for item in news_items:
        sentiment = item.get('sentiment', item.get('composite_score', 0))
        timestamp = item.get('timestamp', reference_time)
        category = item.get('category', 'general')
        
        # Convert string timestamp if needed
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Apply decay
        decayed = apply_temporal_decay(
            sentiment, timestamp, category, reference_time
        )
        
        total_sentiment += decayed.decayed_sentiment
        total_weight += decayed.decay_factor
    
    if total_weight == 0:
        return 0.0
    
    return total_sentiment / total_weight


def batch_apply_decay(
    df: pd.DataFrame,
    sentiment_col: str = 'sentiment',
    timestamp_col: str = 'timestamp',
    category_col: str = 'category',
    reference_time: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Apply temporal decay to DataFrame of news items.
    
    Parameters
    ----------
    df : pd.DataFrame
        News DataFrame
    sentiment_col : str
        Column with sentiment scores
    timestamp_col : str
        Column with timestamps
    category_col : str
        Column with categories
    reference_time : datetime, optional
        Reference time
        
    Returns
    -------
    pd.DataFrame
        DataFrame with decay columns added
    """
    if reference_time is None:
        reference_time = datetime.now()
    
    df = df.copy()
    
    # Ensure timestamp column is datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Calculate age in hours
    df['age_hours'] = (reference_time - df[timestamp_col]).dt.total_seconds() / 3600
    
    # Get half-life for each category
    df['half_life'] = df[category_col].apply(get_half_life)
    
    # Calculate decay factor
    df['decay_factor'] = df.apply(
        lambda row: calculate_decay_factor(row['age_hours'], row['half_life']),
        axis=1
    )
    
    # Apply decay
    df['decayed_sentiment'] = df[sentiment_col] * df['decay_factor']
    
    return df


class TemporalDecayEngine:
    """
    Complete temporal decay engine.
    
    Usage:
        engine = TemporalDecayEngine()
        decayed = engine.apply(sentiment, timestamp, category)
        aggregated = engine.aggregate(news_items)
    """
    
    def __init__(
        self,
        custom_half_lives: Optional[Dict[str, float]] = None
    ):
        """
        Initialize engine.
        
        Parameters
        ----------
        custom_half_lives : dict, optional
            Custom half-life configuration
        """
        self.half_lives = HALF_LIFE_CONFIG.copy()
        
        if custom_half_lives:
            self.half_lives.update(custom_half_lives)
    
    def apply(
        self,
        sentiment: float,
        timestamp: datetime,
        category: str = 'general',
        reference_time: Optional[datetime] = None
    ) -> DecayedSentiment:
        """Apply decay to single sentiment."""
        return apply_temporal_decay(
            sentiment, timestamp, category, reference_time
        )
    
    def aggregate(
        self,
        news_items: List[Dict],
        reference_time: Optional[datetime] = None
    ) -> float:
        """Aggregate with decay."""
        return aggregate_decayed_sentiment(news_items, reference_time)
    
    def to_dataframe(
        self,
        df: pd.DataFrame,
        sentiment_col: str = 'sentiment',
        timestamp_col: str = 'timestamp',
        category_col: str = 'category'
    ) -> pd.DataFrame:
        """Apply decay to DataFrame."""
        return batch_apply_decay(
            df, sentiment_col, timestamp_col, category_col
        )
    
    def get_half_life(self, category: str) -> float:
        """Get half-life for category."""
        return self.half_lives.get(category, self.half_lives['default'])
    
    def remaining_signal(
        self,
        category: str,
        age_hours: float
    ) -> float:
        """Calculate remaining signal strength."""
        half_life = self.get_half_life(category)
        return calculate_decay_factor(age_hours, half_life)


# Convenience functions
def decay_sentiment(
    sentiment: float,
    age_hours: float,
    category: str = 'general'
) -> float:
    """Quick decay calculation."""
    half_life = get_half_life(category)
    decay = calculate_decay_factor(age_hours, half_life)
    return sentiment * decay


def effective_sentiment(news_items: List[Dict]) -> float:
    """Quick aggregated sentiment."""
    return aggregate_decayed_sentiment(news_items)

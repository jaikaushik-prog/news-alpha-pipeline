"""
Alpha Construction - Layer 9

Cross-sectional alpha ranking and portfolio construction.

Alpha Formula:
    Alpha = Rank(Surprise × Adjusted_Sentiment × Liquidity_Filter)

Portfolio Logic:
- Long top-ranked sectors
- Short / underweight bottom-ranked
- Sector-neutral construction
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AlphaSignal:
    """Represents an alpha signal for a sector."""
    sector: str
    raw_alpha: float          # Raw alpha score
    rank: int                 # Cross-sectional rank
    percentile: float         # Percentile rank (0-100)
    position: str             # 'long', 'short', 'neutral'
    weight: float             # Portfolio weight
    confidence: float         # Signal confidence


def calculate_raw_alpha(
    sentiment: float,
    surprise: float,
    liquidity_score: float = 1.0,
    regime_weight: float = 1.0
) -> float:
    """
    Calculate raw alpha score.
    
    Formula: Alpha = Sentiment × Surprise × Liquidity × Regime
    
    Parameters
    ----------
    sentiment : float
        Sentiment score (typically -1 to 1)
    surprise : float
        Surprise/novelty score (0 to 2)
    liquidity_score : float
        Liquidity filter (0 to 1)
    regime_weight : float
        Regime adjustment (0.5 to 1.5)
        
    Returns
    -------
    float
        Raw alpha score
    """
    return sentiment * surprise * liquidity_score * regime_weight


def rank_signals(
    alpha_scores: Dict[str, float]
) -> List[Tuple[str, float, int]]:
    """
    Rank alpha scores cross-sectionally.
    
    Parameters
    ----------
    alpha_scores : dict
        {sector: raw_alpha}
        
    Returns
    -------
    list
        [(sector, score, rank), ...] sorted by score descending
    """
    sorted_scores = sorted(
        alpha_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [
        (sector, score, rank + 1)
        for rank, (sector, score) in enumerate(sorted_scores)
    ]


def calculate_position(
    rank: int,
    total: int,
    long_threshold: float = 0.25,
    short_threshold: float = 0.75
) -> str:
    """
    Determine position based on rank.
    
    Parameters
    ----------
    rank : int
        Sector rank (1 = best)
    total : int
        Total number of sectors
    long_threshold : float
        Top percentile for long
    short_threshold : float
        Bottom percentile for short
        
    Returns
    -------
    str
        'long', 'short', or 'neutral'
    """
    percentile = rank / total
    
    if percentile <= long_threshold:
        return 'long'
    elif percentile >= short_threshold:
        return 'short'
    else:
        return 'neutral'


def calculate_portfolio_weights(
    signals: List[AlphaSignal],
    method: str = 'equal',
    max_weight: float = 0.2
) -> List[AlphaSignal]:
    """
    Calculate portfolio weights from signals.
    
    Parameters
    ----------
    signals : list
        List of AlphaSignal objects
    method : str
        'equal', 'rank', or 'score'
    max_weight : float
        Maximum weight per sector
        
    Returns
    -------
    list
        Signals with weights assigned
    """
    long_signals = [s for s in signals if s.position == 'long']
    short_signals = [s for s in signals if s.position == 'short']
    
    if method == 'equal':
        # Equal weight within long/short
        long_weight = 0.5 / len(long_signals) if long_signals else 0
        short_weight = -0.5 / len(short_signals) if short_signals else 0
        
        for s in long_signals:
            s.weight = min(long_weight, max_weight)
        for s in short_signals:
            s.weight = max(short_weight, -max_weight)
            
    elif method == 'rank':
        # Rank-weighted
        if long_signals:
            total_rank = sum(1 / s.rank for s in long_signals)
            for s in long_signals:
                weight = (1 / s.rank) / total_rank * 0.5
                s.weight = min(weight, max_weight)
                
        if short_signals:
            total_rank = sum(1 / s.rank for s in short_signals)
            for s in short_signals:
                weight = (1 / s.rank) / total_rank * 0.5
                s.weight = max(-weight, -max_weight)
                
    elif method == 'score':
        # Score-weighted
        if long_signals:
            total_score = sum(s.raw_alpha for s in long_signals)
            for s in long_signals:
                weight = s.raw_alpha / total_score * 0.5 if total_score else 0
                s.weight = min(weight, max_weight)
                
        if short_signals:
            total_score = sum(abs(s.raw_alpha) for s in short_signals)
            for s in short_signals:
                weight = abs(s.raw_alpha) / total_score * 0.5 if total_score else 0
                s.weight = max(-weight, -max_weight)
    
    return signals


def generate_alpha_signals(
    sector_data: Dict[str, Dict],
    regime_weight: float = 1.0,
    long_threshold: float = 0.25,
    short_threshold: float = 0.75
) -> List[AlphaSignal]:
    """
    Generate alpha signals from sector data.
    
    Parameters
    ----------
    sector_data : dict
        {sector: {sentiment, surprise, liquidity}}
    regime_weight : float
        Regime adjustment factor
    long_threshold : float
        Top percentile for long
    short_threshold : float
        Bottom percentile for short
        
    Returns
    -------
    list
        List of AlphaSignal objects
    """
    # Calculate raw alphas
    raw_alphas = {}
    for sector, data in sector_data.items():
        raw_alphas[sector] = calculate_raw_alpha(
            sentiment=data.get('sentiment', 0),
            surprise=data.get('surprise', 1),
            liquidity_score=data.get('liquidity', 1),
            regime_weight=regime_weight
        )
    
    # Rank signals
    ranked = rank_signals(raw_alphas)
    total = len(ranked)
    
    # Generate signals
    signals = []
    for sector, score, rank in ranked:
        position = calculate_position(rank, total, long_threshold, short_threshold)
        percentile = (rank / total) * 100
        
        # Confidence based on absolute alpha magnitude
        confidence = min(1.0, abs(score) / 0.5)
        
        signal = AlphaSignal(
            sector=sector,
            raw_alpha=score,
            rank=rank,
            percentile=percentile,
            position=position,
            weight=0.0,  # Will be set by portfolio construction
            confidence=confidence
        )
        signals.append(signal)
    
    # Calculate weights
    signals = calculate_portfolio_weights(signals)
    
    return signals


def signals_to_dataframe(signals: List[AlphaSignal]) -> pd.DataFrame:
    """Convert signals to DataFrame."""
    data = [
        {
            'sector': s.sector,
            'raw_alpha': s.raw_alpha,
            'rank': s.rank,
            'percentile': s.percentile,
            'position': s.position,
            'weight': s.weight,
            'confidence': s.confidence
        }
        for s in signals
    ]
    
    return pd.DataFrame(data)


class AlphaConstructor:
    """
    Complete alpha construction engine.
    
    Usage:
        constructor = AlphaConstructor()
        signals = constructor.generate(sector_data)
        portfolio = constructor.to_portfolio(signals)
    """
    
    def __init__(
        self,
        long_threshold: float = 0.25,
        short_threshold: float = 0.75,
        max_weight: float = 0.2,
        weight_method: str = 'equal'
    ):
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.max_weight = max_weight
        self.weight_method = weight_method
    
    def generate(
        self,
        sector_data: Dict[str, Dict],
        regime_weight: float = 1.0
    ) -> List[AlphaSignal]:
        """Generate alpha signals."""
        signals = generate_alpha_signals(
            sector_data,
            regime_weight,
            self.long_threshold,
            self.short_threshold
        )
        
        return calculate_portfolio_weights(
            signals,
            method=self.weight_method,
            max_weight=self.max_weight
        )
    
    def to_dataframe(self, signals: List[AlphaSignal]) -> pd.DataFrame:
        """Convert to DataFrame."""
        return signals_to_dataframe(signals)
    
    def get_longs(self, signals: List[AlphaSignal]) -> List[str]:
        """Get long sectors."""
        return [s.sector for s in signals if s.position == 'long']
    
    def get_shorts(self, signals: List[AlphaSignal]) -> List[str]:
        """Get short sectors."""
        return [s.sector for s in signals if s.position == 'short']
    
    def summary(self, signals: List[AlphaSignal]) -> Dict:
        """Generate summary statistics."""
        longs = [s for s in signals if s.position == 'long']
        shorts = [s for s in signals if s.position == 'short']
        
        return {
            'total_sectors': len(signals),
            'long_count': len(longs),
            'short_count': len(shorts),
            'neutral_count': len(signals) - len(longs) - len(shorts),
            'long_sectors': [s.sector for s in longs],
            'short_sectors': [s.sector for s in shorts],
            'avg_long_alpha': np.mean([s.raw_alpha for s in longs]) if longs else 0,
            'avg_short_alpha': np.mean([s.raw_alpha for s in shorts]) if shorts else 0,
            'top_signal': signals[0].sector if signals else None,
            'bottom_signal': signals[-1].sector if signals else None
        }


# Convenience function
def get_alpha_portfolio(
    sector_sentiment: Dict[str, float],
    sector_surprise: Optional[Dict[str, float]] = None,
    regime_weight: float = 1.0
) -> pd.DataFrame:
    """
    Quick alpha portfolio generation.
    
    Parameters
    ----------
    sector_sentiment : dict
        {sector: sentiment_score}
    sector_surprise : dict, optional
        {sector: surprise_score}
    regime_weight : float
        Regime adjustment
        
    Returns
    -------
    pd.DataFrame
        Portfolio weights and signals
    """
    if sector_surprise is None:
        sector_surprise = {s: 1.0 for s in sector_sentiment}
    
    sector_data = {
        sector: {
            'sentiment': sector_sentiment.get(sector, 0),
            'surprise': sector_surprise.get(sector, 1.0),
            'liquidity': 1.0
        }
        for sector in sector_sentiment
    }
    
    constructor = AlphaConstructor()
    signals = constructor.generate(sector_data, regime_weight)
    
    return constructor.to_dataframe(signals)

"""
Expectation Gap Modeling ⭐⭐⭐⭐⭐ (ELITE)

Markets move on DIFFERENCE vs EXPECTATION, not raw sentiment.

Formula:
    Expectation_Gap = Actual_Sentiment − Expected_Sentiment

This is how institutional investors actually think:
- Model expectation baseline explicitly
- Compare today's news vs rolling baseline
- Trade the GAP, not the level

"This alone puts you in top 5%"
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import deque

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExpectationGap:
    """Represents the gap between actual and expected sentiment."""
    sector: str
    actual_sentiment: float
    expected_sentiment: float
    gap: float                      # Actual - Expected
    gap_zscore: float              # Standardized gap
    interpretation: str            # 'positive_surprise', 'negative_surprise', 'in_line'
    trade_signal: str              # 'strong_long', 'long', 'neutral', 'short', 'strong_short'


class ExpectationModel:
    """
    Expectation Gap Model - Institutional Grade.
    
    Mirrors earnings surprise logic applied to news sentiment.
    
    Usage:
        model = ExpectationModel()
        
        # Feed historical data to build baseline
        for day_sentiment in historical_data:
            model.update(sector, sentiment)
        
        # Calculate gap for today
        gap = model.calculate_gap(sector, today_sentiment)
    """
    
    def __init__(
        self,
        lookback_days: int = 20,
        decay_alpha: float = 0.1,     # EMA decay factor
        surprise_threshold: float = 1.5  # Z-score threshold for surprise
    ):
        """
        Initialize expectation model.
        
        Parameters
        ----------
        lookback_days : int
            Rolling window for baseline
        decay_alpha : float
            EMA decay (higher = more weight to recent)
        surprise_threshold : float
            Z-score threshold to classify as surprise
        """
        self.lookback = lookback_days
        self.alpha = decay_alpha
        self.threshold = surprise_threshold
        
        # Store rolling data per sector
        self._history: Dict[str, deque] = {}
        self._ema: Dict[str, float] = {}
        self._std: Dict[str, float] = {}
    
    def update(self, sector: str, sentiment: float):
        """
        Update expectation baseline with new observation.
        
        Parameters
        ----------
        sector : str
            Sector name
        sentiment : float
            Observed sentiment score
        """
        # Initialize if needed
        if sector not in self._history:
            self._history[sector] = deque(maxlen=self.lookback)
            self._ema[sector] = sentiment
            self._std[sector] = 0.1  # Initial std
        
        # Add to history
        self._history[sector].append(sentiment)
        
        # Update EMA (exponential moving average)
        self._ema[sector] = (
            self.alpha * sentiment + 
            (1 - self.alpha) * self._ema[sector]
        )
        
        # Update rolling std
        if len(self._history[sector]) >= 5:
            self._std[sector] = np.std(list(self._history[sector]))
    
    def get_expected(self, sector: str) -> float:
        """Get expected (baseline) sentiment for sector."""
        return self._ema.get(sector, 0.0)
    
    def get_std(self, sector: str) -> float:
        """Get rolling std for sector."""
        return max(self._std.get(sector, 0.1), 0.01)  # Avoid division by zero
    
    def calculate_gap(
        self,
        sector: str,
        actual_sentiment: float
    ) -> ExpectationGap:
        """
        Calculate expectation gap.
        
        Parameters
        ----------
        sector : str
            Sector name
        actual_sentiment : float
            Today's observed sentiment
            
        Returns
        -------
        ExpectationGap
            Complete gap analysis
        """
        expected = self.get_expected(sector)
        std = self.get_std(sector)
        
        # Raw gap
        gap = actual_sentiment - expected
        
        # Standardized gap (z-score)
        gap_zscore = gap / std
        
        # Classify
        if gap_zscore > self.threshold:
            interpretation = 'positive_surprise'
            trade_signal = 'strong_long' if gap_zscore > 2 * self.threshold else 'long'
        elif gap_zscore < -self.threshold:
            interpretation = 'negative_surprise'
            trade_signal = 'strong_short' if gap_zscore < -2 * self.threshold else 'short'
        else:
            interpretation = 'in_line'
            trade_signal = 'neutral'
        
        return ExpectationGap(
            sector=sector,
            actual_sentiment=actual_sentiment,
            expected_sentiment=expected,
            gap=gap,
            gap_zscore=gap_zscore,
            interpretation=interpretation,
            trade_signal=trade_signal
        )
    
    def batch_calculate_gaps(
        self,
        sector_sentiment: Dict[str, float]
    ) -> List[ExpectationGap]:
        """Calculate gaps for multiple sectors."""
        gaps = []
        for sector, sentiment in sector_sentiment.items():
            gap = self.calculate_gap(sector, sentiment)
            gaps.append(gap)
        return sorted(gaps, key=lambda x: abs(x.gap_zscore), reverse=True)
    
    def get_surprises(
        self,
        sector_sentiment: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """
        Get positive and negative surprise sectors.
        
        Returns
        -------
        tuple
            (positive_surprises, negative_surprises)
        """
        gaps = self.batch_calculate_gaps(sector_sentiment)
        
        positive = [g.sector for g in gaps if g.interpretation == 'positive_surprise']
        negative = [g.sector for g in gaps if g.interpretation == 'negative_surprise']
        
        return positive, negative


def train_expectation_model(
    historical_sentiment: pd.DataFrame,
    date_col: str = 'date',
    sector_col: str = 'sector',
    sentiment_col: str = 'sentiment'
) -> ExpectationModel:
    """
    Train expectation model on historical data.
    
    Parameters
    ----------
    historical_sentiment : pd.DataFrame
        Historical sector sentiment data
    date_col : str
        Date column name
    sector_col : str
        Sector column name
    sentiment_col : str
        Sentiment score column name
        
    Returns
    -------
    ExpectationModel
        Trained model
    """
    model = ExpectationModel()
    
    # Sort by date
    df = historical_sentiment.sort_values(date_col)
    
    # Train on each day
    for _, row in df.iterrows():
        model.update(row[sector_col], row[sentiment_col])
    
    logger.info(f"Trained expectation model on {len(df)} observations")
    
    return model


def gaps_to_dataframe(gaps: List[ExpectationGap]) -> pd.DataFrame:
    """Convert gaps to DataFrame."""
    return pd.DataFrame([
        {
            'sector': g.sector,
            'actual': g.actual_sentiment,
            'expected': g.expected_sentiment,
            'gap': g.gap,
            'gap_zscore': g.gap_zscore,
            'interpretation': g.interpretation,
            'signal': g.trade_signal
        }
        for g in gaps
    ])


# Convenience function for quick analysis
def calculate_expectation_gaps(
    current_sentiment: Dict[str, float],
    historical_sentiment: Optional[Dict[str, List[float]]] = None,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Quick expectation gap calculation.
    
    Parameters
    ----------
    current_sentiment : dict
        {sector: current_score}
    historical_sentiment : dict, optional
        {sector: [historical_scores]}
    lookback : int
        Lookback window
        
    Returns
    -------
    pd.DataFrame
        Gap analysis
    """
    model = ExpectationModel(lookback_days=lookback)
    
    # Train on historical if provided
    if historical_sentiment:
        for sector, history in historical_sentiment.items():
            for score in history:
                model.update(sector, score)
    else:
        # Use current as baseline (gaps will be ~0)
        for sector, score in current_sentiment.items():
            for _ in range(lookback):
                model.update(sector, score + np.random.normal(0, 0.05))
    
    # Calculate gaps
    gaps = model.batch_calculate_gaps(current_sentiment)
    
    return gaps_to_dataframe(gaps)

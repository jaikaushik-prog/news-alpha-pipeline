"""
Sentiment–Price Divergence Alpha ⭐⭐⭐⭐⭐ (ELITE)

When markets DISAGREE with news → future move.

Definition:
    Divergence = Sign(Sentiment) ≠ Sign(Price Move)

Cases:
- Positive sentiment + price down → absorption (smart money selling into news)
- Negative sentiment + price up → hidden strength (smart money accumulating)

Trade Logic:
    DON'T trade sentiment
    TRADE market DISAGREEMENT with sentiment

"This is how smart money thinks - you're detecting positioning, not opinions"
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


class DivergenceType(Enum):
    """Types of sentiment-price divergence."""
    ABSORPTION = "absorption"              # Positive sentiment, negative price
    HIDDEN_STRENGTH = "hidden_strength"    # Negative sentiment, positive price
    CONFIRMATION = "confirmation"          # Signs match
    NEUTRAL = "neutral"                    # Insufficient data


@dataclass
class DivergenceSignal:
    """Represents a sentiment-price divergence signal."""
    sector: str
    sentiment: float                  # News sentiment
    price_return: float              # Price change (%)
    divergence_type: DivergenceType
    divergence_strength: float       # How strong is the disagreement
    smart_money_signal: str          # 'accumulating', 'distributing', 'neutral'
    trade_action: str                # 'long', 'short', 'hold'
    confidence: float                # Signal confidence


class DivergenceDetector:
    """
    Detects sentiment-price divergence.
    
    This is how institutional traders think:
    - Markets absorbing good news (not rallying) = distribution
    - Markets shrugging off bad news = accumulation
    
    Usage:
        detector = DivergenceDetector()
        
        signal = detector.analyze(
            sector='banking',
            sentiment=0.6,      # Bullish news
            price_return=-1.2   # But price fell
        )
        # Returns: ABSORPTION signal
    """
    
    def __init__(
        self,
        sentiment_threshold: float = 0.15,  # Min sentiment to consider
        price_threshold: float = 0.5,       # Min % move to consider
        divergence_threshold: float = 0.8   # Strength threshold
    ):
        """
        Initialize detector.
        
        Parameters
        ----------
        sentiment_threshold : float
            Minimum absolute sentiment to consider
        price_threshold : float
            Minimum price move (%) to consider
        divergence_threshold : float
            Minimum divergence strength for signal
        """
        self.sentiment_threshold = sentiment_threshold
        self.price_threshold = price_threshold
        self.divergence_threshold = divergence_threshold
        
        # Track historical divergences
        self._history: List[DivergenceSignal] = []
    
    def analyze(
        self,
        sector: str,
        sentiment: float,
        price_return: float,
        volume_ratio: float = 1.0   # Today's volume / avg volume
    ) -> DivergenceSignal:
        """
        Analyze sentiment-price divergence.
        
        Parameters
        ----------
        sector : str
            Sector name
        sentiment : float
            Aggregated sentiment score (-1 to 1)
        price_return : float
            Price return (%)
        volume_ratio : float
            Volume vs average (> 1 = high volume)
            
        Returns
        -------
        DivergenceSignal
            Complete divergence analysis
        """
        # Signs
        sentiment_sign = 1 if sentiment > self.sentiment_threshold else (
            -1 if sentiment < -self.sentiment_threshold else 0
        )
        price_sign = 1 if price_return > self.price_threshold else (
            -1 if price_return < -self.price_threshold else 0
        )
        
        # Determine divergence type
        if sentiment_sign == 0 or price_sign == 0:
            div_type = DivergenceType.NEUTRAL
        elif sentiment_sign != price_sign:
            # Divergence detected
            if sentiment_sign > 0 and price_sign < 0:
                div_type = DivergenceType.ABSORPTION
            else:
                div_type = DivergenceType.HIDDEN_STRENGTH
        else:
            div_type = DivergenceType.CONFIRMATION
        
        # Calculate divergence strength
        strength = self._calculate_strength(sentiment, price_return, volume_ratio)
        
        # Determine smart money signal and trade action
        smart_money, action = self._interpret(div_type, strength)
        
        # Confidence
        confidence = min(1.0, strength / self.divergence_threshold) if strength > 0 else 0
        
        signal = DivergenceSignal(
            sector=sector,
            sentiment=sentiment,
            price_return=price_return,
            divergence_type=div_type,
            divergence_strength=strength,
            smart_money_signal=smart_money,
            trade_action=action,
            confidence=confidence
        )
        
        self._history.append(signal)
        
        return signal
    
    def _calculate_strength(
        self,
        sentiment: float,
        price_return: float,
        volume_ratio: float
    ) -> float:
        """
        Calculate divergence strength.
        
        Strength = |sentiment| × |price_return| × volume_factor
        
        Higher when:
        - Strong sentiment
        - Large price move
        - High volume
        """
        # Base strength
        base = abs(sentiment) * abs(price_return) / 100
        
        # Volume amplifier
        volume_factor = 1 + 0.5 * (volume_ratio - 1)
        volume_factor = max(0.5, min(2.0, volume_factor))
        
        return base * volume_factor
    
    def _interpret(
        self,
        div_type: DivergenceType,
        strength: float
    ) -> Tuple[str, str]:
        """
        Interpret divergence for trading.
        
        Returns
        -------
        tuple
            (smart_money_signal, trade_action)
        """
        if div_type == DivergenceType.ABSORPTION:
            # Good news, price fell
            # Smart money is selling into strength
            if strength > self.divergence_threshold:
                return 'distributing', 'short'
            else:
                return 'distributing', 'hold'
                
        elif div_type == DivergenceType.HIDDEN_STRENGTH:
            # Bad news, price rose
            # Smart money is accumulating on weakness
            if strength > self.divergence_threshold:
                return 'accumulating', 'long'
            else:
                return 'accumulating', 'hold'
                
        elif div_type == DivergenceType.CONFIRMATION:
            # Sentiment and price aligned
            return 'neutral', 'hold'
            
        else:
            return 'neutral', 'hold'
    
    def batch_analyze(
        self,
        sector_data: Dict[str, Dict]
    ) -> List[DivergenceSignal]:
        """
        Analyze multiple sectors.
        
        Parameters
        ----------
        sector_data : dict
            {sector: {sentiment, price_return, volume_ratio}}
            
        Returns
        -------
        list
            List of DivergenceSignal objects
        """
        signals = []
        for sector, data in sector_data.items():
            signal = self.analyze(
                sector=sector,
                sentiment=data.get('sentiment', 0),
                price_return=data.get('price_return', 0),
                volume_ratio=data.get('volume_ratio', 1.0)
            )
            signals.append(signal)
        
        # Sort by strength
        return sorted(signals, key=lambda x: x.divergence_strength, reverse=True)
    
    def get_actionable_signals(
        self,
        signals: List[DivergenceSignal]
    ) -> Dict[str, List[str]]:
        """
        Get actionable trading signals.
        
        Returns
        -------
        dict
            {action: [sectors]}
        """
        result = {'long': [], 'short': [], 'hold': []}
        
        for signal in signals:
            if signal.confidence >= 0.5:  # Only high confidence
                result[signal.trade_action].append(signal.sector)
        
        return result
    
    def get_divergence_summary(self) -> Dict:
        """Get summary of recent divergences."""
        if not self._history:
            return {}
        
        absorption_count = sum(
            1 for s in self._history 
            if s.divergence_type == DivergenceType.ABSORPTION
        )
        hidden_count = sum(
            1 for s in self._history 
            if s.divergence_type == DivergenceType.HIDDEN_STRENGTH
        )
        
        return {
            'total_analyzed': len(self._history),
            'absorption_signals': absorption_count,
            'hidden_strength_signals': hidden_count,
            'confirmation_signals': len(self._history) - absorption_count - hidden_count,
            'avg_strength': np.mean([s.divergence_strength for s in self._history])
        }


def signals_to_dataframe(signals: List[DivergenceSignal]) -> pd.DataFrame:
    """Convert signals to DataFrame."""
    return pd.DataFrame([
        {
            'sector': s.sector,
            'sentiment': s.sentiment,
            'price_return': s.price_return,
            'divergence_type': s.divergence_type.value,
            'strength': s.divergence_strength,
            'smart_money': s.smart_money_signal,
            'action': s.trade_action,
            'confidence': s.confidence
        }
        for s in signals
    ])


def detect_divergences(
    sentiment_scores: Dict[str, float],
    price_returns: Dict[str, float],
    volume_ratios: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Quick divergence detection.
    
    Parameters
    ----------
    sentiment_scores : dict
        {sector: sentiment}
    price_returns : dict
        {sector: return_%}
    volume_ratios : dict, optional
        {sector: volume_ratio}
        
    Returns
    -------
    pd.DataFrame
        Divergence analysis
    """
    detector = DivergenceDetector()
    
    if volume_ratios is None:
        volume_ratios = {s: 1.0 for s in sentiment_scores}
    
    sector_data = {
        sector: {
            'sentiment': sentiment_scores.get(sector, 0),
            'price_return': price_returns.get(sector, 0),
            'volume_ratio': volume_ratios.get(sector, 1.0)
        }
        for sector in sentiment_scores
    }
    
    signals = detector.batch_analyze(sector_data)
    
    return signals_to_dataframe(signals)


# Demo function
def demo_divergence_detection():
    """Demo divergence detection."""
    detector = DivergenceDetector()
    
    # Case 1: ABSORPTION - Good news, price down
    signal1 = detector.analyze(
        sector='banking',
        sentiment=0.7,       # Very positive news
        price_return=-2.1,   # But price fell 2%
        volume_ratio=1.5     # High volume
    )
    print(f"Banking: {signal1.divergence_type.value} → {signal1.trade_action}")
    print(f"  Smart money: {signal1.smart_money_signal}")
    
    # Case 2: HIDDEN STRENGTH - Bad news, price up
    signal2 = detector.analyze(
        sector='pharma',
        sentiment=-0.5,      # Negative news
        price_return=1.8,    # But price rose
        volume_ratio=1.3
    )
    print(f"Pharma: {signal2.divergence_type.value} → {signal2.trade_action}")
    print(f"  Smart money: {signal2.smart_money_signal}")
    
    # Case 3: CONFIRMATION - News and price aligned
    signal3 = detector.analyze(
        sector='it',
        sentiment=0.4,
        price_return=1.2,
        volume_ratio=1.0
    )
    print(f"IT: {signal3.divergence_type.value} → {signal3.trade_action}")
    
    return detector

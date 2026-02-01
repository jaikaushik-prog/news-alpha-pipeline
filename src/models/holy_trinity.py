"""
Holy Trinity Integration ⭐⭐⭐⭐⭐

Combines the three elite differentiators:
1. Expectation Gap
2. Narrative Velocity  
3. Sentiment-Price Divergence

"I modeled expectation-adjusted information shocks and measured 
how markets absorb or reject them under different regimes."
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

from .expectation_gap import ExpectationModel, ExpectationGap
from .narrative_velocity import NarrativeVelocityTracker, NarrativeKinematics
from .divergence import DivergenceDetector, DivergenceSignal, DivergenceType
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HolyTrinitySignal:
    """Combined signal from all three differentiators."""
    sector: str
    
    # Component 1: Expectation Gap
    expectation_gap: float
    gap_zscore: float
    is_surprise: bool
    
    # Component 2: Narrative Velocity
    velocity: float
    acceleration: float
    momentum_phase: str
    is_early_mover: bool
    
    # Component 3: Divergence
    divergence_type: str
    smart_money_signal: str
    
    # Combined
    trinity_score: float        # Unified score (-1 to 1)
    conviction: str             # 'high', 'medium', 'low'
    trade_recommendation: str   # 'strong_long', 'long', 'hold', 'short', 'strong_short'
    rationale: str             # Human-readable explanation


class HolyTrinityModel:
    """
    The Ultimate Differentiator: Holy Trinity Integration.
    
    Combines:
    1. Expectation Gap (surprise vs baseline)
    2. Narrative Velocity (information diffusion dynamics)
    3. Sentiment-Price Divergence (smart money positioning)
    
    This is institutional-grade novelty.
    
    Usage:
        model = HolyTrinityModel()
        
        # Feed historical data
        model.update_historical(sector_data)
        
        # Get today's signals
        signals = model.analyze(current_data)
    """
    
    def __init__(
        self,
        gap_weight: float = 0.35,
        velocity_weight: float = 0.25,
        divergence_weight: float = 0.40
    ):
        """
        Initialize Holy Trinity model.
        
        Parameters
        ----------
        gap_weight : float
            Weight for expectation gap component
        velocity_weight : float
            Weight for narrative velocity component
        divergence_weight : float
            Weight for divergence component
        """
        self.weights = {
            'gap': gap_weight,
            'velocity': velocity_weight,
            'divergence': divergence_weight
        }
        
        # Initialize components
        self.expectation_model = ExpectationModel()
        self.velocity_tracker = NarrativeVelocityTracker()
        self.divergence_detector = DivergenceDetector()
    
    def update_historical(
        self,
        sector: str,
        sentiment: float,
        volume: float = 1.0,
        intensity: float = 0.5
    ):
        """
        Feed historical data to train models.
        
        Parameters
        ----------
        sector : str
            Sector name
        sentiment : float
            Historical sentiment
        volume : float
            Article volume
        intensity : float
            Sentiment intensity
        """
        # Update expectation baseline
        self.expectation_model.update(sector, sentiment)
        
        # Update velocity tracker
        self.velocity_tracker.update(sector, {
            'volume': volume,
            'intensity': intensity
        })
    
    def analyze(
        self,
        sector: str,
        current_sentiment: float,
        current_volume: float,
        current_intensity: float,
        price_return: float,
        volume_ratio: float = 1.0
    ) -> HolyTrinitySignal:
        """
        Analyze sector using Holy Trinity.
        
        Parameters
        ----------
        sector : str
            Sector name
        current_sentiment : float
            Today's aggregated sentiment
        current_volume : float
            Today's article count
        current_intensity : float
            Today's avg intensity
        price_return : float
            Today's price return (%)
        volume_ratio : float
            Trading volume vs average
            
        Returns
        -------
        HolyTrinitySignal
            Complete trinity analysis
        """
        # Component 1: Expectation Gap
        gap = self.expectation_model.calculate_gap(sector, current_sentiment)
        
        # Component 2: Narrative Velocity
        self.velocity_tracker.update(sector, {
            'volume': current_volume,
            'intensity': current_intensity
        })
        kinematics = self.velocity_tracker.get_kinematics(sector)
        
        # Component 3: Divergence
        divergence = self.divergence_detector.analyze(
            sector=sector,
            sentiment=current_sentiment,
            price_return=price_return,
            volume_ratio=volume_ratio
        )
        
        # Calculate unified score
        trinity_score = self._calculate_trinity_score(gap, kinematics, divergence)
        
        # Determine conviction and recommendation
        conviction = self._assess_conviction(gap, kinematics, divergence)
        recommendation = self._generate_recommendation(trinity_score, conviction)
        rationale = self._generate_rationale(gap, kinematics, divergence)
        
        return HolyTrinitySignal(
            sector=sector,
            # Gap
            expectation_gap=gap.gap,
            gap_zscore=gap.gap_zscore,
            is_surprise=gap.interpretation != 'in_line',
            # Velocity
            velocity=kinematics.velocity,
            acceleration=kinematics.acceleration,
            momentum_phase=kinematics.momentum_phase,
            is_early_mover=kinematics.trade_timing == 'early',
            # Divergence
            divergence_type=divergence.divergence_type.value,
            smart_money_signal=divergence.smart_money_signal,
            # Combined
            trinity_score=trinity_score,
            conviction=conviction,
            trade_recommendation=recommendation,
            rationale=rationale
        )
    
    def _calculate_trinity_score(
        self,
        gap: ExpectationGap,
        kinematics: NarrativeKinematics,
        divergence: DivergenceSignal
    ) -> float:
        """
        Calculate unified trinity score.
        
        Score considers:
        - Direction from gap z-score
        - Timing from velocity (early = amplify)
        - Confirmation/Contradiction from divergence
        """
        # Gap component (directional)
        gap_score = np.clip(gap.gap_zscore / 3, -1, 1)
        
        # Velocity component (timing multiplier)
        if kinematics.trade_timing == 'early':
            velocity_mult = 1.3
        elif kinematics.trade_timing == 'late':
            velocity_mult = 0.7
        elif kinematics.momentum_phase == 'priced_in':
            velocity_mult = 0.5
        else:
            velocity_mult = 1.0
        
        # Divergence component
        if divergence.divergence_type == DivergenceType.HIDDEN_STRENGTH:
            div_score = 0.5  # Bullish
        elif divergence.divergence_type == DivergenceType.ABSORPTION:
            div_score = -0.5  # Bearish
        else:
            div_score = 0.0
        
        # Combine
        score = (
            self.weights['gap'] * gap_score +
            self.weights['velocity'] * gap_score * velocity_mult +
            self.weights['divergence'] * div_score
        )
        
        return float(np.clip(score, -1, 1))
    
    def _assess_conviction(
        self,
        gap: ExpectationGap,
        kinematics: NarrativeKinematics,
        divergence: DivergenceSignal
    ) -> str:
        """Assess signal conviction based on agreement."""
        bullish_count = 0
        bearish_count = 0
        
        # Gap
        if gap.gap_zscore > 1:
            bullish_count += 1
        elif gap.gap_zscore < -1:
            bearish_count += 1
        
        # Velocity (early mover)
        if kinematics.trade_timing == 'early':
            if kinematics.velocity > 0:
                bullish_count += 1
            else:
                bearish_count += 1
        
        # Divergence
        if divergence.divergence_type == DivergenceType.HIDDEN_STRENGTH:
            bullish_count += 1
        elif divergence.divergence_type == DivergenceType.ABSORPTION:
            bearish_count += 1
        
        # Agreement = conviction
        if bullish_count >= 2 or bearish_count >= 2:
            return 'high'
        elif bullish_count >= 1 or bearish_count >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendation(self, score: float, conviction: str) -> str:
        """Generate trade recommendation."""
        if conviction == 'high':
            if score > 0.3:
                return 'strong_long'
            elif score < -0.3:
                return 'strong_short'
        
        if score > 0.2:
            return 'long'
        elif score < -0.2:
            return 'short'
        else:
            return 'hold'
    
    def _generate_rationale(
        self,
        gap: ExpectationGap,
        kinematics: NarrativeKinematics,
        divergence: DivergenceSignal
    ) -> str:
        """Generate human-readable rationale."""
        parts = []
        
        # Gap
        if gap.gap_zscore > 1.5:
            parts.append(f"Positive surprise (z={gap.gap_zscore:.1f})")
        elif gap.gap_zscore < -1.5:
            parts.append(f"Negative surprise (z={gap.gap_zscore:.1f})")
        else:
            parts.append("News in-line with expectations")
        
        # Velocity
        if kinematics.trade_timing == 'early':
            parts.append("early mover opportunity")
        elif kinematics.momentum_phase == 'priced_in':
            parts.append("narrative priced in")
        
        # Divergence
        if divergence.divergence_type == DivergenceType.HIDDEN_STRENGTH:
            parts.append("smart money accumulating")
        elif divergence.divergence_type == DivergenceType.ABSORPTION:
            parts.append("smart money distributing")
        
        return "; ".join(parts)
    
    def batch_analyze(
        self,
        sector_data: Dict[str, Dict]
    ) -> List[HolyTrinitySignal]:
        """
        Analyze multiple sectors.
        
        Parameters
        ----------
        sector_data : dict
            {sector: {sentiment, volume, intensity, price_return, volume_ratio}}
        """
        signals = []
        for sector, data in sector_data.items():
            signal = self.analyze(
                sector=sector,
                current_sentiment=data.get('sentiment', 0),
                current_volume=data.get('volume', 1),
                current_intensity=data.get('intensity', 0.5),
                price_return=data.get('price_return', 0),
                volume_ratio=data.get('volume_ratio', 1.0)
            )
            signals.append(signal)
        
        return sorted(signals, key=lambda x: abs(x.trinity_score), reverse=True)


def trinity_to_dataframe(signals: List[HolyTrinitySignal]) -> pd.DataFrame:
    """Convert signals to DataFrame."""
    return pd.DataFrame([
        {
            'sector': s.sector,
            'trinity_score': s.trinity_score,
            'conviction': s.conviction,
            'recommendation': s.trade_recommendation,
            'gap_zscore': s.gap_zscore,
            'is_surprise': s.is_surprise,
            'velocity': s.velocity,
            'is_early': s.is_early_mover,
            'divergence': s.divergence_type,
            'smart_money': s.smart_money_signal,
            'rationale': s.rationale
        }
        for s in signals
    ])


# THE presentation statement
def get_presentation_statement() -> str:
    """The statement that changes perception."""
    return (
        "I modeled expectation-adjusted information shocks and measured "
        "how markets absorb or reject them under different regimes."
    )

"""
Pre-Budget Confidence Score Module.

Combines all sentiment signals into a single tradable signal:
    Confidence = z(PCR_change) - z(IV_spike) + z(FII_flow_intensity)

This interaction term with Budget_Sentiment creates powerful alpha.

Architecture:
┌────────────────────────────────────────────┐
│            PRE-BUDGET SENTIMENT             │
├────────────────────────────────────────────┤
│ 1. Sector Option Metrics (PCR, IV, OI)      │
│ 2. FII/DII Flow Intensity                   │
│ 3. News Headline Sentiment (FinBERT)        │
│ 4. Google Trends Attention Index            │
└────────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────┐
│         PRE-BUDGET CONFIDENCE SCORE         │
├────────────────────────────────────────────┤
│ Composite signal for trading decisions      │
└────────────────────────────────────────────┘
"""

from typing import Dict, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)


def z_score(value: float, mean: float, std: float) -> float:
    """Calculate z-score."""
    if std <= 0:
        return 0.0
    return (value - mean) / std


def calculate_confidence_score(
    pcr_change: float,
    iv_spike: float,
    fii_flow_intensity: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate Pre-Budget Confidence Score.
    
    Formula:
        Confidence = z(PCR_change) - z(IV_spike) + z(FII_flow_intensity)
    
    Interpretation:
        Positive = Bullish (low fear, institutional buying)
        Negative = Bearish (high fear, institutional selling)
    
    Parameters
    ----------
    pcr_change : float
        Change in PCR (1.0 - PCR normalizes to bullish = positive)
    iv_spike : float
        IV change from baseline (positive = fear spike)
    fii_flow_intensity : float
        Signed FII flow intensity (positive = buying)
    weights : dict, optional
        Custom weights for each component
        
    Returns
    -------
    float
        Confidence score (-1 to +1 scale)
    """
    if weights is None:
        weights = {
            'pcr': 0.35,
            'iv': 0.30,
            'flow': 0.35
        }
    
    # Historical norms (for z-scoring)
    # These are approximate values for India VIX / Nifty options
    NORMS = {
        'pcr_mean': 1.0,
        'pcr_std': 0.2,
        'iv_mean': 15.0,
        'iv_std': 5.0,
        'flow_mean': 0.0,
        'flow_std': 0.5
    }
    
    # Calculate z-scores
    pcr_z = z_score(1.0 - pcr_change, 0, NORMS['pcr_std'])  # Invert: low PCR = bullish
    iv_z = z_score(iv_spike, NORMS['iv_mean'], NORMS['iv_std'])  # High IV = bearish
    flow_z = z_score(fii_flow_intensity, NORMS['flow_mean'], NORMS['flow_std'])
    
    # Composite score
    confidence = (
        weights['pcr'] * pcr_z +
        weights['iv'] * (-iv_z) +  # Negate: high IV should reduce confidence
        weights['flow'] * flow_z
    )
    
    # Clip to -1 to +1 range
    confidence = max(-1, min(1, confidence))
    
    return round(confidence, 3)


def get_pre_budget_confidence(
    budget_date: str,
    options_data: Optional[Dict] = None,
    flow_data: Optional[Dict] = None,
    trends_data: Optional[Dict] = None
) -> Dict:
    """
    Calculate comprehensive pre-budget confidence score.
    
    Parameters
    ----------
    budget_date : str
        Budget date in YYYY-MM-DD format
    options_data : dict, optional
        Options sentiment metrics
    flow_data : dict, optional
        FII/DII flow metrics
    trends_data : dict, optional
        Google Trends metrics
        
    Returns
    -------
    dict
        Confidence score and components
    """
    result = {
        'budget_date': budget_date,
        'timestamp': datetime.now().isoformat()
    }
    
    # Default values if data not provided
    pcr_change = options_data.get('pcr_oi', 1.0) if options_data else 1.0
    iv_spike = options_data.get('atm_iv', 15.0) if options_data else 15.0
    fii_intensity = flow_data.get('fii_avg_intensity', 0.0) if flow_data else 0.0
    
    # Direction adjustment for FII
    if flow_data:
        fii_trend = flow_data.get('fii_trend', 'neutral')
        if fii_trend == 'selling':
            fii_intensity = -abs(fii_intensity)
    
    # Calculate confidence
    confidence = calculate_confidence_score(pcr_change, iv_spike, fii_intensity)
    
    result['confidence_score'] = confidence
    result['confidence_regime'] = _classify_confidence(confidence)
    
    # Component breakdown
    result['components'] = {
        'pcr_input': pcr_change,
        'iv_input': iv_spike,
        'flow_input': fii_intensity
    }
    
    # Add trends overlay if available
    if trends_data:
        attention = trends_data.get('attention_score', 0)
        result['attention_overlay'] = attention
        # High attention + High confidence = Strong signal
        result['signal_strength'] = abs(confidence) * (1 + attention)
    
    # Strategy recommendation
    result['strategy_recommendation'] = _generate_strategy(confidence, options_data, flow_data)
    
    return result


def _classify_confidence(score: float) -> str:
    """Classify confidence score into regime."""
    if score > 0.5:
        return 'strong_bullish'
    elif score > 0.2:
        return 'bullish'
    elif score > -0.2:
        return 'neutral'
    elif score > -0.5:
        return 'bearish'
    else:
        return 'strong_bearish'


def _generate_strategy(
    confidence: float,
    options_data: Optional[Dict],
    flow_data: Optional[Dict]
) -> str:
    """Generate strategy recommendation based on signals."""
    regime = _classify_confidence(confidence)
    
    # Get smart money signal if available
    smart_money = flow_data.get('smart_money_signal', 'unclear') if flow_data else 'unclear'
    
    strategies = {
        ('strong_bullish', 'confident_risk_on'): 
            "STRONG BUY. High confidence + FII buying + falling IV. Momentum strategy on positive mentions.",
        ('strong_bullish', 'unclear'):
            "BUY. High confidence but mixed FII signals. Reduced position size recommended.",
        ('bullish', 'confident_risk_on'):
            "BUY. Moderately bullish with institutional support.",
        ('bullish', 'unclear'):
            "LEAN BUY. Watch for confirmation from Budget mentions.",
        ('neutral', 'unclear'):
            "WAIT. Mixed signals. Trade only on strong Budget sentiment.",
        ('bearish', 'defensive_exit'):
            "HEDGE. Consider protective puts. Trade negative mentions short.",
        ('bearish', 'unclear'):
            "LEAN SELL. Reduce exposure. Avoid long momentum.",
        ('strong_bearish', 'defensive_exit'):
            "STRONG SELL/SHORT. High fear + FII selling. Contrarian only on extreme positive surprises.",
        ('strong_bearish', 'unclear'):
            "DEFENSIVE. High cash. Trade only contrarian on positive surprises."
    }
    
    key = (regime, smart_money)
    return strategies.get(key, f"Mixed signals. Regime: {regime}. Wait for clearer direction.")


def interaction_term(
    confidence_score: float,
    budget_sentiment: float
) -> float:
    """
    Calculate interaction term for signal enhancement.
    
    This is the 🔥 upgrade: Budget_Sentiment × Confidence
    
    Interpretation:
        High positive = Strong momentum signal
        High negative = Strong contrarian signal
        Near zero = Weak signal
    
    Parameters
    ----------
    confidence_score : float
        Pre-budget confidence (-1 to +1)
    budget_sentiment : float
        Budget mention sentiment (-1 to +1)
        
    Returns
    -------
    float
        Interaction term
    """
    return confidence_score * budget_sentiment


def get_signal_matrix() -> pd.DataFrame:
    """
    Generate signal interpretation matrix.
    
    Returns
    -------
    pd.DataFrame
        Action matrix for Pre-Confidence × Budget-Sentiment
    """
    data = [
        {'pre_confidence': 'Bullish', 'mention_sentiment': 'Positive', 'action': 'STRONG BUY', 'rationale': 'Aligned signals'},
        {'pre_confidence': 'Bullish', 'mention_sentiment': 'Negative', 'action': 'FADE/WAIT', 'rationale': 'Contradiction - wait for clarity'},
        {'pre_confidence': 'Bullish', 'mention_sentiment': 'Neutral', 'action': 'LEAN BUY', 'rationale': 'Pre-sentiment stronger than no news'},
        {'pre_confidence': 'Bearish', 'mention_sentiment': 'Positive', 'action': 'REVERSAL BUY', 'rationale': 'Positive surprise - high upside'},
        {'pre_confidence': 'Bearish', 'mention_sentiment': 'Negative', 'action': 'STRONG SELL', 'rationale': 'Aligned bearish signals'},
        {'pre_confidence': 'Bearish', 'mention_sentiment': 'Neutral', 'action': 'STAY SHORT', 'rationale': 'No catalyst to reverse'},
        {'pre_confidence': 'Neutral', 'mention_sentiment': 'Positive', 'action': 'BUY', 'rationale': 'Budget drives direction'},
        {'pre_confidence': 'Neutral', 'mention_sentiment': 'Negative', 'action': 'SELL', 'rationale': 'Budget drives direction'},
        {'pre_confidence': 'Neutral', 'mention_sentiment': 'Neutral', 'action': 'NO TRADE', 'rationale': 'No edge'}
    ]
    
    return pd.DataFrame(data)


def create_mock_confidence_score(budget_date: str) -> Dict:
    """
    Create mock confidence score for demonstration.
    
    Parameters
    ----------
    budget_date : str
        Budget date
        
    Returns
    -------
    dict
        Mock confidence data
    """
    np.random.seed(42)
    
    confidence = np.random.uniform(-0.5, 0.5)
    
    return {
        'budget_date': budget_date,
        'confidence_score': round(confidence, 3),
        'confidence_regime': _classify_confidence(confidence),
        'components': {
            'pcr_input': round(np.random.uniform(0.8, 1.3), 2),
            'iv_input': round(np.random.uniform(12, 22), 1),
            'flow_input': round(np.random.uniform(-0.5, 0.5), 3)
        },
        'attention_overlay': round(np.random.uniform(0.2, 0.8), 2),
        'signal_strength': round(np.random.uniform(0.1, 0.8), 2),
        'strategy_recommendation': _generate_strategy(confidence, None, None)
    }

"""
Multi-Factor Sentiment - Layer 4

Decomposes news sentiment into 5 dimensions:
1. Polarity - Positive/Negative
2. Intensity - Strength of language
3. Certainty - Speculative vs decisive
4. Urgency - Immediate vs long-term
5. Risk Tone - Stability vs fragility

Output: S = [polarity, intensity, certainty, urgency, risk] ∈ [-1, 1]^5
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import re

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentVector:
    """5-dimensional sentiment representation."""
    polarity: float      # [-1, 1] negative to positive
    intensity: float     # [0, 1] weak to strong
    certainty: float     # [0, 1] speculative to decisive
    urgency: float       # [0, 1] long-term to immediate
    risk_tone: float     # [-1, 1] fragile to stable
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.polarity,
            self.intensity,
            self.certainty,
            self.urgency,
            self.risk_tone
        ])
    
    def __repr__(self):
        return (f"SentimentVector(polarity={self.polarity:.2f}, "
                f"intensity={self.intensity:.2f}, certainty={self.certainty:.2f}, "
                f"urgency={self.urgency:.2f}, risk_tone={self.risk_tone:.2f})")


# Keyword dictionaries for each factor
POLARITY_KEYWORDS = {
    'positive': [
        'surge', 'rally', 'gain', 'jump', 'soar', 'bullish', 'rise', 'record',
        'beat', 'outperform', 'upgrade', 'profit', 'growth', 'boom', 'strong',
        'optimistic', 'positive', 'recovery', 'strengthen', 'expand'
    ],
    'negative': [
        'fall', 'drop', 'crash', 'plunge', 'bearish', 'decline', 'slump',
        'miss', 'downgrade', 'loss', 'weak', 'concern', 'fear', 'risk',
        'volatile', 'negative', 'recession', 'contraction', 'warning'
    ]
}

INTENSITY_KEYWORDS = {
    'high': [
        'surge', 'plunge', 'crash', 'soar', 'skyrocket', 'collapse',
        'massive', 'huge', 'dramatic', 'extreme', 'sharp', 'severe',
        'unprecedented', 'record-breaking', 'explosive', 'remarkable'
    ],
    'medium': [
        'rise', 'fall', 'gain', 'drop', 'increase', 'decrease',
        'moderate', 'steady', 'gradual', 'notable'
    ],
    'low': [
        'slight', 'marginal', 'minor', 'small', 'modest', 'limited',
        'subdued', 'mild', 'flat', 'unchanged'
    ]
}

CERTAINTY_KEYWORDS = {
    'decisive': [
        'confirms', 'announces', 'declares', 'decides', 'approves',
        'will', 'must', 'definitely', 'certainly', 'guaranteed',
        'official', 'confirmed', 'passed', 'enacted', 'signed'
    ],
    'speculative': [
        'may', 'might', 'could', 'possibly', 'likely', 'expected',
        'rumor', 'speculation', 'sources say', 'reportedly',
        'considering', 'exploring', 'potential', 'if', 'uncertain'
    ]
}

URGENCY_KEYWORDS = {
    'immediate': [
        'today', 'now', 'immediately', 'breaking', 'just',
        'urgent', 'emergency', 'sudden', 'instantly', 'flash',
        'intraday', 'hours', 'minutes'
    ],
    'short_term': [
        'week', 'days', 'soon', 'upcoming', 'near-term',
        'next', 'shortly', 'imminent'
    ],
    'long_term': [
        'year', 'decade', 'long-term', 'future', 'eventual',
        'gradual', 'over time', 'strategic', 'outlook'
    ]
}

RISK_KEYWORDS = {
    'stable': [
        'stable', 'steady', 'resilient', 'strong', 'secure',
        'safe', 'protected', 'robust', 'solid', 'reliable',
        'confidence', 'support', 'backed'
    ],
    'fragile': [
        'volatile', 'unstable', 'risky', 'uncertain', 'fragile',
        'vulnerable', 'exposed', 'threat', 'danger', 'crisis',
        'stress', 'pressure', 'turbulence', 'turmoil'
    ]
}


def _count_keywords(text: str, keywords: List[str]) -> int:
    """Count keyword occurrences in text."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def calculate_polarity(text: str) -> float:
    """
    Calculate polarity score.
    
    Returns
    -------
    float
        Score from -1 (very negative) to +1 (very positive)
    """
    pos_count = _count_keywords(text, POLARITY_KEYWORDS['positive'])
    neg_count = _count_keywords(text, POLARITY_KEYWORDS['negative'])
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    
    return (pos_count - neg_count) / total


def calculate_intensity(text: str) -> float:
    """
    Calculate intensity score.
    
    Returns
    -------
    float
        Score from 0 (weak) to 1 (strong)
    """
    high_count = _count_keywords(text, INTENSITY_KEYWORDS['high'])
    medium_count = _count_keywords(text, INTENSITY_KEYWORDS['medium'])
    low_count = _count_keywords(text, INTENSITY_KEYWORDS['low'])
    
    # Weighted average
    total = high_count + medium_count + low_count
    if total == 0:
        return 0.5  # Neutral intensity
    
    score = (high_count * 1.0 + medium_count * 0.5 + low_count * 0.2) / total
    return min(score, 1.0)


def calculate_certainty(text: str) -> float:
    """
    Calculate certainty score.
    
    Returns
    -------
    float
        Score from 0 (speculative) to 1 (decisive)
    """
    decisive_count = _count_keywords(text, CERTAINTY_KEYWORDS['decisive'])
    speculative_count = _count_keywords(text, CERTAINTY_KEYWORDS['speculative'])
    
    total = decisive_count + speculative_count
    if total == 0:
        return 0.5  # Neutral
    
    return decisive_count / total


def calculate_urgency(text: str) -> float:
    """
    Calculate urgency score.
    
    Returns
    -------
    float
        Score from 0 (long-term) to 1 (immediate)
    """
    immediate_count = _count_keywords(text, URGENCY_KEYWORDS['immediate'])
    short_count = _count_keywords(text, URGENCY_KEYWORDS['short_term'])
    long_count = _count_keywords(text, URGENCY_KEYWORDS['long_term'])
    
    total = immediate_count + short_count + long_count
    if total == 0:
        return 0.5  # Neutral
    
    score = (immediate_count * 1.0 + short_count * 0.6 + long_count * 0.1) / total
    return min(score, 1.0)


def calculate_risk_tone(text: str) -> float:
    """
    Calculate risk tone score.
    
    Returns
    -------
    float
        Score from -1 (fragile/risky) to +1 (stable/safe)
    """
    stable_count = _count_keywords(text, RISK_KEYWORDS['stable'])
    fragile_count = _count_keywords(text, RISK_KEYWORDS['fragile'])
    
    total = stable_count + fragile_count
    if total == 0:
        return 0.0  # Neutral
    
    return (stable_count - fragile_count) / total


def analyze_multifactor_sentiment(text: str) -> SentimentVector:
    """
    Perform multi-factor sentiment analysis.
    
    Parameters
    ----------
    text : str
        Input text (headline or content)
        
    Returns
    -------
    SentimentVector
        5-dimensional sentiment representation
    """
    return SentimentVector(
        polarity=calculate_polarity(text),
        intensity=calculate_intensity(text),
        certainty=calculate_certainty(text),
        urgency=calculate_urgency(text),
        risk_tone=calculate_risk_tone(text)
    )


def batch_analyze_sentiment(texts: List[str]) -> List[SentimentVector]:
    """Analyze multiple texts."""
    return [analyze_multifactor_sentiment(text) for text in texts]


def sentiment_to_dataframe(
    texts: List[str],
    sentiments: List[SentimentVector]
) -> 'pd.DataFrame':
    """Convert sentiments to DataFrame."""
    import pandas as pd
    
    data = []
    for text, sent in zip(texts, sentiments):
        data.append({
            'text': text,
            'polarity': sent.polarity,
            'intensity': sent.intensity,
            'certainty': sent.certainty,
            'urgency': sent.urgency,
            'risk_tone': sent.risk_tone,
            'composite_score': calculate_composite_score(sent)
        })
    
    return pd.DataFrame(data)


def calculate_composite_score(sentiment: SentimentVector) -> float:
    """
    Calculate composite sentiment score.
    
    Weighted combination of factors for a single actionable score.
    
    Returns
    -------
    float
        Composite score from -1 to +1
    """
    weights = {
        'polarity': 0.4,
        'intensity': 0.15,
        'certainty': 0.2,
        'urgency': 0.1,
        'risk_tone': 0.15
    }
    
    # Adjust intensity and certainty to affect magnitude
    base = sentiment.polarity * weights['polarity']
    
    # Intensity amplifies the signal
    intensity_mult = 0.5 + sentiment.intensity * 0.5
    
    # Certainty increases confidence
    certainty_mult = 0.5 + sentiment.certainty * 0.5
    
    # Urgency slightly boosts immediate news
    urgency_boost = sentiment.urgency * weights['urgency']
    
    # Risk tone adds or subtracts
    risk_adj = sentiment.risk_tone * weights['risk_tone']
    
    composite = (base * intensity_mult * certainty_mult) + urgency_boost + risk_adj
    
    return np.clip(composite, -1, 1)


class MultifactorSentimentAnalyzer:
    """
    Complete multi-factor sentiment analyzer.
    
    Usage:
        analyzer = MultifactorSentimentAnalyzer()
        result = analyzer.analyze("Markets surge on positive budget news")
    """
    
    def __init__(self, use_transformers: bool = False):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        use_transformers : bool
            If True, use transformer-based analysis (slower but more accurate)
        """
        self.use_transformers = use_transformers
        self._transformer_model = None
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze text and return full sentiment breakdown.
        
        Returns
        -------
        dict
            Complete sentiment analysis with scores and interpretation
        """
        sentiment = analyze_multifactor_sentiment(text)
        composite = calculate_composite_score(sentiment)
        
        return {
            'text': text,
            'vector': sentiment,
            'polarity': sentiment.polarity,
            'intensity': sentiment.intensity,
            'certainty': sentiment.certainty,
            'urgency': sentiment.urgency,
            'risk_tone': sentiment.risk_tone,
            'composite_score': composite,
            'interpretation': self._interpret(sentiment, composite)
        }
    
    def _interpret(self, sentiment: SentimentVector, composite: float) -> str:
        """Generate human-readable interpretation."""
        # Polarity interpretation
        if sentiment.polarity > 0.3:
            pol_str = "strongly positive"
        elif sentiment.polarity > 0:
            pol_str = "slightly positive"
        elif sentiment.polarity > -0.3:
            pol_str = "slightly negative"
        else:
            pol_str = "strongly negative"
        
        # Intensity
        if sentiment.intensity > 0.7:
            int_str = "high-impact"
        elif sentiment.intensity > 0.4:
            int_str = "moderate"
        else:
            int_str = "low-impact"
        
        # Certainty
        if sentiment.certainty > 0.7:
            cert_str = "confirmed"
        elif sentiment.certainty > 0.4:
            cert_str = "likely"
        else:
            cert_str = "speculative"
        
        # Urgency
        if sentiment.urgency > 0.7:
            urg_str = "immediate action needed"
        elif sentiment.urgency > 0.4:
            urg_str = "near-term relevant"
        else:
            urg_str = "long-term outlook"
        
        return f"{pol_str.capitalize()}, {int_str}, {cert_str} news. {urg_str.capitalize()}."
    
    def batch_analyze(self, texts: List[str]) -> 'pd.DataFrame':
        """Analyze multiple texts and return DataFrame."""
        sentiments = batch_analyze_sentiment(texts)
        return sentiment_to_dataframe(texts, sentiments)

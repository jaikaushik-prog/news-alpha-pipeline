"""
Sector Attribution Engine - Layer 5

Probabilistic sector exposure mapping for news items.

Instead of hard mapping:
    News_i → {Sector_j : Weight_ij}

Sector-Level Aggregation:
    Sector_Sentiment_t = Σ (News_Sentiment_i × Exposure_ij × Surprise_i)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Sector keyword mappings with relevance weights
SECTOR_KEYWORDS = {
    'banking_nbfc': {
        'keywords': [
            ('hdfc', 0.9), ('icici', 0.9), ('sbi', 0.9), ('axis', 0.8),
            ('kotak', 0.8), ('bajaj finance', 0.9), ('bank', 0.7), ('nbfc', 0.9),
            ('rbi', 0.8), ('lending', 0.7), ('loan', 0.7), ('npa', 0.8),
            ('credit growth', 0.8), ('deposit', 0.6), ('interest rate', 0.7)
        ],
        'nifty_index': 'NIFTYBANK',
        'typical_beta': 1.2
    },
    'it_technology': {
        'keywords': [
            ('infosys', 0.95), ('tcs', 0.95), ('wipro', 0.9), ('hcl', 0.9),
            ('tech mahindra', 0.9), ('it services', 0.9), ('software', 0.7),
            ('digital', 0.5), ('cloud', 0.6), ('ai ', 0.5), ('outsourcing', 0.7)
        ],
        'nifty_index': 'NIFTYIT',
        'typical_beta': 0.9
    },
    'pharma_healthcare': {
        'keywords': [
            ('sun pharma', 0.95), ('cipla', 0.95), ('dr reddy', 0.9),
            ('pharma', 0.85), ('drug', 0.7), ('healthcare', 0.7),
            ('hospital', 0.7), ('fda', 0.8), ('medicine', 0.6),
            ('biotech', 0.7), ('generic', 0.7)
        ],
        'nifty_index': 'NIFTYPHARMA',
        'typical_beta': 0.7
    },
    'auto': {
        'keywords': [
            ('maruti', 0.95), ('tata motors', 0.95), ('mahindra', 0.9),
            ('auto', 0.8), ('car', 0.7), ('vehicle', 0.7), ('ev ', 0.8),
            ('electric vehicle', 0.85), ('two-wheeler', 0.8), ('scooter', 0.6)
        ],
        'nifty_index': 'NIFTYAUTO',
        'typical_beta': 1.1
    },
    'infrastructure': {
        'keywords': [
            ('l&t', 0.95), ('larsen', 0.95), ('infra', 0.85),
            ('construction', 0.8), ('road', 0.7), ('highway', 0.7),
            ('bridge', 0.6), ('metro', 0.7), ('railway', 0.7),
            ('cement', 0.7), ('real estate', 0.6)
        ],
        'nifty_index': 'NIFTYINFRA',
        'typical_beta': 1.0
    },
    'energy_power': {
        'keywords': [
            ('reliance', 0.8), ('ongc', 0.95), ('oil', 0.8),
            ('gas', 0.7), ('power', 0.8), ('solar', 0.8),
            ('renewable', 0.8), ('coal', 0.7), ('ntpc', 0.95),
            ('electricity', 0.7), ('refinery', 0.8)
        ],
        'nifty_index': 'NIFTYENERGY',
        'typical_beta': 0.95
    },
    'metals_mining': {
        'keywords': [
            ('tata steel', 0.95), ('jsw steel', 0.95), ('hindalco', 0.95),
            ('vedanta', 0.9), ('metal', 0.85), ('steel', 0.9),
            ('copper', 0.8), ('aluminium', 0.8), ('iron ore', 0.85),
            ('zinc', 0.7), ('mining', 0.8)
        ],
        'nifty_index': 'NIFTYMETAL',
        'typical_beta': 1.3
    },
    'fmcg': {
        'keywords': [
            ('itc', 0.9), ('hul', 0.95), ('hindustan unilever', 0.95),
            ('nestle', 0.9), ('britannia', 0.9), ('fmcg', 0.9),
            ('consumer', 0.6), ('food', 0.5), ('beverage', 0.6)
        ],
        'nifty_index': 'NIFTYFMCG',
        'typical_beta': 0.6
    },
    'realty': {
        'keywords': [
            ('dlf', 0.95), ('godrej properties', 0.95), ('oberoi', 0.9),
            ('realty', 0.9), ('real estate', 0.9), ('property', 0.8),
            ('housing', 0.7), ('residential', 0.7), ('commercial real', 0.8)
        ],
        'nifty_index': 'NIFTYREALTY',
        'typical_beta': 1.4
    },
    'defence': {
        'keywords': [
            ('hal', 0.95), ('bharat electronics', 0.95), ('bel', 0.9),
            ('defence', 0.9), ('defense', 0.9), ('military', 0.8),
            ('aerospace', 0.8), ('missile', 0.8), ('navy', 0.7)
        ],
        'nifty_index': 'NIFTY50',  # No specific index
        'typical_beta': 0.8
    },
    'psu': {
        'keywords': [
            ('psu', 0.9), ('public sector', 0.9), ('government company', 0.8),
            ('disinvestment', 0.9), ('privatization', 0.85)
        ],
        'nifty_index': 'NIFTYPSE',
        'typical_beta': 1.0
    }
}


@dataclass
class SectorExposure:
    """Represents sector exposure for a news item."""
    sector: str
    weight: float
    confidence: float
    matched_keywords: List[str]


def calculate_sector_exposure(
    text: str,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Calculate probabilistic sector exposure for text.
    
    Parameters
    ----------
    text : str
        News headline or content
    normalize : bool
        If True, normalize weights to sum to 1
        
    Returns
    -------
    dict
        {sector: weight} mapping
    """
    text_lower = text.lower()
    exposures = {}
    
    for sector, config in SECTOR_KEYWORDS.items():
        sector_score = 0
        
        for keyword, weight in config['keywords']:
            if keyword in text_lower:
                sector_score += weight
        
        if sector_score > 0:
            exposures[sector] = sector_score
    
    # Normalize
    if normalize and exposures:
        total = sum(exposures.values())
        exposures = {k: v/total for k, v in exposures.items()}
    
    return exposures


def get_detailed_exposure(text: str) -> List[SectorExposure]:
    """
    Get detailed sector exposure with matched keywords.
    
    Parameters
    ----------
    text : str
        News headline or content
        
    Returns
    -------
    list
        List of SectorExposure objects
    """
    text_lower = text.lower()
    exposures = []
    
    for sector, config in SECTOR_KEYWORDS.items():
        matched = []
        total_weight = 0
        
        for keyword, weight in config['keywords']:
            if keyword in text_lower:
                matched.append(keyword)
                total_weight += weight
        
        if matched:
            # Confidence based on number of matches
            confidence = min(1.0, len(matched) / 3)
            
            exposures.append(SectorExposure(
                sector=sector,
                weight=total_weight,
                confidence=confidence,
                matched_keywords=matched
            ))
    
    # Sort by weight
    exposures.sort(key=lambda x: x.weight, reverse=True)
    
    return exposures


def aggregate_sector_sentiment(
    news_items: List[Dict],
    sentiment_key: str = 'composite_score',
    surprise_key: str = 'surprise_score'
) -> Dict[str, float]:
    """
    Aggregate sentiment at sector level.
    
    Formula: Sector_Sentiment = Σ (Sentiment_i × Exposure_ij × Surprise_i)
    
    Parameters
    ----------
    news_items : list
        List of news items with sentiment and surprise scores
    sentiment_key : str
        Key for sentiment score in news items
    surprise_key : str
        Key for surprise score in news items
        
    Returns
    -------
    dict
        {sector: aggregated_sentiment}
    """
    sector_scores = {sector: 0.0 for sector in SECTOR_KEYWORDS}
    sector_weights = {sector: 0.0 for sector in SECTOR_KEYWORDS}
    
    for item in news_items:
        headline = item.get('headline', item.get('text', ''))
        sentiment = item.get(sentiment_key, 0)
        surprise = item.get(surprise_key, 1.0)  # Default to 1 if not available
        
        # Get sector exposures
        exposures = calculate_sector_exposure(headline)
        
        for sector, exposure in exposures.items():
            # Weighted contribution
            contribution = sentiment * exposure * surprise
            sector_scores[sector] += contribution
            sector_weights[sector] += exposure * surprise
    
    # Normalize by weights
    result = {}
    for sector in SECTOR_KEYWORDS:
        if sector_weights[sector] > 0:
            result[sector] = sector_scores[sector] / sector_weights[sector]
        else:
            result[sector] = 0.0
    
    return result


def rank_sectors(sector_sentiment: Dict[str, float]) -> List[Tuple[str, float]]:
    """
    Rank sectors by sentiment.
    
    Returns
    -------
    list
        [(sector, score), ...] sorted by score descending
    """
    return sorted(sector_sentiment.items(), key=lambda x: x[1], reverse=True)


def get_sector_recommendations(
    sector_sentiment: Dict[str, float],
    threshold: float = 0.15
) -> Dict[str, List[str]]:
    """
    Generate sector recommendations based on sentiment.
    
    Parameters
    ----------
    sector_sentiment : dict
        {sector: score} mapping
    threshold : float
        Threshold for strong signals
        
    Returns
    -------
    dict
        {long: [...], short: [...], neutral: [...]}
    """
    recommendations = {
        'long': [],
        'short': [],
        'neutral': []
    }
    
    for sector, score in sector_sentiment.items():
        if score > threshold:
            recommendations['long'].append(sector)
        elif score < -threshold:
            recommendations['short'].append(sector)
        else:
            recommendations['neutral'].append(sector)
    
    return recommendations


class SectorAttributionEngine:
    """
    Complete sector attribution engine.
    
    Usage:
        engine = SectorAttributionEngine()
        exposures = engine.get_exposure("RBI cuts repo rate, banks to benefit")
        sentiment = engine.aggregate(news_items)
    """
    
    def __init__(self):
        self.sectors = list(SECTOR_KEYWORDS.keys())
    
    def get_exposure(self, text: str, detailed: bool = False):
        """Get sector exposure for text."""
        if detailed:
            return get_detailed_exposure(text)
        return calculate_sector_exposure(text)
    
    def aggregate(self, news_items: List[Dict]) -> Dict[str, float]:
        """Aggregate sentiment by sector."""
        return aggregate_sector_sentiment(news_items)
    
    def rank(self, sector_sentiment: Dict[str, float]) -> List[Tuple[str, float]]:
        """Rank sectors by sentiment."""
        return rank_sectors(sector_sentiment)
    
    def recommend(self, sector_sentiment: Dict[str, float]) -> Dict[str, List[str]]:
        """Generate recommendations."""
        return get_sector_recommendations(sector_sentiment)
    
    def to_dataframe(self, sector_sentiment: Dict[str, float]) -> pd.DataFrame:
        """Convert to DataFrame with metadata."""
        data = []
        for sector, score in sector_sentiment.items():
            config = SECTOR_KEYWORDS.get(sector, {})
            data.append({
                'sector': sector,
                'sentiment_score': score,
                'nifty_index': config.get('nifty_index', 'NIFTY50'),
                'typical_beta': config.get('typical_beta', 1.0),
                'expected_impact': score * config.get('typical_beta', 1.0)
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('sentiment_score', ascending=False)
        
        return df

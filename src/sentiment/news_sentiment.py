"""
News Headline Sentiment Analyzer.

Scrapes and analyzes sentiment from major Indian financial news sources:
- Economic Times
- Business Standard  
- LiveMint
- Moneycontrol

Uses FinBERT + VADER for financial sentiment analysis.
Provides sector-level and market-level sentiment scores.

Usage:
    from src.sentiment.news_sentiment import get_market_sentiment
    
    sentiment = get_market_sentiment()
    print(sentiment['overall_score'])
    print(sentiment['bullish_headlines'])
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import re
import time
import pandas as pd
import numpy as np

from ..utils.logging import get_logger
from ..nlp.sentiment import SentimentAnalyzer

logger = get_logger(__name__)


# News sources configuration
NEWS_SOURCES = {
    'economic_times': {
        'name': 'Economic Times',
        'base_url': 'https://economictimes.indiatimes.com',
        'markets_url': 'https://economictimes.indiatimes.com/markets',
        'rss_url': 'https://economictimes.indiatimes.com/rssfeedstopstories.cms',
    },
    'business_standard': {
        'name': 'Business Standard',
        'base_url': 'https://www.business-standard.com',
        'markets_url': 'https://www.business-standard.com/markets',
        'rss_url': 'https://www.business-standard.com/rss/markets-106.rss',
    },
    'livemint': {
        'name': 'LiveMint',
        'base_url': 'https://www.livemint.com',
        'markets_url': 'https://www.livemint.com/market',
        'rss_url': 'https://www.livemint.com/rss/markets',
    },
    'moneycontrol': {
        'name': 'Moneycontrol',
        'base_url': 'https://www.moneycontrol.com',
        'markets_url': 'https://www.moneycontrol.com/news/business/markets/',
    }
}


# Keywords for sector classification
SECTOR_KEYWORDS = {
    'banking_nbfc': ['bank', 'nbfc', 'hdfc', 'icici', 'sbi', 'axis', 'kotak', 'bajaj', 'rbi', 'lending', 'loan', 'npa'],
    'it_technology': ['it', 'infosys', 'tcs', 'wipro', 'hcl', 'tech', 'software', 'digital', 'ai', 'cloud'],
    'pharma_healthcare': ['pharma', 'drug', 'healthcare', 'hospital', 'cipla', 'sun pharma', 'fda', 'medicine'],
    'auto': ['auto', 'car', 'vehicle', 'maruti', 'tata motors', 'mahindra', 'ev', 'electric vehicle'],
    'infrastructure': ['infra', 'construction', 'l&t', 'road', 'highway', 'bridge', 'real estate', 'cement'],
    'energy_power': ['energy', 'power', 'oil', 'gas', 'ongc', 'reliance', 'solar', 'renewable', 'coal'],
    'metals_mining': ['metal', 'steel', 'tata steel', 'jsw', 'hindalco', 'copper', 'aluminium', 'iron ore'],
    'fmcg': ['fmcg', 'consumer', 'itc', 'hul', 'hindustan unilever', 'nestle', 'britannia', 'food'],
    'realty': ['realty', 'real estate', 'property', 'dlf', 'godrej properties', 'housing', 'residential'],
    'defence': ['defence', 'defense', 'hal', 'bharat electronics', 'bel', 'military', 'aerospace'],
}


# Market sentiment keywords
BULLISH_KEYWORDS = [
    'surge', 'rally', 'gain', 'jump', 'soar', 'bullish', 'up', 'rise', 'high', 'record',
    'beat', 'outperform', 'upgrade', 'buy', 'positive', 'growth', 'profit', 'boom'
]

BEARISH_KEYWORDS = [
    'fall', 'drop', 'crash', 'plunge', 'bearish', 'down', 'decline', 'low', 'slump',
    'miss', 'downgrade', 'sell', 'negative', 'loss', 'weak', 'concern', 'fear', 'risk'
]


@dataclass
class NewsHeadline:
    """Represents a single news headline."""
    title: str
    source: str
    url: Optional[str] = None
    timestamp: Optional[datetime] = None
    sentiment_score: float = 0.0
    sectors: List[str] = None


def scrape_headlines_rss(source_key: str, limit: int = 20) -> List[NewsHeadline]:
    """
    Scrape headlines from RSS feed.
    
    Parameters
    ----------
    source_key : str
        Key from NEWS_SOURCES dict
    limit : int
        Maximum headlines to fetch
        
    Returns
    -------
    list
        List of NewsHeadline objects
    """
    source = NEWS_SOURCES.get(source_key)
    if not source or 'rss_url' not in source:
        logger.warning(f"No RSS URL for source: {source_key}")
        return []
    
    try:
        import feedparser
        
        feed = feedparser.parse(source['rss_url'])
        
        headlines = []
        for entry in feed.entries[:limit]:
            headline = NewsHeadline(
                title=entry.get('title', ''),
                source=source['name'],
                url=entry.get('link', ''),
                timestamp=datetime.now(),  # RSS may have 'published'
            )
            headlines.append(headline)
        
        logger.info(f"Fetched {len(headlines)} headlines from {source['name']} RSS")
        return headlines
        
    except ImportError:
        logger.warning("feedparser not installed. Install with: pip install feedparser")
        return []
    except Exception as e:
        logger.error(f"Error fetching RSS from {source_key}: {e}")
        return []


def scrape_headlines_web(source_key: str, limit: int = 20) -> List[NewsHeadline]:
    """
    Scrape headlines from website (fallback when RSS not available).
    
    Parameters
    ----------
    source_key : str
        Key from NEWS_SOURCES dict
    limit : int
        Maximum headlines to fetch
        
    Returns
    -------
    list
        List of NewsHeadline objects
    """
    source = NEWS_SOURCES.get(source_key)
    if not source:
        return []
    
    try:
        import requests
        from bs4 import BeautifulSoup
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(source['markets_url'], headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find headline elements (varies by site)
        headlines = []
        
        # Common headline tag patterns
        for tag in soup.find_all(['h1', 'h2', 'h3', 'a'], limit=limit * 3):
            text = tag.get_text(strip=True)
            if len(text) > 20 and len(text) < 300:  # Filter noise
                headline = NewsHeadline(
                    title=text,
                    source=source['name'],
                    url=tag.get('href', ''),
                    timestamp=datetime.now()
                )
                headlines.append(headline)
                
                if len(headlines) >= limit:
                    break
        
        logger.info(f"Scraped {len(headlines)} headlines from {source['name']} web")
        return headlines
        
    except ImportError:
        logger.warning("requests/beautifulsoup4 not installed")
        return []
    except Exception as e:
        logger.error(f"Error scraping {source_key}: {e}")
        return []


def analyze_headline_sentiment(
    headlines: List[NewsHeadline],
    method: str = 'combined'
) -> List[NewsHeadline]:
    """
    Analyze sentiment for each headline.
    
    Parameters
    ----------
    headlines : list
        List of NewsHeadline objects
    method : str
        Sentiment method: 'vader', 'finbert', or 'combined'
        
    Returns
    -------
    list
        Headlines with sentiment scores
    """
    try:
        analyzer = SentimentAnalyzer(method=method)
    except:
        # Fallback to simple keyword-based sentiment
        logger.warning("SentimentAnalyzer not available, using keyword method")
        analyzer = None
    
    for headline in headlines:
        text = headline.title.lower()
        
        if analyzer:
            try:
                result = analyzer.analyze(headline.title)
                headline.sentiment_score = result.get('sentiment_compound', 0)
            except:
                # Fallback
                headline.sentiment_score = _keyword_sentiment(text)
        else:
            headline.sentiment_score = _keyword_sentiment(text)
        
        # Classify sectors
        headline.sectors = _classify_sectors(text)
    
    return headlines


def _keyword_sentiment(text: str) -> float:
    """Simple keyword-based sentiment scoring."""
    text_lower = text.lower()
    
    bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
    
    total = bullish_count + bearish_count
    if total == 0:
        return 0.0
    
    return (bullish_count - bearish_count) / total


def _classify_sectors(text: str) -> List[str]:
    """Classify headline into sectors based on keywords."""
    text_lower = text.lower()
    sectors = []
    
    for sector, keywords in SECTOR_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                sectors.append(sector)
                break
    
    return sectors


def aggregate_sentiment(headlines: List[NewsHeadline]) -> Dict:
    """
    Aggregate sentiment across all headlines.
    
    Parameters
    ----------
    headlines : list
        Analyzed headlines
        
    Returns
    -------
    dict
        Aggregated sentiment metrics
    """
    if not headlines:
        return {'overall_score': 0, 'headline_count': 0}
    
    scores = [h.sentiment_score for h in headlines]
    
    # Separate bullish/bearish
    bullish = [h for h in headlines if h.sentiment_score > 0.1]
    bearish = [h for h in headlines if h.sentiment_score < -0.1]
    neutral = [h for h in headlines if -0.1 <= h.sentiment_score <= 0.1]
    
    # Sector sentiment
    sector_sentiment = {}
    for sector in SECTOR_KEYWORDS.keys():
        sector_headlines = [h for h in headlines if sector in (h.sectors or [])]
        if sector_headlines:
            sector_sentiment[sector] = np.mean([h.sentiment_score for h in sector_headlines])
    
    result = {
        'overall_score': np.mean(scores),
        'score_std': np.std(scores),
        'headline_count': len(headlines),
        'bullish_count': len(bullish),
        'bearish_count': len(bearish),
        'neutral_count': len(neutral),
        'bullish_ratio': len(bullish) / len(headlines),
        'bearish_ratio': len(bearish) / len(headlines),
        'sentiment_regime': _classify_regime(np.mean(scores)),
        'sector_sentiment': sector_sentiment,
        'top_bullish': [h.title for h in sorted(bullish, key=lambda x: x.sentiment_score, reverse=True)[:5]],
        'top_bearish': [h.title for h in sorted(bearish, key=lambda x: x.sentiment_score)[:5]],
        'timestamp': datetime.now().isoformat()
    }
    
    return result


def _classify_regime(score: float) -> str:
    """Classify overall market sentiment regime."""
    if score > 0.3:
        return 'very_bullish'
    elif score > 0.1:
        return 'bullish'
    elif score > -0.1:
        return 'neutral'
    elif score > -0.3:
        return 'bearish'
    else:
        return 'very_bearish'


def get_market_sentiment(
    sources: Optional[List[str]] = None,
    headlines_per_source: int = 15
) -> Dict:
    """
    Get overall market sentiment from news headlines.
    
    Parameters
    ----------
    sources : list, optional
        Sources to scrape (default: all)
    headlines_per_source : int
        Headlines per source
        
    Returns
    -------
    dict
        Market sentiment analysis
    """
    if sources is None:
        sources = list(NEWS_SOURCES.keys())
    
    all_headlines = []
    
    for source_key in sources:
        # Try RSS first, then web scraping
        headlines = scrape_headlines_rss(source_key, headlines_per_source)
        
        if not headlines:
            headlines = scrape_headlines_web(source_key, headlines_per_source)
        
        all_headlines.extend(headlines)
        time.sleep(0.5)  # Be polite to servers
    
    if not all_headlines:
        logger.warning("No headlines fetched from any source")
        return {'overall_score': 0, 'error': 'No headlines available'}
    
    # Analyze sentiment
    analyzed = analyze_headline_sentiment(all_headlines)
    
    # Aggregate
    sentiment = aggregate_sentiment(analyzed)
    
    logger.info(f"Market sentiment: {sentiment['sentiment_regime']} (score: {sentiment['overall_score']:.3f})")
    
    return sentiment


def get_sector_sentiment(sector: str) -> Dict:
    """
    Get sentiment for a specific sector.
    
    Parameters
    ----------
    sector : str
        Sector name from SECTOR_KEYWORDS
        
    Returns
    -------
    dict
        Sector-specific sentiment
    """
    market_sentiment = get_market_sentiment()
    
    sector_score = market_sentiment.get('sector_sentiment', {}).get(sector)
    
    return {
        'sector': sector,
        'sentiment_score': sector_score,
        'market_overall': market_sentiment['overall_score'],
        'relative_strength': (sector_score - market_sentiment['overall_score']) if sector_score else None
    }


def create_mock_news_sentiment() -> Dict:
    """
    Create mock news sentiment for demonstration.
    
    Returns
    -------
    dict
        Mock sentiment data
    """
    np.random.seed(42)
    
    mock_headlines = [
        NewsHeadline("Nifty hits record high as FIIs turn buyers", "Economic Times", sentiment_score=0.8),
        NewsHeadline("Banking stocks rally on RBI rate decision", "Business Standard", sentiment_score=0.6),
        NewsHeadline("IT sector faces headwinds amid global slowdown", "LiveMint", sentiment_score=-0.4),
        NewsHeadline("Auto sales surge 15% in January", "Moneycontrol", sentiment_score=0.5),
        NewsHeadline("Pharma stocks under pressure on FDA concerns", "Economic Times", sentiment_score=-0.5),
        NewsHeadline("Infrastructure push to boost cement demand", "Business Standard", sentiment_score=0.4),
        NewsHeadline("Metal prices volatile on China demand worries", "LiveMint", sentiment_score=-0.2),
        NewsHeadline("FMCG earnings beat street estimates", "Moneycontrol", sentiment_score=0.6),
        NewsHeadline("Real estate sales pick up in major cities", "Economic Times", sentiment_score=0.3),
        NewsHeadline("Defence orders lift HAL, BEL stocks", "Business Standard", sentiment_score=0.5),
    ]
    
    for h in mock_headlines:
        h.sectors = _classify_sectors(h.title.lower())
    
    return aggregate_sentiment(mock_headlines)


def interpret_for_investing(sentiment: Dict) -> str:
    """
    Generate investment interpretation from sentiment analysis.
    
    Parameters
    ----------
    sentiment : dict
        Sentiment analysis results
        
    Returns
    -------
    str
        Investment recommendation
    """
    score = sentiment.get('overall_score', 0)
    regime = sentiment.get('sentiment_regime', 'neutral')
    bullish_ratio = sentiment.get('bullish_ratio', 0.5)
    
    interpretations = {
        'very_bullish': (
            "🟢 VERY BULLISH sentiment. News coverage is overwhelmingly positive. "
            f"Bullish/Bearish ratio: {bullish_ratio:.0%}. "
            "⚠️ Potential for mean reversion if sentiment is too extreme. "
            "Consider: Momentum trades with tight trailing stops."
        ),
        'bullish': (
            "🟢 BULLISH sentiment. Positive news flow supports market upside. "
            f"Bullish/Bearish ratio: {bullish_ratio:.0%}. "
            "Consider: Accumulate quality stocks on dips."
        ),
        'neutral': (
            "🟡 NEUTRAL sentiment. Mixed news flow, no clear direction. "
            f"Bullish/Bearish ratio: {bullish_ratio:.0%}. "
            "Consider: Wait for clearer signals or trade specific sectors."
        ),
        'bearish': (
            "🔴 BEARISH sentiment. Negative news dominates headlines. "
            f"Bullish/Bearish ratio: {bullish_ratio:.0%}. "
            "Consider: Reduce exposure, hedge positions, or wait for reversal."
        ),
        'very_bearish': (
            "🔴 VERY BEARISH sentiment. Extreme fear in news coverage. "
            f"Bullish/Bearish ratio: {bullish_ratio:.0%}. "
            "⚠️ Potential contrarian opportunity if fundamentals are intact. "
            "Consider: Start building positions gradually."
        )
    }
    
    base = interpretations.get(regime, "Unknown regime")
    
    # Add sector insights
    sector_sentiment = sentiment.get('sector_sentiment', {})
    if sector_sentiment:
        best_sector = max(sector_sentiment.items(), key=lambda x: x[1])
        worst_sector = min(sector_sentiment.items(), key=lambda x: x[1])
        base += f"\n\n📊 Sector Highlights:\n"
        base += f"  Best: {best_sector[0]} (score: {best_sector[1]:.2f})\n"
        base += f"  Worst: {worst_sector[0]} (score: {worst_sector[1]:.2f})"
    
    return base

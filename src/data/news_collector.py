"""
News Data Collector - Layer 1: Data Acquisition

Collects time-stamped financial headlines from multiple sources:
- Economic Times
- Business Standard
- LiveMint
- Moneycontrol

Each headline is categorized as: policy / earnings / macro
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import re
import time
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NewsItem:
    """Represents a single news item with metadata."""
    headline: str
    timestamp: datetime
    source: str
    category: str = 'general'  # policy / earnings / macro / general
    url: Optional[str] = None
    content_hash: str = field(default='', init=False)
    
    def __post_init__(self):
        # Generate content hash for deduplication
        self.content_hash = hashlib.md5(
            self.headline.lower().encode()
        ).hexdigest()[:16]


# News source configurations
NEWS_SOURCES = {
    'economic_times': {
        'name': 'Economic Times',
        'rss_feeds': {
            'markets': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'economy': 'https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms',
            'policy': 'https://economictimes.indiatimes.com/news/economy/policy/rssfeeds/1373401680.cms',
        },
        'reliability': 0.9
    },
    'business_standard': {
        'name': 'Business Standard',
        'rss_feeds': {
            'markets': 'https://www.business-standard.com/rss/markets-106.rss',
            'economy': 'https://www.business-standard.com/rss/economy-102.rss',
        },
        'reliability': 0.85
    },
    'livemint': {
        'name': 'LiveMint',
        'rss_feeds': {
            'markets': 'https://www.livemint.com/rss/markets',
            'economy': 'https://www.livemint.com/rss/economy',
        },
        'reliability': 0.8
    },
    'moneycontrol': {
        'name': 'Moneycontrol',
        'rss_feeds': {
            'markets': 'https://www.moneycontrol.com/rss/marketreports.xml',
            'news': 'https://www.moneycontrol.com/rss/business.xml',
        },
        'reliability': 0.75
    }
}


# Category classification keywords
CATEGORY_KEYWORDS = {
    'policy': [
        'rbi', 'sebi', 'budget', 'policy', 'regulation', 'government', 'ministry',
        'parliament', 'tax', 'gst', 'interest rate', 'inflation target', 'fiscal',
        'monetary policy', 'repo rate', 'crr', 'slr', 'fdi', 'nirmala', 'finance minister'
    ],
    'earnings': [
        'q1', 'q2', 'q3', 'q4', 'quarterly', 'results', 'profit', 'revenue', 'earnings',
        'eps', 'net income', 'operating profit', 'ebitda', 'margin', 'guidance',
        'beat estimates', 'miss estimates', 'dividend', 'bonus'
    ],
    'macro': [
        'gdp', 'inflation', 'cpi', 'wpi', 'iip', 'pmi', 'trade deficit', 'exports',
        'imports', 'current account', 'forex', 'rupee', 'dollar', 'crude oil',
        'unemployment', 'employment', 'manufacturing', 'services'
    ]
}


def classify_category(headline: str) -> str:
    """
    Classify headline into category based on keywords.
    
    Parameters
    ----------
    headline : str
        News headline text
        
    Returns
    -------
    str
        Category: 'policy', 'earnings', 'macro', or 'general'
    """
    text_lower = headline.lower()
    
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[category] = score
    
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    
    return 'general'


def fetch_rss_feed(url: str, source_name: str) -> List[NewsItem]:
    """
    Fetch news items from an RSS feed.
    
    Parameters
    ----------
    url : str
        RSS feed URL
    source_name : str
        Name of the source
        
    Returns
    -------
    list
        List of NewsItem objects
    """
    try:
        import feedparser
        
        feed = feedparser.parse(url)
        
        items = []
        for entry in feed.entries:
            # Parse timestamp
            timestamp = datetime.now()
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                try:
                    timestamp = datetime(*entry.published_parsed[:6])
                except:
                    pass
            
            headline = entry.get('title', '').strip()
            if not headline:
                continue
            
            # Classify category
            category = classify_category(headline)
            
            item = NewsItem(
                headline=headline,
                timestamp=timestamp,
                source=source_name,
                category=category,
                url=entry.get('link', '')
            )
            items.append(item)
        
        return items
        
    except ImportError:
        logger.warning("feedparser not installed. Install with: pip install feedparser")
        return []
    except Exception as e:
        logger.error(f"Error fetching RSS from {url}: {e}")
        return []


def collect_all_news(
    sources: Optional[List[str]] = None,
    max_age_hours: int = 24
) -> List[NewsItem]:
    """
    Collect news from all configured sources.
    
    Parameters
    ----------
    sources : list, optional
        List of source keys (default: all sources)
    max_age_hours : int
        Maximum age of news items in hours
        
    Returns
    -------
    list
        List of NewsItem objects, deduplicated
    """
    if sources is None:
        sources = list(NEWS_SOURCES.keys())
    
    all_items = []
    seen_hashes = set()
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    for source_key in sources:
        source = NEWS_SOURCES.get(source_key)
        if not source:
            continue
        
        logger.info(f"Collecting from {source['name']}...")
        
        for feed_name, feed_url in source.get('rss_feeds', {}).items():
            items = fetch_rss_feed(feed_url, source['name'])
            
            for item in items:
                # Skip duplicates
                if item.content_hash in seen_hashes:
                    continue
                
                # Skip old items
                if item.timestamp < cutoff_time:
                    continue
                
                seen_hashes.add(item.content_hash)
                all_items.append(item)
            
            time.sleep(0.3)  # Rate limiting
    
    # Sort by timestamp (newest first)
    all_items.sort(key=lambda x: x.timestamp, reverse=True)
    
    logger.info(f"Collected {len(all_items)} unique news items")
    return all_items


def filter_by_category(
    items: List[NewsItem],
    categories: List[str]
) -> List[NewsItem]:
    """Filter news items by category."""
    return [item for item in items if item.category in categories]


def to_dataframe(items: List[NewsItem]) -> pd.DataFrame:
    """Convert news items to DataFrame."""
    if not items:
        return pd.DataFrame()
    
    data = [
        {
            'headline': item.headline,
            'timestamp': item.timestamp,
            'source': item.source,
            'category': item.category,
            'url': item.url,
            'content_hash': item.content_hash
        }
        for item in items
    ]
    
    return pd.DataFrame(data)


def save_news_cache(items: List[NewsItem], cache_path: Path):
    """Save news items to JSON cache."""
    data = [
        {
            'headline': item.headline,
            'timestamp': item.timestamp.isoformat(),
            'source': item.source,
            'category': item.category,
            'url': item.url,
            'content_hash': item.content_hash
        }
        for item in items
    ]
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(items)} items to {cache_path}")


def load_news_cache(cache_path: Path) -> List[NewsItem]:
    """Load news items from JSON cache."""
    if not cache_path.exists():
        return []
    
    with open(cache_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = []
    for d in data:
        item = NewsItem(
            headline=d['headline'],
            timestamp=datetime.fromisoformat(d['timestamp']),
            source=d['source'],
            category=d['category'],
            url=d.get('url', '')
        )
        items.append(item)
    
    return items


def get_category_summary(items: List[NewsItem]) -> Dict[str, int]:
    """Get count of items per category."""
    summary = {}
    for item in items:
        summary[item.category] = summary.get(item.category, 0) + 1
    return summary


# Convenience function
def get_latest_news(max_items: int = 50) -> pd.DataFrame:
    """
    Get latest news headlines as DataFrame.
    
    Parameters
    ----------
    max_items : int
        Maximum items to return
        
    Returns
    -------
    pd.DataFrame
        News data
    """
    items = collect_all_news()[:max_items]
    return to_dataframe(items)

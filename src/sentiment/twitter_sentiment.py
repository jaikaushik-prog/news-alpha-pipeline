"""
Twitter/X Sentiment Analysis Module.

Scrapes and analyzes Twitter sentiment for Budget-related keywords.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import re

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Budget-related keywords for tracking
BUDGET_KEYWORDS = {
    'general': [
        'Union Budget', 'Budget 2025', 'Budget2025', 'IndianBudget',
        'Nirmala Sitharaman', 'Finance Minister', 'FM Sitharaman',
        'budget speech', 'budget announcement', 'fiscal policy'
    ],
    'banking_nbfc': [
        'banking sector budget', 'PSU banks budget', 'NBFC budget',
        'bank recapitalization', 'credit growth budget'
    ],
    'infrastructure': [
        'infrastructure budget', 'capex budget', 'roads highways budget',
        'railways budget', 'metro budget', 'construction budget'
    ],
    'it_technology': [
        'IT sector budget', 'tech budget', 'digital india budget',
        'startup budget', 'electronics manufacturing'
    ],
    'pharma_healthcare': [
        'pharma budget', 'healthcare budget', 'Ayushman Bharat',
        'medical budget', 'hospital budget'
    ],
    'auto': [
        'automobile budget', 'EV budget', 'electric vehicle budget',
        'auto sector', 'FAME scheme'
    ],
    'energy_power': [
        'power sector budget', 'renewable energy budget', 'solar budget',
        'green hydrogen', 'energy transition'
    ],
    'realty': [
        'real estate budget', 'housing budget', 'affordable housing',
        'PMAY', 'property budget'
    ],
    'metals_mining': [
        'steel budget', 'metals budget', 'mining budget',
        'PLI scheme metals', 'aluminum budget'
    ],
    'fmcg': [
        'FMCG budget', 'consumer goods budget', 'rural demand budget',
        'GST rate', 'excise duty'
    ],
    'agriculture': [
        'agriculture budget', 'farm budget', 'MSP budget',
        'fertilizer subsidy', 'PM-KISAN', 'agri budget'
    ],
    'defence': [
        'defence budget', 'military budget', 'defense spending',
        'Make in India defence', 'HAL budget'
    ]
}


class TwitterSentimentAnalyzer:
    """
    Analyze Twitter sentiment for Budget-related content.
    
    Note: Requires Twitter API v2 access (Academic Research or Basic tier).
    """
    
    def __init__(self, bearer_token: Optional[str] = None):
        """
        Initialize Twitter analyzer.
        
        Parameters
        ----------
        bearer_token : str, optional
            Twitter API bearer token. If None, will try environment variable.
        """
        self.bearer_token = bearer_token or self._get_token_from_env()
        self.client = None
        
        if self.bearer_token:
            self._init_client()
    
    def _get_token_from_env(self) -> Optional[str]:
        """Get Twitter bearer token from environment."""
        import os
        return os.environ.get('TWITTER_BEARER_TOKEN')
    
    def _init_client(self):
        """Initialize Twitter client."""
        try:
            import tweepy
            self.client = tweepy.Client(bearer_token=self.bearer_token)
            logger.info("Twitter API client initialized")
        except ImportError:
            logger.warning("tweepy not installed. Install with: pip install tweepy")
        except Exception as e:
            logger.error(f"Error initializing Twitter client: {e}")
    
    def search_recent_tweets(
        self,
        query: str,
        max_results: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Search for recent tweets matching query.
        
        Parameters
        ----------
        query : str
            Search query
        max_results : int
            Maximum results to return (max 100 for basic tier)
        start_time : datetime, optional
            Start time for search
        end_time : datetime, optional
            End time for search
            
        Returns
        -------
        list
            List of tweet dictionaries
        """
        if not self.client:
            logger.warning("Twitter client not initialized")
            return []
        
        try:
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                start_time=start_time,
                end_time=end_time,
                tweet_fields=['created_at', 'public_metrics', 'lang']
            )
            
            if not tweets.data:
                return []
            
            results = []
            for tweet in tweets.data:
                results.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'retweets': tweet.public_metrics.get('retweet_count', 0) if tweet.public_metrics else 0,
                    'likes': tweet.public_metrics.get('like_count', 0) if tweet.public_metrics else 0,
                    'lang': tweet.lang
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Parameters
        ----------
        text : str
            Tweet text
            
        Returns
        -------
        dict
            Sentiment scores
        """
        try:
            from ..nlp.sentiment import SentimentAnalyzer
            analyzer = SentimentAnalyzer(method='vader')
            return analyzer.analyze(text)
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {'sentiment_compound': 0}
    
    def get_sector_sentiment(
        self,
        sector: str,
        hours_lookback: int = 24
    ) -> Dict[str, float]:
        """
        Get aggregated sentiment for a sector.
        
        Parameters
        ----------
        sector : str
            Sector name
        hours_lookback : int
            Hours to look back
            
        Returns
        -------
        dict
            Aggregated sentiment metrics
        """
        keywords = BUDGET_KEYWORDS.get(sector, [])
        if not keywords:
            logger.warning(f"No keywords defined for sector: {sector}")
            return {}
        
        # Build query
        query = ' OR '.join([f'"{kw}"' for kw in keywords[:5]])  # Limit to avoid query length issues
        query += ' lang:en -is:retweet'
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_lookback)
        
        tweets = self.search_recent_tweets(
            query=query,
            max_results=100,
            start_time=start_time,
            end_time=end_time
        )
        
        if not tweets:
            return {'n_tweets': 0, 'avg_sentiment': 0, 'engagement': 0}
        
        sentiments = []
        total_engagement = 0
        
        for tweet in tweets:
            sent = self.analyze_sentiment(tweet['text'])
            sentiments.append(sent.get('sentiment_compound', 0))
            total_engagement += tweet['likes'] + tweet['retweets']
        
        return {
            'n_tweets': len(tweets),
            'avg_sentiment': np.mean(sentiments),
            'std_sentiment': np.std(sentiments),
            'positive_pct': sum(1 for s in sentiments if s > 0.05) / len(sentiments),
            'negative_pct': sum(1 for s in sentiments if s < -0.05) / len(sentiments),
            'total_engagement': total_engagement,
            'avg_engagement': total_engagement / len(tweets)
        }


def analyze_pre_budget_twitter_sentiment(
    sectors: List[str] = None,
    hours_lookback: int = 24
) -> pd.DataFrame:
    """
    Analyze Twitter sentiment for all sectors before Budget.
    
    Parameters
    ----------
    sectors : list, optional
        List of sectors (default: all defined sectors)
    hours_lookback : int
        Hours to look back
        
    Returns
    -------
    pd.DataFrame
        Sector-wise sentiment summary
    """
    if sectors is None:
        sectors = list(BUDGET_KEYWORDS.keys())
    
    analyzer = TwitterSentimentAnalyzer()
    
    results = []
    for sector in sectors:
        logger.info(f"Fetching Twitter sentiment for {sector}...")
        sentiment = analyzer.get_sector_sentiment(sector, hours_lookback)
        sentiment['sector'] = sector
        results.append(sentiment)
    
    return pd.DataFrame(results)


def create_mock_twitter_sentiment(budget_date: str) -> pd.DataFrame:
    """
    Create mock Twitter sentiment data for demonstration.
    
    Use this when Twitter API is not available.
    
    Parameters
    ----------
    budget_date : str
        Budget date for context
        
    Returns
    -------
    pd.DataFrame
        Mock sentiment data
    """
    np.random.seed(42)  # For reproducibility
    
    sectors = list(BUDGET_KEYWORDS.keys())
    
    mock_data = []
    for sector in sectors:
        # Generate realistic-looking mock data
        n_tweets = np.random.randint(50, 500)
        base_sentiment = np.random.uniform(-0.2, 0.4)  # Slight positive bias
        
        mock_data.append({
            'sector': sector,
            'n_tweets': n_tweets,
            'avg_sentiment': base_sentiment,
            'std_sentiment': np.random.uniform(0.2, 0.4),
            'positive_pct': 0.5 + base_sentiment,
            'negative_pct': 0.3 - base_sentiment / 2,
            'total_engagement': n_tweets * np.random.randint(5, 50),
            'avg_engagement': np.random.randint(5, 50)
        })
    
    df = pd.DataFrame(mock_data)
    logger.info(f"Created mock Twitter sentiment data for {len(sectors)} sectors")
    return df

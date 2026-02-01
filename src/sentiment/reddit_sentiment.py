"""
Reddit Sentiment Analysis Module.

Scrapes and analyzes Reddit posts/comments for Budget-related sentiment
from Indian investing subreddits.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Target subreddits for Indian Budget sentiment
SUBREDDITS = [
    'IndiaInvestments',
    'indiabusiness', 
    'IndianStockMarket',
    'india'
]

# Budget-related search keywords
BUDGET_SEARCH_TERMS = [
    'union budget',
    'budget 2025',
    'budget 2024',
    'nirmala sitharaman',
    'finance minister budget',
    'budget expectations',
    'budget predictions',
    'budget impact stocks',
    'budget sectors'
]


class RedditSentimentAnalyzer:
    """
    Analyze Reddit sentiment for Budget-related discussions.
    
    Requires Reddit API credentials:
    - CLIENT_ID: Your app's client ID
    - CLIENT_SECRET: Your app's secret
    - USER_AGENT: A unique identifier for your app
    """
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "BudgetSentiment/1.0"
    ):
        """
        Initialize Reddit analyzer.
        
        Parameters
        ----------
        client_id : str, optional
            Reddit API client ID. Falls back to REDDIT_CLIENT_ID env var.
        client_secret : str, optional  
            Reddit API client secret. Falls back to REDDIT_CLIENT_SECRET env var.
        user_agent : str
            Unique identifier for API requests
        """
        self.client_id = client_id or os.environ.get('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.environ.get('REDDIT_CLIENT_SECRET')
        self.user_agent = user_agent
        self.reddit = None
        
        if self.client_id and self.client_secret:
            self._init_client()
    
    def _init_client(self):
        """Initialize Reddit client using PRAW."""
        try:
            import praw
            
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            logger.info("Reddit API client initialized successfully")
            
        except ImportError:
            logger.warning("praw not installed. Install with: pip install praw")
        except Exception as e:
            logger.error(f"Error initializing Reddit client: {e}")
    
    def search_posts(
        self,
        query: str,
        subreddit: str = "IndiaInvestments",
        limit: int = 100,
        time_filter: str = "month"
    ) -> List[Dict]:
        """
        Search for posts matching query.
        
        Parameters
        ----------
        query : str
            Search query
        subreddit : str
            Subreddit to search
        limit : int
            Maximum posts to return
        time_filter : str
            Time filter: 'hour', 'day', 'week', 'month', 'year', 'all'
            
        Returns
        -------
        list
            List of post dictionaries
        """
        if not self.reddit:
            logger.warning("Reddit client not initialized")
            return []
        
        try:
            subreddit_obj = self.reddit.subreddit(subreddit)
            posts = []
            
            for post in subreddit_obj.search(query, limit=limit, time_filter=time_filter):
                posts.append({
                    'id': post.id,
                    'title': post.title,
                    'selftext': post.selftext,
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'subreddit': subreddit,
                    'url': post.url,
                    'author': str(post.author)
                })
            
            logger.info(f"Found {len(posts)} posts for '{query}' in r/{subreddit}")
            return posts
            
        except Exception as e:
            logger.error(f"Error searching Reddit: {e}")
            return []
    
    def get_post_comments(
        self,
        post_id: str,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get comments from a specific post.
        
        Parameters
        ----------
        post_id : str
            Reddit post ID
        limit : int
            Maximum comments to return
            
        Returns
        -------
        list
            List of comment dictionaries
        """
        if not self.reddit:
            return []
        
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Flatten comment tree
            
            comments = []
            for comment in submission.comments.list()[:limit]:
                comments.append({
                    'id': comment.id,
                    'body': comment.body,
                    'score': comment.score,
                    'created_utc': datetime.fromtimestamp(comment.created_utc),
                    'author': str(comment.author)
                })
            
            return comments
            
        except Exception as e:
            logger.error(f"Error getting comments: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using VADER.
        
        Parameters
        ----------
        text : str
            Text to analyze
            
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
            logger.error(f"Sentiment analysis error: {e}")
            return {'sentiment_compound': 0}
    
    def get_budget_sentiment(
        self,
        days_before_budget: int = 7,
        budget_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get aggregated Budget sentiment from Reddit.
        
        Parameters
        ----------
        days_before_budget : int
            Days before budget to analyze
        budget_date : str, optional
            Budget date in YYYY-MM-DD format
            
        Returns
        -------
        pd.DataFrame
            Aggregated sentiment results
        """
        all_posts = []
        
        for subreddit in SUBREDDITS:
            for term in BUDGET_SEARCH_TERMS[:3]:  # Limit to avoid rate limits
                posts = self.search_posts(
                    query=term,
                    subreddit=subreddit,
                    limit=25,
                    time_filter='month'
                )
                all_posts.extend(posts)
        
        if not all_posts:
            logger.warning("No posts found")
            return pd.DataFrame()
        
        # Analyze sentiment for each post
        results = []
        for post in all_posts:
            # Combine title and body for analysis
            text = f"{post['title']} {post['selftext']}"
            sentiment = self.analyze_sentiment(text)
            
            results.append({
                'post_id': post['id'],
                'subreddit': post['subreddit'],
                'title': post['title'][:100],
                'score': post['score'],
                'num_comments': post['num_comments'],
                'created_utc': post['created_utc'],
                'sentiment_compound': sentiment.get('sentiment_compound', 0),
                'engagement': post['score'] + post['num_comments']
            })
        
        df = pd.DataFrame(results)
        
        # Remove duplicates
        df = df.drop_duplicates(subset='post_id')
        
        logger.info(f"Analyzed {len(df)} unique Reddit posts")
        return df
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate summary statistics from sentiment DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Sentiment results from get_budget_sentiment
            
        Returns
        -------
        dict
            Summary statistics
        """
        if df.empty:
            return {}
        
        # Engagement-weighted sentiment
        total_engagement = df['engagement'].sum()
        if total_engagement > 0:
            weighted_sentiment = (df['sentiment_compound'] * df['engagement']).sum() / total_engagement
        else:
            weighted_sentiment = df['sentiment_compound'].mean()
        
        return {
            'n_posts': len(df),
            'avg_sentiment': df['sentiment_compound'].mean(),
            'weighted_sentiment': weighted_sentiment,
            'std_sentiment': df['sentiment_compound'].std(),
            'positive_pct': (df['sentiment_compound'] > 0.05).mean(),
            'negative_pct': (df['sentiment_compound'] < -0.05).mean(),
            'total_engagement': total_engagement,
            'top_subreddit': df.groupby('subreddit')['engagement'].sum().idxmax() if len(df) > 0 else None
        }


def analyze_pre_budget_reddit_sentiment(
    client_id: str = None,
    client_secret: str = None
) -> Dict[str, float]:
    """
    Main function to get pre-Budget Reddit sentiment.
    
    Parameters
    ----------
    client_id : str
        Reddit client ID
    client_secret : str
        Reddit client secret
        
    Returns
    -------
    dict
        Sentiment summary
    """
    analyzer = RedditSentimentAnalyzer(client_id, client_secret)
    
    if not analyzer.reddit:
        logger.warning("Reddit client not available. Using mock data.")
        return create_mock_reddit_sentiment()
    
    df = analyzer.get_budget_sentiment()
    return analyzer.get_sentiment_summary(df)


def create_mock_reddit_sentiment() -> Dict[str, float]:
    """
    Create mock Reddit sentiment data for demonstration.
    
    Returns
    -------
    dict
        Mock sentiment summary
    """
    np.random.seed(42)
    
    mock_data = {
        'n_posts': np.random.randint(50, 200),
        'avg_sentiment': np.random.uniform(0.1, 0.3),  # Generally optimistic pre-Budget
        'weighted_sentiment': np.random.uniform(0.15, 0.35),
        'std_sentiment': np.random.uniform(0.2, 0.4),
        'positive_pct': np.random.uniform(0.5, 0.7),
        'negative_pct': np.random.uniform(0.1, 0.25),
        'total_engagement': np.random.randint(500, 5000),
        'top_subreddit': 'IndiaInvestments'
    }
    
    logger.info("Created mock Reddit sentiment data")
    return mock_data

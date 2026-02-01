"""
Reddit Sentiment Scraper
Uses PRAW (Python Reddit API Wrapper) to scrape Indian finance subreddits.

SETUP:
1. Go to https://www.reddit.com/prefs/apps
2. Create a "script" type app
3. Set environment variables:
   - REDDIT_CLIENT_ID=your_client_id
   - REDDIT_CLIENT_SECRET=your_client_secret
   - REDDIT_USER_AGENT=NewsAnalyser/1.0
"""
import os
import time
from typing import Dict, List, Optional
from collections import Counter

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    print("PRAW not installed. Reddit integration unavailable.")


# Subreddits to monitor
FINANCE_SUBREDDITS = [
    'IndiaInvestments',
    'IndianStreetBets', 
    'IndiaStocks',
    'DalalStreetBets'
]

# Cache for 10 minutes
_reddit_cache: Dict = {}
_cache_ttl = 600


def get_reddit_client() -> Optional['praw.Reddit']:
    """Initialize Reddit client from environment variables."""
    if not PRAW_AVAILABLE:
        return None
    
    client_id = os.environ.get('REDDIT_CLIENT_ID')
    client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
    user_agent = os.environ.get('REDDIT_USER_AGENT', 'NewsAnalyser/1.0')
    
    if not client_id or not client_secret:
        print("Reddit credentials not configured. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET.")
        return None
    
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        return reddit
    except Exception as e:
        print(f"Failed to initialize Reddit client: {e}")
        return None


def scrape_subreddit_posts(subreddit_name: str, limit: int = 25) -> List[Dict]:
    """
    Scrape hot posts from a subreddit.
    Returns list of {title, score, comments, created_utc, url}
    """
    reddit = get_reddit_client()
    if not reddit:
        return []
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        posts = []
        
        for post in subreddit.hot(limit=limit):
            posts.append({
                'title': post.title,
                'score': post.score,
                'comments': post.num_comments,
                'created_utc': post.created_utc,
                'url': post.url,
                'subreddit': subreddit_name
            })
        
        return posts
    except Exception as e:
        print(f"Error scraping r/{subreddit_name}: {e}")
        return []


def get_reddit_sentiment() -> Dict:
    """
    Get overall sentiment from Indian finance subreddits.
    Returns aggregated data with bullish/bearish indicators.
    """
    # Check cache
    cache_key = 'reddit_sentiment'
    if cache_key in _reddit_cache:
        cached = _reddit_cache[cache_key]
        if time.time() - cached['timestamp'] < _cache_ttl:
            return cached['data']
    
    all_posts = []
    for sub in FINANCE_SUBREDDITS:
        posts = scrape_subreddit_posts(sub, limit=15)
        all_posts.extend(posts)
    
    if not all_posts:
        return {
            'available': False,
            'error': 'No Reddit data available. Check API credentials.'
        }
    
    # Simple sentiment keywords
    bullish_keywords = ['buy', 'bull', 'moon', 'green', 'long', 'up', 'rally', 'breakout', 'undervalued']
    bearish_keywords = ['sell', 'bear', 'crash', 'red', 'short', 'down', 'dump', 'overvalued', 'panic']
    
    bullish_count = 0
    bearish_count = 0
    mentioned_stocks = Counter()
    
    for post in all_posts:
        title_lower = post['title'].lower()
        
        # Count sentiment
        for kw in bullish_keywords:
            if kw in title_lower:
                bullish_count += post['score'] + 1  # Weight by upvotes
                break
        
        for kw in bearish_keywords:
            if kw in title_lower:
                bearish_count += post['score'] + 1
                break
        
        # Extract stock mentions
        for stock in ['TCS', 'INFY', 'RELIANCE', 'HDFC', 'NIFTY', 'SENSEX', 'ADANI', 'TATA']:
            if stock.lower() in title_lower or stock in post['title']:
                mentioned_stocks[stock] += 1
    
    total = bullish_count + bearish_count
    if total == 0:
        sentiment_score = 0
    else:
        sentiment_score = (bullish_count - bearish_count) / total
    
    result = {
        'available': True,
        'sentiment_score': round(sentiment_score, 3),
        'sentiment_label': 'bullish' if sentiment_score > 0.2 else 'bearish' if sentiment_score < -0.2 else 'neutral',
        'bullish_signals': bullish_count,
        'bearish_signals': bearish_count,
        'posts_analyzed': len(all_posts),
        'top_mentioned': mentioned_stocks.most_common(5),
        'sample_posts': [p['title'] for p in sorted(all_posts, key=lambda x: x['score'], reverse=True)[:5]]
    }
    
    # Cache it
    _reddit_cache[cache_key] = {
        'timestamp': time.time(),
        'data': result
    }
    
    return result

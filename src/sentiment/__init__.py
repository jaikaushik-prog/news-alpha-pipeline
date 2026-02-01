"""
Sentiment Analysis Package.

This package contains modules for analyzing external sentiment signals
to enhance budget impact analysis.
"""

from .twitter_sentiment import (
    TwitterSentimentAnalyzer,
    analyze_pre_budget_twitter_sentiment,
    create_mock_twitter_sentiment,
    BUDGET_KEYWORDS
)

from .reddit_sentiment import (
    RedditSentimentAnalyzer,
    analyze_pre_budget_reddit_sentiment,
    create_mock_reddit_sentiment,
    SUBREDDITS
)

__all__ = [
    # Twitter
    'TwitterSentimentAnalyzer',
    'analyze_pre_budget_twitter_sentiment',
    'create_mock_twitter_sentiment',
    'BUDGET_KEYWORDS',
    # Reddit
    'RedditSentimentAnalyzer',
    'analyze_pre_budget_reddit_sentiment',
    'create_mock_reddit_sentiment',
    'SUBREDDITS'
]

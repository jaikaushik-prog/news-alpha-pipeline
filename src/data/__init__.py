"""
Data package initialization.
"""
from .news_collector import (
    NewsItem,
    collect_all_news,
    get_latest_news,
    classify_category,
    filter_by_category,
    to_dataframe
)

__all__ = [
    'NewsItem',
    'collect_all_news',
    'get_latest_news',
    'classify_category',
    'filter_by_category',
    'to_dataframe'
]

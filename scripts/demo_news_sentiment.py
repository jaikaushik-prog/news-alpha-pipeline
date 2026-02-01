#!/usr/bin/env python
"""
Demo: News Sentiment Analyzer for Investment Decisions

Analyzes market sentiment from Indian financial news before investing.

Usage:
    python scripts/demo_news_sentiment.py
    
Requirements:
    pip install feedparser beautifulsoup4 requests
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("📰 NEWS SENTIMENT ANALYZER FOR INVESTING")
print("=" * 60)
print()

# =========================================================================
# 1. Try Live News Scraping
# =========================================================================
print("🔄 Fetching live news headlines...")
print("-" * 40)

try:
    from src.sentiment.news_sentiment import (
        get_market_sentiment,
        interpret_for_investing,
        create_mock_news_sentiment
    )
    
    # Try live scraping
    try:
        sentiment = get_market_sentiment(headlines_per_source=10)
        
        if 'error' in sentiment:
            raise Exception(sentiment['error'])
            
        print(f"\n✓ Analyzed {sentiment['headline_count']} headlines from live sources")
        
    except Exception as e:
        print(f"\n⚠️ Live scraping unavailable: {e}")
        print("Using mock data for demonstration...")
        sentiment = create_mock_news_sentiment()
    
    # =========================================================================
    # 2. Display Sentiment Analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("📊 MARKET SENTIMENT ANALYSIS")
    print("=" * 60)
    
    print(f"\n  📈 Overall Score: {sentiment['overall_score']:.3f}")
    print(f"  📊 Regime: {sentiment['sentiment_regime'].upper()}")
    print(f"  🟢 Bullish Headlines: {sentiment['bullish_count']} ({sentiment['bullish_ratio']:.0%})")
    print(f"  🔴 Bearish Headlines: {sentiment['bearish_count']} ({sentiment['bearish_ratio']:.0%})")
    print(f"  🟡 Neutral Headlines: {sentiment['neutral_count']}")
    
    # =========================================================================
    # 3. Sector Breakdown
    # =========================================================================
    print("\n" + "-" * 40)
    print("📂 SECTOR SENTIMENT:")
    print("-" * 40)
    
    sector_sentiment = sentiment.get('sector_sentiment', {})
    
    if sector_sentiment:
        # Sort by sentiment score
        sorted_sectors = sorted(sector_sentiment.items(), key=lambda x: x[1], reverse=True)
        
        for sector, score in sorted_sectors:
            emoji = "🟢" if score > 0.1 else ("🔴" if score < -0.1 else "🟡")
            print(f"  {emoji} {sector:20s}: {score:+.3f}")
    else:
        print("  No sector-specific data available")
    
    # =========================================================================
    # 4. Top Headlines
    # =========================================================================
    print("\n" + "-" * 40)
    print("📰 TOP BULLISH HEADLINES:")
    print("-" * 40)
    
    for headline in sentiment.get('top_bullish', [])[:3]:
        print(f"  ✅ {headline[:60]}...")
    
    print("\n" + "-" * 40)
    print("📰 TOP BEARISH HEADLINES:")
    print("-" * 40)
    
    for headline in sentiment.get('top_bearish', [])[:3]:
        print(f"  ❌ {headline[:60]}...")
    
    # =========================================================================
    # 5. Investment Recommendation
    # =========================================================================
    print("\n" + "=" * 60)
    print("💰 INVESTMENT INTERPRETATION")
    print("=" * 60)
    
    interpretation = interpret_for_investing(sentiment)
    print(f"\n{interpretation}")
    
except ImportError as e:
    print(f"\n❌ Missing dependencies: {e}")
    print("\nInstall with:")
    print("  pip install feedparser beautifulsoup4 requests")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETE")
print("=" * 60)
print("\nTip: Run this before market opens to gauge sentiment!")

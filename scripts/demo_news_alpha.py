#!/usr/bin/env python
"""
Demo: News Alpha Pipeline

Runs the complete institutional-grade news sentiment pipeline:
Information → Surprise → Belief → Market Reaction → Alpha

Usage:
    python scripts/demo_news_alpha.py
    
Requirements:
    pip install feedparser beautifulsoup4 requests
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("🏛️  INSTITUTIONAL NEWS ALPHA PIPELINE")
print("=" * 70)
print()

# =========================================================================
# Run Pipeline
# =========================================================================
try:
    from src.pipeline.news_alpha_pipeline import NewsAlphaPipeline
    
    # Initialize pipeline
    pipeline = NewsAlphaPipeline(
        use_embeddings=False,  # Set True if sentence-transformers installed
        long_threshold=0.25,
        short_threshold=0.75
    )
    
    # Run analysis
    print("🔄 Running pipeline...")
    print("-" * 50)
    
    result = pipeline.run(max_headlines=30, max_age_hours=24)
    
    # =========================================================================
    # Display Results
    # =========================================================================
    pipeline.print_summary(result)
    
    # =========================================================================
    # Detailed Alpha Signals
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 DETAILED ALPHA SIGNALS")
    print("=" * 70)
    
    if not result.alpha_signals.empty:
        print("\n  Sector Rankings by Alpha Score:\n")
        
        for _, row in result.alpha_signals.iterrows():
            if row['position'] == 'long':
                emoji = "🟢 LONG"
            elif row['position'] == 'short':
                emoji = "🔴 SHORT"
            else:
                emoji = "⚪ NEUTRAL"
            
            print(f"  {row['rank']:2d}. {row['sector']:20s} | "
                  f"Alpha: {row['raw_alpha']:+.4f} | "
                  f"{emoji:12s} | "
                  f"Weight: {row['weight']:+.2%}")
    
    # =========================================================================
    # Investment Recommendation
    # =========================================================================
    print("\n" + "=" * 70)
    print("💰 INVESTMENT RECOMMENDATION")
    print("=" * 70)
    
    regime = result.regime
    sentiment = result.effective_sentiment
    
    if regime == 'risk_on':
        print(f"""
  🟢 REGIME: RISK-ON (Sentiment: {sentiment:+.3f})
  
  The news flow is predominantly bullish. Consider:
  • Increase exposure to high-beta sectors
  • Long top-ranked sectors: {', '.join(result.portfolio_recommendation['long_sectors'])}
  • Reduce hedges
        """)
    elif regime == 'risk_off':
        print(f"""
  🔴 REGIME: RISK-OFF (Sentiment: {sentiment:+.3f})
  
  The news flow is predominantly bearish. Consider:
  • Reduce equity exposure
  • Rotate to defensive sectors (FMCG, Pharma)
  • Increase hedges
  • Avoid: {', '.join(result.portfolio_recommendation['short_sectors'])}
        """)
    else:
        print(f"""
  🟡 REGIME: TRANSITIONAL (Sentiment: {sentiment:+.3f})
  
  Mixed news flow signals uncertainty. Consider:
  • Sector rotation over directional bets
  • Long/Short pairs within sectors
  • Focus on fundamentals
        """)
    
    # =========================================================================
    # Sector Heatmap
    # =========================================================================
    print("\n" + "=" * 70)
    print("🗺️  SECTOR SENTIMENT HEATMAP")
    print("=" * 70)
    print()
    
    for sector, score in result.sector_ranking:
        # Create visual bar
        bar_len = int(abs(score) * 30)
        if score > 0:
            bar = "█" * bar_len
            print(f"  {sector:20s} |{'':>15}{bar} +{score:.3f}")
        else:
            bar = "█" * bar_len
            print(f"  {sector:20s} |{bar:>15} {score:.3f}")
    
    print()

except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nMake sure you're running from the project root:")
    print("  cd c:\\Users\\DELL\\Desktop\\budget_speech")
    print("  python scripts/demo_news_alpha.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ PIPELINE DEMO COMPLETE")
print("=" * 70)
print("\nNext Steps:")
print("  1. Install feedparser for live news: pip install feedparser")
print("  2. Install sentence-transformers for embeddings: pip install sentence-transformers")
print("  3. Run before market opens for daily sentiment check")

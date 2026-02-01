#!/usr/bin/env python
"""
Demo: Holy Trinity - Elite Differentiators

The combination that separates you completely:
1. Expectation Gap (surprise vs baseline)
2. Narrative Velocity (information diffusion)
3. Sentiment-Price Divergence (smart money positioning)

Usage:
    python scripts/demo_holy_trinity.py
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("🏆 HOLY TRINITY - INSTITUTIONAL-GRADE ALPHA")
print("=" * 70)
print()
print("\"I modeled expectation-adjusted information shocks and measured")
print("how markets absorb or reject them under different regimes.\"")
print()

# =========================================================================
# Demo Holy Trinity
# =========================================================================
try:
    from src.models.holy_trinity import HolyTrinityModel, trinity_to_dataframe
    
    # Initialize model
    model = HolyTrinityModel()
    
    # Simulate 20 days of historical data for baseline
    print("📊 Training expectation baseline (20 days of history)...")
    print("-" * 50)
    
    import numpy as np
    np.random.seed(42)
    
    sectors = ['banking', 'it', 'pharma', 'auto', 'metals', 'infra']
    
    # Build historical baseline
    for day in range(20):
        for sector in sectors:
            # Simulate historical sentiment
            base = {'banking': 0.1, 'it': 0.05, 'pharma': -0.02, 
                    'auto': 0.08, 'metals': 0.0, 'infra': 0.15}
            sentiment = base.get(sector, 0) + np.random.normal(0, 0.1)
            
            model.update_historical(
                sector=sector,
                sentiment=sentiment,
                volume=10 + np.random.randint(-3, 5),
                intensity=0.5 + np.random.normal(0, 0.1)
            )
    
    print("   ✓ Expectation model trained")
    print("   ✓ Narrative velocity tracker initialized")
    print()
    
    # =========================================================================
    # Today's Scenario (with surprises!)
    # =========================================================================
    print("=" * 70)
    print("📅 TODAY'S MARKET - ANALYZING HOLY TRINITY SIGNALS")
    print("=" * 70)
    print()
    
    # Construct today's data with interesting scenarios
    today_data = {
        'banking': {
            'sentiment': 0.65,      # SURPRISE: Much higher than usual!
            'volume': 35,           # High volume
            'intensity': 0.7,
            'price_return': -1.2,   # But price FELL (DIVERGENCE!)
            'volume_ratio': 1.8
        },
        'it': {
            'sentiment': -0.3,      # Negative news
            'volume': 25,
            'intensity': 0.5,
            'price_return': 1.5,    # But price UP (hidden strength!)
            'volume_ratio': 1.4
        },
        'pharma': {
            'sentiment': 0.1,       # Mildly positive
            'volume': 8,
            'intensity': 0.4,
            'price_return': 0.3,    # Confirmation
            'volume_ratio': 0.9
        },
        'auto': {
            'sentiment': 0.45,      # SURPRISE: Positive vs baseline
            'volume': 18,           # Increasing (early mover)
            'intensity': 0.6,
            'price_return': 0.8,
            'volume_ratio': 1.1
        },
        'metals': {
            'sentiment': -0.4,      # Negative surprise
            'volume': 12,
            'intensity': 0.5,
            'price_return': -2.1,   # Confirmation
            'volume_ratio': 1.2
        },
        'infra': {
            'sentiment': 0.2,       # In-line with expectation
            'volume': 15,           # Stable
            'intensity': 0.5,
            'price_return': 0.5,
            'volume_ratio': 1.0
        }
    }
    
    # Analyze with Holy Trinity
    signals = model.batch_analyze(today_data)
    df = trinity_to_dataframe(signals)
    
    # =========================================================================
    # Display Results
    # =========================================================================
    print("  SECTOR        │ TRINITY  │ CONVICTION │ ACTION       │ KEY INSIGHT")
    print("  " + "─" * 70)
    
    for signal in signals:
        # Emoji based on recommendation
        if 'long' in signal.trade_recommendation:
            emoji = "🟢"
        elif 'short' in signal.trade_recommendation:
            emoji = "🔴"
        else:
            emoji = "⚪"
        
        # Conviction emoji
        conv_emoji = "🔥" if signal.conviction == 'high' else ("⚡" if signal.conviction == 'medium' else "💤")
        
        print(f"  {emoji} {signal.sector:12s} │ {signal.trinity_score:+.3f}  │ {conv_emoji} {signal.conviction:6s}  │ {signal.trade_recommendation:12s} │ {signal.rationale[:40]}")
    
    # =========================================================================
    # Detailed Analysis
    # =========================================================================
    print()
    print("=" * 70)
    print("🔍 DETAILED COMPONENT BREAKDOWN")
    print("=" * 70)
    
    for signal in signals[:3]:  # Top 3
        print(f"\n  📌 {signal.sector.upper()}")
        print(f"  " + "─" * 50)
        print(f"     1️⃣ Expectation Gap: z-score={signal.gap_zscore:+.2f}")
        print(f"        {'⚡ SURPRISE!' if signal.is_surprise else 'In-line with expectations'}")
        print(f"     2️⃣ Narrative Velocity: v={signal.velocity:.2f}, phase={signal.momentum_phase}")
        print(f"        {'🚀 EARLY MOVER!' if signal.is_early_mover else ''}")
        print(f"     3️⃣ Divergence: {signal.divergence_type}")
        print(f"        Smart money: {signal.smart_money_signal}")
        print(f"     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"     📈 TRINITY SCORE: {signal.trinity_score:+.3f} → {signal.trade_recommendation.upper()}")
    
    # =========================================================================
    # Trading Recommendations
    # =========================================================================
    print()
    print("=" * 70)
    print("💰 ACTIONABLE RECOMMENDATIONS")
    print("=" * 70)
    
    longs = [s.sector for s in signals if 'long' in s.trade_recommendation]
    shorts = [s.sector for s in signals if 'short' in s.trade_recommendation]
    high_conviction = [s.sector for s in signals if s.conviction == 'high']
    
    print(f"""
  🟢 LONG CANDIDATES:
     {', '.join(longs) if longs else 'None'}
  
  🔴 SHORT CANDIDATES:
     {', '.join(shorts) if shorts else 'None'}
  
  🔥 HIGH CONVICTION:
     {', '.join(high_conviction) if high_conviction else 'None'}
    """)
    
    # =========================================================================
    # Why This Separates You
    # =========================================================================
    print("=" * 70)
    print("🎓 WHY THIS SEPARATES YOU (INTERVIEW TALKING POINTS)")
    print("=" * 70)
    print("""
  ❌ What everyone else says:
     "I built a sentiment analyzer"
  
  ✅ What YOU say:
     "I modeled expectation-adjusted information shocks and measured
      how markets absorb or reject them under different regimes."
  
  📚 Three Elite Concepts:
  
  1️⃣ EXPECTATION GAP
     → Markets move on DIFFERENCE vs expectation, not raw sentiment
     → Mirrors earnings surprise logic
     → Most projects ignore this completely
  
  2️⃣ NARRATIVE VELOCITY  
     → Second-order thinking (velocity & acceleration)
     → Detects early movers vs priced-in narratives
     → Information diffusion theory
  
  3️⃣ SENTIMENT-PRICE DIVERGENCE
     → When markets disagree with news → future move
     → Detects smart money positioning
     → This is how institutions actually think
    """)

except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nMake sure you're running from the project root:")
    print("  cd c:\\Users\\DELL\\Desktop\\budget_speech")
    print("  python scripts/demo_holy_trinity.py")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("✅ HOLY TRINITY DEMO COMPLETE")
print("=" * 70)

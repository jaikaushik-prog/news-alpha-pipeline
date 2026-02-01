#!/usr/bin/env python
"""
Demo: Pre-Budget Sentiment Analysis

Run this to see the professional-grade sentiment stack in action.

Usage:
    python scripts/demo_prebudget_sentiment.py
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("PRE-BUDGET SENTIMENT ANALYSIS DEMO")
print("=" * 60)

# =========================================================================
# 1. Options Market Sentiment (PCR, IV)
# =========================================================================
print("\n📊 1. OPTIONS MARKET SENTIMENT")
print("-" * 40)

try:
    from src.market.options_sentiment import (
        get_options_sentiment,
        create_mock_options_sentiment
    )
    
    # Try live data first, fallback to mock
    nifty_sentiment = get_options_sentiment('NIFTY')
    
    if not nifty_sentiment:
        print("  Using mock data (live API unavailable)...")
        mock_df = create_mock_options_sentiment()
        print(mock_df[['sector', 'pcr_oi', 'atm_iv', 'options_sentiment_score']].to_string())
    else:
        print(f"  PCR (OI): {nifty_sentiment.get('pcr_oi', 'N/A'):.2f}")
        print(f"  ATM IV: {nifty_sentiment.get('atm_iv', 'N/A'):.1f}%")
        print(f"  IV Skew: {nifty_sentiment.get('iv_skew', 'N/A'):.2f}")
        print(f"  Sentiment Score: {nifty_sentiment.get('options_sentiment_score', 'N/A'):.3f}")
        print(f"  Interpretation: {nifty_sentiment.get('pcr_interpretation', 'N/A')}")
        
except Exception as e:
    print(f"  ⚠️ Options data unavailable: {e}")
    print("  → Run with mock data instead")

# =========================================================================
# 2. FII/DII Flow Intensity
# =========================================================================
print("\n💰 2. FII/DII FLOW INTENSITY")
print("-" * 40)

try:
    from src.market.flow_intensity import (
        create_mock_fii_dii_data,
        calculate_flow_intensity,
        classify_flow_regime
    )
    
    # Create mock data for demo
    fii_dii_df = create_mock_fii_dii_data('2025-01-01', '2025-02-01')
    fii_dii_df = calculate_flow_intensity(fii_dii_df)
    
    latest = fii_dii_df.iloc[-1]
    
    print(f"  FII Net: ₹{latest['fii_net']:,.0f} Cr")
    print(f"  DII Net: ₹{latest['dii_net']:,.0f} Cr")
    print(f"  FII Intensity: {latest['fii_intensity']:.3f}")
    print(f"  DII Intensity: {latest['dii_intensity']:.3f}")
    
    regime = classify_flow_regime(latest['fii_net'], latest['dii_net'])
    print(f"  Combined Regime: {regime['combined_regime']}")
    
except Exception as e:
    print(f"  ⚠️ Flow data error: {e}")

# =========================================================================
# 3. Google Trends Attention
# =========================================================================
print("\n🔍 3. GOOGLE TRENDS ATTENTION")
print("-" * 40)

try:
    from src.sentiment.google_trends import (
        get_pre_budget_attention_signal,
        create_mock_trends_data
    )
    
    # Try live first, fallback to mock
    try:
        attention = get_pre_budget_attention_signal('2025-02-01')
        if attention.get('attention_level') == 'unknown':
            raise Exception("No data")
        print(f"  Attention Level: {attention['attention_level']}")
        print(f"  Avg Spike: {attention.get('avg_spike_pct', 0):.1f}%")
        print(f"  Interpretation: {attention.get('interpretation', 'N/A')[:60]}...")
    except:
        print("  Using mock data...")
        mock_trends = create_mock_trends_data(['general', 'tax', 'infrastructure'])
        print(mock_trends[['sector', 'avg_spike_pct', 'attention_score']].to_string())
        
except Exception as e:
    print(f"  ⚠️ Trends error: {e}")

# =========================================================================
# 4. Pre-Budget Confidence Score (COMPOSITE)
# =========================================================================
print("\n🎯 4. PRE-BUDGET CONFIDENCE SCORE")
print("-" * 40)

try:
    from src.models.confidence_score import (
        get_pre_budget_confidence,
        create_mock_confidence_score,
        get_signal_matrix,
        interaction_term
    )
    
    # Generate mock confidence for demo
    confidence = create_mock_confidence_score('2025-02-01')
    
    print(f"  CONFIDENCE SCORE: {confidence['confidence_score']:.3f}")
    print(f"  Regime: {confidence['confidence_regime']}")
    print(f"  Components:")
    print(f"    - PCR Input: {confidence['components']['pcr_input']}")
    print(f"    - IV Input: {confidence['components']['iv_input']}")
    print(f"    - Flow Input: {confidence['components']['flow_input']}")
    print(f"\n  📌 Strategy: {confidence['strategy_recommendation'][:70]}...")
    
except Exception as e:
    print(f"  ⚠️ Confidence score error: {e}")

# =========================================================================
# 5. Signal Matrix
# =========================================================================
print("\n📋 5. SIGNAL INTERPRETATION MATRIX")
print("-" * 40)

try:
    matrix = get_signal_matrix()
    print(matrix.to_string(index=False))
except Exception as e:
    print(f"  ⚠️ Matrix error: {e}")

# =========================================================================
# 6. Interaction Term Demo
# =========================================================================
print("\n🔥 6. INTERACTION TERM DEMO")
print("-" * 40)

try:
    # Example: Bullish confidence + Positive mention
    confidence_score = 0.4  # Bullish
    mention_sentiment = 0.6  # Positive
    
    signal = interaction_term(confidence_score, mention_sentiment)
    
    print(f"  Confidence Score: {confidence_score}")
    print(f"  Mention Sentiment: {mention_sentiment}")
    print(f"  Interaction Signal: {signal:.3f}")
    print(f"  → {'STRONG' if abs(signal) > 0.2 else 'WEAK'} {'BUY' if signal > 0 else 'SELL'} signal")
    
except Exception as e:
    print(f"  ⚠️ Interaction error: {e}")

print("\n" + "=" * 60)
print("DEMO COMPLETE")
print("=" * 60)
print("\nThis professional-grade stack replaces Twitter/Reddit sentiment")
print("with options-implied signals that reflect REAL CAPITAL positions.")

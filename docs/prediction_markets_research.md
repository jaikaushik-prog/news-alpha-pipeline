# India Prediction Markets Research

## Overview
This document explores prediction market platforms in India that could be used
for pre-Budget sentiment analysis.

## Available Platforms

### 1. Probo (probo.in)
- **Type**: Opinion trading platform
- **Events**: Cricket, politics, news, entertainment
- **Budget Coverage**: Limited — may have general election/policy markets
- **API Access**: No public API available
- **Data Collection**: Manual scraping only (against ToS)
- **Recommendation**: Not suitable for automated analysis

### 2. MPL (Mobile Premier League)
- **Type**: Gaming/fantasy platform
- **Events**: Sports primarily
- **Budget Coverage**: None
- **API Access**: No
- **Recommendation**: Not applicable

### 3. TradeX
- **Type**: Prediction market for sports
- **Budget Coverage**: None
- **Recommendation**: Not applicable

## International Alternatives

### Polymarket (polymarket.com)
- **Type**: Crypto-based prediction market
- **Events**: Global politics, crypto, macro
- **Budget Coverage**: No India-specific Budget markets
- **API Access**: Yes (REST + WebSocket)
- **Recommendation**: Can monitor related macro events (Fed, global risk)

### PredictIt (predictit.org)
- **Type**: US-focused political prediction market
- **Budget Coverage**: No India events
- **API Access**: Limited public data
- **Recommendation**: Not applicable

## Conclusion

**There is no reliable prediction market for Indian Union Budget outcomes.**

### Alternative Signals for Pre-Budget Sentiment:

1. **India VIX** ✅ (Implemented in `src/market/option_iv.py`)
   - Best proxy for market fear/uncertainty before Budget

2. **Twitter/X Sentiment** ✅ (Implemented in `src/sentiment/twitter_sentiment.py`)
   - Sector-wise public opinion tracking

3. **FII/DII Flows** (Future)
   - Institutional positioning before Budget

4. **Nifty Option Chain OI** (Future)
   - Put/Call ratio, max pain, IV skew

5. **Brokerage Research Sentiment** (Future)
   - Scrape pre-Budget reports from ICICI, HDFC, Motilal

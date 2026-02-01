"""
News Alpha Pipeline - FastAPI Backend
Live news analysis with RSS feeds
"""
import os
import sys
import random
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(title="News Alpha Pipeline", version="2.0.0")

# ============== RSS NEWS FETCHER ==============
import urllib.request
import xml.etree.ElementTree as ET

RSS_FEEDS = {
    'economic_times': [
        'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
        'https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms',
    ],
    'moneycontrol': [
        'https://www.moneycontrol.com/rss/latestnews.xml',
        'https://www.moneycontrol.com/rss/marketreports.xml',
    ],
    'livemint': [
        'https://www.livemint.com/rss/markets',
        'https://www.livemint.com/rss/companies',
    ],
}

def fetch_rss_headlines(max_items: int = 50) -> List[Dict]:
    """Fetch live headlines from RSS feeds"""
    headlines = []
    seen = set()
    
    for source, feeds in RSS_FEEDS.items():
        for feed_url in feeds:
            try:
                req = urllib.request.Request(
                    feed_url,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    content = response.read()
                    root = ET.fromstring(content)
                    
                    for item in root.findall('.//item')[:10]:
                        title = item.find('title')
                        if title is not None and title.text:
                            text = title.text.strip()
                            h = hashlib.md5(text.encode()).hexdigest()
                            if h not in seen:
                                seen.add(h)
                                headlines.append({
                                    'headline': text,
                                    'source': source,
                                    'timestamp': datetime.now().isoformat()
                                })
            except Exception as e:
                print(f"Error fetching {feed_url}: {e}")
                continue
    
    return headlines[:max_items]

# ============== SENTIMENT ANALYSIS ==============
POSITIVE_WORDS = {'surge', 'jump', 'gain', 'rise', 'boost', 'profit', 'growth', 'rally', 'bullish', 'up', 'high', 'record', 'strong', 'positive', 'optimistic', 'buy', 'outperform', 'upgrade', 'beat', 'exceed'}
NEGATIVE_WORDS = {'fall', 'drop', 'crash', 'decline', 'loss', 'plunge', 'bearish', 'down', 'low', 'weak', 'negative', 'pessimistic', 'sell', 'underperform', 'downgrade', 'miss', 'fail', 'cut', 'slash', 'crisis'}

def analyze_sentiment(text: str) -> float:
    """Simple but effective sentiment analysis"""
    words = text.lower().split()
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total

# ============== SECTOR DETECTION ==============
SECTOR_KEYWORDS = {
    'banking': ['bank', 'hdfc', 'icici', 'sbi', 'kotak', 'axis', 'rbi', 'npa', 'credit', 'loan', 'deposit'],
    'it': ['it', 'tcs', 'infosys', 'wipro', 'tech', 'software', 'digital', 'cloud', 'ai', 'hcl', 'cognizant'],
    'pharma': ['pharma', 'drug', 'medicine', 'cipla', 'sun', 'lupin', 'dr reddy', 'biocon', 'vaccine', 'fda'],
    'auto': ['auto', 'car', 'vehicle', 'tata motors', 'mahindra', 'maruti', 'ev', 'electric', 'bajaj', 'hero'],
    'metals': ['metal', 'steel', 'tata steel', 'jsw', 'hindalco', 'vedanta', 'aluminium', 'copper', 'zinc', 'iron'],
    'infra': ['infra', 'construction', 'cement', 'larsen', 'ultratech', 'acc', 'ambuja', 'road', 'highway', 'real estate'],
    'fmcg': ['fmcg', 'itc', 'hindustan unilever', 'nestle', 'britannia', 'dabur', 'consumer', 'retail'],
    'energy': ['oil', 'gas', 'reliance', 'ongc', 'bpcl', 'ioc', 'power', 'coal', 'energy', 'renewable', 'solar'],
}

def detect_sectors(text: str) -> List[str]:
    """Detect which sectors a headline relates to"""
    text_lower = text.lower()
    sectors = []
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            sectors.append(sector)
    return sectors if sectors else ['general']

# ============== HOLY TRINITY MODEL ==============
def calculate_trinity_score(sector: str, sentiment: float, news_count: int) -> Dict:
    """Calculate Holy Trinity score for a sector"""
    # 1. Expectation Gap (sentiment surprise)
    baseline = 0.0  # neutral baseline
    expectation_gap = sentiment - baseline
    
    # 2. Narrative Velocity (news volume indicator)
    velocity = min(1.0, news_count / 10)
    narrative_velocity = velocity * abs(sentiment)
    
    # 3. Sentiment-Price Divergence (simulated for now)
    divergence = random.uniform(-0.2, 0.2) * sentiment
    
    # Combine scores
    trinity_score = (0.4 * expectation_gap) + (0.35 * narrative_velocity) + (0.25 * divergence)
    
    # Determine recommendation
    if trinity_score > 0.3:
        recommendation = 'strong_long'
        conviction = 'high'
    elif trinity_score > 0.1:
        recommendation = 'long'
        conviction = 'medium'
    elif trinity_score < -0.3:
        recommendation = 'strong_short'
        conviction = 'high'
    elif trinity_score < -0.1:
        recommendation = 'short'
        conviction = 'medium'
    else:
        recommendation = 'hold'
        conviction = 'low'
    
    rationale = []
    if abs(expectation_gap) > 0.2:
        rationale.append("Sentiment surprise detected")
    if narrative_velocity > 0.3:
        rationale.append("High narrative velocity")
    if abs(divergence) > 0.1:
        rationale.append("Price divergence signal")
    
    return {
        'trinity_score': round(trinity_score, 3),
        'recommendation': recommendation,
        'conviction': conviction,
        'rationale': '; '.join(rationale) if rationale else 'Normal market conditions'
    }

# ============== MAIN ANALYSIS ==============
def run_live_analysis() -> Dict:
    """Run full live analysis pipeline"""
    # Fetch live headlines
    headlines = fetch_rss_headlines(50)
    
    if not headlines:
        raise HTTPException(status_code=503, detail="No news feeds available")
    
    # Analyze each headline
    sector_data = {}
    for h in headlines:
        sentiment = analyze_sentiment(h['headline'])
        sectors = detect_sectors(h['headline'])
        
        for sector in sectors:
            if sector not in sector_data:
                sector_data[sector] = {'sentiments': [], 'count': 0}
            sector_data[sector]['sentiments'].append(sentiment)
            sector_data[sector]['count'] += 1
    
    # Calculate sector signals
    sector_signals = []
    for sector, data in sector_data.items():
        if sector == 'general':
            continue
        avg_sentiment = sum(data['sentiments']) / len(data['sentiments']) if data['sentiments'] else 0
        trinity = calculate_trinity_score(sector, avg_sentiment, data['count'])
        
        sector_signals.append({
            'sector': sector,
            'sentiment': round(avg_sentiment, 2),
            'trinity_score': trinity['trinity_score'],
            'recommendation': trinity['recommendation'],
            'conviction': trinity['conviction'],
            'rationale': trinity['rationale'],
            'news_count': data['count']
        })
    
    # Sort by absolute trinity score
    sector_signals.sort(key=lambda x: abs(x['trinity_score']), reverse=True)
    
    # Calculate overall sentiment
    all_sentiments = [s['sentiment'] for s in sector_signals]
    effective_sentiment = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0
    
    # Determine regime
    if effective_sentiment > 0.15:
        regime = 'bullish'
    elif effective_sentiment < -0.15:
        regime = 'bearish'
    else:
        regime = 'neutral'
    
    # Get recommendations
    long_sectors = [s['sector'] for s in sector_signals if 'long' in s['recommendation']]
    short_sectors = [s['sector'] for s in sector_signals if 'short' in s['recommendation']]
    high_conviction = [s['sector'] for s in sector_signals if s['conviction'] == 'high']
    
    return {
        'timestamp': datetime.now().isoformat(),
        'regime': regime,
        'effective_sentiment': round(effective_sentiment, 2),
        'sector_signals': sector_signals,
        'long_sectors': long_sectors,
        'short_sectors': short_sectors,
        'high_conviction': high_conviction,
        'headlines_analyzed': len(headlines),
        'sources': list(set(h['source'] for h in headlines))
    }

# ============== API ENDPOINTS ==============
class AnalysisRequest(BaseModel):
    max_headlines: int = 50
    max_age_hours: int = 24

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard"""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    raise HTTPException(status_code=404, detail="Dashboard not found")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/analyze")
async def analyze_news(request: AnalysisRequest = None):
    """Run live analysis on current news"""
    try:
        result = run_live_analysis()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/demo")
async def get_demo_data():
    """Return demo data for testing"""
    return {
        "timestamp": datetime.now().isoformat(),
        "regime": "bullish",
        "effective_sentiment": 0.35,
        "sector_signals": [
            {"sector": "auto", "sentiment": 0.45, "trinity_score": 0.675, "recommendation": "strong_long", "conviction": "high", "rationale": "Positive surprise; early mover", "news_count": 8},
            {"sector": "banking", "sentiment": 0.65, "trinity_score": 0.475, "recommendation": "strong_long", "conviction": "high", "rationale": "Positive surprise", "news_count": 12},
            {"sector": "it", "sentiment": -0.30, "trinity_score": -0.475, "recommendation": "strong_short", "conviction": "high", "rationale": "Negative surprise", "news_count": 6},
            {"sector": "metals", "sentiment": -0.40, "trinity_score": -0.375, "recommendation": "short", "conviction": "medium", "rationale": "Negative surprise", "news_count": 4},
            {"sector": "pharma", "sentiment": 0.10, "trinity_score": 0.304, "recommendation": "long", "conviction": "medium", "rationale": "Modest surprise", "news_count": 5},
            {"sector": "infra", "sentiment": 0.05, "trinity_score": 0.060, "recommendation": "hold", "conviction": "low", "rationale": "In-line with expectations", "news_count": 3},
        ],
        "long_sectors": ["auto", "banking", "pharma"],
        "short_sectors": ["it", "metals"],
        "high_conviction": ["auto", "banking", "it"],
        "headlines_analyzed": 47,
        "sources": ["economic_times", "moneycontrol", "livemint"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

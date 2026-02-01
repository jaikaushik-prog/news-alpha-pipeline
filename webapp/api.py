"""
News Alpha Pipeline - FastAPI Backend V2
Live news analysis with expanded RSS feeds including Zerodha Pulse sources,
Google Trends integration, and more Indian financial news outlets.
"""
import os
import sys
import random
import hashlib
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(title="News Alpha Pipeline", version="3.0.0")

# ============== EXPANDED RSS NEWS FEEDS ==============
# Sources aligned with Zerodha Pulse aggregation
import urllib.request
import xml.etree.ElementTree as ET
import json

RSS_FEEDS = {
    # Primary Indian Financial News
    'economic_times': [
        'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
        'https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms',
        'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
        'https://economictimes.indiatimes.com/wealth/rssfeeds/837555174.cms',
    ],
    'moneycontrol': [
        'https://www.moneycontrol.com/rss/latestnews.xml',
        'https://www.moneycontrol.com/rss/marketreports.xml',
        'https://www.moneycontrol.com/rss/business.xml',
    ],
    'livemint': [
        'https://www.livemint.com/rss/markets',
        'https://www.livemint.com/rss/companies',
        'https://www.livemint.com/rss/money',
    ],
    # Additional Sources (Zerodha Pulse sources)
    'business_standard': [
        'https://www.business-standard.com/rss/latest.rss',
        'https://www.business-standard.com/rss/markets-106.rss',
        'https://www.business-standard.com/rss/companies-101.rss',
    ],
    'the_hindu_business': [
        'https://www.thehindu.com/business/feeder/default.rss',
        'https://www.thehindu.com/business/markets/feeder/default.rss',
        'https://www.thehindu.com/business/Industry/feeder/default.rss',
    ],
    'ndtv_profit': [
        'https://feeds.feedburner.com/ndtvprofit-latest',
    ],
    'reuters_india': [
        'https://feeds.reuters.com/reuters/INbusinessNews',
    ],
    'financial_express': [
        'https://www.financialexpress.com/feed/',
        'https://www.financialexpress.com/market/feed/',
    ],
}

# Google Trends - Trending Finance Topics (India)
GOOGLE_TRENDS_API = "https://trends.google.com/trends/api/dailytrends?hl=en-IN&tz=-330&geo=IN"

def fetch_google_trends() -> List[Dict]:
    """Fetch trending topics from Google Trends India"""
    trends = []
    try:
        req = urllib.request.Request(
            GOOGLE_TRENDS_API,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            content = response.read().decode('utf-8')
            # Remove the anti-XSS prefix
            if content.startswith(")]}'"):
                content = content[5:]
            data = json.loads(content)
            
            for day in data.get('default', {}).get('trendingSearchesDays', []):
                for search in day.get('trendingSearches', [])[:5]:
                    title = search.get('title', {}).get('query', '')
                    if any(kw in title.lower() for kw in ['stock', 'nifty', 'sensex', 'bank', 'market', 'share', 'budget', 'rbi', 'rupee']):
                        trends.append({
                            'headline': title,
                            'source': 'google_trends',
                            'timestamp': datetime.now().isoformat(),
                            'traffic': search.get('formattedTraffic', 'N/A')
                        })
    except Exception as e:
        print(f"Google Trends error: {e}")
    
    return trends[:10]


def fetch_zerodha_pulse() -> List[Dict]:
    """Scrape headlines from Zerodha Pulse"""
    headlines = []
    try:
        req = urllib.request.Request(
            'https://pulse.zerodha.com/',
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html'
            }
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            content = response.read().decode('utf-8', errors='ignore')
            
            # Extract headlines using regex
            # Look for title patterns in the HTML
            title_patterns = [
                r'<h2[^>]*class="title"[^>]*>\s*<a[^>]*>([^<]+)</a>',
                r'<h3[^>]*>\s*<a[^>]*>([^<]+)</a>',
                r'class="title"[^>]*>([^<]+)<',
            ]
            
            seen = set()
            for pattern in title_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches[:20]:
                    title = match.strip()
                    if len(title) > 20 and title not in seen:
                        seen.add(title)
                        headlines.append({
                            'headline': title,
                            'source': 'zerodha_pulse',
                            'timestamp': datetime.now().isoformat()
                        })
                        
    except Exception as e:
        print(f"Zerodha Pulse error: {e}")
    
    return headlines[:15]


def fetch_rss_headlines(max_items: int = 100) -> List[Dict]:
    """Fetch live headlines from all RSS feeds"""
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
                    
                    # Handle different RSS formats
                    items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')
                    
                    for item in items[:8]:
                        title = item.find('title') or item.find('{http://www.w3.org/2005/Atom}title')
                        if title is not None and title.text:
                            text = title.text.strip()
                            # Clean up CDATA and HTML
                            text = re.sub(r'<!\[CDATA\[|\]\]>', '', text)
                            text = re.sub(r'<[^>]+>', '', text)
                            text = text.strip()
                            
                            h = hashlib.md5(text.encode()).hexdigest()
                            if h not in seen and len(text) > 15:
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


# ============== ENHANCED SENTIMENT ANALYSIS ==============
POSITIVE_WORDS = {
    # Strong positive
    'surge', 'soar', 'jump', 'skyrocket', 'boom', 'rally', 'breakthrough', 'record', 
    'bullish', 'outperform', 'upgrade', 'beat', 'exceed', 'strong', 'robust',
    # Moderate positive
    'gain', 'rise', 'boost', 'profit', 'growth', 'up', 'high', 'positive', 
    'optimistic', 'buy', 'support', 'recovery', 'improve', 'advance', 'expand',
    'dividend', 'bonus', 'approval', 'deal', 'acquisition', 'merger', 'investment'
}

NEGATIVE_WORDS = {
    # Strong negative
    'crash', 'collapse', 'plunge', 'tumble', 'plummet', 'crisis', 'disaster',
    'bearish', 'downgrade', 'fail', 'default', 'bankruptcy', 'fraud', 'scam',
    # Moderate negative
    'fall', 'drop', 'decline', 'loss', 'down', 'low', 'weak', 'negative',
    'pessimistic', 'sell', 'underperform', 'miss', 'cut', 'slash', 'concern',
    'worry', 'risk', 'volatility', 'correction', 'pressure', 'fear', 'warning'
}

SECTOR_SENTIMENT_MODIFIERS = {
    # Sector-specific sentiment words
    'banking': {'npa': -0.2, 'deposit': 0.1, 'credit': 0.1, 'loan': 0.05, 'rbi': 0.05},
    'pharma': {'fda': 0.1, 'approval': 0.3, 'trial': 0.1, 'patent': 0.1},
    'it': {'contract': 0.2, 'deal': 0.2, 'layoff': -0.3, 'hiring': 0.2},
    'auto': {'ev': 0.2, 'sales': 0.1, 'production': 0.1, 'recall': -0.3},
}


def analyze_sentiment(text: str, sector: str = None) -> float:
    """Enhanced sentiment analysis with sector modifiers"""
    text_lower = text.lower()
    words = text_lower.split()
    
    score = 0.0
    word_count = 0
    
    # Base sentiment from keywords
    for word in words:
        if word in POSITIVE_WORDS:
            score += 1.0
            word_count += 1
        elif word in NEGATIVE_WORDS:
            score -= 1.0
            word_count += 1
    
    # Apply sector-specific modifiers
    if sector and sector in SECTOR_SENTIMENT_MODIFIERS:
        for keyword, modifier in SECTOR_SENTIMENT_MODIFIERS[sector].items():
            if keyword in text_lower:
                score += modifier
                word_count += 0.5
    
    # Normalize
    if word_count == 0:
        return 0.0
    
    normalized = score / max(word_count, 1)
    return max(-1.0, min(1.0, normalized))


# ============== SECTOR DETECTION ==============
SECTOR_KEYWORDS = {
    'banking': ['bank', 'hdfc', 'icici', 'sbi', 'kotak', 'axis', 'rbi', 'npa', 'credit', 'loan', 'deposit', 'psu bank', 'private bank', 'nbfc', 'lender', 'financial services'],
    'it': ['it', 'tcs', 'infosys', 'wipro', 'tech mahindra', 'hcl', 'software', 'digital', 'cloud', 'ai', 'cognizant', 'tech', 'saas', 'cybersecurity', 'data center'],
    'pharma': ['pharma', 'drug', 'medicine', 'cipla', 'sun pharma', 'lupin', 'dr reddy', 'biocon', 'vaccine', 'fda', 'healthcare', 'hospital', 'diagnostic', 'apollo'],
    'auto': ['auto', 'car', 'vehicle', 'tata motors', 'mahindra', 'maruti', 'ev', 'electric vehicle', 'bajaj', 'hero', 'tvs', 'ashok leyland', 'ola', 'ather'],
    'metals': ['metal', 'steel', 'tata steel', 'jsw', 'hindalco', 'vedanta', 'aluminium', 'copper', 'zinc', 'iron', 'coal', 'mining', 'commodity'],
    'infra': ['infra', 'construction', 'cement', 'larsen', 'ultratech', 'acc', 'ambuja', 'road', 'highway', 'real estate', 'realty', 'dlf', 'godrej', 'housing'],
    'fmcg': ['fmcg', 'itc', 'hindustan unilever', 'nestle', 'britannia', 'dabur', 'consumer', 'retail', 'dmart', 'titan', 'marico'],
    'energy': ['oil', 'gas', 'reliance', 'ongc', 'bpcl', 'ioc', 'power', 'coal india', 'energy', 'renewable', 'solar', 'ntpc', 'adani power', 'tata power'],
    'telecom': ['telecom', 'jio', 'airtel', 'vodafone', 'vi', '5g', 'bharti', 'spectrum', 'tower'],
}


def detect_sectors(text: str) -> List[str]:
    """Detect which sectors a headline relates to"""
    text_lower = text.lower()
    sectors = []
    
    for sector, keywords in SECTOR_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                if sector not in sectors:
                    sectors.append(sector)
                break
    
    return sectors if sectors else ['general']


# ============== HOLY TRINITY MODEL V2 ==============
def calculate_trinity_score(sector: str, sentiment: float, news_count: int, total_news: int) -> Dict:
    """Enhanced Holy Trinity score calculation"""
    
    # 1. Expectation Gap (sentiment deviation from neutral)
    expectation_gap = sentiment  # Deviation from 0 (neutral baseline)
    
    # 2. Narrative Velocity (relative news volume + sentiment intensity)
    news_share = news_count / max(total_news, 1)
    velocity = news_share * 5  # Scale factor
    narrative_velocity = velocity * (1 + abs(sentiment))
    
    # 3. Sentiment-Price Divergence (simulated based on extreme sentiment)
    # Strong sentiment without proportional coverage = possible contrarian signal
    if abs(sentiment) > 0.5 and news_share < 0.1:
        divergence = -sentiment * 0.5  # Contrarian signal
    else:
        divergence = sentiment * 0.2
    
    # Weighted combination
    trinity_score = (0.45 * expectation_gap) + (0.30 * narrative_velocity) + (0.25 * divergence)
    
    # Cap the score
    trinity_score = max(-1.0, min(1.0, trinity_score))
    
    # Determine recommendation with thresholds
    if trinity_score > 0.4:
        recommendation = 'strong_long'
        conviction = 'high'
    elif trinity_score > 0.2:
        recommendation = 'long'
        conviction = 'medium'
    elif trinity_score < -0.4:
        recommendation = 'strong_short'
        conviction = 'high'
    elif trinity_score < -0.2:
        recommendation = 'short'
        conviction = 'medium'
    else:
        recommendation = 'hold'
        conviction = 'low'
    
    # Generate rationale
    rationale = []
    if abs(expectation_gap) > 0.3:
        rationale.append(f"{'Positive' if expectation_gap > 0 else 'Negative'} sentiment surprise")
    if narrative_velocity > 0.3:
        rationale.append(f"High news velocity ({news_count} articles)")
    if abs(divergence) > 0.15:
        rationale.append("Divergence signal detected")
    if news_share > 0.15:
        rationale.append("Dominant sector in news flow")
    
    return {
        'trinity_score': round(trinity_score, 3),
        'recommendation': recommendation,
        'conviction': conviction,
        'rationale': '; '.join(rationale) if rationale else 'Normal market conditions',
        'components': {
            'expectation_gap': round(expectation_gap, 3),
            'narrative_velocity': round(narrative_velocity, 3),
            'divergence': round(divergence, 3)
        }
    }


# ============== MAIN ANALYSIS ==============
def run_live_analysis() -> Dict:
    """Run full live analysis pipeline with all sources"""
    
    # Fetch from all sources
    rss_headlines = fetch_rss_headlines(100)
    pulse_headlines = fetch_zerodha_pulse()
    trends_headlines = fetch_google_trends()
    
    # Combine all headlines
    all_headlines = rss_headlines + pulse_headlines + trends_headlines
    
    if not all_headlines:
        raise HTTPException(status_code=503, detail="No news feeds available")
    
    # Analyze each headline
    sector_data = {}
    for h in all_headlines:
        sectors = detect_sectors(h['headline'])
        
        for sector in sectors:
            if sector == 'general':
                continue
            if sector not in sector_data:
                sector_data[sector] = {'headlines': [], 'sentiments': [], 'count': 0, 'sources': set()}
            
            sentiment = analyze_sentiment(h['headline'], sector)
            sector_data[sector]['headlines'].append(h['headline'])
            sector_data[sector]['sentiments'].append(sentiment)
            sector_data[sector]['count'] += 1
            sector_data[sector]['sources'].add(h['source'])
    
    total_sector_news = sum(d['count'] for d in sector_data.values())
    
    # Calculate sector signals
    sector_signals = []
    for sector, data in sector_data.items():
        if data['count'] == 0:
            continue
            
        avg_sentiment = sum(data['sentiments']) / len(data['sentiments'])
        trinity = calculate_trinity_score(sector, avg_sentiment, data['count'], total_sector_news)
        
        sector_signals.append({
            'sector': sector,
            'sentiment': round(avg_sentiment, 3),
            'trinity_score': trinity['trinity_score'],
            'recommendation': trinity['recommendation'],
            'conviction': trinity['conviction'],
            'rationale': trinity['rationale'],
            'news_count': data['count'],
            'sources': list(data['sources'])[:3],
            'top_headlines': data['headlines'][:3]
        })
    
    # Sort by absolute trinity score
    sector_signals.sort(key=lambda x: abs(x['trinity_score']), reverse=True)
    
    # Calculate overall market sentiment
    if sector_signals:
        weighted_sentiment = sum(s['sentiment'] * s['news_count'] for s in sector_signals) / sum(s['news_count'] for s in sector_signals)
    else:
        weighted_sentiment = 0
    
    # Determine regime
    if weighted_sentiment > 0.2:
        regime = 'bullish'
    elif weighted_sentiment < -0.2:
        regime = 'bearish'
    else:
        regime = 'neutral'
    
    # Get recommendations
    long_sectors = [s['sector'] for s in sector_signals if 'long' in s['recommendation']]
    short_sectors = [s['sector'] for s in sector_signals if 'short' in s['recommendation']]
    high_conviction = [s['sector'] for s in sector_signals if s['conviction'] == 'high']
    
    # Compile sources used
    all_sources = set()
    for s in sector_signals:
        all_sources.update(s.get('sources', []))
    
    return {
        'timestamp': datetime.now().isoformat(),
        'regime': regime,
        'effective_sentiment': round(weighted_sentiment, 3),
        'sector_signals': sector_signals,
        'long_sectors': long_sectors,
        'short_sectors': short_sectors,
        'high_conviction': high_conviction,
        'headlines_analyzed': len(all_headlines),
        'sources': list(all_sources),
        'source_breakdown': {
            'rss_feeds': len(rss_headlines),
            'zerodha_pulse': len(pulse_headlines),
            'google_trends': len(trends_headlines)
        }
    }


# ============== API ENDPOINTS ==============
class AnalysisRequest(BaseModel):
    max_headlines: int = 100
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
    return {"status": "healthy", "version": "3.0.0", "timestamp": datetime.now().isoformat()}


@app.post("/api/analyze")
async def analyze_news(request: AnalysisRequest = None):
    """Run live analysis on current news from all sources"""
    try:
        result = run_live_analysis()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sources")
async def list_sources():
    """List all available news sources"""
    return {
        "rss_feeds": list(RSS_FEEDS.keys()),
        "other_sources": ["zerodha_pulse", "google_trends"],
        "total_feed_urls": sum(len(urls) for urls in RSS_FEEDS.values())
    }


@app.get("/api/demo")
async def get_demo_data():
    """Return demo data for testing"""
    return {
        "timestamp": datetime.now().isoformat(),
        "regime": "bullish",
        "effective_sentiment": 0.35,
        "sector_signals": [
            {"sector": "auto", "sentiment": 0.45, "trinity_score": 0.675, "recommendation": "strong_long", "conviction": "high", "rationale": "Positive surprise; early mover", "news_count": 8, "sources": ["economic_times", "moneycontrol"]},
            {"sector": "banking", "sentiment": 0.65, "trinity_score": 0.475, "recommendation": "strong_long", "conviction": "high", "rationale": "Positive surprise", "news_count": 12, "sources": ["business_standard", "livemint"]},
            {"sector": "it", "sentiment": -0.30, "trinity_score": -0.475, "recommendation": "strong_short", "conviction": "high", "rationale": "Negative surprise", "news_count": 6, "sources": ["ndtv_profit"]},
            {"sector": "metals", "sentiment": -0.40, "trinity_score": -0.375, "recommendation": "short", "conviction": "medium", "rationale": "Negative surprise", "news_count": 4, "sources": ["reuters_india"]},
            {"sector": "pharma", "sentiment": 0.10, "trinity_score": 0.304, "recommendation": "long", "conviction": "medium", "rationale": "Modest surprise", "news_count": 5, "sources": ["the_hindu_business"]},
            {"sector": "infra", "sentiment": 0.05, "trinity_score": 0.060, "recommendation": "hold", "conviction": "low", "rationale": "In-line with expectations", "news_count": 3, "sources": ["financial_express"]},
        ],
        "long_sectors": ["auto", "banking", "pharma"],
        "short_sectors": ["it", "metals"],
        "high_conviction": ["auto", "banking", "it"],
        "headlines_analyzed": 85,
        "sources": ["economic_times", "moneycontrol", "livemint", "business_standard", "ndtv_profit", "zerodha_pulse", "google_trends"],
        "source_breakdown": {"rss_feeds": 65, "zerodha_pulse": 15, "google_trends": 5}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
News Alpha Pipeline - FastAPI Backend V3
Enhanced sentiment analysis with VADER, expanded keywords, bigrams, and position weighting.
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

import threading
import time
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(title="News Alpha Pipeline", version="3.1.0")

# ============== KEEP-ALIVE PINGER ==============
def run_pinger():
    """Periodically ping the Render URL to prevent spin-down"""
    target_url = "https://news-alpha-pipeline.onrender.com/health"
    print(f"Starting Keep-Alive Pinger for: {target_url}")
    
    while True:
        try:
            time.sleep(600) # Ping every 10 minutes
            response = requests.get(target_url, timeout=10)
            print(f"Keep-Alive Ping: {response.status_code}")
        except Exception as e:
            print(f"Keep-Alive Ping Failed: {e}")

# Start pinger in background thread
# We only want one pinger running, so we check if we are in the main process
# However, uvicorn reloader might duplicate this. Ideally, this runs on the server.
# For simplicity, we start it as a daemon thread.
pinger_thread = threading.Thread(target=run_pinger, daemon=True)
pinger_thread.start()

@app.get("/health")
def health_check():
    """Health check endpoint for pinger"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# ============== RSS NEWS FEEDS ==============
import urllib.request
import xml.etree.ElementTree as ET
import json

RSS_FEEDS = {
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
    'business_standard': [
        'https://www.business-standard.com/rss/latest.rss',
        'https://www.business-standard.com/rss/markets-106.rss',
        'https://www.business-standard.com/rss/companies-101.rss',
    ],
    'the_hindu_business': [
        'https://www.thehindu.com/business/feeder/default.rss',
        'https://www.thehindu.com/business/markets/feeder/default.rss',
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

GOOGLE_TRENDS_API = "https://trends.google.com/trends/api/dailytrends?hl=en-IN&tz=-330&geo=IN"


# ============== VADER SENTIMENT ANALYZER ==============
class VADERSentiment:
    """
    VADER (Valence Aware Dictionary and sEntiment Reasoner)
    Lightweight rule-based sentiment analysis optimized for social media and news.
    """
    
    def __init__(self):
        # VADER lexicon - word: (positive, negative, neutral) valence
        self.lexicon = self._build_lexicon()
        
        # Booster words that modify sentiment
        self.boosters = {
            'absolutely': 0.293, 'amazingly': 0.293, 'awfully': 0.293,
            'completely': 0.293, 'considerably': 0.293, 'decidedly': 0.293,
            'deeply': 0.293, 'enormously': 0.293, 'entirely': 0.293,
            'especially': 0.293, 'exceptionally': 0.293, 'extremely': 0.293,
            'greatly': 0.293, 'highly': 0.293, 'hugely': 0.293,
            'incredibly': 0.293, 'intensely': 0.293, 'majorly': 0.293,
            'massively': 0.293, 'particularly': 0.293, 'purely': 0.293,
            'quite': 0.293, 'really': 0.293, 'remarkably': 0.293,
            'significantly': 0.293, 'substantially': 0.293, 'thoroughly': 0.293,
            'totally': 0.293, 'tremendously': 0.293, 'unbelievably': 0.293,
            'unusually': 0.293, 'utterly': 0.293, 'very': 0.293,
            # Dampeners
            'almost': -0.293, 'barely': -0.293, 'hardly': -0.293,
            'just enough': -0.293, 'kind of': -0.293, 'kinda': -0.293,
            'kindof': -0.293, 'less': -0.293, 'little': -0.293,
            'marginally': -0.293, 'occasionally': -0.293, 'partly': -0.293,
            'scarcely': -0.293, 'slightly': -0.293, 'somewhat': -0.293,
            'sort of': -0.293, 'sorta': -0.293, 'sortof': -0.293,
        }
        
        # Negation words
        self.negations = {
            'not', 'isnt', "isn't", 'arent', "aren't", 'wasnt', "wasn't",
            'werent', "weren't", 'hasnt', "hasn't", 'havent', "haven't",
            'hadnt', "hadn't", 'wont', "won't", 'wouldnt', "wouldn't",
            'dont', "don't", 'doesnt', "doesn't", 'didnt', "didn't",
            'cant', "can't", 'couldnt', "couldn't", 'shouldnt', "shouldn't",
            'mightnt', "mightn't", 'mustnt', "mustn't", 'neither', 'never',
            'no', 'nobody', 'none', 'noone', 'nor', 'nothing', 'nowhere',
            'without', 'rarely', 'seldom', 'despite'
        }
    
    def _build_lexicon(self) -> Dict[str, float]:
        """Build comprehensive financial sentiment lexicon"""
        lexicon = {}
        
        # Strong Positive (0.6 - 1.0)
        strong_positive = {
            'surge': 0.9, 'soar': 0.9, 'skyrocket': 0.95, 'boom': 0.85,
            'rally': 0.8, 'breakthrough': 0.85, 'record': 0.7, 'bullish': 0.8,
            'outperform': 0.75, 'beat': 0.7, 'exceed': 0.7, 'exceptional': 0.8,
            'outstanding': 0.8, 'remarkable': 0.75, 'stellar': 0.85,
            'blockbuster': 0.85, 'tremendous': 0.8, 'phenomenal': 0.85,
            'spectacular': 0.85, 'magnificent': 0.8, 'brilliant': 0.8,
        }
        
        # Moderate Positive (0.3 - 0.6)
        moderate_positive = {
            'gain': 0.5, 'rise': 0.45, 'boost': 0.5, 'profit': 0.55,
            'growth': 0.5, 'up': 0.35, 'high': 0.4, 'positive': 0.5,
            'optimistic': 0.55, 'buy': 0.4, 'support': 0.4, 'recovery': 0.5,
            'improve': 0.45, 'advance': 0.45, 'expand': 0.45, 'upgrade': 0.55,
            'dividend': 0.5, 'bonus': 0.5, 'approval': 0.55, 'deal': 0.45,
            'acquisition': 0.45, 'merger': 0.4, 'investment': 0.45,
            'strong': 0.5, 'robust': 0.5, 'solid': 0.45, 'stable': 0.4,
            'healthy': 0.5, 'successful': 0.55, 'winner': 0.6, 'best': 0.55,
            'top': 0.45, 'lead': 0.4, 'leader': 0.45, 'momentum': 0.45,
            'upbeat': 0.5, 'confident': 0.5, 'encouraging': 0.5,
        }
        
        # Mild Positive (0.1 - 0.3)
        mild_positive = {
            'steady': 0.25, 'unchanged': 0.15, 'maintain': 0.2, 'hold': 0.15,
            'continues': 0.2, 'expected': 0.2, 'inline': 0.2, 'target': 0.2,
        }
        
        # Strong Negative (-0.6 to -1.0)
        strong_negative = {
            'crash': -0.95, 'collapse': -0.9, 'plunge': -0.85, 'tumble': -0.8,
            'plummet': -0.85, 'crisis': -0.85, 'disaster': -0.9, 'bearish': -0.8,
            'downgrade': -0.7, 'fail': -0.75, 'default': -0.85, 'bankruptcy': -0.95,
            'fraud': -0.9, 'scam': -0.9, 'scandal': -0.8, 'catastrophe': -0.9,
            'devastating': -0.85, 'terrible': -0.8, 'horrible': -0.8,
            'worst': -0.85, 'dismal': -0.75, 'abysmal': -0.8, 'dire': -0.75,
        }
        
        # Moderate Negative (-0.3 to -0.6)
        moderate_negative = {
            'fall': -0.5, 'drop': -0.5, 'decline': -0.5, 'loss': -0.55,
            'down': -0.4, 'low': -0.4, 'weak': -0.5, 'negative': -0.5,
            'pessimistic': -0.55, 'sell': -0.45, 'underperform': -0.55,
            'miss': -0.5, 'cut': -0.45, 'slash': -0.5, 'concern': -0.45,
            'worry': -0.45, 'risk': -0.4, 'volatility': -0.35, 'correction': -0.45,
            'pressure': -0.4, 'fear': -0.5, 'warning': -0.5, 'trouble': -0.5,
            'struggle': -0.45, 'slow': -0.35, 'slump': -0.55, 'selloff': -0.55,
            'drag': -0.4, 'hurt': -0.5, 'pain': -0.5, 'disappointing': -0.5,
            'missed': -0.45, 'below': -0.35, 'lower': -0.4, 'weaken': -0.5,
        }
        
        # Mild Negative (-0.1 to -0.3)
        mild_negative = {
            'uncertain': -0.25, 'unclear': -0.2, 'mixed': -0.15, 'flat': -0.15,
            'cautious': -0.2, 'muted': -0.2, 'tepid': -0.25, 'subdued': -0.25,
        }
        
        # Merge all
        for words, mult in [(strong_positive, 1), (moderate_positive, 1), 
                            (mild_positive, 1), (strong_negative, 1),
                            (moderate_negative, 1), (mild_negative, 1)]:
            for word, score in words.items():
                lexicon[word] = score
        
        return lexicon
    
    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze sentiment with VADER-style scoring"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        sentiments = []
        
        for i, word in enumerate(words):
            if word in self.lexicon:
                score = self.lexicon[word]
                
                # Check for negation in previous 3 words
                start = max(0, i - 3)
                preceding = words[start:i]
                if any(neg in self.negations for neg in preceding):
                    score *= -0.74  # VADER negation coefficient
                
                # Check for boosters/dampeners
                if i > 0 and words[i-1] in self.boosters:
                    score += self.boosters[words[i-1]] * (1 if score > 0 else -1)
                
                sentiments.append(score)
        
        if not sentiments:
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
        
        # Calculate compound score (normalized)
        total = sum(sentiments)
        compound = total / ((total ** 2 + 15) ** 0.5)  # VADER normalization
        compound = max(-1, min(1, compound))
        
        pos = sum(s for s in sentiments if s > 0) / len(sentiments) if sentiments else 0
        neg = abs(sum(s for s in sentiments if s < 0)) / len(sentiments) if sentiments else 0
        neu = 1 - (pos + neg)
        
        return {'compound': compound, 'pos': pos, 'neg': neg, 'neu': max(0, neu)}


# Initialize VADER
vader = VADERSentiment()


# ============== BIGRAM MATCHING ==============
BIGRAM_SENTIMENTS = {
    # Strong Positive Bigrams
    ('record', 'high'): 0.8, ('all', 'time'): 0.6, ('strong', 'buy'): 0.85,
    ('beat', 'expectations'): 0.8, ('exceeds', 'estimates'): 0.8,
    ('better', 'expected'): 0.7, ('above', 'consensus'): 0.7,
    ('positive', 'outlook'): 0.7, ('bright', 'future'): 0.7,
    ('massive', 'growth'): 0.8, ('robust', 'demand'): 0.7,
    ('solid', 'performance'): 0.65, ('strong', 'earnings'): 0.75,
    ('profit', 'surge'): 0.8, ('revenue', 'jump'): 0.75,
    ('market', 'rally'): 0.7, ('bull', 'run'): 0.75,
    ('upward', 'trend'): 0.6, ('positive', 'momentum'): 0.65,
    ('buying', 'opportunity'): 0.7, ('undervalued', 'stock'): 0.65,
    ('dividend', 'increase'): 0.65, ('stock', 'split'): 0.55,
    ('buyback', 'announced'): 0.6, ('fdi', 'inflow'): 0.6,
    
    # Strong Negative Bigrams
    ('record', 'low'): -0.8, ('heavy', 'losses'): -0.85,
    ('profit', 'warning'): -0.8, ('earnings', 'miss'): -0.75,
    ('below', 'expectations'): -0.7, ('guidance', 'cut'): -0.75,
    ('market', 'crash'): -0.9, ('sell', 'off'): -0.7,
    ('bear', 'market'): -0.75, ('downward', 'trend'): -0.6,
    ('job', 'cuts'): -0.7, ('layoffs', 'announced'): -0.75,
    ('debt', 'crisis'): -0.85, ('credit', 'downgrade'): -0.8,
    ('rating', 'downgrade'): -0.75, ('negative', 'outlook'): -0.7,
    ('fii', 'outflow'): -0.65, ('capital', 'flight'): -0.7,
    ('margin', 'pressure'): -0.6, ('cost', 'overrun'): -0.6,
    ('liquidity', 'crunch'): -0.75, ('default', 'risk'): -0.8,
    ('bankruptcy', 'filing'): -0.9, ('fraud', 'allegations'): -0.85,
    
    # Neutral/Context Bigrams
    ('interest', 'rate'): 0.0, ('central', 'bank'): 0.0,
    ('quarterly', 'results'): 0.0, ('fiscal', 'year'): 0.0,
}


def get_bigram_score(text: str) -> float:
    """Calculate sentiment from bigram matches"""
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    total_score = 0.0
    matches = 0
    
    for i in range(len(words) - 1):
        bigram = (words[i], words[i+1])
        if bigram in BIGRAM_SENTIMENTS:
            total_score += BIGRAM_SENTIMENTS[bigram]
            matches += 1
    
    return total_score / max(matches, 1) if matches > 0 else 0.0


# ============== POSITION WEIGHTING ==============
def get_position_weight(position: int, total: int) -> float:
    """
    Headlines at the top of news feeds are more important.
    First 20% get 1.5x weight, next 30% get 1.2x, rest get 1.0x
    """
    if total == 0:
        return 1.0
    
    relative_pos = position / total
    
    if relative_pos <= 0.2:
        return 1.5  # Top 20% - most important
    elif relative_pos <= 0.5:
        return 1.2  # Next 30% - moderately important
    else:
        return 1.0  # Rest - normal weight


# ============== SECTOR KEYWORDS (EXPANDED) ==============
SECTOR_KEYWORDS = {
    'banking': [
        'bank', 'hdfc', 'icici', 'sbi', 'kotak', 'axis', 'rbi', 'npa', 'credit', 
        'loan', 'deposit', 'psu bank', 'private bank', 'nbfc', 'lender', 
        'financial services', 'idfc', 'bandhan', 'indusind', 'yes bank', 
        'federal bank', 'rbl', 'au bank', 'interest rate', 'repo', 'monetary policy',
        'casa', 'nim', 'slippage', 'provision', 'write-off', 'gross npa', 'net npa'
    ],
    'it': [
        'it', 'tcs', 'infosys', 'wipro', 'tech mahindra', 'hcl', 'software', 
        'digital', 'cloud', 'ai', 'cognizant', 'tech', 'saas', 'cybersecurity',
        'data center', 'ltimindtree', 'mphasis', 'persistent', 'coforge', 
        'birlasoft', 'ltts', 'cyient', 'zensar', 'outsourcing', 'attrition',
        'deal wins', 'large deals', 'digital transformation'
    ],
    'pharma': [
        'pharma', 'drug', 'medicine', 'cipla', 'sun pharma', 'lupin', 
        'dr reddy', 'biocon', 'vaccine', 'fda', 'healthcare', 'hospital', 
        'diagnostic', 'apollo', 'fortis', 'max healthcare', 'natco', 'aurobindo',
        'torrent pharma', 'zydus', 'alkem', 'laurus', 'divi', 'api', 'anda',
        'generic', 'biosimilar', '483', 'warning letter', 'eir', 'clinical trial'
    ],
    'auto': [
        'auto', 'car', 'vehicle', 'tata motors', 'mahindra', 'maruti', 'ev', 
        'electric vehicle', 'bajaj', 'hero', 'tvs', 'ashok leyland', 'ola', 
        'ather', 'eicher', 'escorts', 'sml isuzu', 'force motors', 'automobile',
        'two wheeler', 'passenger vehicle', 'commercial vehicle', 'ev sales',
        'chip shortage', 'auto sales', 'wholesale', 'retail', 'dealer'
    ],
    'metals': [
        'metal', 'steel', 'tata steel', 'jsw', 'hindalco', 'vedanta', 
        'aluminium', 'copper', 'zinc', 'iron', 'coal', 'mining', 'commodity',
        'sail', 'nmdc', 'nalco', 'jindal', 'jspl', 'hind copper', 'moil',
        'gold', 'silver', 'iron ore', 'coking coal', 'base metals'
    ],
    'infra': [
        'infra', 'construction', 'cement', 'larsen', 'ultratech', 'acc', 
        'ambuja', 'road', 'highway', 'real estate', 'realty', 'dlf', 'godrej',
        'housing', 'oberoi', 'prestige', 'brigade', 'sobha', 'macrotech',
        'l&t', 'irb', 'knr', 'pnc', 'nhai', 'order book', 'awarding'
    ],
    'fmcg': [
        'fmcg', 'itc', 'hindustan unilever', 'nestle', 'britannia', 'dabur', 
        'consumer', 'retail', 'dmart', 'titan', 'marico', 'godrej consumer',
        'tata consumer', 'varun beverages', 'colgate', 'p&g', 'emami',
        'patanjali', 'avenue supermarts', 'volume growth', 'price hike'
    ],
    'energy': [
        'oil', 'gas', 'reliance', 'ongc', 'bpcl', 'ioc', 'power', 'coal india', 
        'energy', 'renewable', 'solar', 'ntpc', 'adani power', 'tata power',
        'gail', 'petronet', 'hpcl', 'mrpl', 'chennai petro', 'adani green',
        'jswenergy', 'torrent power', 'cesc', 'power grid', 'crude', 'lng',
        'refining margin', 'grm', 'pll'
    ],
    'telecom': [
        'telecom', 'jio', 'airtel', 'vodafone', 'vi', '5g', 'bharti', 
        'spectrum', 'tower', 'indus towers', 'tata communications', 'arpu',
        'subscriber', 'data consumption', 'tariff hike', 'agr'
    ],
    'defence': [
        'defence', 'hal', 'bel', 'bharat dynamics', 'mazagon dock', 'cochin shipyard',
        'grse', 'bdl', 'midhani', 'paras defence', 'data patterns', 'astra',
        'military', 'defense', 'order win', 'procurement', 'make in india'
    ],
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


# ============== ENHANCED SENTIMENT ANALYSIS ==============
SECTOR_SENTIMENT_MODIFIERS = {
    'banking': {'npa': -0.25, 'slippage': -0.2, 'provision': -0.15, 
                'casa': 0.15, 'nim': 0.1, 'credit growth': 0.2},
    'pharma': {'fda': 0.1, 'approval': 0.3, 'patent': 0.1, '483': -0.25,
               'warning letter': -0.35, 'clinical trial': 0.15},
    'it': {'deal win': 0.25, 'large deal': 0.3, 'attrition': -0.2, 
           'layoff': -0.35, 'hiring': 0.2, 'fresher': 0.15},
    'auto': {'ev': 0.2, 'recall': -0.35, 'chip shortage': -0.25, 
             'price hike': -0.1, 'volume growth': 0.2},
    'metals': {'china demand': 0.2, 'price rally': 0.25, 'dumping': -0.2},
    'infra': {'order win': 0.25, 'order book': 0.2, 'awarding': 0.2},
    'fmcg': {'volume growth': 0.25, 'price hike': 0.1, 'rural demand': 0.15},
    'energy': {'crude': 0.0, 'grm': 0.15, 'refining margin': 0.15},
    'telecom': {'arpu': 0.2, 'tariff hike': 0.2, 'subscriber': 0.1},
}


def analyze_sentiment_enhanced(text: str, sector: str = None, position: int = 0, total: int = 1) -> Dict:
    """
    Enhanced sentiment analysis combining:
    1. VADER sentiment
    2. Bigram matching
    3. Sector-specific modifiers
    4. Position weighting
    """
    # 1. VADER base sentiment
    vader_result = vader.analyze(text)
    vader_score = vader_result['compound']
    
    # 2. Bigram modifier
    bigram_score = get_bigram_score(text)
    
    # 3. Sector modifier
    sector_modifier = 0.0
    text_lower = text.lower()
    if sector and sector in SECTOR_SENTIMENT_MODIFIERS:
        for keyword, mod in SECTOR_SENTIMENT_MODIFIERS[sector].items():
            if keyword in text_lower:
                sector_modifier += mod
    
    # 4. Position weight
    position_weight = get_position_weight(position, total)
    
    # Combine scores (weighted average)
    combined_score = (
        (vader_score * 0.5) +      # 50% VADER
        (bigram_score * 0.3) +      # 30% Bigrams
        (sector_modifier * 0.2)     # 20% Sector modifiers
    )
    
    # Apply position weight
    weighted_score = combined_score * position_weight
    
    # Normalize to [-1, 1]
    final_score = max(-1.0, min(1.0, weighted_score))
    
    return {
        'score': final_score,
        'vader': vader_score,
        'bigram': bigram_score,
        'sector_mod': sector_modifier,
        'position_weight': position_weight,
        'components': vader_result
    }


# ============== NEWS FETCHING ==============
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
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36', 'Accept': 'text/html'}
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            content = response.read().decode('utf-8', errors='ignore')
            
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


def fetch_single_rss_feed(args) -> List[Dict]:
    """Fetch headlines from a single RSS feed"""
    source, feed_url = args
    headlines = []
    try:
        req = urllib.request.Request(
            feed_url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read()
            root = ET.fromstring(content)
            
            items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')
            
            for item in items[:8]:
                title = item.find('title') or item.find('{http://www.w3.org/2005/Atom}title')
                if title is not None and title.text:
                    text = title.text.strip()
                    text = re.sub(r'<!\[CDATA\[|\]\]>', '', text)
                    text = re.sub(r'<[^>]+>', '', text)
                    text = text.strip()
                    
                    if len(text) > 15:
                        headlines.append({
                            'headline': text,
                            'source': source,
                            'timestamp': datetime.now().isoformat()
                        })
    except Exception as e:
        print(f"Error fetching {feed_url}: {e}")
    
    return headlines


def fetch_rss_headlines(max_items: int = 100) -> List[Dict]:
    """Fetch live headlines from all RSS feeds using concurrent requests"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Prepare all feed URLs with their sources
    feed_tasks = []
    for source, feeds in RSS_FEEDS.items():
        for feed_url in feeds:
            feed_tasks.append((source, feed_url))
    
    all_headlines = []
    seen = set()
    
    # Fetch concurrently with max 10 workers
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_single_rss_feed, task): task for task in feed_tasks}
        
        for future in as_completed(futures, timeout=30):
            try:
                results = future.result()
                for h in results:
                    h_hash = hashlib.md5(h['headline'].encode()).hexdigest()
                    if h_hash not in seen:
                        seen.add(h_hash)
                        all_headlines.append(h)
            except Exception as e:
                print(f"Feed fetch error: {e}")
                continue
    
    print(f"RSS feeds fetched: {len(all_headlines)} headlines from {len(set(h['source'] for h in all_headlines))} sources")
    return all_headlines[:max_items]


def _old_fetch_rss_headlines(max_items: int = 100) -> List[Dict]:
    """OLD: Sequential fetch - kept for reference"""
    headlines = []
    seen = set()
    
    for source, feeds in RSS_FEEDS.items():
        for feed_url in feeds:
            try:
                req = urllib.request.Request(
                    feed_url,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    content = response.read()
                    root = ET.fromstring(content)
                    
                    items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')
                    
                    for item in items[:8]:
                        title = item.find('title') or item.find('{http://www.w3.org/2005/Atom}title')
                        if title is not None and title.text:
                            text = title.text.strip()
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


# ============== TRINITY SCORE ==============
def calculate_trinity_score(sector: str, sentiment: float, news_count: int, total_news: int) -> Dict:
    """Enhanced Holy Trinity score calculation"""
    
    expectation_gap = sentiment
    news_share = news_count / max(total_news, 1)
    velocity = news_share * 5
    narrative_velocity = velocity * (1 + abs(sentiment))
    
    if abs(sentiment) > 0.5 and news_share < 0.1:
        divergence = -sentiment * 0.5
    else:
        divergence = sentiment * 0.2
    
    trinity_score = (0.45 * expectation_gap) + (0.30 * narrative_velocity) + (0.25 * divergence)
    trinity_score = max(-1.0, min(1.0, trinity_score))
    
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
    }


# ============== NER LAYER (Always Available) ==============
try:
    from src.nlp.ner import extract_companies, get_company_name, COMPANY_ALIASES
    print(f"NER Layer loaded successfully with {len(COMPANY_ALIASES)} company aliases")
except ImportError as e:
    print(f"NER Layer unavailable: {e}")
    def extract_companies(text): return []
    def get_company_name(sym): return sym
    COMPANY_ALIASES = {}

# ============== SEMANTIC LAYER ==============
try:
    from src.nlp.embeddings import get_embeddings
    from src.nlp.surprise_score import SurpriseModel
    from src.models.sector_attribution import SectorAttributionModel
    SEMANTIC_LAYER_AVAILABLE = True
    surprise_model = SurpriseModel(memory_window=200)
    attribution_model = SectorAttributionModel()
    print("Semantic Layer (Surprise + Attribution) initialized successfully")
except ImportError as e:
    print(f"Semantic Layer unavailable: {e}")
    SEMANTIC_LAYER_AVAILABLE = False
    surprise_model = None
    attribution_model = None


# ============== ANALYTICS UTILS ==============
def generate_word_cloud(headlines: List[str]) -> List[Dict]:
    """Generate frequency map for word cloud"""
    stop_words = {
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about', 
        'into', 'over', 'after', 'the', 'and', 'a', 'an', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'but',
        'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
        'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
        'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
        'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
        'now', 'india', 'share', 'price', 
        'live', 'updates', 'today', 
        'ltd', 'limited', 'corp', 'inc', 'co', 'bse', 'nse', 'gain', 'lose', 'falls', 'rises'
    }
    
    word_freq = {}
    for text in headlines:
        # Extract words (3+ chars)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by freq
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50]
    
    # Normalize size 10-100
    if not sorted_words:
        return []
        
    max_freq = sorted_words[0][1]
    min_freq = sorted_words[-1][1]
    
    return [
        {'text': word, 'value': 10 + ((freq - min_freq) / max(1, max_freq - min_freq)) * 90}
        for word, freq in sorted_words
    ]

def extract_hot_stocks(headlines: List[str]) -> List[Dict]:
    """
    Extract most mentioned stocks from headlines using simple keyword matching.
    Returns list of {symbol, name, mentions}
    """
    # Simple keyword to symbol mapping (most common Indian stocks)
    stock_keywords = {
        'tcs': ('TCS', 'Tata Consultancy Services'),
        'infosys': ('INFY', 'Infosys'),
        'reliance': ('RELIANCE', 'Reliance Industries'),
        'hdfc': ('HDFCBANK', 'HDFC Bank'),
        'icici': ('ICICIBANK', 'ICICI Bank'),
        'sbi': ('SBIN', 'State Bank of India'),
        'wipro': ('WIPRO', 'Wipro'),
        'bharti': ('BHARTIARTL', 'Bharti Airtel'),
        'airtel': ('BHARTIARTL', 'Bharti Airtel'),
        'adani': ('ADANIENT', 'Adani Enterprises'),
        'tata motors': ('TATAMOTORS', 'Tata Motors'),
        'maruti': ('MARUTI', 'Maruti Suzuki'),
        'nifty': ('NIFTY50', 'Nifty 50'),
        'sensex': ('SENSEX', 'BSE Sensex'),
        'itc': ('ITC', 'ITC Ltd'),
        'bajaj': ('BAJFINANCE', 'Bajaj Finance'),
        'asian paints': ('ASIANPAINT', 'Asian Paints'),
        'hul': ('HINDUNILVR', 'Hindustan Unilever'),
        'larsen': ('LT', 'Larsen & Toubro'),
        'l&t': ('LT', 'Larsen & Toubro'),
        'titan': ('TITAN', 'Titan Company'),
        'mahindra': ('M&M', 'Mahindra & Mahindra'),
        'kotak': ('KOTAKBANK', 'Kotak Mahindra Bank'),
        'axis': ('AXISBANK', 'Axis Bank'),
    }
    
    stock_mentions = {}
    
    for headline in headlines:
        headline_lower = headline.lower()
        for keyword, (symbol, name) in stock_keywords.items():
            if keyword in headline_lower:
                if symbol not in stock_mentions:
                    stock_mentions[symbol] = {'name': name, 'mentions': 0}
                stock_mentions[symbol]['mentions'] += 1
    
    # Convert to list and sort by mentions
    hot_stocks = [
        {'symbol': symbol, 'name': data['name'], 'mentions': data['mentions']}
        for symbol, data in stock_mentions.items()
    ]
    hot_stocks.sort(key=lambda x: x['mentions'], reverse=True)
    
    return hot_stocks[:10]  # Top 10


def generate_ai_summary(sector_signals: List[Dict], top_stocks: List[Dict], regime: str) -> str:
    """
    Generate an AI-powered executive summary of the market analysis.
    Uses template-based generation (no external API needed).
    """
    if not sector_signals:
        return "Insufficient data for market summary. Run analysis to populate."
    
    # Find top performing and worst performing sectors
    sorted_sectors = sorted(sector_signals, key=lambda x: x['sentiment'], reverse=True)
    best_sector = sorted_sectors[0] if sorted_sectors else None
    worst_sector = sorted_sectors[-1] if len(sorted_sectors) > 1 else None
    
    # Count recommendations
    long_count = sum(1 for s in sector_signals if 'long' in s.get('recommendation', ''))
    short_count = sum(1 for s in sector_signals if 'short' in s.get('recommendation', ''))
    
    # Build summary
    parts = []
    
    # Opening line based on regime
    if regime == 'bullish':
        parts.append("Markets show BULLISH momentum today.")
    elif regime == 'bearish':
        parts.append("Markets are under BEARISH pressure today.")
    else:
        parts.append("Markets are trading in a NEUTRAL range today.")
    
    # Sector highlights
    if best_sector:
        sent_str = f"+{best_sector['sentiment']:.2f}" if best_sector['sentiment'] >= 0 else f"{best_sector['sentiment']:.2f}"
        parts.append(f"{best_sector['sector'].upper()} leads with {sent_str} sentiment.")
    
    if worst_sector and worst_sector['sentiment'] < 0:
        parts.append(f"{worst_sector['sector'].upper()} faces headwinds ({worst_sector['sentiment']:.2f}).")
    
    # Stock mentions
    if top_stocks:
        top_stock = top_stocks[0]
        parts.append(f"Most active: {top_stock['symbol']} ({top_stock['mentions']} mentions).")
    
    # Recommendation summary
    if long_count > short_count:
        parts.append(f"Bias: {long_count} long signals vs {short_count} short.")
    elif short_count > long_count:
        parts.append(f"Caution: {short_count} short signals outweigh {long_count} long.")
    
    return " ".join(parts)


# ============== MAIN ANALYSIS ==============

def run_live_analysis() -> Dict:
    """Run full live analysis pipeline with enhanced sentiment and semantic surprise"""
    
    rss_headlines = fetch_rss_headlines(100)
    pulse_headlines = fetch_zerodha_pulse()
    trends_headlines = fetch_google_trends()
    
    all_headlines = rss_headlines + pulse_headlines + trends_headlines
    total_headlines = len(all_headlines)
    
    if not all_headlines:
        raise HTTPException(status_code=503, detail="No news feeds available")
    
    # Process batch embeddings if available
    headline_embeddings = {}
    if SEMANTIC_LAYER_AVAILABLE:
        try:
            texts = [h['headline'] for h in all_headlines]
            embeddings = get_embeddings(texts) # Returns np.array
            for i, h in enumerate(all_headlines):
                h_hash = hashlib.md5(h['headline'].encode()).hexdigest()
                if i < len(embeddings):
                   headline_embeddings[h_hash] = embeddings[i]
        except Exception as e:
            print(f"Embedding generation failed: {e}")
    
    # Analyze each headline
    sector_data = {}
    company_data = {} # {symbol: {mentions: 0, sentiments: [], surprise: []}}
    
    for idx, h in enumerate(all_headlines):
        h_hash = hashlib.md5(h['headline'].encode()).hexdigest()
        
        # 1. Semantic Surpise & Attribution
        surprise_val = 0.0
        semantic_sectors = {}
        
        if SEMANTIC_LAYER_AVAILABLE and h_hash in headline_embeddings:
            emb = headline_embeddings[h_hash]
            
            # Surprise
            surp_res = surprise_model.calculate_surprise(emb)
            surprise_val = surp_res['surprise_score']
            surprise_model.update(emb)
            
            # Attribution (Probabilistic)
            semantic_sectors = attribution_model.get_sector_decomposition(h['headline'], headline_embedding=emb)
        
        h['surprise_score'] = surprise_val
        
        # 2. Keyword Sector Detection
        keyword_sectors = detect_sectors(h['headline'])
        
        # Merge sectors (Union of Keyword and Semantic)
        final_sectors = semantic_sectors.copy()
        for k_sec in keyword_sectors:
            if k_sec != 'general' and k_sec not in final_sectors:
                final_sectors[k_sec] = 1.0 
        
        # 3. Company Extraction
        companies = extract_companies(h['headline'])
        
        # Determine Sentiment for this headline
        # Use 'general' sector sentiment if no specific sector, or avg of sectors
        sentiment_result = analyze_sentiment_enhanced(
            h['headline'], 
            sector=list(final_sectors.keys())[0] if final_sectors else None,
            position=idx,
            total=total_headlines
        )
        sent_score = sentiment_result['score']
        
        # Update Company Data
        for comp in companies:
            if comp not in company_data:
                company_data[comp] = {'mentions': 0, 'sentiments': [], 'surprise': []}
            company_data[comp]['mentions'] += 1
            company_data[comp]['sentiments'].append(sent_score)
            company_data[comp]['surprise'].append(surprise_val)

        if not final_sectors:
            continue

        for sector, weight in final_sectors.items():
            if sector not in sector_data:
                sector_data[sector] = {'headlines': [], 'sentiments': [], 'surprise': [], 'count': 0, 'sources': set()}
            
            sector_data[sector]['headlines'].append(h['headline'])
            sector_data[sector]['sentiments'].append(sent_score)
            sector_data[sector]['surprise'].append(surprise_val)
            sector_data[sector]['count'] += 1
            sector_data[sector]['sources'].add(h['source'])
            
    # Calculate sector signals
    total_sector_news = sum(d['count'] for d in sector_data.values())
    sector_signals = []
    
    for sector, data in sector_data.items():
        if data['count'] == 0:
            continue
            
        avg_sentiment = sum(data['sentiments']) / len(data['sentiments'])
        avg_surprise = sum(data['surprise']) / len(data['surprise']) if data['surprise'] else 0.0
        
        trinity = calculate_trinity_score(sector, avg_sentiment, data['count'], total_sector_news)
        
        # Boost recommendation if High Surprise + High Sentiment
        if avg_surprise > 0.7 and abs(avg_sentiment) > 0.3:
            trinity['rationale'] += "; High Information Novelty"
            if trinity['conviction'] == 'medium':
                trinity['conviction'] = 'high'
        
        sector_signals.append({
            'sector': sector,
            'sentiment': round(avg_sentiment, 3),
            'surprise_score': round(avg_surprise, 3),
            'trinity_score': trinity['trinity_score'],
            'recommendation': trinity['recommendation'],
            'conviction': trinity['conviction'],
            'rationale': trinity['rationale'],
            'news_count': data['count'],
            'sources': list(data['sources'])[:3],
            'top_headlines': data['headlines'][:3]
        })
        
        
    # --- Simple Stock Extraction (Keyword-based) ---
    top_stocks = extract_hot_stocks([h['headline'] for h in all_headlines])
    
    # Word Cloud
    word_cloud = generate_word_cloud([h['headline'] for h in all_headlines])
    
    print(f"DEBUG: Found {len(top_stocks)} stocks and {len(word_cloud)} word cloud items")
    if top_stocks:
        print(f"DEBUG Top Stock: {top_stocks[0]}")
    
    sector_signals.sort(key=lambda x: abs(x['trinity_score']), reverse=True)
    
    effective_sentiment = sum(s['sentiment'] * s['news_count'] for s in sector_signals) / max(1, total_sector_news) if sector_signals else 0
    regime = 'bullish' if effective_sentiment > 0.1 else 'bearish' if effective_sentiment < -0.1 else 'neutral'
    
    # Generate AI Summary
    ai_summary = generate_ai_summary(sector_signals, top_stocks, regime)
    
    # Prepare headlines for timeline (top 20)
    headlines_timeline = []
    for h in all_headlines[:20]:
        sent = analyze_sentiment_enhanced(h['headline'])
        headlines_timeline.append({
            'text': h['headline'],
            'source': h.get('source', 'Unknown'),
            'timestamp': h.get('timestamp', datetime.now().isoformat()),
            'sentiment': sent['score'],
            'sentiment_label': 'positive' if sent['score'] > 0.1 else 'negative' if sent['score'] < -0.1 else 'neutral'
        })
    
    all_sources = set()
    for s in sector_signals:
        all_sources.update(s.get('sources', []))
    
    return {
        'timestamp': datetime.now().isoformat(),
        'headlines_analyzed': total_headlines,
        'effective_sentiment': round(effective_sentiment, 3),
        'regime': regime,
        'ai_summary': ai_summary,
        'headlines': headlines_timeline,
        'sector_signals': sector_signals,
        'long_sectors': [s['sector'] for s in sector_signals if 'long' in s['recommendation']],
        'short_sectors': [s['sector'] for s in sector_signals if 'short' in s['recommendation']],
        'high_conviction': [s['sector'] for s in sector_signals if s['conviction'] == 'high'],
        'top_stocks': top_stocks,
        'word_cloud': word_cloud,
        'sources': list(all_sources),
        'source_breakdown': {
            'rss_feeds': len(rss_headlines),
            'zerodha_pulse': len(pulse_headlines),
            'google_trends': len(trends_headlines)
        },
        'analysis_version': '4.0 (AI Summary + Timeline)'
    }


# ============== API ENDPOINTS ==============
class AnalysisRequest(BaseModel):
    max_headlines: int = 100
    max_age_hours: int = 24


@app.get("/debug/ner")
def debug_ner():
    """Debug endpoint to test NER extraction"""
    test_headline = "TCS and Infosys report earnings while Nifty 50 drops"
    companies = extract_companies(test_headline)
    return {
        "headline": test_headline,
        "extracted": companies,
        "version": "3.2",
        "aliases_count": len(COMPANY_ALIASES),
        "indices_working": 'NIFTY50' in companies
    }


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    raise HTTPException(status_code=404, detail="Dashboard not found")


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "3.1.0", "timestamp": datetime.now().isoformat()}


@app.post("/api/analyze")
async def analyze_news(request: AnalysisRequest = None):
    try:
        result = run_live_analysis()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sources")
async def list_sources():
    return {
        "rss_feeds": list(RSS_FEEDS.keys()),
        "other_sources": ["zerodha_pulse", "google_trends"],
        "total_feed_urls": sum(len(urls) for urls in RSS_FEEDS.values())
    }


@app.get("/api/demo")
async def get_demo_data():
    return {
        "timestamp": datetime.now().isoformat(),
        "regime": "bullish",
        "effective_sentiment": 0.35,
        "sector_signals": [
            {"sector": "auto", "sentiment": 0.45, "trinity_score": 0.675, "recommendation": "strong_long", "conviction": "high", "rationale": "Positive surprise; High news velocity"},
            {"sector": "banking", "sentiment": 0.65, "trinity_score": 0.475, "recommendation": "strong_long", "conviction": "high", "rationale": "Positive surprise"},
            {"sector": "it", "sentiment": -0.30, "trinity_score": -0.475, "recommendation": "strong_short", "conviction": "high", "rationale": "Negative surprise"},
        ],
        "long_sectors": ["auto", "banking"],
        "short_sectors": ["it"],
        "high_conviction": ["auto", "banking", "it"],
        "headlines_analyzed": 85,
        "analysis_version": "3.1 (VADER + Bigrams + Position Weighting)"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

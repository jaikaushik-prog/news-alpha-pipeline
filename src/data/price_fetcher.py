"""
Live Stock Price Fetcher
Uses yfinance for free stock price data (no API key required)
"""
import functools
import time
from typing import Dict, Optional

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("yfinance not installed. Price data unavailable.")


# NSE stock symbol mapping (add .NS suffix for yfinance)
NSE_SYMBOL_MAP = {
    'TCS': 'TCS.NS',
    'INFY': 'INFY.NS',
    'RELIANCE': 'RELIANCE.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'SBIN': 'SBIN.NS',
    'WIPRO': 'WIPRO.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
    'ADANIENT': 'ADANIENT.NS',
    'TATAMOTORS': 'TATAMOTORS.NS',
    'MARUTI': 'MARUTI.NS',
    'NIFTY50': '^NSEI',
    'SENSEX': '^BSESN',
    'ITC': 'ITC.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
    'ASIANPAINT': 'ASIANPAINT.NS',
    'HINDUNILVR': 'HINDUNILVR.NS',
    'LT': 'LT.NS',
    'TITAN': 'TITAN.NS',
    'M&M': 'M&M.NS',
    'KOTAKBANK': 'KOTAKBANK.NS',
    'AXISBANK': 'AXISBANK.NS',
    # Indicies
    'BANKNIFTY': '^NSEBANK',
    'NIFTYIT': '^CNXIT',
    'NIFTYAUTO': '^CNXAUTO',
    'NIFTYPHARMA': '^CNXPHARMA',
    'NIFTYFMCG': '^CNXFMCG',
    'NIFTYMETAL': '^CNXMETAL',
    'NIFTYINFRA': '^CNXINFRA',
    'NIFTYPSE': '^CNXPSE',
    # Commodities & Forex
    'USDINR': 'INR=X',
    'GOLD': 'GC=F',
    'CRUDEOIL': 'CL=F',
}

# Cache prices for 5 minutes to avoid rate limiting
_price_cache: Dict[str, Dict] = {}
_cache_ttl = 300  # 5 minutes


def get_stock_price(symbol: str) -> Optional[Dict]:
    """
    Get current price and change for a stock symbol.
    Returns: {price: float, change: float, change_pct: float} or None
    """
    if not YFINANCE_AVAILABLE:
        return None
    
    # Check cache
    cache_key = symbol.upper()
    if cache_key in _price_cache:
        cached = _price_cache[cache_key]
        if time.time() - cached['timestamp'] < _cache_ttl:
            return cached['data']
    
    # Convert to yfinance symbol
    yf_symbol = NSE_SYMBOL_MAP.get(cache_key, f"{cache_key}.NS")
    
    try:
        ticker = yf.Ticker(yf_symbol)
        info = ticker.fast_info
        
        current_price = info.last_price
        prev_close = info.previous_close
        
        if current_price and prev_close:
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            data = {
                'price': round(current_price, 2),
                'change': round(change, 2),
                'change_pct': round(change_pct, 2)
            }
            
            # Cache it
            _price_cache[cache_key] = {
                'timestamp': time.time(),
                'data': data
            }
            
            return data
    except Exception as e:
        print(f"Price fetch error for {symbol}: {e}")
    
    return None


def get_multiple_prices(symbols: list) -> Dict[str, Dict]:
    """
    Get prices for multiple symbols.
    Returns dict of {symbol: price_data}
    """
    results = {}
    for symbol in symbols[:10]:  # Limit to 10 to avoid rate limiting
        price_data = get_stock_price(symbol)
        if price_data:
            results[symbol] = price_data
    return results


def enrich_stocks_with_prices(stocks: list) -> list:
    """
    Add price data to a list of stock dictionaries.
    Each stock dict should have a 'symbol' key.
    """
    for stock in stocks:
        symbol = stock.get('symbol', '')
        price_data = get_stock_price(symbol)
        if price_data:
            stock['price'] = price_data['price']
            stock['change'] = price_data['change']
            stock['change_pct'] = price_data['change_pct']
        else:
            stock['price'] = None
            stock['change'] = None
            stock['change_pct'] = None
    return stocks

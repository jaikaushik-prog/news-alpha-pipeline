
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from webapp.api import extract_hot_stocks
from src.data.price_fetcher import enrich_stocks_with_prices, get_stock_price

def test_stock_extraction_and_pricing():
    print("--- Testing Stock Extraction ---")
    
    # Test headlines that should trigger matches with the new keywords
    test_headlines = [
        "Nifty hits all time high as markets rally",
        "Rupee falls against dollar amid global cues",
        "Gold prices surge as investors seek safe haven",
        "IT sector outlook positive says analyst",
        "Banking stocks lead the gains today",
        "Oil prices stable after recent volatility",
        "TCS and Infosys announce new deals",
        "Budget expectations drive market volatility"
    ]
    
    print(f"Testing with {len(test_headlines)} headlines...")
    
    extracted = extract_hot_stocks(test_headlines)
    print(f"Extracted {len(extracted)} stocks:")
    for stock in extracted:
        print(f"  - {stock['symbol']} ({stock['name']}): {stock['mentions']} mentions")
        
    if not extracted:
        print("FAIL: No stocks extracted via keywords.")
        return

    print("\n--- Testing Price Fetching (yfinance) ---")
    try:
        enriched = enrich_stocks_with_prices(extracted)
        print("Enrichment complete. Results:")
        for stock in enriched:
            price = stock.get('price')
            change = stock.get('change_pct')
            print(f"  - {stock['symbol']}: ₹{price} ({change}%)")
            
            if price is None:
                print(f"    WARNING: Price is None for {stock['symbol']}")
    except Exception as e:
        print(f"Enrichment FAILED: {e}")

    print("\n--- Testing Single Stock Fetch ---")
    tcs = get_stock_price("TCS")
    print(f"TCS Check: {tcs}")

if __name__ == "__main__":
    test_stock_extraction_and_pricing()

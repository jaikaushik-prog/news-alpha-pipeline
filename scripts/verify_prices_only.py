
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.price_fetcher import get_stock_price

def verify_prices():
    test_symbols = ['BANKNIFTY', 'NIFTY50', 'USDINR', 'GOLD', 'TCS', 'RELIANCE']
    print(f"Testing price fetch for: {test_symbols}")
    
    for sym in test_symbols:
        try:
            data = get_stock_price(sym)
            if data:
                print(f"✅ {sym}: {data['price']} ({data['change_pct']}%)")
            else:
                print(f"❌ {sym}: FAILED (None)")
        except Exception as e:
            print(f"❌ {sym}: ERROR {e}")

if __name__ == "__main__":
    verify_prices()

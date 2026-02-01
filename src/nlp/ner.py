"""
Named Entity Recognition (NER) for Indian Financial Markets
Identifies companies, indices, and financial entities in news headlines.
"""
import re
from typing import List, Dict, Tuple, Optional

# Mapping of common company names/aliases to their stock tickers/symbols
COMPANY_ALIASES = {
    # IT Services
    'tcs': 'TCS', 'tata consultancy': 'TCS',
    'infosys': 'INFY', 'infy': 'INFY',
    'wipro': 'WIPRO',
    'hcl': 'HCLTECH', 'hcl tech': 'HCLTECH', 'hcl technologies': 'HCLTECH',
    'tech mahindra': 'TECHM', 'techm': 'TECHM',
    'ltimindtree': 'LTIM', 'mindtree': 'LTIM',
    
    # Banking & Finance
    'hdfc bank': 'HDFCBANK', 'hdfc': 'HDFCBANK',
    'icici': 'ICICIBANK', 'icici bank': 'ICICIBANK',
    'sbi': 'SBIN', 'state bank of india': 'SBIN',
    'kotak': 'KOTAKBANK', 'kotak bank': 'KOTAKBANK', 'kotak mahindra': 'KOTAKBANK',
    'axis bank': 'AXISBANK', 'axis': 'AXISBANK',
    'bajaj finance': 'BAJFINANCE',
    'bajaj finserv': 'BAJAJFINSV',
    
    # Reliance / Energy
    'reliance': 'RELIANCE', 'ril': 'RELIANCE', 'reliance industries': 'RELIANCE',
    'adani ent': 'ADANIENT', 'adani enterprises': 'ADANIENT',
    'adani ports': 'ADANIPORTS',
    'adani green': 'ADANIGREEN',
    'adani power': 'ADANIPOWER',
    'ongc': 'ONGC',
    'ntpc': 'NTPC',
    'power grid': 'POWERGRID',
    'coal india': 'COALINDIA',
    
    # Auto
    'tata motors': 'TATAMOTORS', 'tamo': 'TATAMOTORS',
    'mahindra': 'M&M', 'm&m': 'M&M', 'mahindra & mahindra': 'M&M',
    'maruti': 'MARUTI', 'maruti suzuki': 'MARUTI',
    'bajaj auto': 'BAJAJ-AUTO',
    'eicher': 'EICHERMOT', 'royal enfield': 'EICHERMOT',
    'tvs': 'TVSMOTOR',
    'hero': 'HEROMOTOCO', 'hero motocorp': 'HEROMOTOCO',
    
    # FMCG
    'itc': 'ITC',
    'hul': 'HINDUNILVR', 'hindustan unilever': 'HINDUNILVR',
    'nestle': 'NESTLEIND',
    'britannia': 'BRITANNIA',
    'titan': 'TITAN',
    
    # Infra / Materials
    'l&t': 'LT', 'larsen': 'LT', 'larsen & toubro': 'LT',
    'ultratech': 'ULTRACEMCO',
    'asian paints': 'ASIANPAINT',
    'tata steel': 'TATASTEEL',
    'jsw steel': 'JSWSTEEL',
    'hindalco': 'HINDALCO',
    
    # Pharma
    'sun pharma': 'SUNPHARMA',
    'cipla': 'CIPLA',
    'dr reddy': 'DRREDDY', 'dr. reddy': 'DRREDDY',
    'divi': 'DIVISLAB', 'divis': 'DIVISLAB',
    
    # Telecom
    'airtel': 'BHARTIARTL', 'bharti airtel': 'BHARTIARTL',
    'geo': 'RELIANCE', # Jio often implies Reliance
    'vodafone': 'IDEA', 'vi': 'IDEA', 'idea': 'IDEA',

    # Indices & Commodities (Treating as "Stocks" for this widget)
    'nifty': 'NIFTY50', 'nifty 50': 'NIFTY50',
    'sensex': 'SENSEX', 'bse sensex': 'SENSEX',
    'bank nifty': 'BANKNIFTY', 'banknifty': 'BANKNIFTY',
    'gold': 'GOLD',
    'silver': 'SILVER',
    'crude': 'CRUDEOIL', 'oil': 'CRUDEOIL',
}

# Inverted index for reverse lookups (Symbol -> Name)
SYMBOL_TO_NAME = {
    'TCS': 'Tata Consultancy Services',
    'INFY': 'Infosys Ltd',
    'WIPRO': 'Wipro Ltd',
    'HCLTECH': 'HCL Technologies',
    'TECHM': 'Tech Mahindra',
    'HDFCBANK': 'HDFC Bank',
    'ICICIBANK': 'ICICI Bank',
    'SBIN': 'State Bank of India',
    'KOTAKBANK': 'Kotak Mahindra Bank',
    'AXISBANK': 'Axis Bank',
    'RELIANCE': 'Reliance Industries',
    'ADANIENT': 'Adani Enterprises',
    'TATAMOTORS': 'Tata Motors',
    'M&M': 'Mahindra & Mahindra',
    'MARUTI': 'Maruti Suzuki',
    'ITC': 'ITC Ltd',
    'HINDUNILVR': 'Hindustan Unilever',
    'LT': 'Larsen & Toubro',
    'BHARTIARTL': 'Bharti Airtel',
    'NIFTY50': 'Nifty 50 Index',
    'SENSEX': 'BSE Sensex Index',
    'BANKNIFTY': 'Bank Nifty Index',
    'GOLD': 'Gold (Commodity)',
    'SILVER': 'Silver (Commodity)',
    'CRUDEOIL': 'Crude Oil',
}



def extract_companies(text: str) -> List[str]:
    """
    Extract list of stock symbols from text.
    Returns list of unique symbols found (e.g., ['TCS', 'INFY'])
    """
    text_lower = text.lower()
    found_symbols = set()
    
    # 1. Exact matches from alias map
    # Sort aliases by length (desc) to match longer phrases first 
    # (e.g., 'tata motors' before 'tata')
    sorted_aliases = sorted(COMPANY_ALIASES.keys(), key=len, reverse=True)
    
    for alias in sorted_aliases:
        # Use regex boundary matching to avoid partial words 
        # (e.g. 'sun' shouldn't match 'sunday')
        # Escape alias for regex safety
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, text_lower):
            found_symbols.add(COMPANY_ALIASES[alias])
            
    return list(found_symbols)

def get_company_name(symbol: str) -> str:
    """Get full company name from symbol"""
    return SYMBOL_TO_NAME.get(symbol, symbol)

if __name__ == "__main__":
    # Test
    headlines = [
        "TCS and Infosys report strong Q3 earnings beat",
        "Reliance to demerge financial services arm",
        "HDFC Bank share price jumps on merger news",
        "Maruti Suzuki sales drop, but Tata Motors gains share",
        "Adani Ent falls 10% after report",
        "RBI policy to impact banking stocks like SBI and ICICI"
    ]
    
    print("Testing Company NER:")
    for h in headlines:
        print(f"\nHeadline: {h}")
        extracted = extract_companies(h)
        print(f"Entities: {extracted}")

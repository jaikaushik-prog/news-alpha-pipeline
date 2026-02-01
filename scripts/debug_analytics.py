
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from webapp.api import fetch_zerodha_pulse, fetch_rss_headlines, generate_word_cloud
from src.nlp.ner import extract_companies

def debug_analytics():
    print("Fetching headlines...")
    pulse = fetch_zerodha_pulse()
    rss = fetch_rss_headlines(50)
    
    all_headlines = pulse + rss
    print(f"Total headlines: {len(all_headlines)}")
    
    if not all_headlines:
        print("No headlines found!")
        return

    print("\nSample Headlines:")
    for h in all_headlines[:5]:
        print(f"- {h['headline']}")
        
    print("\nTesting Company Extraction:")
    companies_found = []
    for h in all_headlines:
        comps = extract_companies(h['headline'])
        if comps:
            companies_found.extend(comps)
            print(f"MATCH: '{h['headline']}' -> {comps}")
            
    print(f"\nTotal Companies Found: {len(companies_found)}")
    
    print("\nTesting Word Cloud:")
    cloud = generate_word_cloud([h['headline'] for h in all_headlines])
    print(f"Word Cloud Items: {len(cloud)}")
    if cloud:
        print("Top 5 Cloud Items:")
        for item in cloud[:5]:
            print(item)

if __name__ == "__main__":
    debug_analytics()

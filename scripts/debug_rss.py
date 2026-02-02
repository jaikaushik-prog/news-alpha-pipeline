
import sys
import os
import urllib.request
import xml.etree.ElementTree as ET
import concurrent.futures
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RSS_FEEDS = {
    'economic_times': [
        'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
    ],
    'moneycontrol': [
        'https://www.moneycontrol.com/rss/latestnews.xml',
    ],
    'mint': [
        'https://www.livemint.com/rss/markets',
    ]
}

def fetch_feed(url):
    print(f"Fetching {url}...")
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read()
            # Try parsing
            try:
                root = ET.fromstring(content)
                items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')
                return True, len(items), url
            except Exception as e:
                return False, f"Parse Error: {e}", url
    except Exception as e:
        return False, f"Net Error: {e}", url

def main():
    print("Testing RSS Feed Fetching...")
    
    urls = []
    for source, feed_urls in RSS_FEEDS.items():
        urls.extend(feed_urls)
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_feed, url) for url in urls]
        for future in concurrent.futures.as_completed(futures):
            success, result, url = future.result()
            if success:
                print(f"SUCCESS: {url} - Found {result} items")
            else:
                print(f"FAILED: {url} - {result}")

if __name__ == "__main__":
    main()

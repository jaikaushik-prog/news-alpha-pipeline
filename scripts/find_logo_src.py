
import requests
from bs4 import BeautifulSoup
import urllib.parse

def find_logo():
    url = "https://www.wallstreetbitsp.org/"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')
        
        images = soup.find_all('img')
        print(f"Found {len(images)} images.")
        
        for img in images:
            src = img.get('src')
            if src:
                full_url = urllib.parse.urljoin(url, src)
                print(f"Image: {full_url} | Alt: {img.get('alt', '')} | Class: {img.get('class', '')}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_logo()

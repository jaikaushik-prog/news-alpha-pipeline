
import requests
import json
import sys

def verify_sources():
    print("Calling /api/analyze...")
    try:
        response = requests.post("http://localhost:8000/api/analyze", json={})
        if response.status_code == 200:
            data = response.json()
            headlines = data.get('headlines', [])
            print(f"Received {len(headlines)} headlines in timeline.")
            
            print("\nTop 10 Headlines Sources:")
            for i, h in enumerate(headlines[:10]):
                print(f"{i+1}. {h.get('source')} - {h.get('text')[:30]}...")
            
            sources = set(h.get('source') for h in headlines[:20])
            print(f"\nUnique sources in top 20: {sources}")
            
            if len(sources) > 1:
                print("\nSUCCESS: Multiple sources found in trending list.")
            else:
                print("\nFAILURE: Only one source found.")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    verify_sources()

"""
Gemini AI Integration for Market Summaries
Uses Google's Gemini API for intelligent, contextual market analysis.

Setup:
- Set GEMINI_API_KEY environment variable
"""
import os
from typing import Dict, List, Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("google-generativeai not installed. Gemini AI unavailable.")


def initialize_gemini() -> bool:
    """Initialize Gemini with API key from environment."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("GEMINI_API_KEY not set. Falling back to template-based summary.")
        return False
    
    if not GEMINI_AVAILABLE:
        return False
    
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"Gemini initialization failed: {e}")
        return False


def generate_gemini_summary(
    sector_signals: List[Dict],
    top_stocks: List[Dict],
    regime: str,
    headlines: List[str] = None
) -> Optional[str]:
    """
    Generate an intelligent market summary using Gemini.
    
    Args:
        sector_signals: List of sector data with sentiment, score, recommendation
        top_stocks: List of most mentioned stocks
        regime: Current market regime (bullish/neutral/bearish)
        headlines: Optional list of recent headlines for context
    
    Returns:
        AI-generated summary string or None if failed
    """
    if not initialize_gemini():
        return None
    
    # Build context for Gemini
    sectors_context = "\n".join([
        f"- {s['sector'].upper()}: Sentiment {s['sentiment']:.2f}, Signal: {s['recommendation']}, Conviction: {s['conviction']}"
        for s in sector_signals[:7]
    ])
    
    stocks_context = "\n".join([
        f"- {s['symbol']} ({s['name']}): {s['mentions']} mentions"
        for s in top_stocks[:5]
    ]) if top_stocks else "No specific stocks heavily mentioned."
    
    headlines_context = ""
    if headlines:
        headlines_context = "\nRecent Headlines:\n" + "\n".join([f"- {h}" for h in headlines[:5]])
    
    prompt = f"""You are a financial market analyst. Generate a concise, professional 2-3 sentence market intelligence brief based on this data:

Market Regime: {regime.upper()}

Sector Analysis:
{sectors_context}

Most Active Stocks:
{stocks_context}
{headlines_context}

Instructions:
- Be concise (2-3 sentences max)
- Start with the overall market mood
- Highlight the best and worst performing sectors
- Mention any notable stocks if relevant
- Use professional financial language
- Include specific numbers where impactful
- End with a forward-looking insight or caution if appropriate

Generate the market brief:"""

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=200,
                temperature=0.7
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None


# Cache for summary to avoid repeated API calls
_summary_cache: Dict = {}
_cache_ttl = 300  # 5 minutes


def get_cached_gemini_summary(
    sector_signals: List[Dict],
    top_stocks: List[Dict],
    regime: str,
    headlines: List[str] = None
) -> Optional[str]:
    """Get summary with caching to reduce API calls."""
    import time
    import hashlib
    
    # Create cache key from inputs
    cache_key = hashlib.md5(
        f"{regime}{len(sector_signals)}{len(top_stocks or [])}".encode()
    ).hexdigest()
    
    # Check cache
    if cache_key in _summary_cache:
        cached = _summary_cache[cache_key]
        if time.time() - cached['timestamp'] < _cache_ttl:
            return cached['summary']
    
    # Generate new summary
    summary = generate_gemini_summary(sector_signals, top_stocks, regime, headlines)
    
    if summary:
        import time
        _summary_cache[cache_key] = {
            'timestamp': time.time(),
            'summary': summary
        }
    
    return summary

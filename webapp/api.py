"""
News Alpha Pipeline - FastAPI Backend

Deployable REST API for the institutional news alpha pipeline.

Run:
    uvicorn webapp.api:app --reload --port 8000

Endpoints:
    GET  /                      → Dashboard page
    GET  /api/health            → Health check
    POST /api/analyze           → Run full pipeline
    GET  /api/sectors           → Get sector sentiment
    GET  /api/signals           → Get trading signals
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Initialize FastAPI
app = FastAPI(
    title="News Alpha Pipeline",
    description="Institutional-grade news sentiment analysis for alpha generation",
    version="1.0.0"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================================
# Models
# =========================================================================

class AnalysisRequest(BaseModel):
    max_headlines: int = 50
    max_age_hours: int = 24


class SectorSignal(BaseModel):
    sector: str
    sentiment: float
    trinity_score: float
    recommendation: str
    conviction: str
    rationale: str


class AnalysisResponse(BaseModel):
    timestamp: str
    regime: str
    effective_sentiment: float
    sector_signals: List[SectorSignal]
    long_sectors: List[str]
    short_sectors: List[str]
    high_conviction: List[str]
    headlines_analyzed: int


# =========================================================================
# Global state (for demo - use Redis/DB in production)
# =========================================================================

_latest_analysis: Optional[Dict] = None


# =========================================================================
# API Endpoints
# =========================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
async def run_analysis(request: AnalysisRequest):
    """
    Run the full news alpha pipeline.
    
    Returns sector signals with Holy Trinity scoring.
    """
    global _latest_analysis
    
    try:
        # Import pipeline components
        from src.pipeline.news_alpha_pipeline import NewsAlphaPipeline
        from src.models.holy_trinity import HolyTrinityModel
        
        # Run main pipeline
        pipeline = NewsAlphaPipeline()
        result = pipeline.run(
            max_headlines=request.max_headlines,
            max_age_hours=request.max_age_hours
        )
        
        # Initialize Holy Trinity
        trinity = HolyTrinityModel()
        
        # Build historical baseline from analyzed items
        for item in result.analyzed_items[:20]:
            for sector in ['banking', 'it', 'pharma', 'auto', 'metals', 'infra']:
                trinity.update_historical(
                    sector=sector,
                    sentiment=item.get('sentiment', 0),
                    volume=1,
                    intensity=0.5
                )
        
        # Prepare sector data for Trinity analysis
        sector_data = {}
        for sector, score in result.sector_sentiment.items():
            sector_data[sector] = {
                'sentiment': score,
                'volume': sum(1 for i in result.analyzed_items if sector in i.get('headline', '').lower()),
                'intensity': abs(score),
                'price_return': 0,  # Would need real price data
                'volume_ratio': 1.0
            }
        
        # Analyze with Trinity
        trinity_signals = trinity.batch_analyze(sector_data)
        
        # Build response
        sector_signals = []
        long_sectors = []
        short_sectors = []
        high_conviction = []
        
        for sig in trinity_signals:
            sector_signals.append(SectorSignal(
                sector=sig.sector,
                sentiment=result.sector_sentiment.get(sig.sector, 0),
                trinity_score=sig.trinity_score,
                recommendation=sig.trade_recommendation,
                conviction=sig.conviction,
                rationale=sig.rationale
            ))
            
            if 'long' in sig.trade_recommendation:
                long_sectors.append(sig.sector)
            elif 'short' in sig.trade_recommendation:
                short_sectors.append(sig.sector)
            
            if sig.conviction == 'high':
                high_conviction.append(sig.sector)
        
        response = AnalysisResponse(
            timestamp=datetime.now().isoformat(),
            regime=result.regime,
            effective_sentiment=result.effective_sentiment,
            sector_signals=sector_signals,
            long_sectors=long_sectors,
            short_sectors=short_sectors,
            high_conviction=high_conviction,
            headlines_analyzed=len(result.analyzed_items)
        )
        
        _latest_analysis = response.dict()
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sectors")
async def get_sectors():
    """Get latest sector sentiment."""
    if _latest_analysis is None:
        return {"message": "No analysis run yet", "sectors": []}
    
    return {
        "timestamp": _latest_analysis.get("timestamp"),
        "sectors": _latest_analysis.get("sector_signals", [])
    }


@app.get("/api/signals")
async def get_signals():
    """Get trading signals."""
    if _latest_analysis is None:
        return {"message": "No analysis run yet", "signals": {}}
    
    return {
        "timestamp": _latest_analysis.get("timestamp"),
        "regime": _latest_analysis.get("regime"),
        "long": _latest_analysis.get("long_sectors", []),
        "short": _latest_analysis.get("short_sectors", []),
        "high_conviction": _latest_analysis.get("high_conviction", [])
    }


@app.get("/api/demo")
async def demo_analysis():
    """
    Demo endpoint with mock data.
    
    Use this when news scraping is unavailable.
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "regime": "bullish",
        "effective_sentiment": 0.35,
        "sector_signals": [
            {"sector": "banking", "sentiment": 0.65, "trinity_score": 0.475, "recommendation": "strong_long", "conviction": "high", "rationale": "Positive surprise; early mover"},
            {"sector": "auto", "sentiment": 0.45, "trinity_score": 0.675, "recommendation": "strong_long", "conviction": "high", "rationale": "Positive surprise; early mover"},
            {"sector": "it", "sentiment": -0.30, "trinity_score": -0.475, "recommendation": "strong_short", "conviction": "high", "rationale": "Negative surprise; smart money accumulating"},
            {"sector": "pharma", "sentiment": 0.10, "trinity_score": 0.304, "recommendation": "long", "conviction": "medium", "rationale": "Modest surprise"},
            {"sector": "metals", "sentiment": -0.40, "trinity_score": -0.475, "recommendation": "short", "conviction": "medium", "rationale": "Negative surprise"},
            {"sector": "infra", "sentiment": 0.05, "trinity_score": 0.060, "recommendation": "hold", "conviction": "low", "rationale": "In-line with expectations"}
        ],
        "long_sectors": ["banking", "auto", "pharma"],
        "short_sectors": ["it", "metals"],
        "high_conviction": ["banking", "auto", "it"],
        "headlines_analyzed": 47
    }


# =========================================================================
# Serve Frontend
# =========================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard."""
    dashboard_path = project_root / "webapp" / "index.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    else:
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)


# Mount static files
static_path = project_root / "webapp" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# =========================================================================
# Run with uvicorn
# =========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

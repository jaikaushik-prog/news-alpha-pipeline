# News Alpha Pipeline - Webapp

Web dashboard for the institutional-grade news sentiment analysis pipeline.

## Quick Start

```powershell
# Install dependencies
pip install fastapi uvicorn

# Run the server
cd c:\Users\DELL\Desktop\budget_speech
uvicorn webapp.api:app --reload --port 8000

# Open browser
# http://localhost:8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/health` | Health check |
| POST | `/api/analyze` | Run full pipeline |
| GET | `/api/sectors` | Get sector sentiment |
| GET | `/api/signals` | Get trading signals |
| GET | `/api/demo` | Demo with mock data |

## Files

- `api.py` - FastAPI backend
- `index.html` - Dashboard frontend

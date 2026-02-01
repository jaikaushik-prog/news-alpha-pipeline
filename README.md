# 🏆 News Alpha Pipeline

Institutional-grade news sentiment analysis for alpha generation.


## 📁 Project Structure

```
budget_speech/
├── webapp/
│   ├── api.py          # FastAPI backend
│   └── index.html      # Dashboard frontend
├── src/
│   ├── data/           # News collection
│   ├── nlp/            # Text processing & sentiment
│   ├── models/         # Holy Trinity models
│   ├── strategy/       # Alpha construction
│   └── pipeline/       # Orchestration
├── requirements.txt    # Dependencies
├── Procfile           # Deployment command
└── render.yaml        # Render.com config
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| POST | `/api/analyze` | Run full pipeline |
| GET | `/api/demo` | Demo data |
| GET | `/api/health` | Health check |

## 🏆 Holy Trinity Architecture

1. **Expectation Gap** - Sentiment surprise vs baseline
2. **Narrative Velocity** - Speed of information spread  
3. **Sentiment-Price Divergence** - Smart money detection

---
Built with FastAPI + Python 🐍

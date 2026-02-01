# 🏆 News Alpha Pipeline

Institutional-grade news sentiment analysis for alpha generation.

## 🚀 Quick Start (Local)

```powershell
# Install dependencies
pip install -r requirements.txt

# Run server
python -m uvicorn webapp.api:app --reload --port 8000

# Open browser: http://localhost:8000
```

## 🌐 Deploy to Internet

### Option 1: Render.com (FREE - Recommended)

1. **Push to GitHub**
   ```powershell
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/news-alpha-pipeline.git
   git push -u origin main
   ```

2. **Deploy on Render**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repo
   - It will auto-detect `render.yaml`
   - Click "Create Web Service"
   - Wait 2-3 minutes for deployment

3. **Your app is live at:** `https://news-alpha-pipeline.onrender.com`

### Option 2: Railway.app (FREE)

```powershell
npm install -g @railway/cli
railway login
railway init
railway up
```

### Option 3: Ngrok (Quick temporary URL)

```powershell
# While server is running locally:
ngrok http 8000
# Get public URL like: https://abc123.ngrok.io
```

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

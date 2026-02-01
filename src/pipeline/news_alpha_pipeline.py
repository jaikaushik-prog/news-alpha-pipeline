"""
News Alpha Pipeline - Unified Orchestrator

Complete pipeline: Information → Surprise → Belief → Market Reaction → Alpha

Layers:
1. Data Acquisition
2. Text Preprocessing  
3. Semantic Embeddings + Surprise
4. Multi-Factor Sentiment
5. Sector Attribution
6. Temporal Decay
7. Regime Conditioning
8. Market Impact
9. Alpha Construction
"""

from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ..data.news_collector import collect_all_news, NewsItem, to_dataframe
from ..nlp.text_preprocessor import TextPreprocessor, preprocess_headlines
from ..nlp.multifactor_sentiment import MultifactorSentimentAnalyzer, SentimentVector
from ..models.sector_attribution import SectorAttributionEngine, aggregate_sector_sentiment
from ..models.temporal_decay import TemporalDecayEngine, aggregate_decayed_sentiment
from ..strategy.alpha_construction import AlphaConstructor, get_alpha_portfolio
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Complete pipeline output."""
    timestamp: datetime
    headline_count: int
    sector_sentiment: Dict[str, float]
    sector_ranking: List[tuple]
    alpha_signals: pd.DataFrame
    portfolio_recommendation: Dict
    effective_sentiment: float
    regime: str
    errors: List[str]


class NewsAlphaPipeline:
    """
    Complete News → Alpha Pipeline.
    
    Usage:
        pipeline = NewsAlphaPipeline()
        result = pipeline.run()
        print(result.portfolio_recommendation)
    """
    
    def __init__(
        self,
        use_embeddings: bool = False,  # Requires sentence-transformers
        regime_weight: float = 1.0,
        long_threshold: float = 0.25,
        short_threshold: float = 0.75
    ):
        self.use_embeddings = use_embeddings
        self.regime_weight = regime_weight
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.sentiment_analyzer = MultifactorSentimentAnalyzer()
        self.sector_engine = SectorAttributionEngine()
        self.decay_engine = TemporalDecayEngine()
        self.alpha_constructor = AlphaConstructor(
            long_threshold=long_threshold,
            short_threshold=short_threshold
        )
        
        # Optional: Embedding model (lazy load)
        self._embedding_model = None
        
        self.errors = []
    
    def run(
        self,
        max_headlines: int = 50,
        max_age_hours: int = 24
    ) -> PipelineResult:
        """
        Run complete pipeline.
        
        Parameters
        ----------
        max_headlines : int
            Maximum headlines to process
        max_age_hours : int
            Maximum age of news
            
        Returns
        -------
        PipelineResult
            Complete pipeline output
        """
        logger.info("=" * 50)
        logger.info("STARTING NEWS ALPHA PIPELINE")
        logger.info("=" * 50)
        
        self.errors = []
        
        # Layer 1: Collect News
        logger.info("Layer 1: Collecting news...")
        try:
            news_items = collect_all_news(max_age_hours=max_age_hours)
            news_items = news_items[:max_headlines]
        except Exception as e:
            logger.error(f"News collection failed: {e}")
            self.errors.append(f"Layer 1: {e}")
            news_items = []
        
        if not news_items:
            logger.warning("No news items collected, using mock data")
            news_items = self._create_mock_news()
        
        # Layer 2: Preprocess
        logger.info(f"Layer 2: Preprocessing {len(news_items)} headlines...")
        headlines = [item.headline for item in news_items]
        
        # Layer 3-4: Sentiment Analysis
        logger.info("Layers 3-4: Analyzing sentiment...")
        analyzed_items = []
        
        for item in news_items:
            try:
                sentiment = self.sentiment_analyzer.analyze(item.headline)
                analyzed_items.append({
                    'headline': item.headline,
                    'timestamp': item.timestamp,
                    'source': item.source,
                    'category': item.category,
                    'sentiment': sentiment['composite_score'],
                    'polarity': sentiment['polarity'],
                    'intensity': sentiment['intensity'],
                    'certainty': sentiment['certainty'],
                    'urgency': sentiment['urgency'],
                    'risk_tone': sentiment['risk_tone'],
                    'surprise': 1.0  # Default if not using embeddings
                })
            except Exception as e:
                self.errors.append(f"Sentiment error: {e}")
                continue
        
        # Layer 5: Sector Attribution
        logger.info("Layer 5: Attributing to sectors...")
        sector_sentiment = aggregate_sector_sentiment(
            analyzed_items,
            sentiment_key='sentiment',
            surprise_key='surprise'
        )
        
        # Layer 6: Temporal Decay
        logger.info("Layer 6: Applying temporal decay...")
        effective_sentiment = aggregate_decayed_sentiment(analyzed_items)
        
        # Layer 7: Regime Conditioning
        logger.info("Layer 7: Checking regime...")
        regime = self._detect_regime(effective_sentiment)
        regime_weight = self._get_regime_weight(regime)
        
        # Layer 9: Alpha Construction
        logger.info("Layer 9: Constructing alpha signals...")
        
        # Prepare sector data
        sector_data = {
            sector: {
                'sentiment': score,
                'surprise': 1.0,
                'liquidity': 1.0
            }
            for sector, score in sector_sentiment.items()
        }
        
        alpha_signals = self.alpha_constructor.generate(sector_data, regime_weight)
        alpha_df = self.alpha_constructor.to_dataframe(alpha_signals)
        
        # Generate recommendations
        portfolio = self.alpha_constructor.summary(alpha_signals)
        
        # Sector ranking
        sector_ranking = self.sector_engine.rank(sector_sentiment)
        
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 50)
        
        return PipelineResult(
            timestamp=datetime.now(),
            headline_count=len(analyzed_items),
            sector_sentiment=sector_sentiment,
            sector_ranking=sector_ranking,
            alpha_signals=alpha_df,
            portfolio_recommendation=portfolio,
            effective_sentiment=effective_sentiment,
            regime=regime,
            errors=self.errors
        )
    
    def _detect_regime(self, sentiment: float) -> str:
        """Detect market regime from sentiment."""
        if sentiment > 0.3:
            return 'risk_on'
        elif sentiment < -0.3:
            return 'risk_off'
        else:
            return 'transitional'
    
    def _get_regime_weight(self, regime: str) -> float:
        """Get regime adjustment weight."""
        weights = {
            'risk_on': 1.2,
            'risk_off': 0.8,
            'transitional': 1.0
        }
        return weights.get(regime, 1.0)
    
    def _create_mock_news(self) -> List[NewsItem]:
        """Create mock news for demonstration."""
        now = datetime.now()
        
        mock_headlines = [
            ("RBI keeps repo rate unchanged, maintains accommodative stance", "policy"),
            ("Infosys beats Q3 estimates, raises FY24 guidance", "earnings"),
            ("Banking stocks rally on better-than-expected credit growth", "general"),
            ("Auto sales surge 18% in January on festive demand", "macro"),
            ("Pharma sector faces headwinds on FDA inspection concerns", "general"),
            ("Infrastructure push: Govt allocates Rs 10 lakh cr for roads", "policy"),
            ("FIIs turn net buyers after 3 months of selling", "macro"),
            ("Metal stocks under pressure on China demand worries", "general"),
            ("IT hiring freeze continues amid global slowdown fears", "general"),
            ("Reliance announces major renewable energy investment", "earnings"),
        ]
        
        items = []
        for headline, category in mock_headlines:
            items.append(NewsItem(
                headline=headline,
                timestamp=now,
                source="Mock News",
                category=category
            ))
        
        return items
    
    def print_summary(self, result: PipelineResult):
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print("📰 NEWS ALPHA PIPELINE RESULTS")
        print("=" * 60)
        
        print(f"\n📊 Analysis Summary:")
        print(f"   Headlines Analyzed: {result.headline_count}")
        print(f"   Effective Sentiment: {result.effective_sentiment:+.3f}")
        print(f"   Market Regime: {result.regime.upper()}")
        
        print(f"\n📈 Top Sectors (Long Candidates):")
        for sector, score in result.sector_ranking[:3]:
            emoji = "🟢" if score > 0 else "🔴"
            print(f"   {emoji} {sector}: {score:+.3f}")
        
        print(f"\n📉 Bottom Sectors (Short Candidates):")
        for sector, score in result.sector_ranking[-3:]:
            emoji = "🟢" if score > 0 else "🔴"
            print(f"   {emoji} {sector}: {score:+.3f}")
        
        print(f"\n💼 Portfolio Recommendation:")
        print(f"   LONG: {', '.join(result.portfolio_recommendation['long_sectors']) or 'None'}")
        print(f"   SHORT: {', '.join(result.portfolio_recommendation['short_sectors']) or 'None'}")
        
        if result.errors:
            print(f"\n⚠️ Warnings: {len(result.errors)}")
        
        print("\n" + "=" * 60)


# Convenience function
def run_pipeline() -> PipelineResult:
    """Quick pipeline execution."""
    pipeline = NewsAlphaPipeline()
    return pipeline.run()

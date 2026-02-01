"""
FinBERT Sentiment Analyzer - Production Upgrade

Uses HuggingFace FinBERT for financial sentiment analysis.
Falls back to keyword-based if FinBERT unavailable.

FinBERT: Pre-trained on financial text (SEC filings, analyst reports)
Much more accurate than generic BERT or keyword matching.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FinBERTResult:
    """FinBERT analysis result."""
    text: str
    sentiment: str            # 'positive', 'negative', 'neutral'
    score: float              # -1 to +1
    confidence: float         # 0 to 1
    probabilities: Dict[str, float]  # {positive, negative, neutral}


class FinBERTAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial text.
    
    Uses ProsusAI/finbert from HuggingFace.
    
    Usage:
        analyzer = FinBERTAnalyzer()
        result = analyzer.analyze("Markets rally on positive earnings")
        print(result.score)  # 0.85
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize FinBERT analyzer.
        
        Parameters
        ----------
        device : str
            'auto', 'cpu', or 'cuda'
        """
        self.device = device
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._initialized = False
        self._use_fallback = False
    
    def _load_model(self):
        """Lazy load FinBERT model."""
        if self._initialized:
            return
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
            import torch
            
            logger.info("Loading FinBERT model (ProsusAI/finbert)...")
            
            # Determine device
            if self.device == 'auto':
                device = 0 if torch.cuda.is_available() else -1
            elif self.device == 'cuda':
                device = 0
            else:
                device = -1
            
            # Load model and tokenizer
            model_name = "ProsusAI/finbert"
            
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Create pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model,
                tokenizer=self._tokenizer,
                device=device,
                top_k=None  # Return all scores
            )
            
            self._initialized = True
            logger.info("✓ FinBERT loaded successfully")
            
        except ImportError as e:
            logger.warning(f"transformers not installed: {e}")
            logger.warning("Falling back to keyword-based sentiment")
            self._use_fallback = True
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Error loading FinBERT: {e}")
            logger.warning("Falling back to keyword-based sentiment")
            self._use_fallback = True
            self._initialized = True
    
    def analyze(self, text: str) -> FinBERTResult:
        """
        Analyze text sentiment using FinBERT.
        
        Parameters
        ----------
        text : str
            Financial text to analyze
            
        Returns
        -------
        FinBERTResult
            Complete sentiment analysis
        """
        self._load_model()
        
        if self._use_fallback:
            return self._fallback_analyze(text)
        
        try:
            # Truncate if too long
            if len(text) > 500:
                text = text[:500]
            
            # Get predictions
            results = self._pipeline(text)[0]
            
            # Parse results
            probs = {r['label'].lower(): r['score'] for r in results}
            
            # Determine sentiment and score
            sentiment = max(probs, key=probs.get)
            confidence = probs[sentiment]
            
            # Convert to -1 to +1 score
            score = probs.get('positive', 0) - probs.get('negative', 0)
            
            return FinBERTResult(
                text=text[:100] + "..." if len(text) > 100 else text,
                sentiment=sentiment,
                score=score,
                confidence=confidence,
                probabilities=probs
            )
            
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return self._fallback_analyze(text)
    
    def _fallback_analyze(self, text: str) -> FinBERTResult:
        """Keyword-based fallback."""
        text_lower = text.lower()
        
        positive_kw = [
            'surge', 'rally', 'gain', 'jump', 'beat', 'profit', 'growth',
            'bullish', 'upgrade', 'outperform', 'strong', 'positive'
        ]
        negative_kw = [
            'fall', 'drop', 'crash', 'plunge', 'miss', 'loss', 'weak',
            'bearish', 'downgrade', 'underperform', 'decline', 'negative'
        ]
        
        pos_count = sum(1 for kw in positive_kw if kw in text_lower)
        neg_count = sum(1 for kw in negative_kw if kw in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            score = 0
            sentiment = 'neutral'
        else:
            score = (pos_count - neg_count) / total
            sentiment = 'positive' if score > 0.1 else ('negative' if score < -0.1 else 'neutral')
        
        return FinBERTResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            sentiment=sentiment,
            score=score,
            confidence=0.6,  # Lower confidence for fallback
            probabilities={
                'positive': max(0, score),
                'negative': max(0, -score),
                'neutral': 1 - abs(score)
            }
        )
    
    def batch_analyze(self, texts: List[str]) -> List[FinBERTResult]:
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]
    
    def analyze_with_aspects(self, text: str) -> Dict:
        """
        Analyze with aspect breakdown.
        
        Returns sentiment + key aspects detected.
        """
        result = self.analyze(text)
        
        # Detect aspects
        aspects = {
            'earnings': any(kw in text.lower() for kw in ['earnings', 'profit', 'revenue', 'q1', 'q2', 'q3', 'q4']),
            'guidance': any(kw in text.lower() for kw in ['guidance', 'outlook', 'forecast', 'expects']),
            'macro': any(kw in text.lower() for kw in ['gdp', 'inflation', 'rbi', 'rate', 'policy']),
            'deal': any(kw in text.lower() for kw in ['acquisition', 'merger', 'deal', 'partnership']),
            'management': any(kw in text.lower() for kw in ['ceo', 'cfo', 'management', 'board', 'appoint'])
        }
        
        return {
            'sentiment': result.sentiment,
            'score': result.score,
            'confidence': result.confidence,
            'aspects': [k for k, v in aspects.items() if v]
        }


# Global instance for convenience
_finbert_instance = None


def get_finbert_analyzer() -> FinBERTAnalyzer:
    """Get or create global FinBERT instance."""
    global _finbert_instance
    if _finbert_instance is None:
        _finbert_instance = FinBERTAnalyzer()
    return _finbert_instance


def analyze_sentiment_finbert(text: str) -> float:
    """Quick FinBERT sentiment score."""
    analyzer = get_finbert_analyzer()
    result = analyzer.analyze(text)
    return result.score


def batch_analyze_finbert(texts: List[str]) -> List[float]:
    """Quick batch analysis."""
    analyzer = get_finbert_analyzer()
    results = analyzer.batch_analyze(texts)
    return [r.score for r in results]

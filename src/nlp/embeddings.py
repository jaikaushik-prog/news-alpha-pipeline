"""
Semantic Embeddings & Surprise Score - Layer 3

Financial text embeddings using sentence transformers.
Calculates surprise/novelty score for each headline.

Surprise_t = Distance(Embedding_t, Mean(Embeddings_{t-N}))

Higher surprise = less priced-in information = more alpha potential
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Model configurations
EMBEDDING_MODELS = {
    'finbert': 'ProsusAI/finbert',
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    'mpnet': 'sentence-transformers/all-mpnet-base-v2'
}

DEFAULT_MODEL = 'minilm'  # Good balance of speed and quality


class EmbeddingModel:
    """
    Sentence embedding model for financial text.
    
    Uses SentenceTransformers for efficient embedding generation.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize embedding model.
        
        Parameters
        ----------
        model_name : str
            Model key from EMBEDDING_MODELS or HuggingFace path
        """
        self.model_name = EMBEDDING_MODELS.get(model_name, model_name)
        self.model = None
        self._dimension = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self.model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            test_embedding = self.model.encode("test")
            self._dimension = len(test_embedding)
            
            logger.info(f"Model loaded. Embedding dimension: {self._dimension}")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            self._load_model()
        return self._dimension
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Parameters
        ----------
        texts : list
            List of texts to encode
        batch_size : int
            Batch size for encoding
            
        Returns
        -------
        np.ndarray
            Embeddings of shape (n_texts, embedding_dim)
        """
        self._load_model()
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100
        )
        
        return np.array(embeddings)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text."""
        return self.encode([text])[0]


def calculate_surprise_score(
    current_embedding: np.ndarray,
    historical_embeddings: np.ndarray,
    method: str = 'cosine'
) -> float:
    """
    Calculate surprise/novelty score for a news item.
    
    Surprise = Distance from historical centroid
    
    Parameters
    ----------
    current_embedding : np.ndarray
        Embedding of current news item
    historical_embeddings : np.ndarray
        Embeddings of historical news items (N x dim)
    method : str
        Distance method: 'cosine', 'euclidean', 'mahalanobis'
        
    Returns
    -------
    float
        Surprise score (higher = more novel)
    """
    if len(historical_embeddings) == 0:
        return 1.0  # First item is maximally surprising
    
    # Calculate centroid of historical embeddings
    centroid = np.mean(historical_embeddings, axis=0)
    
    if method == 'cosine':
        # Cosine distance (1 - cosine similarity)
        similarity = np.dot(current_embedding, centroid) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(centroid) + 1e-8
        )
        surprise = 1 - similarity
        
    elif method == 'euclidean':
        # Normalized euclidean distance
        distance = np.linalg.norm(current_embedding - centroid)
        # Normalize by average historical distance
        avg_distance = np.mean([
            np.linalg.norm(emb - centroid) for emb in historical_embeddings
        ])
        surprise = distance / (avg_distance + 1e-8)
        
    elif method == 'mahalanobis':
        # Mahalanobis distance (accounts for covariance)
        try:
            cov = np.cov(historical_embeddings.T)
            cov_inv = np.linalg.pinv(cov)
            diff = current_embedding - centroid
            surprise = np.sqrt(np.dot(np.dot(diff, cov_inv), diff))
        except:
            # Fallback to euclidean
            surprise = np.linalg.norm(current_embedding - centroid)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Clip to reasonable range
    return float(np.clip(surprise, 0, 2))


def batch_calculate_surprise(
    embeddings: np.ndarray,
    lookback: int = 20,
    method: str = 'cosine'
) -> np.ndarray:
    """
    Calculate surprise scores for a batch of embeddings.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings in chronological order (oldest first)
    lookback : int
        Number of historical items to compare against
    method : str
        Distance method
        
    Returns
    -------
    np.ndarray
        Surprise scores for each embedding
    """
    n = len(embeddings)
    surprises = np.zeros(n)
    
    for i in range(n):
        # Get historical window
        start_idx = max(0, i - lookback)
        historical = embeddings[start_idx:i]
        
        surprises[i] = calculate_surprise_score(
            embeddings[i],
            historical,
            method=method
        )
    
    return surprises


class SurpriseScorer:
    """
    Calculates news surprise/novelty scores.
    
    Usage:
        scorer = SurpriseScorer()
        scores = scorer.score_headlines(headlines)
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        lookback: int = 20,
        distance_method: str = 'cosine'
    ):
        self.embedding_model = EmbeddingModel(model_name)
        self.lookback = lookback
        self.distance_method = distance_method
        
        # Cache for historical embeddings
        self._embedding_cache = []
    
    def score_headlines(
        self,
        headlines: List[str],
        timestamps: Optional[List[datetime]] = None
    ) -> pd.DataFrame:
        """
        Score headlines for surprise/novelty.
        
        Parameters
        ----------
        headlines : list
            List of headline texts
        timestamps : list, optional
            Timestamps (for sorting)
            
        Returns
        -------
        pd.DataFrame
            Headlines with embeddings and surprise scores
        """
        if not headlines:
            return pd.DataFrame()
        
        # Sort by timestamp if provided
        if timestamps:
            sorted_pairs = sorted(zip(timestamps, headlines))
            headlines = [h for _, h in sorted_pairs]
        
        # Generate embeddings
        logger.info(f"Encoding {len(headlines)} headlines...")
        embeddings = self.embedding_model.encode(headlines)
        
        # Calculate surprise scores
        logger.info("Calculating surprise scores...")
        surprises = batch_calculate_surprise(
            embeddings,
            lookback=self.lookback,
            method=self.distance_method
        )
        
        # Build result DataFrame
        result = pd.DataFrame({
            'headline': headlines,
            'surprise_score': surprises,
            'embedding_dim': [embeddings.shape[1]] * len(headlines)
        })
        
        # Store embeddings as list for later use
        result['embedding'] = [emb for emb in embeddings]
        
        logger.info(f"Scored {len(headlines)} headlines. Avg surprise: {surprises.mean():.3f}")
        
        return result
    
    def add_to_cache(self, embeddings: np.ndarray):
        """Add embeddings to historical cache."""
        self._embedding_cache.extend(list(embeddings))
        
        # Limit cache size
        max_cache = self.lookback * 10
        if len(self._embedding_cache) > max_cache:
            self._embedding_cache = self._embedding_cache[-max_cache:]
    
    def score_single(self, headline: str) -> Tuple[float, np.ndarray]:
        """
        Score single headline against cache.
        
        Returns
        -------
        tuple
            (surprise_score, embedding)
        """
        embedding = self.embedding_model.encode_single(headline)
        
        if not self._embedding_cache:
            return 1.0, embedding
        
        historical = np.array(self._embedding_cache[-self.lookback:])
        surprise = calculate_surprise_score(
            embedding,
            historical,
            method=self.distance_method
        )
        
        return surprise, embedding


# Convenience functions
def embed_headlines(headlines: List[str], model: str = DEFAULT_MODEL) -> np.ndarray:
    """Quick embedding generation."""
    model = EmbeddingModel(model)
    return model.encode(headlines)


def get_surprise_scores(headlines: List[str], lookback: int = 20) -> np.ndarray:
    """Quick surprise scoring."""
    scorer = SurpriseScorer(lookback=lookback)
    df = scorer.score_headlines(headlines)
    return df['surprise_score'].values

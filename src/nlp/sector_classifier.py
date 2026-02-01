"""
Sector classification module.

Assigns soft sector probabilities to each sentence.
Supports both keyword-weighted and embedding-based approaches.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd
import yaml

from ..utils.logging import get_logger

logger = get_logger(__name__)


class SectorClassifier:
    """
    Classifies sentences into sectors with soft probabilities.
    """
    
    def __init__(
        self,
        sectors_config: Optional[Dict] = None,
        method: str = 'keyword_weighted'
    ):
        """
        Initialize the sector classifier.
        
        Parameters
        ----------
        sectors_config : dict, optional
            Sector definitions with keywords
        method : str
            Classification method: 'keyword_weighted' or 'embedding'
        """
        self.method = method
        
        if sectors_config is None:
            sectors_config = self._load_default_config()
        
        self.sectors = sectors_config.get('sectors', {})
        self.sector_names = list(self.sectors.keys())
        
        # Precompile keyword patterns
        self._compile_patterns()
        
        # Initialize embedding model if needed
        self.embedding_model = None
        if method == 'embedding':
            self._load_embedding_model()
    
    def _load_default_config(self) -> Dict:
        """Load default sector configuration."""
        config_path = Path(__file__).parents[2] / "config" / "sectors.yaml"
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        return {'sectors': {}}
    
    def _compile_patterns(self):
        """Compile regex patterns for keyword matching."""
        self.patterns = {}
        
        for sector, info in self.sectors.items():
            keywords = info.get('keywords', [])
            if keywords:
                # Create pattern that matches any keyword
                pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
                self.patterns[sector] = re.compile(pattern, re.IGNORECASE)
    
    def _load_embedding_model(self):
        """Load sentence transformer model for embedding-based classification."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create sector embeddings from keywords/descriptions
            self.sector_embeddings = {}
            for sector, info in self.sectors.items():
                keywords = info.get('keywords', [])
                name = info.get('name', sector)
                
                # Combine name and keywords for sector representation
                sector_text = f"{name}: {', '.join(keywords)}"
                self.sector_embeddings[sector] = self.embedding_model.encode(sector_text)
            
            logger.info("Loaded sentence transformer model")
            
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to keyword method")
            self.method = 'keyword_weighted'
    
    def classify_keyword(self, sentence: str) -> Dict[str, float]:
        """
        Classify sentence using keyword matching.
        
        Parameters
        ----------
        sentence : str
            Input sentence
            
        Returns
        -------
        dict
            Sector probabilities
        """
        scores = {}
        total_score = 0
        
        for sector, pattern in self.patterns.items():
            matches = pattern.findall(sentence)
            score = len(matches)
            scores[sector] = score
            total_score += score
        
        # Normalize to probabilities
        if total_score > 0:
            probs = {s: score / total_score for s, score in scores.items()}
        else:
            # Uniform distribution if no matches
            probs = {s: 0.0 for s in self.sector_names}
        
        return probs
    
    def classify_embedding(self, sentence: str) -> Dict[str, float]:
        """
        Classify sentence using embedding similarity.
        
        Parameters
        ----------
        sentence : str
            Input sentence
            
        Returns
        -------
        dict
            Sector probabilities
        """
        if self.embedding_model is None:
            return self.classify_keyword(sentence)
        
        # Get sentence embedding
        sent_embedding = self.embedding_model.encode(sentence)
        
        # Calculate cosine similarity with each sector
        similarities = {}
        for sector, sector_emb in self.sector_embeddings.items():
            sim = np.dot(sent_embedding, sector_emb) / (
                np.linalg.norm(sent_embedding) * np.linalg.norm(sector_emb)
            )
            similarities[sector] = max(0, sim)  # Clip negative similarities
        
        # Softmax to get probabilities
        total = sum(similarities.values())
        if total > 0:
            probs = {s: sim / total for s, sim in similarities.items()}
        else:
            probs = {s: 0.0 for s in self.sector_names}
        
        return probs
    
    def classify(self, sentence: str) -> Dict[str, float]:
        """
        Classify sentence into sectors.
        
        Parameters
        ----------
        sentence : str
            Input sentence
            
        Returns
        -------
        dict
            Sector probabilities
        """
        if self.method == 'embedding':
            return self.classify_embedding(sentence)
        else:
            return self.classify_keyword(sentence)
    
    def classify_batch(
        self,
        sentences: List[str],
        min_probability: float = 0.1
    ) -> pd.DataFrame:
        """
        Classify multiple sentences.
        
        Parameters
        ----------
        sentences : list
            List of sentences
        min_probability : float
            Minimum probability to include
            
        Returns
        -------
        pd.DataFrame
            DataFrame with sector probabilities for each sentence
        """
        results = []
        
        for i, sentence in enumerate(sentences):
            probs = self.classify(sentence)
            
            # Filter by minimum probability
            probs = {s: p for s, p in probs.items() if p >= min_probability}
            
            results.append({
                'sentence_idx': i,
                'text': sentence[:100] + '...' if len(sentence) > 100 else sentence,
                **probs
            })
        
        df = pd.DataFrame(results)
        
        # Fill NaN with 0 for sectors
        for sector in self.sector_names:
            if sector not in df.columns:
                df[sector] = 0.0
            else:
                df[sector] = df[sector].fillna(0.0)
        
        return df
    
    def get_top_sectors(
        self,
        sentence: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get top-k sectors for a sentence.
        
        Parameters
        ----------
        sentence : str
            Input sentence
        top_k : int
            Number of top sectors to return
            
        Returns
        -------
        list
            List of (sector, probability) tuples
        """
        probs = self.classify(sentence)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:top_k]


def classify_sentences(
    sentences_df: pd.DataFrame,
    text_column: str = 'text',
    method: str = 'keyword_weighted'
) -> pd.DataFrame:
    """
    Add sector probabilities to sentences DataFrame.
    
    Parameters
    ----------
    sentences_df : pd.DataFrame
        DataFrame with sentences
    text_column : str
        Name of text column
    method : str
        Classification method
        
    Returns
    -------
    pd.DataFrame
        DataFrame with sector probability columns added
    """
    classifier = SectorClassifier(method=method)
    
    sentences = sentences_df[text_column].tolist()
    probs_df = classifier.classify_batch(sentences)
    
    # Merge with original DataFrame
    result = sentences_df.copy()
    
    for sector in classifier.sector_names:
        if sector in probs_df.columns:
            result[f'prob_{sector}'] = probs_df[sector].values
    
    logger.info(f"Added {len(classifier.sector_names)} sector probability columns")
    
    return result

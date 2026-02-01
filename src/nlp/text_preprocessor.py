"""
Text Preprocessor - Layer 2: Preprocessing

Financial text normalization and cleaning:
- Duplicate detection across sources
- Financial named entity recognition
- Finance-specific stopword removal
- Lemmatization
"""

from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import re
import hashlib
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Finance-specific stopwords (common but uninformative)
FINANCE_STOPWORDS = {
    'stock', 'share', 'market', 'trade', 'trading', 'nse', 'bse',
    'today', 'yesterday', 'week', 'month', 'year', 'live', 'update',
    'news', 'report', 'says', 'said', 'according', 'sources',
    'rs', 'crore', 'lakh', 'inr', 'usd'
}


# Named entity patterns
NER_PATTERNS = {
    'company': r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*(?:\s+(?:Ltd|Limited|Corp|Inc|Industries|Pharma|Bank|Finance))?)\b',
    'ticker': r'\b([A-Z]{2,10})\b',
    'amount': r'(?:Rs\.?|₹|INR)\s*[\d,]+(?:\.\d+)?\s*(?:crore|lakh|billion|million)?',
    'percentage': r'[\d.]+\s*%',
    'date': r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
}


# Entity type mappings for policy detection
POLICY_ENTITIES = [
    'rbi', 'sebi', 'budget', 'government', 'ministry', 'parliament',
    'nirmala sitharaman', 'finance minister', 'pm modi', 'modi government'
]


def normalize_text(text: str) -> str:
    """
    Normalize financial text.
    
    Parameters
    ----------
    text : str
        Raw text
        
    Returns
    -------
    str
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Standardize punctuation
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    # Standardize numbers
    text = re.sub(r'(\d),(\d)', r'\1\2', text)  # Remove commas in numbers
    
    return text.strip()


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from text.
    
    Parameters
    ----------
    text : str
        Input text
        
    Returns
    -------
    dict
        Entities by type
    """
    entities = {}
    
    for entity_type, pattern in NER_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            entities[entity_type] = list(set(matches))
    
    return entities


def detect_policy_mention(text: str) -> bool:
    """Check if text mentions policy-related entities."""
    text_lower = text.lower()
    return any(entity in text_lower for entity in POLICY_ENTITIES)


def remove_duplicates(
    texts: List[str],
    similarity_threshold: float = 0.85
) -> Tuple[List[str], List[int]]:
    """
    Remove near-duplicate texts.
    
    Uses simple character n-gram similarity.
    
    Parameters
    ----------
    texts : list
        List of texts
    similarity_threshold : float
        Minimum similarity to consider duplicate
        
    Returns
    -------
    tuple
        (unique_texts, original_indices)
    """
    if not texts:
        return [], []
    
    def get_ngrams(text: str, n: int = 3) -> Set[str]:
        """Get character n-grams from text."""
        text = text.lower()
        return set(text[i:i+n] for i in range(len(text)-n+1))
    
    def similarity(text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between texts."""
        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
    
    unique_texts = []
    original_indices = []
    
    for i, text in enumerate(texts):
        is_duplicate = False
        
        for unique_text in unique_texts:
            if similarity(text, unique_text) >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_texts.append(text)
            original_indices.append(i)
    
    logger.info(f"Removed {len(texts) - len(unique_texts)} duplicates from {len(texts)} texts")
    
    return unique_texts, original_indices


def calculate_information_density(text: str) -> float:
    """
    Calculate information density score.
    
    Higher score = more information-dense text.
    
    Parameters
    ----------
    text : str
        Input text
        
    Returns
    -------
    float
        Information density score (0-1)
    """
    words = text.lower().split()
    
    if len(words) < 3:
        return 0.0
    
    # Count non-stopword words
    content_words = [w for w in words if w not in FINANCE_STOPWORDS and len(w) > 2]
    
    # Count numbers and percentages
    numbers = len(re.findall(r'\d+', text))
    
    # Count entities
    entities = extract_entities(text)
    entity_count = sum(len(v) for v in entities.values())
    
    # Calculate density
    word_ratio = len(content_words) / len(words)
    number_bonus = min(numbers * 0.1, 0.3)
    entity_bonus = min(entity_count * 0.05, 0.2)
    
    density = word_ratio + number_bonus + entity_bonus
    
    return min(density, 1.0)


def preprocess_headlines(
    headlines: List[str],
    remove_dups: bool = True,
    min_density: float = 0.3
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for headlines.
    
    Parameters
    ----------
    headlines : list
        Raw headlines
    remove_dups : bool
        Whether to remove duplicates
    min_density : float
        Minimum information density
        
    Returns
    -------
    pd.DataFrame
        Preprocessed data with features
    """
    results = []
    
    for i, headline in enumerate(headlines):
        # Normalize
        normalized = normalize_text(headline)
        
        # Calculate density
        density = calculate_information_density(headline)
        
        # Skip low-density headlines
        if density < min_density:
            continue
        
        # Extract entities
        entities = extract_entities(headline)
        
        results.append({
            'original': headline,
            'normalized': normalized,
            'info_density': density,
            'is_policy': detect_policy_mention(headline),
            'entity_count': sum(len(v) for v in entities.values()),
            'entities': entities
        })
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Remove duplicates
    if remove_dups and len(df) > 1:
        unique_texts, indices = remove_duplicates(df['normalized'].tolist())
        df = df.iloc[indices].reset_index(drop=True)
    
    logger.info(f"Preprocessed {len(df)} headlines (from {len(headlines)} original)")
    
    return df


class TextPreprocessor:
    """
    Text preprocessing pipeline for financial news.
    
    Usage:
        preprocessor = TextPreprocessor()
        df = preprocessor.process(headlines)
    """
    
    def __init__(
        self,
        remove_duplicates: bool = True,
        min_info_density: float = 0.3,
        custom_stopwords: Optional[Set[str]] = None
    ):
        self.remove_duplicates = remove_duplicates
        self.min_info_density = min_info_density
        self.stopwords = FINANCE_STOPWORDS.copy()
        
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
    
    def process(self, headlines: List[str]) -> pd.DataFrame:
        """Process list of headlines."""
        return preprocess_headlines(
            headlines,
            remove_dups=self.remove_duplicates,
            min_density=self.min_info_density
        )
    
    def normalize(self, text: str) -> str:
        """Normalize single text."""
        return normalize_text(text)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text."""
        return extract_entities(text)

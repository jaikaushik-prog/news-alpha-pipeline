"""
Sentiment analysis module.

Extracts polarity and subjectivity from budget speech sentences.
"""

from typing import Dict, List, Optional, Tuple
import re
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


class SentimentAnalyzer:
    """
    Analyzes sentiment of budget speech sentences.
    """
    
    def __init__(self, method: str = 'vader'):
        """
        Initialize sentiment analyzer.
        
        Parameters
        ----------
        method : str
            Sentiment method: 'vader', 'textblob', or 'combined'
        """
        self.method = method
        self.vader = None
        
        if method in ['vader', 'combined']:
            self._load_vader()
            
        if method in ['finbert', 'combined']:
            self._load_finbert()
    
    def _load_vader(self):
        """Load VADER sentiment analyzer."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
            logger.info("Loaded VADER sentiment analyzer")
        except ImportError:
            logger.warning("vaderSentiment not installed")
            
    def _load_finbert(self):
        """Load FinBERT sentiment analyzer."""
        try:
            from transformers import pipeline
            self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1) # CPU by default
            logger.info("Loaded FinBERT sentiment analyzer")
        except ImportError:
            logger.warning("transformers or torch not installed")
            self.finbert = None
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}")
            self.finbert = None
    
    def analyze_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        dict
            Sentiment scores (neg, neu, pos, compound)
        """
        if self.vader is None:
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
        
        return self.vader.polarity_scores(text)
        
    def analyze_finbert(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using FinBERT.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        dict
            Sentiment scores (positive, negative, neutral)
        """
        if self.finbert is None:
            return {'finbert_pos': 0.0, 'finbert_neg': 0.0, 'finbert_neu': 1.0}
            
        try:
            # FinBERT returns list of dicts: [{'label': 'positive', 'score': 0.95}]
            result = self.finbert(text)[0]
            label = result['label'].lower()
            score = result['score']
            
            scores = {'finbert_pos': 0.0, 'finbert_neg': 0.0, 'finbert_neu': 0.0}
            
            if label == 'positive':
                scores['finbert_pos'] = score
            elif label == 'negative':
                scores['finbert_neg'] = score
            else:
                scores['finbert_neu'] = score
                
            return scores
        except Exception as e:
            logger.error(f"FinBERT analysis error: {e}")
            return {'finbert_pos': 0.0, 'finbert_neg': 0.0, 'finbert_neu': 1.0}
    
    def analyze_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        dict
            Polarity and subjectivity scores
        """
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            
            return {
                'polarity': blob.sentiment.polarity,  # -1 to 1
                'subjectivity': blob.sentiment.subjectivity  # 0 to 1
            }
        except ImportError:
            logger.warning("TextBlob not installed")
            return {'polarity': 0.0, 'subjectivity': 0.5}
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using configured method.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        dict
            Sentiment scores
        """
        results = {}
        
        if self.method in ['vader', 'combined']:
            vader_scores = self.analyze_vader(text)
            results['sentiment_compound'] = vader_scores['compound']
            results['sentiment_pos'] = vader_scores['pos']
            results['sentiment_neg'] = vader_scores['neg']
            results['sentiment_neu'] = vader_scores['neu']

        if self.method in ['finbert', 'combined']:
            finbert_scores = self.analyze_finbert(text)
            results['finbert_pos'] = finbert_scores['finbert_pos']
            results['finbert_neg'] = finbert_scores['finbert_neg']
            results['finbert_neu'] = finbert_scores['finbert_neu']
            # Composite score: Positive - Negative
            results['finbert_compound'] = results['finbert_pos'] - results['finbert_neg']
        
        if self.method in ['textblob', 'combined']:
            tb_scores = self.analyze_textblob(text)
            results['polarity'] = tb_scores['polarity']
            results['subjectivity'] = tb_scores['subjectivity']
        
        return results
    
    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for multiple texts.
        
        Parameters
        ----------
        texts : list
            List of texts
            
        Returns
        -------
        pd.DataFrame
            DataFrame with sentiment scores
        """
        # For FinBERT, batch processing is faster but requires different pipeline usage
        # Sticking to loop for simplicity unless method is pure finbert
        results = []
        
        for text in texts:
            scores = self.analyze(text)
            results.append(scores)
        
        return pd.DataFrame(results)


def extract_fiscal_mentions(text: str) -> Dict[str, any]:
    """
    Extract fiscal magnitude mentions from text.
    
    Parameters
    ----------
    text : str
        Input text
        
    Returns
    -------
    dict
        Extracted fiscal information
    """
    results = {
        'has_rupee_amount': False,
        'rupee_amounts': [],
        'has_percentage': False,
        'percentages': [],
        'has_crore': False,
        'has_lakh': False,
        'fiscal_intensity': 0.0
    }
    
    # Find rupee amounts (₹ or Rs.)
    rupee_pattern = r'[₹Rs\.]+\s*([\d,\.]+)\s*(crore|lakh|million|billion)?'
    rupee_matches = re.findall(rupee_pattern, text, re.IGNORECASE)
    
    if rupee_matches:
        results['has_rupee_amount'] = True
        results['rupee_amounts'] = rupee_matches
    
    # Find percentages
    pct_pattern = r'(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)'
    pct_matches = re.findall(pct_pattern, text, re.IGNORECASE)
    
    if pct_matches:
        results['has_percentage'] = True
        results['percentages'] = [float(p) for p in pct_matches]
    
    # Check for crore/lakh mentions
    results['has_crore'] = bool(re.search(r'crore', text, re.IGNORECASE))
    results['has_lakh'] = bool(re.search(r'lakh', text, re.IGNORECASE))
    
    # Calculate fiscal intensity score
    intensity = 0
    if results['has_rupee_amount']:
        intensity += 2
    if results['has_percentage']:
        intensity += 1.5
    if results['has_crore']:
        intensity += 1
    if results['has_lakh']:
        intensity += 0.5
    
    results['fiscal_intensity'] = intensity
    
    return results


def add_sentiment_features(
    sentences_df: pd.DataFrame,
    text_column: str = 'text',
    method: str = 'combined'
) -> pd.DataFrame:
    """
    Add sentiment features to sentences DataFrame.
    
    Parameters
    ----------
    sentences_df : pd.DataFrame
        DataFrame with sentences
    text_column : str
        Name of text column
    method : str
        Sentiment method
        
    Returns
    -------
    pd.DataFrame
        DataFrame with sentiment columns added
    """
    analyzer = SentimentAnalyzer(method=method)
    
    # Get sentiment scores
    texts = sentences_df[text_column].tolist()
    sentiment_df = analyzer.analyze_batch(texts)
    
    # Add to original DataFrame
    result = sentences_df.copy()
    for col in sentiment_df.columns:
        result[col] = sentiment_df[col].values
    
    # Add fiscal mentions
    fiscal_features = []
    for text in texts:
        features = extract_fiscal_mentions(text)
        fiscal_features.append({
            'has_rupee_amount': features['has_rupee_amount'],
            'has_percentage': features['has_percentage'],
            'fiscal_intensity': features['fiscal_intensity']
        })
    
    fiscal_df = pd.DataFrame(fiscal_features)
    for col in fiscal_df.columns:
        result[col] = fiscal_df[col].values
    
    logger.info(f"Added sentiment and fiscal features")
    
    return result


def analyze_sentiment(text: str, method: str = 'combined') -> Dict[str, float]:
    """
    Analyze sentiment of a single text string.
    
    Parameters
    ----------
    text : str
        Input text
    method : str
        Sentiment method
        
    Returns
    -------
    dict
        Sentiment scores
    """
    analyzer = SentimentAnalyzer(method=method)
    return analyzer.analyze(text)

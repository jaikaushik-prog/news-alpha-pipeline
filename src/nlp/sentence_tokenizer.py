"""
Sentence tokenization module.

Tokenizes budget speech text into ordered sentences with metadata.
"""

from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

import pandas as pd

from ..utils.logging import get_logger
from ..utils.time_utils import IST

logger = get_logger(__name__)


@dataclass
class Sentence:
    """Represents a single sentence from the budget speech."""
    sentence_id: int
    text: str
    position: int  # 0-indexed position in speech
    word_count: int
    char_count: int
    estimated_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            'sentence_id': self.sentence_id,
            'text': self.text,
            'position': self.position,
            'word_count': self.word_count,
            'char_count': self.char_count,
            'estimated_timestamp': self.estimated_timestamp
        }


def tokenize_sentences_nltk(text: str) -> List[str]:
    """
    Tokenize text into sentences using NLTK.
    
    Parameters
    ----------
    text : str
        Input text
        
    Returns
    -------
    list
        List of sentences
    """
    try:
        import nltk
        try:
            sentences = nltk.sent_tokenize(text)
            return sentences
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            sentences = nltk.sent_tokenize(text)
            return sentences
    except ImportError:
        logger.warning("NLTK not installed, using regex tokenizer")
        return tokenize_sentences_regex(text)


def tokenize_sentences_regex(text: str) -> List[str]:
    """
    Tokenize text into sentences using regex (fallback).
    
    Parameters
    ----------
    text : str
        Input text
        
    Returns
    -------
    list
        List of sentences
    """
    # Simple sentence splitting on period, question mark, exclamation
    # followed by space and capital letter
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    
    # Clean up
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def clean_sentence(sentence: str) -> str:
    """
    Clean a single sentence.
    
    Parameters
    ----------
    sentence : str
        Input sentence
        
    Returns
    -------
    str
        Cleaned sentence
    """
    # Remove extra whitespace
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # Strip leading/trailing whitespace
    sentence = sentence.strip()
    
    return sentence


def is_valid_sentence(sentence: str, min_words: int = 3) -> bool:
    """
    Check if a sentence is valid for analysis.
    
    Parameters
    ----------
    sentence : str
        Input sentence
    min_words : int
        Minimum word count
        
    Returns
    -------
    bool
        True if valid
    """
    # Check minimum length
    words = sentence.split()
    if len(words) < min_words:
        return False
    
    # Must contain at least one alphabetic character
    if not re.search(r'[a-zA-Z]', sentence):
        return False
    
    # Skip common non-content patterns
    skip_patterns = [
        r'^[\d\s\.\,\-\(\)]+$',  # Only numbers/punctuation
        r'^Page \d+$',
        r'^\d+\.$',  # Just a number with period
    ]
    
    for pattern in skip_patterns:
        if re.match(pattern, sentence, re.IGNORECASE):
            return False
    
    return True


def tokenize_speech(
    text: str,
    speech_start: Optional[datetime] = None,
    speech_duration_minutes: float = 120.0,
    min_words: int = 3
) -> List[Sentence]:
    """
    Tokenize budget speech into sentences with metadata.
    
    Parameters
    ----------
    text : str
        Speech text
    speech_start : datetime, optional
        When the speech started
    speech_duration_minutes : float
        Total speech duration in minutes
    min_words : int
        Minimum words for valid sentence
        
    Returns
    -------
    list
        List of Sentence objects
    """
    # Tokenize
    raw_sentences = tokenize_sentences_nltk(text)
    
    logger.info(f"Initial tokenization: {len(raw_sentences)} sentences")
    
    # Clean and filter
    valid_sentences = []
    for i, sent in enumerate(raw_sentences):
        cleaned = clean_sentence(sent)
        if is_valid_sentence(cleaned, min_words):
            valid_sentences.append(cleaned)
    
    logger.info(f"After filtering: {len(valid_sentences)} valid sentences")
    
    # Create Sentence objects with timestamps
    sentences = []
    total = len(valid_sentences)
    
    for i, text in enumerate(valid_sentences):
        # Estimate timestamp based on position
        timestamp = None
        if speech_start is not None:
            if total > 1:
                progress = i / (total - 1)
            else:
                progress = 0
            elapsed = progress * speech_duration_minutes
            timestamp = speech_start + timedelta(minutes=elapsed)
        
        sentence = Sentence(
            sentence_id=i,
            text=text,
            position=i,
            word_count=len(text.split()),
            char_count=len(text),
            estimated_timestamp=timestamp
        )
        sentences.append(sentence)
    
    return sentences


def sentences_to_dataframe(sentences: List[Sentence]) -> pd.DataFrame:
    """
    Convert list of Sentence objects to DataFrame.
    
    Parameters
    ----------
    sentences : list
        List of Sentence objects
        
    Returns
    -------
    pd.DataFrame
        DataFrame with sentence data
    """
    data = [s.to_dict() for s in sentences]
    return pd.DataFrame(data)


def process_speech(
    pdf_path: Path,
    speech_start: datetime,
    speech_duration_minutes: float,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Full pipeline: PDF -> tokenized sentences DataFrame.
    
    Parameters
    ----------
    pdf_path : Path
        Path to PDF file
    speech_start : datetime
        Speech start time
    speech_duration_minutes : float
        Speech duration
    output_path : Path, optional
        Where to save CSV output
        
    Returns
    -------
    pd.DataFrame
        Tokenized sentences
    """
    from .pdf_to_text import extract_text_from_pdf
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        logger.error(f"No text extracted from {pdf_path}")
        return pd.DataFrame()
    
    # Tokenize
    sentences = tokenize_speech(
        text,
        speech_start=speech_start,
        speech_duration_minutes=speech_duration_minutes
    )
    
    # Convert to DataFrame
    df = sentences_to_dataframe(sentences)
    
    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} sentences to {output_path}")
    
    return df

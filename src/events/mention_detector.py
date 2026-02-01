"""
Mention detection module.

Detects first material mention of each sector in budget speech.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


def calculate_cumulative_attention(
    sentences_df: pd.DataFrame,
    sector: str,
    prob_col_prefix: str = 'prob_',
    weight_col: str = 'importance_weight'
) -> pd.Series:
    """
    Calculate cumulative attention intensity for a sector.
    
    Parameters
    ----------
    sentences_df : pd.DataFrame
        Sentences with sector probabilities
    sector : str
        Sector name
    prob_col_prefix : str
        Prefix for probability columns
    weight_col : str
        Column with importance weights
        
    Returns
    -------
    pd.Series
        Cumulative attention intensity
    """
    prob_col = f"{prob_col_prefix}{sector}"
    
    if prob_col not in sentences_df.columns:
        return pd.Series(0.0, index=sentences_df.index)
    
    # Weighted probability
    if weight_col in sentences_df.columns:
        weighted = sentences_df[prob_col] * sentences_df[weight_col]
    else:
        weighted = sentences_df[prob_col]
    
    return weighted.cumsum()


def detect_first_mention(
    sentences_df: pd.DataFrame,
    sector: str,
    threshold: float = 0.5,
    prob_col_prefix: str = 'prob_',
    weight_col: str = 'importance_weight'
) -> Optional[Dict]:
    """
    Detect first material mention of a sector.
    
    Material mention is when cumulative attention exceeds threshold.
    
    Parameters
    ----------
    sentences_df : pd.DataFrame
        Sentences with sector probabilities and timestamps
    sector : str
        Sector name
    threshold : float
        Cumulative attention threshold
    prob_col_prefix : str
        Prefix for probability columns
    weight_col : str
        Importance weight column
        
    Returns
    -------
    dict or None
        First mention info or None if not found
    """
    cumulative = calculate_cumulative_attention(
        sentences_df, sector, prob_col_prefix, weight_col
    )
    
    # Find first crossing of threshold
    above_threshold = cumulative >= threshold
    
    if not above_threshold.any():
        return None
    
    first_idx = above_threshold.idxmax()
    row = sentences_df.loc[first_idx]
    
    # Calculate average sentiment for the context window (e.g., +/- 2 sentences)
    window_start = max(0, first_idx - 2)
    window_end = min(len(sentences_df), first_idx + 3)
    
    if 'finbert_compound' in sentences_df.columns:
        avg_sentiment = sentences_df.iloc[window_start:window_end]['finbert_compound'].mean()
    elif 'sentiment_compound' in sentences_df.columns:
        avg_sentiment = sentences_df.iloc[window_start:window_end]['sentiment_compound'].mean()
    else:
        avg_sentiment = 0.0
    
    return {
        'sector': sector,
        'sentence_idx': first_idx,
        'sentence_position': row.get('position', first_idx),
        'estimated_timestamp': row.get('estimated_timestamp'),
        'cumulative_attention': cumulative.loc[first_idx],
        'sentence_text': row.get('text', '')[:200],
        'probability': row.get(f'{prob_col_prefix}{sector}', 0.0),
        'avg_sentiment': avg_sentiment
    }


def detect_all_sector_mentions(
    sentences_df: pd.DataFrame,
    sectors: Optional[List[str]] = None,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Detect first mentions for all sectors.
    
    Parameters
    ----------
    sentences_df : pd.DataFrame
        Sentences with sector probabilities
    sectors : list, optional
        List of sectors to detect. If None, auto-detect from columns.
    threshold : float
        Cumulative attention threshold
        
    Returns
    -------
    pd.DataFrame
        First mention info for each sector
    """
    # Auto-detect sectors from probability columns
    if sectors is None:
        prob_cols = [c for c in sentences_df.columns if c.startswith('prob_')]
        sectors = [c.replace('prob_', '') for c in prob_cols]
    
    mentions = []
    
    for sector in sectors:
        mention = detect_first_mention(sentences_df, sector, threshold)
        if mention is not None:
            mentions.append(mention)
    
    if not mentions:
        return pd.DataFrame()
    
    df = pd.DataFrame(mentions)
    
    # Sort by timestamp/position
    if 'estimated_timestamp' in df.columns:
        df = df.sort_values('estimated_timestamp')
    elif 'sentence_position' in df.columns:
        df = df.sort_values('sentence_position')
    
    logger.info(f"Detected first mentions for {len(df)} sectors")
    
    return df


def detect_mention_clusters(
    sentences_df: pd.DataFrame,
    sector: str,
    prob_threshold: float = 0.2,
    cluster_gap: int = 3
) -> List[Dict]:
    """
    Detect clusters of mentions for a sector.
    
    A cluster is a sequence of sentences with probability above threshold.
    
    Parameters
    ----------
    sentences_df : pd.DataFrame
        Sentences with sector probabilities
    sector : str
        Sector name
    prob_threshold : float
        Minimum probability to be part of cluster
    cluster_gap : int
        Max gap between sentences in same cluster
        
    Returns
    -------
    list
        List of cluster dictionaries
    """
    prob_col = f'prob_{sector}'
    
    if prob_col not in sentences_df.columns:
        return []
    
    # Find sentences above threshold
    above_threshold = sentences_df[prob_col] >= prob_threshold
    
    if not above_threshold.any():
        return []
    
    # Group into clusters
    clusters = []
    current_cluster = None
    last_mention_idx = None
    
    for idx, row in sentences_df.iterrows():
        if row[prob_col] >= prob_threshold:
            if current_cluster is None:
                # Start new cluster
                current_cluster = {
                    'sector': sector,
                    'start_idx': idx,
                    'end_idx': idx,
                    'start_position': row.get('position', idx),
                    'sentences': [idx],
                    'total_attention': row[prob_col] * row.get('importance_weight', 1.0),
                    'start_timestamp': row.get('estimated_timestamp')
                }
            elif last_mention_idx is not None and (idx - last_mention_idx) <= cluster_gap:
                # Extend current cluster
                current_cluster['end_idx'] = idx
                current_cluster['sentences'].append(idx)
                current_cluster['total_attention'] += row[prob_col] * row.get('importance_weight', 1.0)
            else:
                # End current cluster and start new one
                clusters.append(current_cluster)
                current_cluster = {
                    'sector': sector,
                    'start_idx': idx,
                    'end_idx': idx,
                    'start_position': row.get('position', idx),
                    'sentences': [idx],
                    'total_attention': row[prob_col] * row.get('importance_weight', 1.0),
                    'start_timestamp': row.get('estimated_timestamp')
                }
            
            last_mention_idx = idx
    
    # Don't forget last cluster
    if current_cluster is not None:
        clusters.append(current_cluster)
    
    # Add cluster statistics
    for cluster in clusters:
        cluster['n_sentences'] = len(cluster['sentences'])
        cluster['avg_attention'] = cluster['total_attention'] / cluster['n_sentences']
    
    return clusters


def build_mention_timeline(
    sentences_df: pd.DataFrame,
    sectors: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Build a timeline of sector mentions.
    
    Parameters
    ----------
    sentences_df : pd.DataFrame
        Sentences with sector probabilities
    sectors : list, optional
        List of sectors
        
    Returns
    -------
    pd.DataFrame
        Timeline of mentions
    """
    # Auto-detect sectors
    if sectors is None:
        prob_cols = [c for c in sentences_df.columns if c.startswith('prob_')]
        sectors = [c.replace('prob_', '') for c in prob_cols]
    
    timeline = []
    
    for _, row in sentences_df.iterrows():
        # Find dominant sector for this sentence
        sector_probs = {s: row.get(f'prob_{s}', 0.0) for s in sectors}
        
        if max(sector_probs.values()) > 0.1:  # Only if some sector relevance
            dominant = max(sector_probs, key=sector_probs.get)
            
            timeline.append({
                'sentence_idx': row.name if hasattr(row, 'name') else None,
                'position': row.get('position'),
                'timestamp': row.get('estimated_timestamp'),
                'dominant_sector': dominant,
                'dominant_prob': sector_probs[dominant],
                'n_relevant_sectors': sum(1 for p in sector_probs.values() if p > 0.1),
                'importance_weight': row.get('importance_weight', 1.0)
            })
    
    return pd.DataFrame(timeline)

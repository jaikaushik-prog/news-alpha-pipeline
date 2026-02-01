"""
Certainty and actionability scoring module.

Scores sentences for:
- Linguistic certainty (how confident is the statement)
- Actionability (how likely to result in concrete action)
"""

from typing import Dict, List
import re
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Certainty indicators
CERTAINTY_POSITIVE = [
    'will', 'shall', 'must', 'committed', 'ensure', 'guarantee',
    'definitely', 'certainly', 'undoubtedly', 'absolutely',
    'determined', 'resolved', 'pledge', 'promise'
]

CERTAINTY_NEGATIVE = [
    'may', 'might', 'could', 'possibly', 'perhaps', 'maybe',
    'consider', 'explore', 'examine', 'review', 'study',
    'likely', 'potentially', 'tentatively'
]

# Actionability indicators
ACTION_VERBS = [
    'allocate', 'invest', 'provide', 'launch', 'establish', 'create',
    'build', 'construct', 'develop', 'implement', 'introduce',
    'announce', 'sanction', 'approve', 'fund', 'finance',
    'set up', 'roll out', 'operationalize', 'inaugurate'
]

# Target/goal indicators
TARGET_WORDS = [
    'target', 'goal', 'objective', 'aim', 'mission',
    'by 2025', 'by 2030', 'next year', 'this year',
    'within', 'timeline', 'deadline'
]


def calculate_certainty_score(text: str) -> Dict[str, float]:
    """
    Calculate certainty score for a sentence.
    
    Parameters
    ----------
    text : str
        Input text
        
    Returns
    -------
    dict
        Certainty scores
    """
    text_lower = text.lower()
    
    # Count positive certainty indicators
    pos_count = sum(1 for word in CERTAINTY_POSITIVE if word in text_lower)
    
    # Count negative certainty indicators (hedging)
    neg_count = sum(1 for word in CERTAINTY_NEGATIVE if word in text_lower)
    
    # Calculate raw score
    if pos_count + neg_count == 0:
        raw_score = 0.5  # Neutral
    else:
        raw_score = pos_count / (pos_count + neg_count)
    
    # Adjust based on statement type
    # Imperative sentences tend to be more certain
    if text.strip().endswith('!'):
        raw_score = min(1.0, raw_score + 0.1)
    
    # Questions are less certain
    if '?' in text:
        raw_score = max(0.0, raw_score - 0.2)
    
    return {
        'certainty_score': raw_score,
        'certainty_positive_count': pos_count,
        'certainty_negative_count': neg_count
    }


def calculate_actionability_score(text: str) -> Dict[str, float]:
    """
    Calculate actionability score for a sentence.
    
    Parameters
    ----------
    text : str
        Input text
        
    Returns
    -------
    dict
        Actionability scores
    """
    text_lower = text.lower()
    
    # Count action verbs
    action_count = sum(1 for verb in ACTION_VERBS if verb in text_lower)
    
    # Count target mentions
    target_count = sum(1 for word in TARGET_WORDS if word in text_lower)
    
    # Check for specific allocations (rupee amounts)
    has_amount = bool(re.search(r'[₹Rs\.]', text))
    
    # Calculate score
    score = 0.0
    
    if action_count > 0:
        score += 0.3 * min(action_count, 3)  # Cap at 3 verbs
    
    if target_count > 0:
        score += 0.2 * min(target_count, 2)  # Cap at 2 targets
    
    if has_amount:
        score += 0.3  # Specific amounts are very actionable
    
    # Normalize to 0-1
    score = min(1.0, score)
    
    return {
        'actionability_score': score,
        'action_verb_count': action_count,
        'target_mention_count': target_count,
        'has_specific_amount': has_amount
    }


def classify_statement_type(text: str) -> str:
    """
    Classify the type of budget statement.
    
    Parameters
    ----------
    text : str
        Input text
        
    Returns
    -------
    str
        Statement type
    """
    text_lower = text.lower()
    
    # Check for announcement patterns
    if any(phrase in text_lower for phrase in [
        'i announce', 'we announce', 'i propose', 'we propose',
        'i am pleased', 'happy to announce', 'introduce'
    ]):
        return 'announcement'
    
    # Check for allocation patterns
    if any(phrase in text_lower for phrase in [
        'allocate', 'provision', 'outlay', 'expenditure of'
    ]):
        return 'allocation'
    
    # Check for target/goal patterns
    if any(phrase in text_lower for phrase in [
        'target', 'goal', 'aim to', 'objective'
    ]):
        return 'target'
    
    # Check for reform patterns
    if any(phrase in text_lower for phrase in [
        'reform', 'restructure', 'revamp', 'overhaul', 'streamline'
    ]):
        return 'reform'
    
    # Check for review/status patterns
    if any(phrase in text_lower for phrase in [
        'achieved', 'grew', 'increased', 'declined', 'rose to'
    ]):
        return 'status'
    
    return 'general'


def calculate_importance_weight(
    certainty_score: float,
    actionability_score: float,
    fiscal_intensity: float,
    statement_type: str
) -> float:
    """
    Calculate overall importance weight for attention intensity.
    
    Parameters
    ----------
    certainty_score : float
        Certainty score (0-1)
    actionability_score : float
        Actionability score (0-1)
    fiscal_intensity : float
        Fiscal magnitude intensity
    statement_type : str
        Type of statement
        
    Returns
    -------
    float
        Importance weight
    """
    # Base weight from certainty and actionability
    base_weight = 0.4 * certainty_score + 0.3 * actionability_score
    
    # Boost from fiscal intensity
    fiscal_boost = min(0.3, fiscal_intensity * 0.1)
    
    # Type-based adjustment
    type_multipliers = {
        'announcement': 1.3,
        'allocation': 1.4,
        'target': 1.2,
        'reform': 1.1,
        'status': 0.8,
        'general': 1.0
    }
    
    type_mult = type_multipliers.get(statement_type, 1.0)
    
    weight = (base_weight + fiscal_boost) * type_mult
    
    return min(2.0, weight)


def add_certainty_actionability_features(
    sentences_df: pd.DataFrame,
    text_column: str = 'text'
) -> pd.DataFrame:
    """
    Add certainty, actionability, and importance features to DataFrame.
    
    Parameters
    ----------
    sentences_df : pd.DataFrame
        DataFrame with sentences
    text_column : str
        Name of text column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with new features added
    """
    result = sentences_df.copy()
    
    # Calculate features for each sentence
    certainty_scores = []
    actionability_scores = []
    statement_types = []
    importance_weights = []
    
    for _, row in result.iterrows():
        text = row[text_column]
        
        # Certainty
        cert = calculate_certainty_score(text)
        certainty_scores.append(cert)
        
        # Actionability
        action = calculate_actionability_score(text)
        actionability_scores.append(action)
        
        # Statement type
        stmt_type = classify_statement_type(text)
        statement_types.append(stmt_type)
        
        # Importance weight
        fiscal_intensity = row.get('fiscal_intensity', 0.0)
        importance = calculate_importance_weight(
            cert['certainty_score'],
            action['actionability_score'],
            fiscal_intensity,
            stmt_type
        )
        importance_weights.append(importance)
    
    # Add to DataFrame
    cert_df = pd.DataFrame(certainty_scores)
    action_df = pd.DataFrame(actionability_scores)
    
    for col in cert_df.columns:
        result[col] = cert_df[col].values
    
    for col in action_df.columns:
        result[col] = action_df[col].values
    
    result['statement_type'] = statement_types
    result['importance_weight'] = importance_weights
    
    logger.info("Added certainty, actionability, and importance features")
    
    return result

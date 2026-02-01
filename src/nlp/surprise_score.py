"""
Surprise Score Modeling
Calculates the 'surprise' or novelty of a news item based on its semantic distance 
from the recent narrative history.

Surprise_t = ||Embedding_t - Mean(Embeddings_{t-N:t-1})||₂
"""
import sys
import numpy as np
import pickle
from collections import deque
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class SurpriseModel:
    """
    Models the 'surprise' of new information by comparing it to a 
    memory of recent semantic states.
    """
    
    def __init__(self, memory_window: int = 200, decay_factor: float = 0.99):
        """
        Args:
            memory_window: Number of recent news items to keep in memory for baseline
            decay_factor: Weight decay for running mean (not used in simple window version)
        """
        self.memory_window = memory_window
        self.memory_embeddings = deque(maxlen=memory_window)
        self.memory_timestamps = deque(maxlen=memory_window)
        self.baseline_vector = None
        
    def update(self, embedding: np.ndarray, timestamp: Optional[datetime] = None):
        """Update memory with new observation"""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.memory_embeddings.append(embedding)
        self.memory_timestamps.append(timestamp)
        
        # Update baseline (simple mean of window)
        # Using stack to handle list of arrays efficiently
        if len(self.memory_embeddings) > 0:
            self.baseline_vector = np.mean(np.stack(self.memory_embeddings), axis=0)
            
    def calculate_surprise(self, embedding: np.ndarray) -> Dict[str, float]:
        """
        Calculate surprise score for a single embedding.
        
        Returns:
            Dict containing 'surprise_score' (0-1) and raw distance
        """
        if self.baseline_vector is None or len(self.memory_embeddings) < 5:
            # High surprise/uncertainty if we have no history
            return {
                'surprise_score': 0.5, 
                'distance': 0.0,
                'status': 'cold_start'
            }
            
        # Euclidean distance from narrative centroid
        distance = np.linalg.norm(embedding - self.baseline_vector)
        
        # Normalize distance (heuristic based on cosine space or empirical)
        # For normalized embeddings (length 1), max euclidean distance is 2.
        # Typical distance for unrelated news is ~1.0-1.2
        # Related news is ~0.3-0.6
        
        # Sigmoid-like scaling to 0-1
        # Center around 0.8 (novelty threshold), steepness 5
        score = 1 / (1 + np.exp(-5 * (distance - 0.8)))
        
        return {
            'surprise_score': round(float(score), 3),
            'distance': round(float(distance), 3),
            'status': 'active'
        }

    def get_narrative_health(self) -> Dict:
        """Get stats about the current memory state"""
        if not self.memory_embeddings:
            return {'status': 'empty'}
            
        # Diversity of memory (avg distance from mean)
        diversity = np.mean([np.linalg.norm(e - self.baseline_vector) for e in self.memory_embeddings])
        
        return {
            'memory_size': len(self.memory_embeddings),
            'diversity': round(float(diversity), 3),
            'last_update': self.memory_timestamps[-1].isoformat() if self.memory_timestamps else None
        }

if __name__ == "__main__":
    # Test
    print("Testing Surprise Score Model...")
    from src.nlp.embeddings import get_embeddings
    
    model = SurpriseModel(memory_window=10)
    
    # Simulate a narrative: "Inflation and Rates"
    history = [
        "RBI expected to hike rates",
        "Inflation remains high in Q3",
        "Central bank hawkish on policy",
        "Bond yields rise expecting rate hike",
        "Repo rate hike likely next week"
    ]
    
    print("Building narrative memory...")
    for h in history:
        emb = get_embeddings(h)
        model.update(emb)
        
    print(f"Baseline established. Memory size: {len(model.memory_embeddings)}")
    
    # Test Case 1: Expected News (Low Surprise)
    news_expected = "RBI hikes repo rate as predicted"
    emb_expected = get_embeddings(news_expected)
    res_expected = model.calculate_surprise(emb_expected)
    print(f"\nNews: '{news_expected}'")
    print(f"Surprise: {res_expected['surprise_score']} (Distance: {res_expected['distance']})")
    
    # Test Case 2: Unexpected News (High Surprise)
    news_shock = "Government announces massive tax cut for autos"
    emb_shock = get_embeddings(news_shock)
    res_shock = model.calculate_surprise(emb_shock)
    print(f"\nNews: '{news_shock}'")
    print(f"Surprise: {res_shock['surprise_score']} (Distance: {res_shock['distance']})")

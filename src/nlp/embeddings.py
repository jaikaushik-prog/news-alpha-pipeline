"""
Semantic Embeddings Layer
Generates dense vector representations for news headlines using Sentence Transformers.
"""
import os
import sys
import pickle
import hashlib
from typing import List, Dict, Union, Optional
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class EmbeddingGenerator:
    """
    Generates and manages semantic embeddings for text.
    Includes caching to prevent redundant computation.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the SentenceTransformer model
            cache_dir: Directory to store embedding cache (default: data/cache/embeddings)
        """
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing EmbeddingGenerator with {model_name} on {self.device}...")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Setup cache
        if cache_dir:
            self.cache_path = Path(cache_dir)
        else:
            self.cache_path = PROJECT_ROOT / 'data' / 'cache' / 'embeddings'
            
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_path / f"emb_cache_{model_name.replace('/', '_')}.pkl"
        self.cache = self._load_cache()
        self._cache_dirty = False
        
    def _load_cache(self) -> Dict[str, np.ndarray]:
        """Load embeddings from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                return {}
        return {}
        
    def save_cache(self):
        """Save embeddings to disk"""
        if self._cache_dirty:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
                self._cache_dirty = False
                print(f"Saved {len(self.cache)} embeddings to cache")
            except Exception as e:
                print(f"Error saving cache: {e}")
    
    def _get_hash(self, text: str) -> str:
        """Generate MD5 hash for text key"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def generate(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for a string or list of strings.
        
        Args:
            texts: Single text string or list of strings
            
        Returns:
            Numpy array of embeddings
        """
        single_input = isinstance(texts, str)
        if single_input:
            text_list = [texts]
        else:
            text_list = texts
            
        # Clean inputs
        text_list = [t.strip() for t in text_list if t.strip()]
        if not text_list:
            return np.array([])
            
        # Check cache
        embeddings = []
        to_compute_indices = []
        to_compute_texts = []
        
        for i, text in enumerate(text_list):
            h = self._get_hash(text)
            if h in self.cache:
                embeddings.append(self.cache[h])
            else:
                embeddings.append(None) # Placeholder
                to_compute_indices.append(i)
                to_compute_texts.append(text)
        
        # Compute missing embeddings
        if to_compute_texts:
            print(f"Computing embeddings for {len(to_compute_texts)} new items...")
            new_embeddings = self.model.encode(to_compute_texts, convert_to_numpy=True)
            
            for i, idx in enumerate(to_compute_indices):
                emb = new_embeddings[i]
                text = to_compute_texts[i]
                h = self._get_hash(text)
                
                # Update cache and result list
                self.cache[h] = emb
                embeddings[idx] = emb
                
            self._cache_dirty = True
            
        result = np.array(embeddings)
        
        # Periodically save cache if it grows too much (simple strategy)
        if self._cache_dirty and len(to_compute_texts) > 100:
            self.save_cache()
            
        return result[0] if single_input else result
        
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        emb1 = self.generate(text1)
        emb2 = self.generate(text2)
        
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Singleton instance
_generator = None

def get_embeddings(texts: Union[str, List[str]], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Helper function to get embeddings using singleton generator"""
    global _generator
    if _generator is None:
        _generator = EmbeddingGenerator(model_name=model_name)
    return _generator.generate(texts)

if __name__ == "__main__":
    # Test
    print("Testing Embedding Generator...")
    gen = EmbeddingGenerator()
    
    headlines = [
        "RBI raises repo rate by 25 bps",
        "Central bank hikes interest rates",
        "TCS posts strong Q3 profit growth",
        "Infosys earnings beat estimates",
        "Nifty hits all-time high"
    ]
    
    embs = gen.generate(headlines)
    print(f"Generated {len(embs)} embeddings of shape {embs[0].shape}")
    
    sim = gen.similarity(headlines[0], headlines[1])
    print(f"\nSimilarity between:")
    print(f"1. {headlines[0]}")
    print(f"2. {headlines[1]}")
    print(f"Score: {sim:.4f}")
    
    sim_diff = gen.similarity(headlines[0], headlines[2])
    print(f"\nSimilarity between:")
    print(f"1. {headlines[0]}")
    print(f"2. {headlines[2]}")
    print(f"Score: {sim_diff:.4f}")
    
    gen.save_cache()

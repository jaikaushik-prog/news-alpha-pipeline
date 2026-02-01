"""
Semantic Sector Attribution Model
Maps news headlines to sectors probabilistically using semantic embeddings.
"""
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.embeddings import get_embeddings

class SectorAttributionModel:
    """
    Calculates the exposure of a news item to various sectors based on 
    semantic similarity to sector definitions.
    """
    
    def __init__(self):
        self.sector_profiles = self._define_sectors()
        self.sector_embeddings = {}
        self._initialize_embeddings()
        
    def _define_sectors(self) -> Dict[str, str]:
        """
        Define canonical descriptions for each sector.
        These are used to create the 'reference vectors' for the sectors.
        """
        return {
            'banking': "Banking sector, interest rates, RBI policy, credit growth, NPA, loans, digital payments, fintech regulation, PSU banks, private banks, monetary policy, liquidity.",
            'it': "Information Technology, software services, AI, cloud computing, digital transformation, H1B visas, outsourcing, tech hiring, attrition, deal wins, large implementation.",
            'auto': "Automotive industry, electric vehicles, EV sales, passenger cars, two wheelers, commercial vehicles, auto components, chip shortage, vehicle scraping policy.",
            'pharma': "Pharmaceuticals, healthcare, generic drugs, USFDA approval, clinical trials, vaccine, hospitals, diagnostics, API manufacturing, drug pricing.",
            'energy': "Oil and gas, crude prices, refining margins, renewable energy, solar power, green hydrogen, thermal power, coal, electricity demand, offshore drilling.",
            'fmcg': "Fast moving consumer goods, rural demand, consumption, retail prices, inflation impact, volume growth, packaged food, personal care, staples.",
            'metals': "Metals and mining, steel prices, iron ore, aluminium, copper, china demand, commodity cycle, export duty, infrastructure demand.",
            'infra': "Infrastructure, construction, roads, highways, cement, real estate, affordable housing, order book, capital expenditure, capex, logistics.",
            'telecom': "Telecommunications, 5G rollout, spectrum auction, ARPU, tariff hikes, subscriber growth, adjusted gross revenue, mobile data.",
            'defence': "Defence manufacturing, indigenization, order book, export orders, military modernization, drones, missiles, naval ships, aircraft."
        }

    def _initialize_embeddings(self):
        """Compute the centroid embedding for each sector profile"""
        print("Initializing Sector Attribution Model...")
        for sector, description in self.sector_profiles.items():
            # semantic vector for the description
            emb = get_embeddings(description)
            self.sector_embeddings[sector] = emb
            
    def get_sector_decomposition(self, headline: str, headline_embedding: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Decompose a headline into sector weights.
        Returns a dictionary of {sector: weight} where weight is [0, 1].
        """
        if headline_embedding is None:
            headline_embedding = get_embeddings(headline)
            
        weights = {}
        
        for sector, sector_emb in self.sector_embeddings.items():
            # Cosine similarity
            similarity = np.dot(headline_embedding, sector_emb) / (
                np.linalg.norm(headline_embedding) * np.linalg.norm(sector_emb)
            )
            
            # ReLU activation + scaling
            # We want only positive correlations, and we want to amplify the signal
            # Typical similarity for related topics is 0.3-0.6
            
            if similarity > 0.2:
                # Scale 0.2 -> 0.0, 0.6 -> 1.0
                adjusted_score = (similarity - 0.2) * 2.5
                weights[sector] = round(max(0.0, min(1.0, float(adjusted_score))), 3)
            else:
                weights[sector] = 0.0
                
        # Normalize? Not necessarily. A headline can impact multiple sectors.
        # But we probably want to filter out low noise.
        
        filtered_weights = {k: v for k, v in weights.items() if v > 0.15}
        
        # Sort by weight desc
        return dict(sorted(filtered_weights.items(), key=lambda item: item[1], reverse=True))

if __name__ == "__main__":
    # Test
    model = SectorAttributionModel()
    
    test_headlines = [
        "RBI raises repo rate by 50 basis points to fight inflation",
        "TCS signs multi-million dollar cloud deal with UK bank",
        "Steel prices crash as China demand slows down",
        "Government announces new PLI scheme for EV batteries",
        "Crude oil jumps to $90 on supply concerns"
    ]
    
    print("\nTesting Sector Attribution:")
    for h in test_headlines:
        print(f"\nHeadline: {h}")
        weights = model.get_sector_decomposition(h)
        print(f"Attribution: {weights}")

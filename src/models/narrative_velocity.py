"""
Narrative Velocity & Acceleration ⭐⭐⭐⭐

Not WHAT is being said — HOW FAST it's spreading.

Formulas:
    Narrative_Velocity = d(sentiment_volume)/dt
    Narrative_Acceleration = d²(sentiment_volume)/dt²

Insight Table:
| Situation                         | Meaning        |
|-----------------------------------|----------------|
| High sentiment, low velocity      | Priced in      |
| Moderate sentiment, high accel    | Early move     |
| Falling velocity                  | Narrative dying|

This is second-order thinking + information diffusion theory.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass  
class NarrativeKinematics:
    """Velocity and acceleration of a narrative/sector."""
    sector: str
    current_volume: float           # Current sentiment volume
    velocity: float                 # First derivative (change rate)
    acceleration: float             # Second derivative (change in change)
    momentum_phase: str             # 'emerging', 'accelerating', 'peaking', 'fading', 'priced_in'
    trade_timing: str               # 'early', 'on_time', 'late', 'avoid'


class NarrativeVelocityTracker:
    """
    Tracks velocity and acceleration of sector narratives.
    
    Inspired by information diffusion theory.
    
    Usage:
        tracker = NarrativeVelocityTracker()
        
        # Update with daily sector volume/intensity
        tracker.update('banking', {
            'volume': 25,        # Number of articles
            'intensity': 0.6    # Average sentiment intensity
        })
        
        # Get kinematics
        kinematics = tracker.get_kinematics('banking')
    """
    
    def __init__(
        self,
        lookback: int = 10,
        smoothing: float = 0.3
    ):
        """
        Initialize tracker.
        
        Parameters
        ----------
        lookback : int
            Days of history for derivatives
        smoothing : float
            Exponential smoothing factor
        """
        self.lookback = lookback
        self.smoothing = smoothing
        
        # Time series per sector
        self._volume_history: Dict[str, List[float]] = {}
        self._intensity_history: Dict[str, List[float]] = {}
        self._timestamps: Dict[str, List[datetime]] = {}
    
    def update(
        self,
        sector: str,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ):
        """
        Update sector with new observation.
        
        Parameters
        ----------
        sector : str
            Sector name
        metrics : dict
            {'volume': count, 'intensity': avg_intensity}
        timestamp : datetime, optional
            Observation timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Initialize if needed
        if sector not in self._volume_history:
            self._volume_history[sector] = []
            self._intensity_history[sector] = []
            self._timestamps[sector] = []
        
        # Append new data
        volume = metrics.get('volume', 0)
        intensity = metrics.get('intensity', 0)
        
        self._volume_history[sector].append(volume)
        self._intensity_history[sector].append(intensity)
        self._timestamps[sector].append(timestamp)
        
        # Keep only recent history
        if len(self._volume_history[sector]) > self.lookback * 2:
            self._volume_history[sector] = self._volume_history[sector][-self.lookback * 2:]
            self._intensity_history[sector] = self._intensity_history[sector][-self.lookback * 2:]
            self._timestamps[sector] = self._timestamps[sector][-self.lookback * 2:]
    
    def _calculate_derivative(self, series: List[float], order: int = 1) -> float:
        """
        Calculate derivative using finite differences.
        
        Parameters
        ----------
        series : list
            Time series data
        order : int
            1 for velocity, 2 for acceleration
            
        Returns
        -------
        float
            Derivative value
        """
        if len(series) < order + 1:
            return 0.0
        
        arr = np.array(series[-self.lookback:])
        
        if order == 1:
            # First derivative: recent change
            if len(arr) < 2:
                return 0.0
            # Weighted by recency
            diffs = np.diff(arr)
            weights = np.exp(np.linspace(-1, 0, len(diffs)))
            velocity = np.average(diffs, weights=weights)
            return float(velocity)
            
        elif order == 2:
            # Second derivative: change in change
            if len(arr) < 3:
                return 0.0
            diffs = np.diff(arr)
            second_diffs = np.diff(diffs)
            if len(second_diffs) > 0:
                weights = np.exp(np.linspace(-1, 0, len(second_diffs)))
                acceleration = np.average(second_diffs, weights=weights)
                return float(acceleration)
            return 0.0
        
        return 0.0
    
    def get_kinematics(self, sector: str) -> NarrativeKinematics:
        """
        Get velocity and acceleration for sector.
        
        Parameters
        ----------
        sector : str
            Sector name
            
        Returns
        -------
        NarrativeKinematics
            Complete kinematic analysis
        """
        volume_history = self._volume_history.get(sector, [])
        intensity_history = self._intensity_history.get(sector, [])
        
        if not volume_history:
            return NarrativeKinematics(
                sector=sector,
                current_volume=0,
                velocity=0,
                acceleration=0,
                momentum_phase='unknown',
                trade_timing='avoid'
            )
        
        # Current state
        current_volume = volume_history[-1] if volume_history else 0
        current_intensity = intensity_history[-1] if intensity_history else 0
        
        # Combine volume and intensity into "narrative strength"
        combined = [v * i for v, i in zip(volume_history, intensity_history)]
        
        # Calculate derivatives
        velocity = self._calculate_derivative(combined, order=1)
        acceleration = self._calculate_derivative(combined, order=2)
        
        # Classify momentum phase
        phase, timing = self._classify_phase(current_volume, velocity, acceleration)
        
        return NarrativeKinematics(
            sector=sector,
            current_volume=current_volume,
            velocity=velocity,
            acceleration=acceleration,
            momentum_phase=phase,
            trade_timing=timing
        )
    
    def _classify_phase(
        self,
        volume: float,
        velocity: float,
        acceleration: float
    ) -> Tuple[str, str]:
        """
        Classify narrative phase and trading timing.
        
        Returns
        -------
        tuple
            (phase, timing)
        """
        # Thresholds (can be tuned)
        vol_threshold = 0.5
        accel_threshold = 0.3
        
        if velocity > vol_threshold and acceleration > accel_threshold:
            # Rising fast and accelerating
            return 'emerging', 'early'
        
        elif velocity > vol_threshold and acceleration < -accel_threshold:
            # Rising but decelerating
            return 'peaking', 'on_time'
        
        elif velocity < -vol_threshold and acceleration < -accel_threshold:
            # Falling and accelerating down
            return 'fading', 'late'
        
        elif abs(velocity) < vol_threshold and volume > 0:
            # Stable high volume
            return 'priced_in', 'avoid'
        
        elif velocity > 0 and abs(acceleration) < accel_threshold:
            # Steady rise
            return 'accelerating', 'on_time'
        
        else:
            return 'stable', 'neutral'
    
    def get_all_kinematics(self) -> List[NarrativeKinematics]:
        """Get kinematics for all tracked sectors."""
        return [self.get_kinematics(s) for s in self._volume_history.keys()]
    
    def get_early_movers(self) -> List[str]:
        """Get sectors where it's early to enter."""
        kinematics = self.get_all_kinematics()
        return [k.sector for k in kinematics if k.trade_timing == 'early']
    
    def get_priced_in(self) -> List[str]:
        """Get sectors that are already priced in."""
        kinematics = self.get_all_kinematics()
        return [k.sector for k in kinematics if k.momentum_phase == 'priced_in']


def kinematics_to_dataframe(kinematics: List[NarrativeKinematics]) -> pd.DataFrame:
    """Convert kinematics to DataFrame."""
    return pd.DataFrame([
        {
            'sector': k.sector,
            'volume': k.current_volume,
            'velocity': k.velocity,
            'acceleration': k.acceleration,
            'phase': k.momentum_phase,
            'timing': k.trade_timing
        }
        for k in kinematics
    ])


def analyze_narrative_dynamics(
    sector_timeseries: Dict[str, pd.DataFrame],
    volume_col: str = 'article_count',
    intensity_col: str = 'avg_intensity'
) -> pd.DataFrame:
    """
    Analyze narrative dynamics from time series.
    
    Parameters
    ----------
    sector_timeseries : dict
        {sector: DataFrame with date, volume, intensity}
    volume_col : str
        Column for volume
    intensity_col : str
        Column for intensity
        
    Returns
    -------
    pd.DataFrame
        Narrative dynamics analysis
    """
    tracker = NarrativeVelocityTracker()
    
    # Feed all data
    for sector, df in sector_timeseries.items():
        for _, row in df.iterrows():
            tracker.update(sector, {
                'volume': row.get(volume_col, 1),
                'intensity': row.get(intensity_col, 0.5)
            })
    
    return kinematics_to_dataframe(tracker.get_all_kinematics())


# Quick demonstration function
def demo_narrative_velocity():
    """Demo the narrative velocity tracker."""
    tracker = NarrativeVelocityTracker()
    
    # Simulate emerging narrative in banking
    for i in range(10):
        tracker.update('banking', {
            'volume': 5 + i * 3,  # Increasing
            'intensity': 0.4 + i * 0.05
        })
    
    # Simulate priced-in IT narrative
    for i in range(10):
        tracker.update('it', {
            'volume': 20,  # Stable high
            'intensity': 0.6
        })
    
    # Simulate fading pharma narrative
    for i in range(10):
        tracker.update('pharma', {
            'volume': 30 - i * 3,  # Decreasing
            'intensity': 0.5 - i * 0.03
        })
    
    print("Narrative Dynamics:")
    for k in tracker.get_all_kinematics():
        print(f"  {k.sector}: {k.momentum_phase} (timing: {k.trade_timing})")
        print(f"    velocity={k.velocity:.2f}, accel={k.acceleration:.2f}")
    
    return tracker

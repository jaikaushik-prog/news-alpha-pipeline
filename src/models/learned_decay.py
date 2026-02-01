"""
Learned Half-Lives Estimation ⭐⭐⭐⭐

Instead of manually assigning decay, LEARN half-life from data.

Method:
    1. For each news event, track abnormal returns over time
    2. Fit exponential decay: Impact(t) ≈ e^(−t / τ)
    3. Estimate τ (half-life) per news category

This is:
- Adaptive
- Statistically grounded
- Very few projects do this properly
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HalfLifeEstimate:
    """Estimated half-life for a news category."""
    category: str
    half_life_hours: float
    decay_rate: float           # τ in e^(-t/τ)
    r_squared: float            # Goodness of fit
    sample_size: int
    confidence: str             # 'high', 'medium', 'low'


class HalfLifeLearner:
    """
    Learn information half-lives from historical data.
    
    Instead of manual assignment (8h for policy, 24h for earnings),
    this estimates decay empirically from price data.
    
    Usage:
        learner = HalfLifeLearner(price_data_dir="c:/path/to/stocks")
        
        # Add historical events
        learner.add_event(
            timestamp=datetime(2024, 1, 15, 10, 0),
            category='earnings',
            stocks=['HDFCBANK', 'ICICIBANK'],
            sentiment=0.7
        )
        
        # Estimate half-lives
        estimates = learner.estimate_half_lives()
    """
    
    def __init__(
        self,
        price_data_dir: str,
        max_horizon_hours: int = 48,   # Max time to track impact
        min_events: int = 10           # Min events to estimate
    ):
        """
        Initialize learner.
        
        Parameters
        ----------
        price_data_dir : str
            Directory containing stock CSV files
        max_horizon_hours : int
            Maximum hours to track impact
        min_events : int
            Minimum events needed for reliable estimate
        """
        self.price_dir = Path(price_data_dir)
        self.max_horizon = max_horizon_hours
        self.min_events = min_events
        
        # Store events
        self._events: List[Dict] = []
        
        # Cache loaded price data
        self._price_cache: Dict[str, pd.DataFrame] = {}
    
    def load_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load stock price data from CSV."""
        if symbol in self._price_cache:
            return self._price_cache[symbol]
        
        # Try different file patterns
        patterns = [
            f"{symbol}.csv",
            f"{symbol.upper()}.csv",
            f"{symbol.lower()}.csv"
        ]
        
        for pattern in patterns:
            filepath = self.price_dir / pattern
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath, parse_dates=['date'])
                    df = df.set_index('date').sort_index()
                    self._price_cache[symbol] = df
                    return df
                except Exception as e:
                    logger.warning(f"Error loading {symbol}: {e}")
        
        return None
    
    def add_event(
        self,
        timestamp: datetime,
        category: str,
        stocks: List[str],
        sentiment: float
    ):
        """
        Add a news event for analysis.
        
        Parameters
        ----------
        timestamp : datetime
            When the news occurred
        category : str
            News category (earnings, policy, macro, etc.)
        stocks : list
            Affected stocks
        sentiment : float
            Sentiment score (-1 to 1)
        """
        self._events.append({
            'timestamp': timestamp,
            'category': category,
            'stocks': stocks,
            'sentiment': sentiment
        })
    
    def calculate_abnormal_returns(
        self,
        symbol: str,
        event_time: datetime,
        hours: int = 48
    ) -> Optional[pd.Series]:
        """
        Calculate abnormal returns after an event.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        event_time : datetime
            Event timestamp
        hours : int
            Hours to track
            
        Returns
        -------
        pd.Series
            Returns indexed by hours since event
        """
        df = self.load_stock_data(symbol)
        if df is None:
            return None
        
        try:
            # Find event time in data
            event_dt = pd.Timestamp(event_time).tz_localize(None)
            
            # Get data around event
            mask = (df.index >= event_dt) & (df.index <= event_dt + timedelta(hours=hours))
            event_data = df.loc[mask, 'close']
            
            if len(event_data) < 2:
                return None
            
            # Calculate returns from event start
            base_price = event_data.iloc[0]
            returns = (event_data - base_price) / base_price * 100
            
            # Convert index to hours since event
            hours_since = (returns.index - event_dt).total_seconds() / 3600
            returns.index = hours_since
            
            return returns
            
        except Exception as e:
            logger.debug(f"Error calculating returns for {symbol}: {e}")
            return None
    
    def estimate_decay_curve(
        self,
        category: str
    ) -> Optional[HalfLifeEstimate]:
        """
        Estimate decay curve for a category.
        
        Uses exponential fitting: Impact(t) = A × e^(-t/τ)
        """
        # Get events for category
        cat_events = [e for e in self._events if e['category'] == category]
        
        if len(cat_events) < self.min_events:
            logger.info(f"Insufficient events for {category}: {len(cat_events)}")
            return None
        
        # Collect all return series
        all_returns = []
        
        for event in cat_events:
            for stock in event['stocks']:
                returns = self.calculate_abnormal_returns(
                    stock, 
                    event['timestamp'],
                    self.max_horizon
                )
                if returns is not None and len(returns) > 10:
                    # Normalize by sentiment direction
                    if event['sentiment'] < 0:
                        returns = -returns
                    all_returns.append(returns)
        
        if len(all_returns) < 5:
            return None
        
        # Aggregate returns by hour
        hourly_impact = {}
        for returns in all_returns:
            for hour, ret in returns.items():
                if hour >= 0:
                    hour_bin = int(hour)
                    if hour_bin not in hourly_impact:
                        hourly_impact[hour_bin] = []
                    hourly_impact[hour_bin].append(abs(ret))
        
        if len(hourly_impact) < 5:
            return None
        
        # Average impact per hour
        hours = sorted(hourly_impact.keys())
        impacts = [np.mean(hourly_impact[h]) for h in hours]
        
        # Fit exponential decay
        try:
            def exp_decay(t, a, tau):
                return a * np.exp(-t / tau)
            
            # Initial guess
            popt, pcov = curve_fit(
                exp_decay,
                hours,
                impacts,
                p0=[impacts[0], 8.0],  # Initial: amplitude, 8h half-life
                bounds=([0, 0.5], [np.inf, 100]),
                maxfev=5000
            )
            
            amplitude, tau = popt
            
            # Calculate R-squared
            fitted = [exp_decay(h, amplitude, tau) for h in hours]
            ss_res = sum((i - f) ** 2 for i, f in zip(impacts, fitted))
            ss_tot = sum((i - np.mean(impacts)) ** 2 for i in impacts)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Half-life = τ × ln(2)
            half_life = tau * np.log(2)
            
            # Confidence based on R² and sample size
            if r_squared > 0.7 and len(all_returns) > 20:
                confidence = 'high'
            elif r_squared > 0.4 and len(all_returns) > 10:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            return HalfLifeEstimate(
                category=category,
                half_life_hours=half_life,
                decay_rate=tau,
                r_squared=r_squared,
                sample_size=len(all_returns),
                confidence=confidence
            )
            
        except Exception as e:
            logger.warning(f"Curve fitting failed for {category}: {e}")
            return None
    
    def estimate_all_half_lives(self) -> Dict[str, HalfLifeEstimate]:
        """Estimate half-lives for all categories."""
        categories = set(e['category'] for e in self._events)
        
        estimates = {}
        for cat in categories:
            est = self.estimate_decay_curve(cat)
            if est is not None:
                estimates[cat] = est
                logger.info(f"  {cat}: {est.half_life_hours:.1f}h (R²={est.r_squared:.2f})")
        
        return estimates


def learn_half_lives_from_data(
    events_df: pd.DataFrame,
    price_data_dir: str,
    timestamp_col: str = 'timestamp',
    category_col: str = 'category',
    stocks_col: str = 'stocks',
    sentiment_col: str = 'sentiment'
) -> Dict[str, float]:
    """
    Convenience function to learn half-lives from events DataFrame.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        DataFrame with event data
    price_data_dir : str
        Directory with stock CSVs
    
    Returns
    -------
    dict
        {category: half_life_hours}
    """
    learner = HalfLifeLearner(price_data_dir)
    
    for _, row in events_df.iterrows():
        learner.add_event(
            timestamp=row[timestamp_col],
            category=row[category_col],
            stocks=row[stocks_col] if isinstance(row[stocks_col], list) else [row[stocks_col]],
            sentiment=row[sentiment_col]
        )
    
    estimates = learner.estimate_all_half_lives()
    
    return {cat: est.half_life_hours for cat, est in estimates.items()}


def get_default_half_lives() -> Dict[str, float]:
    """
    Get default half-lives (used when no data available).
    
    These are manually calibrated values.
    """
    return {
        'policy': 8.0,
        'earnings': 24.0,
        'macro': 72.0,
        'breaking': 4.0,
        'general': 12.0
    }


def create_adaptive_decay_config(
    learned: Dict[str, float],
    defaults: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Create adaptive decay config, falling back to defaults.
    
    Parameters
    ----------
    learned : dict
        Learned half-lives from data
    defaults : dict, optional
        Default half-lives for categories not in learned
        
    Returns
    -------
    dict
        Combined half-lives
    """
    if defaults is None:
        defaults = get_default_half_lives()
    
    config = defaults.copy()
    config.update(learned)
    
    return config

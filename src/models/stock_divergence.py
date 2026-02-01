"""
Stock-Level Divergence Detector

Uses actual stock price data to detect sentiment-price divergence at the stock level.
More granular than sector-level analysis.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .divergence import DivergenceDetector, DivergenceSignal, DivergenceType
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StockDivergence:
    """Stock-level divergence signal."""
    symbol: str
    sentiment: float
    price_return: float           # Daily return
    volume_ratio: float           # Volume vs 20-day avg
    divergence_type: str          # 'absorption', 'hidden_strength', 'confirmation'
    smart_money: str              # 'accumulating', 'distributing', 'neutral'
    signal: str                   # 'long', 'short', 'hold'
    confidence: float


class StockDivergenceDetector:
    """
    Detect divergence at stock level using actual price data.
    
    Usage:
        detector = StockDivergenceDetector("c:/path/to/stocks")
        
        signal = detector.analyze(
            symbol='HDFCBANK',
            sentiment=0.65,
            date=datetime(2024, 1, 15)
        )
    """
    
    def __init__(self, price_data_dir: str):
        """
        Initialize detector.
        
        Parameters
        ----------
        price_data_dir : str
            Directory containing stock CSV files
        """
        self.price_dir = Path(price_data_dir)
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._detector = DivergenceDetector()
    
    def load_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load and cache stock data."""
        if symbol in self._price_cache:
            return self._price_cache[symbol]
        
        filepath = self.price_dir / f"{symbol}.csv"
        if not filepath.exists():
            filepath = self.price_dir / f"{symbol.upper()}.csv"
        
        if not filepath.exists():
            return None
        
        try:
            df = pd.read_csv(filepath, parse_dates=['date'])
            
            # Handle timezone if present
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            
            df = df.set_index('date').sort_index()
            self._price_cache[symbol] = df
            return df
            
        except Exception as e:
            logger.warning(f"Error loading {symbol}: {e}")
            return None
    
    def get_daily_data(self, symbol: str, date: datetime) -> Optional[Dict]:
        """
        Get daily OHLCV data for a specific date.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        date : datetime
            Target date
            
        Returns
        -------
        dict or None
            {open, high, low, close, volume, return, volume_ratio}
        """
        df = self.load_stock_data(symbol)
        if df is None:
            return None
        
        # Convert date to match index
        target_date = pd.Timestamp(date).normalize()
        
        # Get intraday data for the date
        mask = (df.index >= target_date) & (df.index < target_date + timedelta(days=1))
        day_data = df.loc[mask]
        
        if day_data.empty:
            # Try nearby dates
            nearby = df.loc[(df.index >= target_date - timedelta(days=5)) & 
                           (df.index < target_date + timedelta(days=5))]
            if nearby.empty:
                return None
            day_data = nearby.iloc[-75:]  # Approximately 1 day of 5-min data
        
        # Aggregate to daily OHLCV
        daily = {
            'open': day_data['open'].iloc[0],
            'high': day_data['high'].max(),
            'low': day_data['low'].min(),
            'close': day_data['close'].iloc[-1],
            'volume': day_data['volume'].sum()
        }
        
        # Calculate return
        if len(day_data) > 1:
            daily['return'] = (daily['close'] - daily['open']) / daily['open'] * 100
        else:
            daily['return'] = 0
        
        # Calculate volume ratio (vs 20-day average)
        # Get last 20 trading days of volume
        end_idx = df.index.get_indexer([day_data.index[0]], method='nearest')[0]
        start_idx = max(0, end_idx - 20 * 75)  # ~20 days of 5-min data
        hist_volume = df.iloc[start_idx:end_idx]['volume'].sum() / 20
        
        daily['volume_ratio'] = daily['volume'] / hist_volume if hist_volume > 0 else 1.0
        
        return daily
    
    def analyze(
        self,
        symbol: str,
        sentiment: float,
        date: Optional[datetime] = None
    ) -> Optional[StockDivergence]:
        """
        Analyze stock for divergence.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        sentiment : float
            News sentiment for this stock (-1 to 1)
        date : datetime, optional
            Date to analyze (default: latest available)
            
        Returns
        -------
        StockDivergence or None
        """
        daily = self.get_daily_data(symbol, date or datetime.now())
        if daily is None:
            return None
        
        # Use base divergence detector
        signal = self._detector.analyze(
            sector=symbol,
            sentiment=sentiment,
            price_return=daily['return'],
            volume_ratio=daily['volume_ratio']
        )
        
        return StockDivergence(
            symbol=symbol,
            sentiment=sentiment,
            price_return=daily['return'],
            volume_ratio=daily['volume_ratio'],
            divergence_type=signal.divergence_type.value,
            smart_money=signal.smart_money_signal,
            signal=signal.trade_action,
            confidence=signal.confidence
        )
    
    def batch_analyze(
        self,
        stock_sentiment: Dict[str, float],
        date: Optional[datetime] = None
    ) -> List[StockDivergence]:
        """
        Analyze multiple stocks.
        
        Parameters
        ----------
        stock_sentiment : dict
            {symbol: sentiment_score}
        date : datetime, optional
            Date to analyze
            
        Returns
        -------
        list
            List of StockDivergence signals
        """
        signals = []
        for symbol, sentiment in stock_sentiment.items():
            signal = self.analyze(symbol, sentiment, date)
            if signal is not None:
                signals.append(signal)
        
        # Sort by confidence
        return sorted(signals, key=lambda x: x.confidence, reverse=True)
    
    def get_actionable_stocks(
        self,
        signals: List[StockDivergence],
        min_confidence: float = 0.5
    ) -> Dict[str, List[str]]:
        """
        Get actionable stock recommendations.
        
        Returns
        -------
        dict
            {action: [symbols]}
        """
        result = {'long': [], 'short': [], 'hold': []}
        
        for signal in signals:
            if signal.confidence >= min_confidence:
                result[signal.signal].append(signal.symbol)
        
        return result


def divergence_to_dataframe(signals: List[StockDivergence]) -> pd.DataFrame:
    """Convert signals to DataFrame."""
    return pd.DataFrame([
        {
            'symbol': s.symbol,
            'sentiment': s.sentiment,
            'return': s.price_return,
            'volume_ratio': s.volume_ratio,
            'divergence': s.divergence_type,
            'smart_money': s.smart_money,
            'signal': s.signal,
            'confidence': s.confidence
        }
        for s in signals
    ])


# Mapping of stocks to sectors for sector-level aggregation
STOCK_SECTOR_MAP = {
    # Banking
    'HDFCBANK': 'banking', 'ICICIBANK': 'banking', 'KOTAKBANK': 'banking',
    'AXISBANK': 'banking', 'SBIN': 'banking', 'INDUSINDBK': 'banking',
    'BANKBARODA': 'banking', 'PNB': 'banking',
    
    # IT
    'TCS': 'it', 'INFY': 'it', 'WIPRO': 'it', 'HCLTECH': 'it',
    'TECHM': 'it', 'LTIM': 'it', 'MPHASIS': 'it', 'COFORGE': 'it',
    
    # Pharma
    'SUNPHARMA': 'pharma', 'DRREDDY': 'pharma', 'CIPLA': 'pharma',
    'DIVISLAB': 'pharma', 'LUPIN': 'pharma', 'AUROPHARMA': 'pharma',
    
    # Auto
    'MARUTI': 'auto', 'TATAMOTORS': 'auto', 'M&M': 'auto',
    'BAJAJ-AUTO': 'auto', 'EICHERMOT': 'auto', 'HEROMOTOCO': 'auto',
    
    # Metals
    'TATASTEEL': 'metals', 'HINDALCO': 'metals', 'JSWSTEEL': 'metals',
    'VEDL': 'metals', 'COALINDIA': 'metals', 'NMDC': 'metals',
    
    # Infra
    'LT': 'infra', 'ADANIPORTS': 'infra', 'ULTRACEMCO': 'infra',
    'GRASIM': 'infra', 'SHREECEM': 'infra', 'ACC': 'infra',
    
    # FMCG
    'ITC': 'fmcg', 'HINDUNILVR': 'fmcg', 'NESTLEIND': 'fmcg',
    'BRITANNIA': 'fmcg', 'DABUR': 'fmcg', 'MARICO': 'fmcg',
    
    # Energy
    'RELIANCE': 'energy', 'ONGC': 'energy', 'BPCL': 'energy',
    'IOC': 'energy', 'GAIL': 'energy', 'NTPC': 'energy',
}


def get_sector_for_stock(symbol: str) -> str:
    """Get sector for a stock symbol."""
    return STOCK_SECTOR_MAP.get(symbol.upper(), 'general')

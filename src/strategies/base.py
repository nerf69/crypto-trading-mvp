from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd
from datetime import datetime

# Import constants for enterprise-grade configuration
try:
    from ..constants import (
        HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD,
        MAX_POSITION_SIZE, MEDIUM_POSITION_SIZE, MIN_POSITION_SIZE
    )
except ImportError:
    # Fallback values if constants module not available
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD = 0.6
    MAX_POSITION_SIZE = 1.0
    MEDIUM_POSITION_SIZE = 0.66
    MIN_POSITION_SIZE = 0.33

class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    timestamp: datetime
    pair: str
    signal: SignalType
    price: float
    confidence: float  # 0-1 score
    strategy_name: str
    reason: str
    indicators: dict  # Store relevant indicator values
    
    def position_size_recommendation(self) -> float:
        """Recommend position size based on confidence using enterprise constants"""
        if self.confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return MAX_POSITION_SIZE
        elif self.confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
            return MEDIUM_POSITION_SIZE
        else:
            return MIN_POSITION_SIZE

class Strategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def calculate_signal(self, df: pd.DataFrame, pair: str) -> TradingSignal:
        """
        Calculate trading signal based on data
        
        Args:
            df: DataFrame with OHLCV data and indicators
            pair: Trading pair symbol
            
        Returns:
            TradingSignal object
        """
        pass
    
    def add_required_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add any indicators required by this strategy
        Override in subclasses to add specific indicators
        """
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that required data is present"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in df.columns for col in required_columns)
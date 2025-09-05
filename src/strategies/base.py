from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd
from datetime import datetime

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
        """Recommend position size based on confidence"""
        if self.confidence >= 0.8:
            return 1.0  # 100% of available capital
        elif self.confidence >= 0.6:
            return 0.66  # 66%
        else:
            return 0.33  # 33%

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
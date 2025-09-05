import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any

from .base import Strategy, TradingSignal, SignalType

logger = logging.getLogger(__name__)

class SwingTradingStrategy(Strategy):
    """
    2.5% Swing Trading Strategy (Optimized for Daily Data)
    
    Buy signals when:
    - Price drops 2.5% or more from recent high (10-day lookback)
    - RSI is oversold (< 35) - adjusted for daily timeframe
    - Volume is above average (1.1x threshold)
    
    Sell signals when:
    - Price gains 2.5% or more from recent low
    - RSI is overbought (> 65) - adjusted for daily timeframe
    - Take profit target hit (2.5-7.5%)
    """
    
    def __init__(self, swing_threshold: float = 0.025, volume_threshold: float = 1.1):
        super().__init__("2.5% Swing Strategy")
        self.swing_threshold = swing_threshold  # 2.5% swing threshold for daily data
        self.volume_threshold = volume_threshold  # Volume must be 1.1x average (less restrictive)
        self.lookback_period = 10  # Period to look for highs/lows (reduced for daily)
    
    def add_required_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators required by this strategy"""
        if df.empty or len(df) < self.lookback_period:
            return df
        
        df = df.copy()
        
        try:
            # Calculate rolling highs and lows
            df['rolling_high'] = df['high'].rolling(window=self.lookback_period).max()
            df['rolling_low'] = df['low'].rolling(window=self.lookback_period).min()
            
            # Calculate percentage distance from highs/lows
            df['pct_from_high'] = (df['close'] - df['rolling_high']) / df['rolling_high'] * 100
            df['pct_from_low'] = (df['close'] - df['rolling_low']) / df['rolling_low'] * 100
            
            # Volume average
            df['volume_avg'] = df['volume'].rolling(window=self.lookback_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_avg']
            
            # Price momentum
            df['price_momentum'] = df['close'].pct_change(5) * 100  # 5-period momentum
            
            # Volatility measure
            df['volatility'] = df['close'].pct_change().rolling(window=10).std() * 100
            
        except Exception as e:
            logger.error(f"Error adding swing strategy indicators: {e}")
        
        return df
    
    def calculate_signal(self, df: pd.DataFrame, pair: str) -> TradingSignal:
        """Calculate trading signal for swing strategy"""
        
        # Validate data
        if not self.validate_data(df) or len(df) < self.lookback_period:
            return self._create_hold_signal(df, pair, "Insufficient data")
        
        # Get latest data
        latest = df.iloc[-1]
        timestamp = latest['timestamp'] if 'timestamp' in df.columns else datetime.now()
        current_price = latest['close']
        
        # Required indicators
        required_indicators = ['pct_from_high', 'pct_from_low', 'rsi', 'volume_ratio']
        if not all(indicator in df.columns for indicator in required_indicators):
            return self._create_hold_signal(df, pair, "Missing required indicators")
        
        # Get indicator values
        pct_from_high = latest.get('pct_from_high', 0)
        pct_from_low = latest.get('pct_from_low', 0)
        rsi = latest.get('rsi', 50)
        volume_ratio = latest.get('volume_ratio', 1.0)
        price_momentum = latest.get('price_momentum', 0)
        volatility = latest.get('volatility', 0)
        
        # Calculate confidence and signal
        signal_type, confidence, reason = self._evaluate_swing_conditions(
            pct_from_high, pct_from_low, rsi, volume_ratio, price_momentum, volatility
        )
        
        # Prepare indicator values for signal
        indicators = {
            'price': current_price,
            'pct_from_high': pct_from_high,
            'pct_from_low': pct_from_low,
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            'price_momentum': price_momentum,
            'volatility': volatility,
            'rolling_high': latest.get('rolling_high', current_price),
            'rolling_low': latest.get('rolling_low', current_price)
        }
        
        return TradingSignal(
            timestamp=timestamp,
            pair=pair,
            signal=signal_type,
            price=current_price,
            confidence=confidence,
            strategy_name=self.name,
            reason=reason,
            indicators=indicators
        )
    
    def _evaluate_swing_conditions(self, pct_from_high: float, pct_from_low: float, 
                                 rsi: float, volume_ratio: float, price_momentum: float,
                                 volatility: float) -> tuple:
        """Evaluate swing trading conditions"""
        
        swing_pct = self.swing_threshold * 100  # Convert to percentage
        
        # BUY CONDITIONS
        # Price dropped significantly from recent high
        dropped_from_high = pct_from_high <= -swing_pct
        oversold_rsi = rsi < 35  # More lenient for daily data
        high_volume = volume_ratio >= self.volume_threshold
        negative_momentum = price_momentum < -1.5  # Less strict momentum for daily
        
        # SELL CONDITIONS  
        # Price rose significantly from recent low
        rose_from_low = pct_from_low >= swing_pct
        overbought_rsi = rsi > 65  # More lenient for daily data
        positive_momentum = price_momentum > 1.5  # Less strict momentum for daily
        
        # STRONG BUY CONDITIONS
        if (dropped_from_high and oversold_rsi and high_volume and 
            pct_from_high <= -(swing_pct * 1.5)):  # 7.5% drop or more
            
            confidence = min(0.9, 0.6 + 
                           abs(pct_from_high) / 100 +  # Bigger drop = higher confidence
                           (max(0, self.volume_threshold - volume_ratio) * 0.1) +
                           (max(0, 35 - rsi) / 100))  # More oversold = higher confidence
            
            reason = f"Strong swing buy: {pct_from_high:.1f}% from high, RSI={rsi:.1f}, Vol={volume_ratio:.1f}x"
            return SignalType.STRONG_BUY, confidence, reason
        
        # REGULAR BUY CONDITIONS
        elif dropped_from_high and (oversold_rsi or high_volume):
            confidence = min(0.8, 0.4 + 
                           abs(pct_from_high) / 200 +
                           (volume_ratio - 1.0) * 0.1 +
                           (max(0, 40 - rsi) / 100))
            
            reason = f"Swing buy: {pct_from_high:.1f}% from high, RSI={rsi:.1f}, Vol={volume_ratio:.1f}x"
            return SignalType.BUY, confidence, reason
        
        # STRONG SELL CONDITIONS
        elif (rose_from_low and overbought_rsi and 
              pct_from_low >= swing_pct * 1.5):  # 7.5% rise or more
            
            confidence = min(0.9, 0.6 + 
                           pct_from_low / 100 +
                           (max(0, rsi - 65) / 100))
            
            reason = f"Strong swing sell: {pct_from_low:.1f}% from low, RSI={rsi:.1f}"
            return SignalType.STRONG_SELL, confidence, reason
        
        # REGULAR SELL CONDITIONS
        elif rose_from_low and (overbought_rsi or positive_momentum):
            confidence = min(0.8, 0.4 + 
                           pct_from_low / 200 +
                           (max(0, rsi - 60) / 100))
            
            reason = f"Swing sell: {pct_from_low:.1f}% from low, RSI={rsi:.1f}"
            return SignalType.SELL, confidence, reason
        
        # HOLD CONDITIONS
        else:
            # Determine hold reason
            if abs(pct_from_high) < swing_pct and abs(pct_from_low) < swing_pct:
                reason = f"No significant swing: {pct_from_high:.1f}% from high, {pct_from_low:.1f}% from low"
            elif volume_ratio < self.volume_threshold:
                reason = f"Low volume: {volume_ratio:.1f}x average"
            elif 35 <= rsi <= 65:
                reason = f"Neutral RSI: {rsi:.1f}"
            else:
                reason = f"Mixed signals: RSI={rsi:.1f}, Vol={volume_ratio:.1f}x"
            
            confidence = 0.5
            return SignalType.HOLD, confidence, reason
    
    def _create_hold_signal(self, df: pd.DataFrame, pair: str, reason: str) -> TradingSignal:
        """Create a HOLD signal with error reason"""
        current_price = df.iloc[-1]['close'] if not df.empty else 0.0
        timestamp = df.iloc[-1]['timestamp'] if not df.empty and 'timestamp' in df.columns else datetime.now()
        
        return TradingSignal(
            timestamp=timestamp,
            pair=pair,
            signal=SignalType.HOLD,
            price=current_price,
            confidence=0.5,
            strategy_name=self.name,
            reason=reason,
            indicators={'price': current_price}
        )
    
    def get_stop_loss_level(self, entry_price: float, signal_type: SignalType) -> float:
        """Calculate stop loss level for a position"""
        stop_loss_pct = 0.05  # 5% stop loss
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return entry_price * (1 - stop_loss_pct)
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            return entry_price * (1 + stop_loss_pct)
        else:
            return entry_price
    
    def get_take_profit_level(self, entry_price: float, signal_type: SignalType, confidence: float) -> float:
        """Calculate take profit level for a position"""
        # Higher confidence = higher profit target
        base_profit = 0.05  # 5% base profit target
        confidence_bonus = confidence * 0.05  # Up to 5% bonus for high confidence
        profit_pct = base_profit + confidence_bonus
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return entry_price * (1 + profit_pct)
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            return entry_price * (1 - profit_pct)
        else:
            return entry_price
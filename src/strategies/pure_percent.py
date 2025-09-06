import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any

from .base import Strategy, TradingSignal, SignalType

logger = logging.getLogger(__name__)

class Pure5PercentStrategy(Strategy):
    """
    Pure 5% Up/Down Strategy - Mirrors Coinbase Daily Alerts
    
    This strategy exactly replicates the simple approach you described:
    - BUY when price drops 5% or more from recent high (like Coinbase alerts)
    - SELL when price rises 5% or more from entry price  
    - No complex technical indicators - pure price action
    - Simple daily alert-based approach
    
    The strategy can be customized with different percentage thresholds.
    """
    
    def __init__(self, drop_threshold: float = 0.05, rise_threshold: float = 0.05, 
                 lookback_days: int = 7):
        super().__init__(f"Pure {drop_threshold*100:.0f}% Strategy")
        self.drop_threshold = drop_threshold  # 5% = 0.05
        self.rise_threshold = rise_threshold  # 5% = 0.05  
        self.lookback_days = lookback_days    # Days to look back for highs/lows
    
    def add_required_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add minimal indicators required by this strategy"""
        if df.empty or len(df) < self.lookback_days:
            return df
        
        df = df.copy()
        
        try:
            # Calculate rolling highs and lows over lookback period
            df['rolling_high'] = df['high'].rolling(window=self.lookback_days, min_periods=1).max()
            df['rolling_low'] = df['low'].rolling(window=self.lookback_days, min_periods=1).min()
            
            # Calculate percentage drop from recent high
            df['pct_drop_from_high'] = (df['rolling_high'] - df['close']) / df['rolling_high'] * 100
            
            # Calculate percentage rise from recent low
            df['pct_rise_from_low'] = (df['close'] - df['rolling_low']) / df['rolling_low'] * 100
            
            # Track if we hit the drop threshold (like Coinbase alert trigger)
            df['hit_drop_threshold'] = df['pct_drop_from_high'] >= (self.drop_threshold * 100)
            
            # Track if we hit the rise threshold (sell signal)
            df['hit_rise_threshold'] = df['pct_rise_from_low'] >= (self.rise_threshold * 100)
            
            # Simple price momentum for confidence
            df['price_change_1d'] = df['close'].pct_change() * 100
            df['price_change_3d'] = df['close'].pct_change(3) * 100
            
        except Exception as e:
            logger.error(f"Error adding Pure5Percent strategy indicators: {e}")
        
        return df
    
    def calculate_signal(self, df: pd.DataFrame, pair: str) -> TradingSignal:
        """Calculate pure percentage-based trading signal"""
        
        # Validate data
        if not self.validate_data(df) or len(df) < self.lookback_days:
            return self._create_hold_signal(df, pair, "Insufficient data")
        
        # Get latest data
        latest = df.iloc[-1]
        timestamp = latest['timestamp'] if 'timestamp' in df.columns else datetime.now()
        current_price = latest['close']
        
        # Required indicators
        required_indicators = ['pct_drop_from_high', 'pct_rise_from_low', 'rolling_high', 'rolling_low']
        if not all(indicator in df.columns for indicator in required_indicators):
            return self._create_hold_signal(df, pair, "Missing required indicators")
        
        # Get indicator values
        pct_drop_from_high = latest.get('pct_drop_from_high', 0)
        pct_rise_from_low = latest.get('pct_rise_from_low', 0)
        rolling_high = latest.get('rolling_high', current_price)
        rolling_low = latest.get('rolling_low', current_price)
        price_change_1d = latest.get('price_change_1d', 0)
        price_change_3d = latest.get('price_change_3d', 0)
        
        # Calculate signal and confidence
        signal_type, confidence, reason = self._evaluate_pure_percent_conditions(
            pct_drop_from_high, pct_rise_from_low, rolling_high, rolling_low,
            current_price, price_change_1d, price_change_3d
        )
        
        # Prepare indicator values for signal
        indicators = {
            'price': current_price,
            'pct_drop_from_high': pct_drop_from_high,
            'pct_rise_from_low': pct_rise_from_low,
            'rolling_high': rolling_high,
            'rolling_low': rolling_low,
            'drop_threshold': self.drop_threshold * 100,
            'rise_threshold': self.rise_threshold * 100,
            'price_change_1d': price_change_1d,
            'price_change_3d': price_change_3d
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
    
    def _evaluate_pure_percent_conditions(self, pct_drop_from_high: float, pct_rise_from_low: float,
                                        rolling_high: float, rolling_low: float, current_price: float,
                                        price_change_1d: float, price_change_3d: float) -> tuple:
        """Evaluate pure percentage-based conditions (mimics Coinbase alerts)"""
        
        drop_threshold_pct = self.drop_threshold * 100
        rise_threshold_pct = self.rise_threshold * 100
        
        # BUY CONDITIONS - Coinbase-style "5% down" alert
        
        # STRONG BUY: Dropped significantly more than threshold
        if pct_drop_from_high >= drop_threshold_pct * 1.5:  # 7.5% for 5% threshold
            confidence = min(0.95, 0.7 + (pct_drop_from_high - drop_threshold_pct) / 100)
            reason = f"Strong buy alert: {pct_drop_from_high:.1f}% drop from ${rolling_high:.2f} high (target: {drop_threshold_pct:.0f}%+)"
            return SignalType.STRONG_BUY, confidence, reason
        
        # REGULAR BUY: Hit the drop threshold (exactly like Coinbase alert)
        elif pct_drop_from_high >= drop_threshold_pct:
            # Higher confidence for bigger drops
            confidence = min(0.85, 0.65 + (pct_drop_from_high - drop_threshold_pct) / 200)
            reason = f"Buy alert: {pct_drop_from_high:.1f}% drop from ${rolling_high:.2f} high (target: {drop_threshold_pct:.0f}%)"
            return SignalType.BUY, confidence, reason
        
        # SELL CONDITIONS - Take profit when up 5%
        
        # STRONG SELL: Rose significantly more than threshold  
        elif pct_rise_from_low >= rise_threshold_pct * 1.5:  # 7.5% for 5% threshold
            confidence = min(0.95, 0.7 + (pct_rise_from_low - rise_threshold_pct) / 100)
            reason = f"Strong sell alert: {pct_rise_from_low:.1f}% rise from ${rolling_low:.2f} low (target: {rise_threshold_pct:.0f}%+)"
            return SignalType.STRONG_SELL, confidence, reason
        
        # REGULAR SELL: Hit the rise threshold (take profit)
        elif pct_rise_from_low >= rise_threshold_pct:
            confidence = min(0.85, 0.65 + (pct_rise_from_low - rise_threshold_pct) / 200)
            reason = f"Sell alert: {pct_rise_from_low:.1f}% rise from ${rolling_low:.2f} low (target: {rise_threshold_pct:.0f}%)"
            return SignalType.SELL, confidence, reason
        
        # HOLD CONDITIONS
        else:
            # Provide informative hold reasons
            if pct_drop_from_high < drop_threshold_pct * 0.5:  # Less than half threshold
                reason = f"No significant drop: only {pct_drop_from_high:.1f}% from high (need {drop_threshold_pct:.0f}%)"
            elif pct_rise_from_low < rise_threshold_pct * 0.5:  # Less than half rise threshold
                reason = f"No significant rise: only {pct_rise_from_low:.1f}% from low (target {rise_threshold_pct:.0f}%)"
            else:
                reason = f"Between thresholds: {pct_drop_from_high:.1f}% from high, {pct_rise_from_low:.1f}% from low"
            
            # Slight bias based on recent price action
            if price_change_3d < -2:  # Dropping trend
                confidence = 0.55  # Slightly bullish bias
            elif price_change_3d > 2:  # Rising trend  
                confidence = 0.45  # Slightly bearish bias
            else:
                confidence = 0.5  # Neutral
            
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
        """Calculate stop loss level (slightly wider for pure strategy)"""
        # Slightly wider stops since this strategy doesn't use technical confirmation
        stop_loss_pct = 0.06  # 6% stop loss
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return entry_price * (1 - stop_loss_pct)
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            return entry_price * (1 + stop_loss_pct)
        else:
            return entry_price
    
    def get_take_profit_level(self, entry_price: float, signal_type: SignalType, confidence: float) -> float:
        """Calculate take profit level (matches the rise threshold)"""
        # Take profit should match our rise threshold (e.g., 5%)
        base_profit = self.rise_threshold  # Use the same percentage as our threshold
        confidence_bonus = confidence * 0.02  # Small bonus for high confidence
        profit_pct = base_profit + confidence_bonus
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return entry_price * (1 + profit_pct)
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            return entry_price * (1 - profit_pct)
        else:
            return entry_price


class DynamicPercentStrategy(Strategy):
    """
    Dynamic Percentage Strategy - Adjustable thresholds
    
    This strategy allows you to easily test different percentage thresholds
    and can adapt based on market volatility.
    """
    
    def __init__(self, drop_threshold: float = 0.03, rise_threshold: float = 0.03, 
                 lookback_days: int = 5, adapt_to_volatility: bool = False):
        super().__init__(f"Dynamic {drop_threshold*100:.0f}% Strategy")
        self.drop_threshold = drop_threshold
        self.rise_threshold = rise_threshold
        self.lookback_days = lookback_days
        self.adapt_to_volatility = adapt_to_volatility
    
    def add_required_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators including volatility adaptation"""
        if df.empty or len(df) < self.lookback_days:
            return df
        
        df = df.copy()
        
        try:
            # Basic percentage calculations
            df['rolling_high'] = df['high'].rolling(window=self.lookback_days, min_periods=1).max()
            df['rolling_low'] = df['low'].rolling(window=self.lookback_days, min_periods=1).min()
            
            # Volatility adaptation
            if self.adapt_to_volatility:
                # Calculate recent volatility (20-day standard deviation of daily returns)
                df['daily_returns'] = df['close'].pct_change()
                df['volatility'] = df['daily_returns'].rolling(window=20, min_periods=5).std()
                df['avg_volatility'] = df['volatility'].rolling(window=50, min_periods=10).mean()
                
                # Adjust thresholds based on volatility
                # If volatility is high, use larger thresholds; if low, use smaller
                df['volatility_multiplier'] = (df['volatility'] / df['avg_volatility']).fillna(1.0)
                df['adaptive_drop_threshold'] = self.drop_threshold * df['volatility_multiplier']
                df['adaptive_rise_threshold'] = self.rise_threshold * df['volatility_multiplier']
            else:
                df['adaptive_drop_threshold'] = self.drop_threshold
                df['adaptive_rise_threshold'] = self.rise_threshold
            
            # Calculate percentage drops and rises using adaptive thresholds
            df['pct_drop_from_high'] = (df['rolling_high'] - df['close']) / df['rolling_high'] * 100
            df['pct_rise_from_low'] = (df['close'] - df['rolling_low']) / df['rolling_low'] * 100
            
            # Dynamic threshold checks
            df['hit_adaptive_drop'] = df['pct_drop_from_high'] >= (df['adaptive_drop_threshold'] * 100)
            df['hit_adaptive_rise'] = df['pct_rise_from_low'] >= (df['adaptive_rise_threshold'] * 100)
            
        except Exception as e:
            logger.error(f"Error adding DynamicPercent strategy indicators: {e}")
        
        return df
    
    def calculate_signal(self, df: pd.DataFrame, pair: str) -> TradingSignal:
        """Calculate dynamic percentage-based signal"""
        
        if not self.validate_data(df) or len(df) < self.lookback_days:
            return self._create_hold_signal(df, pair, "Insufficient data")
        
        latest = df.iloc[-1]
        timestamp = latest['timestamp'] if 'timestamp' in df.columns else datetime.now()
        current_price = latest['close']
        
        # Get values
        pct_drop_from_high = latest.get('pct_drop_from_high', 0)
        pct_rise_from_low = latest.get('pct_rise_from_low', 0)
        adaptive_drop_threshold = latest.get('adaptive_drop_threshold', self.drop_threshold) * 100
        adaptive_rise_threshold = latest.get('adaptive_rise_threshold', self.rise_threshold) * 100
        
        # Calculate signal using adaptive thresholds
        signal_type, confidence, reason = self._evaluate_dynamic_conditions(
            pct_drop_from_high, pct_rise_from_low, 
            adaptive_drop_threshold, adaptive_rise_threshold,
            current_price
        )
        
        indicators = {
            'price': current_price,
            'pct_drop_from_high': pct_drop_from_high,
            'pct_rise_from_low': pct_rise_from_low,
            'adaptive_drop_threshold': adaptive_drop_threshold,
            'adaptive_rise_threshold': adaptive_rise_threshold,
        }
        
        if self.adapt_to_volatility:
            indicators['volatility_multiplier'] = latest.get('volatility_multiplier', 1.0)
        
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
    
    def _evaluate_dynamic_conditions(self, pct_drop_from_high: float, pct_rise_from_low: float,
                                   adaptive_drop_threshold: float, adaptive_rise_threshold: float,
                                   current_price: float) -> tuple:
        """Evaluate dynamic percentage conditions"""
        
        # BUY CONDITIONS
        if pct_drop_from_high >= adaptive_drop_threshold * 1.4:  # Strong buy
            confidence = min(0.9, 0.7 + (pct_drop_from_high - adaptive_drop_threshold) / 100)
            reason = f"Strong buy: {pct_drop_from_high:.1f}% drop (adaptive threshold: {adaptive_drop_threshold:.1f}%)"
            return SignalType.STRONG_BUY, confidence, reason
        
        elif pct_drop_from_high >= adaptive_drop_threshold:  # Regular buy
            confidence = min(0.8, 0.6 + (pct_drop_from_high - adaptive_drop_threshold) / 200)
            reason = f"Buy: {pct_drop_from_high:.1f}% drop (adaptive threshold: {adaptive_drop_threshold:.1f}%)"
            return SignalType.BUY, confidence, reason
        
        # SELL CONDITIONS  
        elif pct_rise_from_low >= adaptive_rise_threshold * 1.4:  # Strong sell
            confidence = min(0.9, 0.7 + (pct_rise_from_low - adaptive_rise_threshold) / 100)
            reason = f"Strong sell: {pct_rise_from_low:.1f}% rise (adaptive threshold: {adaptive_rise_threshold:.1f}%)"
            return SignalType.STRONG_SELL, confidence, reason
        
        elif pct_rise_from_low >= adaptive_rise_threshold:  # Regular sell
            confidence = min(0.8, 0.6 + (pct_rise_from_low - adaptive_rise_threshold) / 200)
            reason = f"Sell: {pct_rise_from_low:.1f}% rise (adaptive threshold: {adaptive_rise_threshold:.1f}%)"
            return SignalType.SELL, confidence, reason
        
        # HOLD
        else:
            reason = f"Hold: {pct_drop_from_high:.1f}% from high, {pct_rise_from_low:.1f}% from low (thresholds: {adaptive_drop_threshold:.1f}%/{adaptive_rise_threshold:.1f}%)"
            return SignalType.HOLD, 0.5, reason
    
    def _create_hold_signal(self, df: pd.DataFrame, pair: str, reason: str) -> TradingSignal:
        """Create a HOLD signal"""
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
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any

from .base import Strategy, TradingSignal, SignalType

logger = logging.getLogger(__name__)

class MACDStrategy(Strategy):
    """
    MACD-based Trading Strategy
    
    Buy signals when:
    - MACD line crosses above signal line (bullish crossover)
    - MACD histogram turns positive
    - MACD line crosses above zero line
    - Bullish divergence between price and MACD
    
    Sell signals when:
    - MACD line crosses below signal line (bearish crossover)  
    - MACD histogram turns negative
    - MACD line crosses below zero line
    - Bearish divergence between price and MACD
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACD Strategy")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.divergence_lookback = 15  # Periods to look back for divergence
    
    def add_required_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators required by this strategy"""
        if df.empty or len(df) < max(self.slow_period, self.signal_period) + 10:
            return df
        
        df = df.copy()
        
        try:
            # MACD crossover signals
            if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
                # Crossover detection
                df['macd_above_signal'] = df['macd'] > df['macd_signal']
                df['macd_crossover'] = df['macd_above_signal'].astype(int).diff()  # 1=bullish, -1=bearish
                
                # Zero line crossover
                df['macd_above_zero'] = df['macd'] > 0
                df['macd_zero_crossover'] = df['macd_above_zero'].astype(int).diff()
                
                # Histogram momentum
                df['histogram_momentum'] = df['macd_histogram'].diff()
                df['histogram_positive'] = df['macd_histogram'] > 0
                df['histogram_increasing'] = df['histogram_momentum'] > 0
                
                # MACD momentum (rate of change)
                df['macd_momentum'] = df['macd'].diff()
                df['macd_roc'] = df['macd'].pct_change(periods=5) * 100
                
                # Signal line momentum
                df['signal_momentum'] = df['macd_signal'].diff()
                
                # MACD divergence detection
                df['macd_divergence'] = self._detect_macd_divergence(df)
                
                # MACD strength (distance from signal line)
                df['macd_strength'] = abs(df['macd'] - df['macd_signal'])
                
        except Exception as e:
            logger.error(f"Error adding MACD strategy indicators: {e}")
        
        return df
    
    def _detect_macd_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Detect MACD divergence patterns"""
        divergence = pd.Series(0, index=df.index)  # 0=no divergence, 1=bullish, -1=bearish
        
        if len(df) < self.divergence_lookback * 2:
            return divergence
        
        try:
            for i in range(self.divergence_lookback, len(df)):
                # Get recent price and MACD data
                recent_prices = df['close'].iloc[i-self.divergence_lookback:i+1]
                recent_macd = df['macd'].iloc[i-self.divergence_lookback:i+1]
                
                # Find local extremes
                price_min_idx = recent_prices.idxmin()
                price_max_idx = recent_prices.idxmax()
                macd_min_idx = recent_macd.idxmin()
                macd_max_idx = recent_macd.idxmax()
                
                current_idx = df.index[i]
                
                # Bullish divergence: price makes lower low, MACD makes higher low
                if (price_min_idx == current_idx and macd_min_idx != current_idx and
                    recent_macd.iloc[-1] > recent_macd.min()):
                    divergence.iloc[i] = 1
                
                # Bearish divergence: price makes higher high, MACD makes lower high
                elif (price_max_idx == current_idx and macd_max_idx != current_idx and
                      recent_macd.iloc[-1] < recent_macd.max()):
                    divergence.iloc[i] = -1
        
        except Exception as e:
            logger.debug(f"Error detecting MACD divergence: {e}")
        
        return divergence
    
    def calculate_signal(self, df: pd.DataFrame, pair: str) -> TradingSignal:
        """Calculate trading signal for MACD strategy"""
        
        # Validate data
        if not self.validate_data(df) or len(df) < max(self.slow_period, self.signal_period) + 10:
            return self._create_hold_signal(df, pair, "Insufficient data")
        
        # Check for required MACD indicators
        required_indicators = ['macd', 'macd_signal', 'macd_histogram']
        if not all(indicator in df.columns for indicator in required_indicators):
            return self._create_hold_signal(df, pair, "Missing MACD indicators")
        
        # Get latest data
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        timestamp = latest['timestamp'] if 'timestamp' in df.columns else datetime.now()
        current_price = latest['close']
        
        # Get indicator values
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        macd_histogram = latest.get('macd_histogram', 0)
        macd_crossover = latest.get('macd_crossover', 0)
        macd_zero_crossover = latest.get('macd_zero_crossover', 0)
        histogram_momentum = latest.get('histogram_momentum', 0)
        macd_momentum = latest.get('macd_momentum', 0)
        macd_divergence = latest.get('macd_divergence', 0)
        macd_strength = latest.get('macd_strength', 0)
        
        # Get previous values for trend confirmation
        prev_macd = previous.get('macd', 0)
        prev_macd_signal = previous.get('macd_signal', 0)
        prev_histogram = previous.get('macd_histogram', 0)
        
        # Calculate signal and confidence
        signal_type, confidence, reason = self._evaluate_macd_conditions(
            macd, macd_signal, macd_histogram, macd_crossover, macd_zero_crossover,
            histogram_momentum, macd_momentum, macd_divergence, macd_strength,
            prev_macd, prev_macd_signal, prev_histogram
        )
        
        # Prepare indicator values for signal
        indicators = {
            'price': current_price,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram,
            'macd_crossover': macd_crossover,
            'macd_zero_crossover': macd_zero_crossover,
            'histogram_momentum': histogram_momentum,
            'macd_momentum': macd_momentum,
            'macd_divergence': macd_divergence,
            'macd_strength': macd_strength
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
    
    def _evaluate_macd_conditions(self, macd: float, macd_signal: float, macd_histogram: float,
                                macd_crossover: float, macd_zero_crossover: float,
                                histogram_momentum: float, macd_momentum: float,
                                macd_divergence: float, macd_strength: float,
                                prev_macd: float, prev_macd_signal: float, 
                                prev_histogram: float) -> tuple:
        """Evaluate MACD trading conditions"""
        
        # STRONG BUY CONDITIONS
        # Multiple bullish signals align
        if (macd_crossover == 1 and  # Fresh bullish crossover
            macd_zero_crossover == 1 and  # MACD crosses above zero
            macd_divergence == 1):  # Bullish divergence
            
            confidence = min(0.95, 0.8 + 
                           (macd_strength / 100) +  # Stronger crossover
                           max(0, macd_momentum) / 50)
            
            reason = "Strong MACD buy: bullish crossover + zero cross + divergence"
            return SignalType.STRONG_BUY, confidence, reason
        
        # Strong bullish crossover with momentum
        elif (macd_crossover == 1 and histogram_momentum > 0 and
              macd > 0 and macd_momentum > 0):
            
            confidence = min(0.9, 0.7 + 
                           (macd_strength / 150) +
                           max(0, histogram_momentum) / 100)
            
            reason = f"Strong MACD buy: fresh crossover, MACD={macd:.4f} > 0, rising momentum"
            return SignalType.STRONG_BUY, confidence, reason
        
        # REGULAR BUY CONDITIONS
        elif macd_crossover == 1:  # Fresh bullish crossover
            base_confidence = 0.6
            
            # Add confidence based on additional conditions
            if macd > 0:
                base_confidence += 0.1  # Above zero line
            if histogram_momentum > 0:
                base_confidence += 0.08  # Histogram improving
            if macd_momentum > 0:
                base_confidence += 0.05  # MACD momentum positive
            if macd_divergence == 1:
                base_confidence += 0.12  # Bullish divergence
            
            confidence = min(0.85, base_confidence)
            reason = f"MACD buy: bullish crossover, MACD={macd:.4f}, signal={macd_signal:.4f}"
            return SignalType.BUY, confidence, reason
        
        # Zero line crossover (delayed but confirmed signal)
        elif (macd_zero_crossover == 1 and macd > macd_signal and 
              macd_histogram > 0):
            
            confidence = min(0.8, 0.65 + (macd_strength / 200))
            reason = f"MACD buy: zero crossover, MACD={macd:.4f} > signal"
            return SignalType.BUY, confidence, reason
        
        # Histogram turning positive (early signal)
        elif (prev_histogram <= 0 and macd_histogram > 0 and
              histogram_momentum > 0 and macd > macd_signal):
            
            confidence = min(0.75, 0.55 + max(0, histogram_momentum) / 100)
            reason = f"MACD buy: histogram turned positive, momentum={histogram_momentum:.4f}"
            return SignalType.BUY, confidence, reason
        
        # STRONG SELL CONDITIONS
        # Multiple bearish signals align
        elif (macd_crossover == -1 and  # Fresh bearish crossover
              macd_zero_crossover == -1 and  # MACD crosses below zero
              macd_divergence == -1):  # Bearish divergence
            
            confidence = min(0.95, 0.8 + 
                           (macd_strength / 100) +
                           abs(min(0, macd_momentum)) / 50)
            
            reason = "Strong MACD sell: bearish crossover + zero cross + divergence"
            return SignalType.STRONG_SELL, confidence, reason
        
        # Strong bearish crossover with momentum
        elif (macd_crossover == -1 and histogram_momentum < 0 and
              macd < 0 and macd_momentum < 0):
            
            confidence = min(0.9, 0.7 + 
                           (macd_strength / 150) +
                           abs(min(0, histogram_momentum)) / 100)
            
            reason = f"Strong MACD sell: fresh crossover, MACD={macd:.4f} < 0, falling momentum"
            return SignalType.STRONG_SELL, confidence, reason
        
        # REGULAR SELL CONDITIONS
        elif macd_crossover == -1:  # Fresh bearish crossover
            base_confidence = 0.6
            
            # Add confidence based on additional conditions
            if macd < 0:
                base_confidence += 0.1  # Below zero line
            if histogram_momentum < 0:
                base_confidence += 0.08  # Histogram deteriorating
            if macd_momentum < 0:
                base_confidence += 0.05  # MACD momentum negative
            if macd_divergence == -1:
                base_confidence += 0.12  # Bearish divergence
            
            confidence = min(0.85, base_confidence)
            reason = f"MACD sell: bearish crossover, MACD={macd:.4f}, signal={macd_signal:.4f}"
            return SignalType.SELL, confidence, reason
        
        # Zero line crossover down
        elif (macd_zero_crossover == -1 and macd < macd_signal and 
              macd_histogram < 0):
            
            confidence = min(0.8, 0.65 + (macd_strength / 200))
            reason = f"MACD sell: zero crossover down, MACD={macd:.4f} < signal"
            return SignalType.SELL, confidence, reason
        
        # Histogram turning negative
        elif (prev_histogram >= 0 and macd_histogram < 0 and
              histogram_momentum < 0 and macd < macd_signal):
            
            confidence = min(0.75, 0.55 + abs(min(0, histogram_momentum)) / 100)
            reason = f"MACD sell: histogram turned negative, momentum={histogram_momentum:.4f}"
            return SignalType.SELL, confidence, reason
        
        # HOLD CONDITIONS
        else:
            # Determine hold reason and bias
            if abs(macd - macd_signal) < 0.001:  # Lines too close
                reason = f"MACD neutral: lines converging, MACD={macd:.4f}≈signal={macd_signal:.4f}"
                confidence = 0.5
            elif macd > macd_signal and macd > 0:
                reason = f"MACD bullish hold: above signal and zero, MACD={macd:.4f}"
                confidence = 0.55  # Slightly bullish bias
            elif macd < macd_signal and macd < 0:
                reason = f"MACD bearish hold: below signal and zero, MACD={macd:.4f}"
                confidence = 0.45  # Slightly bearish bias
            else:
                reason = f"MACD mixed: MACD={macd:.4f}, signal={macd_signal:.4f}, hist={macd_histogram:.4f}"
                confidence = 0.5
            
            return SignalType.HOLD, confidence, reason
    
    def _create_hold_signal(self, df: pd.DataFrame, pair: str, reason: str) -> TradingSignal:
        """Create a HOLD signal with error reason"""
        current_price = df.iloc[-1]['close'] if not df.empty else 0.0
        timestamp = df.iloc[-1]['timestamp'] if not df.empty and 'timestamp' in df.columns else datetime.now()
        
        macd_values = {
            'macd': df.iloc[-1].get('macd', 0) if not df.empty else 0,
            'macd_signal': df.iloc[-1].get('macd_signal', 0) if not df.empty else 0,
            'macd_histogram': df.iloc[-1].get('macd_histogram', 0) if not df.empty else 0
        }
        
        return TradingSignal(
            timestamp=timestamp,
            pair=pair,
            signal=SignalType.HOLD,
            price=current_price,
            confidence=0.5,
            strategy_name=self.name,
            reason=reason,
            indicators={'price': current_price, **macd_values}
        )
    
    def get_stop_loss_level(self, entry_price: float, signal_type: SignalType) -> float:
        """Calculate stop loss level for MACD-based position"""
        # Moderate stops for MACD strategy
        stop_loss_pct = 0.045  # 4.5% stop loss
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return entry_price * (1 - stop_loss_pct)
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            return entry_price * (1 + stop_loss_pct)
        else:
            return entry_price
    
    def get_take_profit_level(self, entry_price: float, signal_type: SignalType, confidence: float) -> float:
        """Calculate take profit level for MACD-based position"""
        # MACD can provide good trend following, so allow for bigger moves
        base_profit = 0.08  # 8% base profit target
        confidence_bonus = confidence * 0.06  # Up to 6% bonus
        profit_pct = base_profit + confidence_bonus
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return entry_price * (1 + profit_pct)
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            return entry_price * (1 - profit_pct)
        else:
            return entry_price
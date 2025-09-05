import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any

from .base import Strategy, TradingSignal, SignalType

logger = logging.getLogger(__name__)

class RSIStrategy(Strategy):
    """
    RSI-based Trading Strategy (Optimized for Daily Data)
    
    Buy signals when:
    - RSI drops below oversold threshold (35) - adjusted for daily timeframe
    - RSI shows bullish divergence
    - Price confirms with upward momentum
    
    Sell signals when:
    - RSI rises above overbought threshold (65) - adjusted for daily timeframe
    - RSI shows bearish divergence
    - Price confirms with downward momentum
    """
    
    def __init__(self, oversold_threshold: float = 35, overbought_threshold: float = 65, 
                 rsi_period: int = 14):
        super().__init__("RSI Strategy")
        self.oversold_threshold = oversold_threshold  # More lenient for daily data
        self.overbought_threshold = overbought_threshold  # More lenient for daily data
        self.rsi_period = rsi_period
        self.divergence_lookback = 7  # Reduced for daily timeframe
    
    def add_required_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators required by this strategy"""
        if df.empty or len(df) < self.rsi_period:
            return df
        
        df = df.copy()
        
        try:
            # RSI momentum and direction
            if 'rsi' in df.columns:
                df['rsi_momentum'] = df['rsi'].diff()
                df['rsi_direction'] = np.where(df['rsi_momentum'] > 0, 1, -1)
                
                # RSI moving average for smoother signals
                df['rsi_sma'] = df['rsi'].rolling(window=5).mean()
                
                # RSI extremes
                df['rsi_oversold'] = df['rsi'] < self.oversold_threshold
                df['rsi_overbought'] = df['rsi'] > self.overbought_threshold
                
                # RSI divergence detection
                df['rsi_divergence'] = self._detect_divergence(df)
            
            # Price momentum for confirmation (adjusted for daily)
            df['price_momentum_3'] = df['close'].pct_change(2) * 100  # Shorter for daily
            df['price_momentum_5'] = df['close'].pct_change(3) * 100  # Shorter for daily
            
            # Stochastic for additional confirmation
            if 'stoch_k' in df.columns:
                df['stoch_oversold'] = df['stoch_k'] < 20
                df['stoch_overbought'] = df['stoch_k'] > 80
            
        except Exception as e:
            logger.error(f"Error adding RSI strategy indicators: {e}")
        
        return df
    
    def _detect_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Detect RSI divergence patterns"""
        divergence = pd.Series(0, index=df.index)  # 0=no divergence, 1=bullish, -1=bearish
        
        if len(df) < self.divergence_lookback * 2:
            return divergence
        
        try:
            for i in range(self.divergence_lookback, len(df)):
                # Get recent price and RSI data
                recent_prices = df['close'].iloc[i-self.divergence_lookback:i+1]
                recent_rsi = df['rsi'].iloc[i-self.divergence_lookback:i+1]
                
                # Find local extremes
                price_min_idx = recent_prices.idxmin()
                price_max_idx = recent_prices.idxmax()
                rsi_min_idx = recent_rsi.idxmin()
                rsi_max_idx = recent_rsi.idxmax()
                
                current_idx = df.index[i]
                
                # Bullish divergence: price makes lower low, RSI makes higher low
                if (price_min_idx == current_idx and rsi_min_idx != current_idx and
                    recent_rsi.iloc[-1] > recent_rsi.min()):
                    divergence.iloc[i] = 1
                
                # Bearish divergence: price makes higher high, RSI makes lower high  
                elif (price_max_idx == current_idx and rsi_max_idx != current_idx and
                      recent_rsi.iloc[-1] < recent_rsi.max()):
                    divergence.iloc[i] = -1
        
        except Exception as e:
            logger.debug(f"Error detecting RSI divergence: {e}")
        
        return divergence
    
    def calculate_signal(self, df: pd.DataFrame, pair: str) -> TradingSignal:
        """Calculate trading signal for RSI strategy"""
        
        # Validate data
        if not self.validate_data(df) or len(df) < self.rsi_period:
            return self._create_hold_signal(df, pair, "Insufficient data")
        
        # Check for required RSI indicator
        if 'rsi' not in df.columns:
            return self._create_hold_signal(df, pair, "Missing RSI indicator")
        
        # Get latest data
        latest = df.iloc[-1]
        timestamp = latest['timestamp'] if 'timestamp' in df.columns else datetime.now()
        current_price = latest['close']
        
        # Get indicator values
        rsi = latest.get('rsi', 50)
        rsi_momentum = latest.get('rsi_momentum', 0)
        rsi_divergence = latest.get('rsi_divergence', 0)
        price_momentum_3 = latest.get('price_momentum_3', 0)
        price_momentum_5 = latest.get('price_momentum_5', 0)
        stoch_k = latest.get('stoch_k', 50)
        
        # Calculate signal and confidence
        signal_type, confidence, reason = self._evaluate_rsi_conditions(
            rsi, rsi_momentum, rsi_divergence, price_momentum_3, price_momentum_5, stoch_k
        )
        
        # Prepare indicator values for signal
        indicators = {
            'price': current_price,
            'rsi': rsi,
            'rsi_momentum': rsi_momentum,
            'rsi_divergence': rsi_divergence,
            'price_momentum_3': price_momentum_3,
            'price_momentum_5': price_momentum_5,
            'stoch_k': stoch_k
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
    
    def _evaluate_rsi_conditions(self, rsi: float, rsi_momentum: float, rsi_divergence: float,
                               price_momentum_3: float, price_momentum_5: float, 
                               stoch_k: float) -> tuple:
        """Evaluate RSI trading conditions"""
        
        # STRONG BUY CONDITIONS
        if (rsi <= 30 and rsi_momentum > 0 and  # Very oversold with upward RSI momentum (adjusted)
            (rsi_divergence > 0 or price_momentum_3 > 0)):  # Bullish divergence or price confirmation
            
            confidence = min(0.9, 0.7 + 
                           (35 - rsi) / 100 +  # More oversold = higher confidence (adjusted)
                           max(0, rsi_momentum) / 10 +
                           max(0, price_momentum_3) / 20)
            
            reason = f"Strong RSI buy: RSI={rsi:.1f} (oversold), momentum={rsi_momentum:.2f}"
            if rsi_divergence > 0:
                reason += ", bullish divergence"
            
            return SignalType.STRONG_BUY, confidence, reason
        
        # REGULAR BUY CONDITIONS
        elif (rsi <= self.oversold_threshold and 
              (rsi_momentum > 0 or price_momentum_5 > 0.5 or stoch_k < 25)):  # More lenient
            
            confirmation_score = 0
            if rsi_momentum > 0:
                confirmation_score += 0.15
            if price_momentum_5 > 0.5:  # More lenient
                confirmation_score += 0.1  
            if stoch_k < 25:  # More lenient
                confirmation_score += 0.1
            
            confidence = min(0.8, 0.5 + 
                           (self.oversold_threshold - rsi) / 100 +
                           confirmation_score)
            
            reason = f"RSI buy: RSI={rsi:.1f} (oversold)"
            if rsi_momentum > 0:
                reason += f", rising momentum={rsi_momentum:.2f}"
            if price_momentum_5 > 0.5:
                reason += f", price momentum={price_momentum_5:.1f}%"
            
            return SignalType.BUY, confidence, reason
        
        # STRONG SELL CONDITIONS
        elif (rsi >= 70 and rsi_momentum < 0 and  # Very overbought with downward RSI momentum (adjusted)
              (rsi_divergence < 0 or price_momentum_3 < 0)):  # Bearish divergence or price confirmation
            
            confidence = min(0.9, 0.7 + 
                           (rsi - 65) / 100 +  # More overbought = higher confidence (adjusted)
                           abs(min(0, rsi_momentum)) / 10 +
                           abs(min(0, price_momentum_3)) / 20)
            
            reason = f"Strong RSI sell: RSI={rsi:.1f} (overbought), momentum={rsi_momentum:.2f}"
            if rsi_divergence < 0:
                reason += ", bearish divergence"
            
            return SignalType.STRONG_SELL, confidence, reason
        
        # REGULAR SELL CONDITIONS
        elif (rsi >= self.overbought_threshold and
              (rsi_momentum < 0 or price_momentum_5 < -0.5 or stoch_k > 75)):  # More lenient
            
            confirmation_score = 0
            if rsi_momentum < 0:
                confirmation_score += 0.15
            if price_momentum_5 < -0.5:  # More lenient
                confirmation_score += 0.1
            if stoch_k > 75:  # More lenient
                confirmation_score += 0.1
            
            confidence = min(0.8, 0.5 + 
                           (rsi - self.overbought_threshold) / 100 +
                           confirmation_score)
            
            reason = f"RSI sell: RSI={rsi:.1f} (overbought)"
            if rsi_momentum < 0:
                reason += f", falling momentum={rsi_momentum:.2f}"
            if price_momentum_5 < -0.5:
                reason += f", price momentum={price_momentum_5:.1f}%"
            
            return SignalType.SELL, confidence, reason
        
        # HOLD CONDITIONS
        else:
            # Determine hold reason
            if self.oversold_threshold < rsi < self.overbought_threshold:
                reason = f"RSI neutral: {rsi:.1f} (normal range)"
            elif rsi <= self.oversold_threshold and rsi_momentum <= 0:
                reason = f"RSI oversold but falling: {rsi:.1f}, momentum={rsi_momentum:.2f}"
            elif rsi >= self.overbought_threshold and rsi_momentum >= 0:
                reason = f"RSI overbought but rising: {rsi:.1f}, momentum={rsi_momentum:.2f}"
            else:
                reason = f"Mixed RSI signals: {rsi:.1f}, momentum={rsi_momentum:.2f}"
            
            # Slight preference based on RSI level
            if rsi < 40:
                confidence = 0.55  # Slightly bullish
            elif rsi > 60:
                confidence = 0.45  # Slightly bearish  
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
            indicators={'price': current_price, 'rsi': df.iloc[-1].get('rsi', 50) if not df.empty else 50}
        )
    
    def get_stop_loss_level(self, entry_price: float, signal_type: SignalType) -> float:
        """Calculate stop loss level for RSI-based position"""
        # Tighter stops for RSI strategy since it's more precise
        stop_loss_pct = 0.04  # 4% stop loss
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return entry_price * (1 - stop_loss_pct)
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            return entry_price * (1 + stop_loss_pct)
        else:
            return entry_price
    
    def get_take_profit_level(self, entry_price: float, signal_type: SignalType, confidence: float) -> float:
        """Calculate take profit level for RSI-based position"""
        # More conservative profit targets for RSI
        base_profit = 0.06  # 6% base profit target
        confidence_bonus = confidence * 0.04  # Up to 4% bonus
        profit_pct = base_profit + confidence_bonus
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return entry_price * (1 + profit_pct)
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            return entry_price * (1 - profit_pct)
        else:
            return entry_price
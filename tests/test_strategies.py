"""
Unit tests for trading strategies.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.strategies.base import Strategy, TradingSignal, SignalType
from src.strategies.swing import SwingTradingStrategy
from src.strategies.rsi import RSIStrategy
from src.strategies.macd import MACDStrategy
from src.strategies.pure_percent import Pure5PercentStrategy, DynamicPercentStrategy


class TestBaseStrategy:
    """Test the base Strategy class and TradingSignal"""
    
    def test_trading_signal_creation(self):
        """Test TradingSignal creation and properties"""
        timestamp = datetime.now()
        signal = TradingSignal(
            timestamp=timestamp,
            pair="BTC-USD",
            signal=SignalType.BUY,
            price=50000.0,
            confidence=0.8,
            strategy_name="Test Strategy",
            reason="Test reason",
            indicators={"rsi": 30, "macd": 0.5}
        )
        
        assert signal.timestamp == timestamp
        assert signal.pair == "BTC-USD"
        assert signal.signal == SignalType.BUY
        assert signal.price == 50000.0
        assert signal.confidence == 0.8
        assert signal.strategy_name == "Test Strategy"
        assert signal.reason == "Test reason"
        assert signal.indicators["rsi"] == 30
    
    def test_position_size_recommendation(self):
        """Test position size recommendations based on confidence"""
        signal_high = TradingSignal(
            timestamp=datetime.now(),
            pair="BTC-USD",
            signal=SignalType.BUY,
            price=50000.0,
            confidence=0.9,  # High confidence
            strategy_name="Test",
            reason="Test",
            indicators={}
        )
        
        signal_medium = TradingSignal(
            timestamp=datetime.now(),
            pair="BTC-USD",
            signal=SignalType.BUY,
            price=50000.0,
            confidence=0.7,  # Medium confidence
            strategy_name="Test",
            reason="Test",
            indicators={}
        )
        
        signal_low = TradingSignal(
            timestamp=datetime.now(),
            pair="BTC-USD",
            signal=SignalType.BUY,
            price=50000.0,
            confidence=0.5,  # Low confidence
            strategy_name="Test",
            reason="Test",
            indicators={}
        )
        
        assert signal_high.position_size_recommendation() == 1.0
        assert signal_medium.position_size_recommendation() == 0.66
        assert signal_low.position_size_recommendation() == 0.33
    
    def test_strategy_abstract_methods(self):
        """Test that Strategy is properly abstract"""
        with pytest.raises(TypeError):
            Strategy("Test Strategy")  # Cannot instantiate abstract class
    
    def test_strategy_validate_data(self, sample_ohlcv_data):
        """Test base strategy data validation"""
        # Create a concrete strategy for testing
        class TestStrategy(Strategy):
            def calculate_signal(self, df, pair):
                return TradingSignal(
                    timestamp=datetime.now(),
                    pair=pair,
                    signal=SignalType.HOLD,
                    price=100.0,
                    confidence=0.5,
                    strategy_name=self.name,
                    reason="Test",
                    indicators={}
                )
        
        strategy = TestStrategy("Test Strategy")
        
        # Valid data
        assert strategy.validate_data(sample_ohlcv_data) is True
        
        # Missing required columns
        invalid_df = sample_ohlcv_data.drop('close', axis=1)
        assert strategy.validate_data(invalid_df) is False
        
        # Empty DataFrame
        assert strategy.validate_data(pd.DataFrame()) is False


class TestSwingTradingStrategy:
    """Test SwingTradingStrategy"""
    
    def test_init_default_parameters(self):
        """Test SwingTradingStrategy initialization with defaults"""
        strategy = SwingTradingStrategy()
        
        assert strategy.name == "2.5% Swing Strategy"
        assert strategy.swing_threshold == 0.025
        assert strategy.volume_threshold == 1.1
        assert strategy.lookback_period == 10
    
    def test_init_custom_parameters(self):
        """Test SwingTradingStrategy with custom parameters"""
        strategy = SwingTradingStrategy(swing_threshold=0.03, volume_threshold=1.5)
        
        assert strategy.swing_threshold == 0.03
        assert strategy.volume_threshold == 1.5
    
    def test_add_required_indicators(self, sample_ohlcv_data):
        """Test that required indicators are added"""
        strategy = SwingTradingStrategy()
        
        df_with_indicators = strategy.add_required_indicators(sample_ohlcv_data)
        
        expected_indicators = ['rolling_high', 'rolling_low', 'pct_from_high', 
                             'pct_from_low', 'volume_avg', 'volume_ratio', 
                             'price_momentum', 'volatility']
        
        for indicator in expected_indicators:
            assert indicator in df_with_indicators.columns
    
    def test_add_required_indicators_insufficient_data(self):
        """Test required indicators with insufficient data"""
        strategy = SwingTradingStrategy()
        
        # Create data with fewer rows than lookback period
        small_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='D'),
            'open': [100] * 5,
            'high': [105] * 5,
            'low': [95] * 5,
            'close': [102] * 5,
            'volume': [1000] * 5
        })
        
        result = strategy.add_required_indicators(small_df)
        assert len(result) == 5  # Should return original data
    
    def test_calculate_signal_strong_buy(self, strategy_test_data):
        """Test strong buy signal generation"""
        strategy = SwingTradingStrategy(swing_threshold=0.02)  # 2% threshold
        
        # Create data that should trigger strong buy
        df = strategy_test_data.copy()
        
        # Add required indicators manually for testing
        df['rolling_high'] = df['high'].rolling(window=10).max()
        df['rolling_low'] = df['low'].rolling(window=10).min()
        df['pct_from_high'] = (df['close'] - df['rolling_high']) / df['rolling_high'] * 100
        df['pct_from_low'] = (df['close'] - df['rolling_low']) / df['rolling_low'] * 100
        df['volume_avg'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_avg']
        df['price_momentum'] = df['close'].pct_change(5) * 100
        df['volatility'] = df['close'].pct_change().rolling(window=10).std() * 100
        df['rsi'] = 25  # Oversold
        
        # Set conditions for strong buy (7.5% drop from high, high volume)
        df.iloc[-1, df.columns.get_loc('pct_from_high')] = -4.0  # 4% drop (> 2.5%)
        df.iloc[-1, df.columns.get_loc('volume_ratio')] = 1.5  # High volume
        df.iloc[-1, df.columns.get_loc('rsi')] = 25  # Oversold
        
        signal = strategy.calculate_signal(df, "BTC-USD")
        
        assert signal.signal in [SignalType.BUY, SignalType.STRONG_BUY]
        assert signal.confidence > 0.4
        assert "swing" in signal.reason.lower()
    
    def test_calculate_signal_hold(self, strategy_test_data):
        """Test hold signal generation"""
        strategy = SwingTradingStrategy()
        
        # Create neutral conditions
        df = strategy_test_data.copy()
        df = strategy.add_required_indicators(df)
        
        # Add RSI for complete indicator set
        df['rsi'] = 50  # Neutral
        
        # Set neutral conditions
        if 'pct_from_high' in df.columns and 'pct_from_low' in df.columns:
            df.iloc[-1, df.columns.get_loc('pct_from_high')] = -1.0  # Small drop
            df.iloc[-1, df.columns.get_loc('pct_from_low')] = 1.0   # Small rise
            df.iloc[-1, df.columns.get_loc('volume_ratio')] = 1.0   # Average volume
        
        signal = strategy.calculate_signal(df, "BTC-USD")
        
        assert signal.signal == SignalType.HOLD
        assert 0.4 <= signal.confidence <= 0.6
    
    def test_calculate_signal_insufficient_data(self):
        """Test signal calculation with insufficient data"""
        strategy = SwingTradingStrategy()
        
        # Create minimal data
        minimal_df = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [102],
            'volume': [1000]
        })
        
        signal = strategy.calculate_signal(minimal_df, "BTC-USD")
        
        assert signal.signal == SignalType.HOLD
        assert "Insufficient data" in signal.reason
    
    def test_get_stop_loss_level(self):
        """Test stop loss calculation"""
        strategy = SwingTradingStrategy()
        
        entry_price = 100.0
        
        # Long position
        stop_loss_long = strategy.get_stop_loss_level(entry_price, SignalType.BUY)
        assert stop_loss_long < entry_price
        assert abs(stop_loss_long - entry_price * 0.95) < 0.01
        
        # Short position
        stop_loss_short = strategy.get_stop_loss_level(entry_price, SignalType.SELL)
        assert stop_loss_short > entry_price
        assert abs(stop_loss_short - entry_price * 1.05) < 0.01
    
    def test_get_take_profit_level(self):
        """Test take profit calculation"""
        strategy = SwingTradingStrategy()
        
        entry_price = 100.0
        
        # High confidence trade
        take_profit_high = strategy.get_take_profit_level(entry_price, SignalType.BUY, 0.9)
        assert take_profit_high > entry_price
        
        # Low confidence trade
        take_profit_low = strategy.get_take_profit_level(entry_price, SignalType.BUY, 0.6)
        assert take_profit_low > entry_price
        
        # High confidence should have higher profit target
        assert take_profit_high > take_profit_low


class TestRSIStrategy:
    """Test RSIStrategy"""
    
    def test_init_default_parameters(self):
        """Test RSI strategy initialization"""
        strategy = RSIStrategy()
        
        assert strategy.name == "RSI Strategy"
        assert strategy.oversold_threshold == 35
        assert strategy.overbought_threshold == 65
        assert strategy.rsi_period == 14
    
    def test_init_custom_parameters(self):
        """Test RSI strategy with custom parameters"""
        strategy = RSIStrategy(oversold_threshold=30, overbought_threshold=70, rsi_period=21)
        
        assert strategy.oversold_threshold == 30
        assert strategy.overbought_threshold == 70
        assert strategy.rsi_period == 21
    
    def test_add_required_indicators(self, sample_ohlcv_data):
        """Test RSI required indicators"""
        strategy = RSIStrategy()
        
        # Add basic RSI first
        df = sample_ohlcv_data.copy()
        df['rsi'] = 50 + np.random.normal(0, 15, len(df))  # Mock RSI
        df['rsi'] = df['rsi'].clip(0, 100)
        
        df_with_indicators = strategy.add_required_indicators(df)
        
        expected_indicators = ['rsi_momentum', 'rsi_direction', 'rsi_sma', 
                             'rsi_oversold', 'rsi_overbought', 'rsi_divergence']
        
        for indicator in expected_indicators:
            assert indicator in df_with_indicators.columns
    
    def test_calculate_signal_strong_buy_oversold(self, sample_ohlcv_data):
        """Test strong buy signal when RSI is very oversold"""
        strategy = RSIStrategy()
        
        # Prepare data with RSI indicators
        df = sample_ohlcv_data.copy()
        df['rsi'] = 25  # Very oversold
        df['rsi_momentum'] = 2.0  # Rising
        df['rsi_divergence'] = 1  # Bullish divergence
        df['price_momentum_3'] = 1.0  # Price rising
        df['stoch_k'] = 20  # Oversold
        
        signal = strategy.calculate_signal(df, "BTC-USD")
        
        assert signal.signal in [SignalType.BUY, SignalType.STRONG_BUY]
        assert signal.confidence > 0.6
        assert "rsi" in signal.reason.lower()
    
    def test_calculate_signal_strong_sell_overbought(self, sample_ohlcv_data):
        """Test strong sell signal when RSI is very overbought"""
        strategy = RSIStrategy()
        
        # Prepare data with RSI indicators
        df = sample_ohlcv_data.copy()
        df['rsi'] = 75  # Very overbought
        df['rsi_momentum'] = -2.0  # Falling
        df['rsi_divergence'] = -1  # Bearish divergence
        df['price_momentum_3'] = -1.0  # Price falling
        df['stoch_k'] = 80  # Overbought
        
        signal = strategy.calculate_signal(df, "BTC-USD")
        
        assert signal.signal in [SignalType.SELL, SignalType.STRONG_SELL]
        assert signal.confidence > 0.6
        assert "rsi" in signal.reason.lower()
    
    def test_calculate_signal_hold_neutral(self, sample_ohlcv_data):
        """Test hold signal when RSI is neutral"""
        strategy = RSIStrategy()
        
        # Neutral RSI conditions
        df = sample_ohlcv_data.copy()
        df['rsi'] = 50  # Neutral
        df['rsi_momentum'] = 0.1  # Minimal movement
        df['rsi_divergence'] = 0  # No divergence
        df['price_momentum_3'] = 0.1  # Minimal price movement
        df['stoch_k'] = 50  # Neutral
        
        signal = strategy.calculate_signal(df, "BTC-USD")
        
        assert signal.signal == SignalType.HOLD
        assert "neutral" in signal.reason.lower()
    
    def test_detect_divergence(self, sample_ohlcv_data):
        """Test RSI divergence detection"""
        strategy = RSIStrategy()
        
        # Create data with potential divergence
        df = sample_ohlcv_data.copy()
        
        # Add RSI that could show divergence
        df['rsi'] = np.random.uniform(30, 70, len(df))
        
        divergence = strategy._detect_divergence(df)
        
        assert isinstance(divergence, pd.Series)
        assert len(divergence) == len(df)
        assert divergence.dtype in [int, float]
        assert all(val in [-1, 0, 1] for val in divergence.unique())


class TestMACDStrategy:
    """Test MACDStrategy"""
    
    def test_init_default_parameters(self):
        """Test MACD strategy initialization"""
        strategy = MACDStrategy()
        
        assert strategy.name == "MACD Strategy"
        assert strategy.fast_period == 12
        assert strategy.slow_period == 26
        assert strategy.signal_period == 9
    
    def test_add_required_indicators(self, sample_ohlcv_data):
        """Test MACD required indicators"""
        strategy = MACDStrategy()
        
        # Add basic MACD indicators first
        df = sample_ohlcv_data.copy()
        df['macd'] = np.random.normal(0, 2, len(df))
        df['macd_signal'] = np.random.normal(0, 1.5, len(df))
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        df_with_indicators = strategy.add_required_indicators(df)
        
        expected_indicators = ['macd_above_signal', 'macd_crossover', 'macd_above_zero',
                             'macd_zero_crossover', 'histogram_momentum', 'macd_momentum']
        
        for indicator in expected_indicators:
            assert indicator in df_with_indicators.columns
    
    def test_calculate_signal_bullish_crossover(self, sample_ohlcv_data):
        """Test bullish MACD crossover signal"""
        strategy = MACDStrategy()
        
        # Set up bullish crossover
        df = sample_ohlcv_data.copy()
        df['macd'] = 0.5
        df['macd_signal'] = 0.3
        df['macd_histogram'] = 0.2
        df['macd_crossover'] = 1  # Fresh bullish crossover
        df['macd_zero_crossover'] = 0
        df['histogram_momentum'] = 0.1  # Improving
        df['macd_momentum'] = 0.1  # Rising
        df['macd_divergence'] = 0
        df['macd_strength'] = 0.2
        
        signal = strategy.calculate_signal(df, "BTC-USD")
        
        assert signal.signal in [SignalType.BUY, SignalType.STRONG_BUY]
        assert signal.confidence > 0.5
        assert "macd" in signal.reason.lower()
        assert "crossover" in signal.reason.lower()
    
    def test_calculate_signal_bearish_crossover(self, sample_ohlcv_data):
        """Test bearish MACD crossover signal"""
        strategy = MACDStrategy()
        
        # Set up bearish crossover
        df = sample_ohlcv_data.copy()
        df['macd'] = -0.5
        df['macd_signal'] = -0.3
        df['macd_histogram'] = -0.2
        df['macd_crossover'] = -1  # Fresh bearish crossover
        df['macd_zero_crossover'] = 0
        df['histogram_momentum'] = -0.1  # Deteriorating
        df['macd_momentum'] = -0.1  # Falling
        df['macd_divergence'] = 0
        df['macd_strength'] = 0.2
        
        signal = strategy.calculate_signal(df, "BTC-USD")
        
        assert signal.signal in [SignalType.SELL, SignalType.STRONG_SELL]
        assert signal.confidence > 0.5
        assert "macd" in signal.reason.lower()
        assert "crossover" in signal.reason.lower()
    
    def test_detect_macd_divergence(self, sample_ohlcv_data):
        """Test MACD divergence detection"""
        strategy = MACDStrategy()
        
        # Create data for divergence testing
        df = sample_ohlcv_data.copy()
        df['macd'] = np.random.normal(0, 1, len(df))
        
        divergence = strategy._detect_macd_divergence(df)
        
        assert isinstance(divergence, pd.Series)
        assert len(divergence) == len(df)
        assert all(val in [-1, 0, 1] for val in divergence.unique())


class TestPure5PercentStrategy:
    """Test Pure5PercentStrategy"""
    
    def test_init_default_parameters(self):
        """Test Pure5PercentStrategy initialization"""
        strategy = Pure5PercentStrategy()
        
        assert strategy.name == "Pure 5% Strategy"
        assert strategy.drop_threshold == 0.05
        assert strategy.rise_threshold == 0.05
        assert strategy.lookback_days == 7
    
    def test_init_custom_parameters(self):
        """Test Pure5PercentStrategy with custom parameters"""
        strategy = Pure5PercentStrategy(drop_threshold=0.03, rise_threshold=0.07, lookback_days=10)
        
        assert strategy.drop_threshold == 0.03
        assert strategy.rise_threshold == 0.07
        assert strategy.lookback_days == 10
        assert strategy.name == "Pure 3% Strategy"
    
    def test_add_required_indicators(self, sample_ohlcv_data):
        """Test Pure5Percent required indicators"""
        strategy = Pure5PercentStrategy()
        
        df_with_indicators = strategy.add_required_indicators(sample_ohlcv_data)
        
        expected_indicators = ['rolling_high', 'rolling_low', 'pct_drop_from_high',
                             'pct_rise_from_low', 'hit_drop_threshold', 'hit_rise_threshold']
        
        for indicator in expected_indicators:
            assert indicator in df_with_indicators.columns
    
    def test_calculate_signal_strong_buy_large_drop(self, sample_ohlcv_data):
        """Test strong buy signal on large price drop"""
        strategy = Pure5PercentStrategy(drop_threshold=0.05)  # 5%
        
        df = strategy.add_required_indicators(sample_ohlcv_data)
        
        # Simulate 8% drop (should trigger strong buy)
        df.iloc[-1, df.columns.get_loc('pct_drop_from_high')] = 8.0  # 8% drop
        df.iloc[-1, df.columns.get_loc('price_change_3d')] = -5.0  # Recent decline
        
        signal = strategy.calculate_signal(df, "BTC-USD")
        
        assert signal.signal in [SignalType.BUY, SignalType.STRONG_BUY]
        assert signal.confidence > 0.6
        assert "drop" in signal.reason.lower()
    
    def test_calculate_signal_sell_large_rise(self, sample_ohlcv_data):
        """Test sell signal on large price rise"""
        strategy = Pure5PercentStrategy(rise_threshold=0.05)  # 5%
        
        df = strategy.add_required_indicators(sample_ohlcv_data)
        
        # Simulate 6% rise (should trigger sell)
        df.iloc[-1, df.columns.get_loc('pct_rise_from_low')] = 6.0  # 6% rise
        
        signal = strategy.calculate_signal(df, "BTC-USD")
        
        assert signal.signal in [SignalType.SELL, SignalType.STRONG_SELL]
        assert signal.confidence > 0.6
        assert "rise" in signal.reason.lower()
    
    def test_calculate_signal_hold_no_threshold(self, sample_ohlcv_data):
        """Test hold signal when thresholds not met"""
        strategy = Pure5PercentStrategy()
        
        df = strategy.add_required_indicators(sample_ohlcv_data)
        
        # Small movements (below thresholds)
        df.iloc[-1, df.columns.get_loc('pct_drop_from_high')] = 2.0  # 2% drop (< 5%)
        df.iloc[-1, df.columns.get_loc('pct_rise_from_low')] = 3.0   # 3% rise (< 5%)
        
        signal = strategy.calculate_signal(df, "BTC-USD")
        
        assert signal.signal == SignalType.HOLD
        assert "No significant" in signal.reason or "Between thresholds" in signal.reason


class TestDynamicPercentStrategy:
    """Test DynamicPercentStrategy"""
    
    def test_init_with_volatility_adaptation(self):
        """Test DynamicPercentStrategy with volatility adaptation"""
        strategy = DynamicPercentStrategy(adapt_to_volatility=True)
        
        assert strategy.adapt_to_volatility is True
        assert "Dynamic" in strategy.name
    
    def test_add_required_indicators_with_volatility(self, sample_ohlcv_data):
        """Test dynamic strategy indicators with volatility adaptation"""
        strategy = DynamicPercentStrategy(adapt_to_volatility=True)
        
        df_with_indicators = strategy.add_required_indicators(sample_ohlcv_data)
        
        volatility_indicators = ['daily_returns', 'volatility', 'avg_volatility',
                               'volatility_multiplier', 'adaptive_drop_threshold', 
                               'adaptive_rise_threshold']
        
        for indicator in volatility_indicators:
            assert indicator in df_with_indicators.columns
    
    def test_adaptive_thresholds(self, sample_ohlcv_data):
        """Test that adaptive thresholds change with volatility"""
        strategy = DynamicPercentStrategy(drop_threshold=0.05, adapt_to_volatility=True)
        
        df = strategy.add_required_indicators(sample_ohlcv_data)
        
        if 'adaptive_drop_threshold' in df.columns:
            # Adaptive thresholds should vary from base threshold
            adaptive_thresholds = df['adaptive_drop_threshold'].dropna()
            if len(adaptive_thresholds) > 1:
                # Should have some variation (not all identical)
                assert len(adaptive_thresholds.unique()) > 1 or adaptive_thresholds.iloc[0] != strategy.drop_threshold


class TestStrategyEdgeCases:
    """Test edge cases across all strategies"""
    
    def test_empty_dataframe_handling(self):
        """Test all strategies handle empty DataFrame gracefully"""
        strategies = [
            SwingTradingStrategy(),
            RSIStrategy(),
            MACDStrategy(),
            Pure5PercentStrategy()
        ]
        
        empty_df = pd.DataFrame()
        
        for strategy in strategies:
            signal = strategy.calculate_signal(empty_df, "BTC-USD")
            assert signal.signal == SignalType.HOLD
            assert "Insufficient data" in signal.reason or "Missing" in signal.reason
    
    def test_missing_indicators_handling(self, sample_ohlcv_data):
        """Test strategies handle missing indicators gracefully"""
        strategies = [
            SwingTradingStrategy(),
            RSIStrategy(),
            MACDStrategy(),
            Pure5PercentStrategy()
        ]
        
        # Basic OHLCV data without any indicators
        basic_df = sample_ohlcv_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        for strategy in strategies:
            signal = strategy.calculate_signal(basic_df, "BTC-USD")
            assert signal.signal == SignalType.HOLD
            # Should handle missing indicators gracefully
    
    def test_extreme_confidence_values(self):
        """Test that confidence values stay within bounds"""
        strategies = [
            SwingTradingStrategy(),
            RSIStrategy(),
            MACDStrategy(),
            Pure5PercentStrategy()
        ]
        
        # Create extreme test data
        extreme_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='D'),
            'open': [100] * 50,
            'high': [200] * 50,  # 100% daily increases
            'low': [50] * 50,    # 50% daily decreases
            'close': [100] * 50,
            'volume': [1000000] * 50  # Very high volume
        })
        
        for strategy in strategies:
            try:
                # Add required indicators
                df_with_indicators = strategy.add_required_indicators(extreme_data)
                signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
                
                # Confidence should be between 0 and 1
                assert 0 <= signal.confidence <= 1
                
                # Price should be reasonable
                assert signal.price > 0
                assert np.isfinite(signal.price)
                
            except Exception as e:
                pytest.fail(f"{strategy.name} failed on extreme data: {e}")
    
    def test_nan_indicator_handling(self, sample_ohlcv_data):
        """Test strategies handle NaN indicators gracefully"""
        strategy = SwingTradingStrategy()
        
        df = strategy.add_required_indicators(sample_ohlcv_data)
        
        # Introduce NaN values in indicators
        if 'pct_from_high' in df.columns:
            df.iloc[-1, df.columns.get_loc('pct_from_high')] = np.nan
        if 'volume_ratio' in df.columns:
            df.iloc[-1, df.columns.get_loc('volume_ratio')] = np.nan
        
        # Should handle NaN gracefully without crashing
        signal = strategy.calculate_signal(df, "BTC-USD")
        assert signal is not None
        assert isinstance(signal.confidence, (int, float))
        assert not np.isnan(signal.confidence)
    
    def test_signal_type_consistency(self, strategy_test_data):
        """Test that signal types are consistent with confidence levels"""
        strategies = [
            SwingTradingStrategy(),
            RSIStrategy(),
            MACDStrategy(),
            Pure5PercentStrategy()
        ]
        
        for strategy in strategies:
            df_with_indicators = strategy.add_required_indicators(strategy_test_data)
            signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
            
            # STRONG signals should have higher confidence than regular signals
            if signal.signal == SignalType.STRONG_BUY:
                assert signal.confidence >= 0.7  # Strong signals should have high confidence
            elif signal.signal == SignalType.STRONG_SELL:
                assert signal.confidence >= 0.7
            elif signal.signal in [SignalType.BUY, SignalType.SELL]:
                assert signal.confidence >= 0.5  # Regular signals should have reasonable confidence
            
            # All signals should have valid reasons
            assert isinstance(signal.reason, str)
            assert len(signal.reason) > 0
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


class TestRSIStrategyBoundaryConditions:
    """Test RSI strategy boundary conditions and algorithmic edge cases"""
    
    def test_rsi_extreme_overbought_conditions(self):
        """Test RSI strategy with extreme overbought conditions"""
        strategy = RSIStrategy(oversold_threshold=30, overbought_threshold=70, rsi_period=14)
        
        # Create data that should produce RSI near 100 (extreme overbought)
        uptrend_prices = [100 * (1.05 ** i) for i in range(30)]  # 5% daily gains
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(uptrend_prices), freq='D'),
            'open': uptrend_prices,
            'high': [p * 1.02 for p in uptrend_prices],
            'low': [p * 0.98 for p in uptrend_prices],
            'close': uptrend_prices,
            'volume': [1000] * len(uptrend_prices)
        })
        
        df_with_indicators = strategy.add_required_indicators(df)
        signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
        
        # Should generate SELL signal with extreme overbought conditions
        assert signal.signal in [SignalType.SELL, SignalType.STRONG_SELL]
        assert signal.confidence >= 0.7  # Should be high confidence
        assert "overbought" in signal.reason.lower() or "sell" in signal.reason.lower()
    
    def test_rsi_extreme_oversold_conditions(self):
        """Test RSI strategy with extreme oversold conditions"""
        strategy = RSIStrategy(oversold_threshold=30, overbought_threshold=70, rsi_period=14)
        
        # Create data that should produce RSI near 0 (extreme oversold)
        downtrend_prices = [100 * (0.95 ** i) for i in range(30)]  # 5% daily losses
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(downtrend_prices), freq='D'),
            'open': downtrend_prices,
            'high': [p * 1.02 for p in downtrend_prices],
            'low': [p * 0.98 for p in downtrend_prices],
            'close': downtrend_prices,
            'volume': [1000] * len(downtrend_prices)
        })
        
        df_with_indicators = strategy.add_required_indicators(df)
        signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
        
        # Should generate BUY signal with extreme oversold conditions
        assert signal.signal in [SignalType.BUY, SignalType.STRONG_BUY]
        assert signal.confidence >= 0.7  # Should be high confidence
        assert "oversold" in signal.reason.lower() or "buy" in signal.reason.lower()
    
    def test_rsi_divergence_detection_accuracy(self):
        """Test RSI divergence detection boundary conditions"""
        strategy = RSIStrategy()
        
        # Create bullish divergence scenario: price makes lower lows, RSI makes higher lows
        base_prices = [100, 90, 95, 85, 92]  # Price lower lows
        # Extend with gradual recovery to create RSI higher lows
        divergence_prices = base_prices + [88, 94, 89, 96, 91, 98]  # RSI should show divergence
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(divergence_prices), freq='D'),
            'open': divergence_prices,
            'high': [p * 1.01 for p in divergence_prices],
            'low': [p * 0.99 for p in divergence_prices],
            'close': divergence_prices,
            'volume': [1000] * len(divergence_prices)
        })
        
        df_with_indicators = strategy.add_required_indicators(df)
        
        # Check divergence detection logic if present
        if 'rsi' in df_with_indicators.columns:
            rsi_values = df_with_indicators['rsi'].dropna()
            if len(rsi_values) >= 5:
                # RSI divergence algorithm should be stable and not crash
                signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
                assert signal is not None
                assert 0 <= signal.confidence <= 1
    
    def test_rsi_threshold_boundary_values(self):
        """Test RSI strategy with boundary threshold values"""
        # Test extreme threshold values
        extreme_strategies = [
            RSIStrategy(oversold_threshold=5, overbought_threshold=95),   # Very extreme
            RSIStrategy(oversold_threshold=45, overbought_threshold=55),  # Very tight
            RSIStrategy(oversold_threshold=25, overbought_threshold=75),  # Standard
        ]
        
        # Create neutral market data
        neutral_prices = [100 + np.sin(i/10) for i in range(50)]  # Gentle oscillation
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(neutral_prices), freq='D'),
            'open': neutral_prices,
            'high': [p * 1.005 for p in neutral_prices],
            'low': [p * 0.995 for p in neutral_prices],
            'close': neutral_prices,
            'volume': [1000] * len(neutral_prices)
        })
        
        for strategy in extreme_strategies:
            df_with_indicators = strategy.add_required_indicators(df)
            signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
            
            # Should handle all threshold configurations without errors
            assert signal is not None
            assert isinstance(signal.signal, SignalType)
            assert 0 <= signal.confidence <= 1
    
    def test_rsi_with_zero_price_changes(self):
        """Test RSI calculation with periods of zero price change"""
        strategy = RSIStrategy()
        
        # Create data with flat periods (zero price changes)
        flat_prices = [100] * 20 + [105] * 10 + [105] * 15  # Flat periods
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(flat_prices), freq='D'),
            'open': flat_prices,
            'high': [max(p * 1.001, p + 0.01) for p in flat_prices],  # Minimal variation
            'low': [min(p * 0.999, p - 0.01) for p in flat_prices],
            'close': flat_prices,
            'volume': [1000] * len(flat_prices)
        })
        
        df_with_indicators = strategy.add_required_indicators(df)
        
        # Should handle flat periods without division by zero or NaN errors
        signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
        assert signal is not None
        
        if 'rsi' in df_with_indicators.columns:
            rsi_values = df_with_indicators['rsi'].dropna()
            if len(rsi_values) > 0:
                # RSI should be either NaN (acceptable) or around 50 for flat periods
                for rsi in rsi_values:
                    assert np.isnan(rsi) or 0 <= rsi <= 100


class TestMACDStrategyBoundaryConditions:
    """Test MACD strategy boundary conditions"""
    
    def test_macd_extreme_divergence_conditions(self):
        """Test MACD with extreme price divergence"""
        strategy = MACDStrategy()
        
        # Create extreme trending data
        extreme_trend = [100 * (1.1 ** (i/10)) for i in range(50)]  # Exponential growth
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(extreme_trend), freq='D'),
            'open': extreme_trend,
            'high': [p * 1.02 for p in extreme_trend],
            'low': [p * 0.98 for p in extreme_trend],
            'close': extreme_trend,
            'volume': [1000] * len(extreme_trend)
        })
        
        df_with_indicators = strategy.add_required_indicators(df)
        signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
        
        # Should handle extreme trending conditions
        assert signal is not None
        assert 0 <= signal.confidence <= 1
        
        # MACD should show strong trend signals
        if signal.signal in [SignalType.BUY, SignalType.STRONG_BUY]:
            assert signal.confidence >= 0.6  # Strong trend should have good confidence
    
    def test_macd_zero_line_crossing_accuracy(self):
        """Test MACD zero-line crossing detection accuracy"""
        strategy = MACDStrategy()
        
        # Create pattern that crosses zero line: down trend then up trend
        crossing_pattern = (
            [100 - i for i in range(25)] +  # Downtrend (MACD below zero)
            [75 + i for i in range(25)]      # Uptrend (MACD above zero)
        )
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(crossing_pattern), freq='D'),
            'open': crossing_pattern,
            'high': [p * 1.01 for p in crossing_pattern],
            'low': [p * 0.99 for p in crossing_pattern],
            'close': crossing_pattern,
            'volume': [1000] * len(crossing_pattern)
        })
        
        df_with_indicators = strategy.add_required_indicators(df)
        
        # Check for zero-line crossing logic
        if all(col in df_with_indicators.columns for col in ['macd', 'macd_signal']):
            macd_values = df_with_indicators['macd'].dropna()
            
            if len(macd_values) > 10:
                # Should have both positive and negative MACD values
                has_positive = any(val > 0 for val in macd_values)
                has_negative = any(val < 0 for val in macd_values)
                
                # This pattern should show crossovers
                signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
                assert signal is not None
                
                # If we detect a trend change, confidence should be reasonable
                if signal.signal != SignalType.HOLD:
                    assert signal.confidence >= 0.5
    
    def test_macd_histogram_extreme_values(self):
        """Test MACD histogram with extreme values"""
        strategy = MACDStrategy()
        
        # Create data with rapid changes to produce extreme histogram values
        rapid_changes = []
        base = 100
        for i in range(40):
            if i % 4 == 0:
                base *= 1.1  # 10% jump every 4th period
            else:
                base *= 0.995  # Small decline other periods
            rapid_changes.append(base)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(rapid_changes), freq='D'),
            'open': rapid_changes,
            'high': [p * 1.005 for p in rapid_changes],
            'low': [p * 0.995 for p in rapid_changes],
            'close': rapid_changes,
            'volume': [1000] * len(rapid_changes)
        })
        
        df_with_indicators = strategy.add_required_indicators(df)
        
        # Should handle rapid changes without errors
        signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
        assert signal is not None
        assert np.isfinite(signal.price)
        assert np.isfinite(signal.confidence)
    
    def test_macd_signal_line_convergence(self):
        """Test MACD behavior when MACD and signal lines converge"""
        strategy = MACDStrategy()
        
        # Create sideways market that should produce converging MACD lines
        sideways_prices = [100 + 2 * np.sin(i/5) for i in range(60)]  # Oscillating around 100
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(sideways_prices), freq='D'),
            'open': sideways_prices,
            'high': [p * 1.01 for p in sideways_prices],
            'low': [p * 0.99 for p in sideways_prices],
            'close': sideways_prices,
            'volume': [1000] * len(sideways_prices)
        })
        
        df_with_indicators = strategy.add_required_indicators(df)
        signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
        
        # Should handle convergent conditions (likely HOLD)
        assert signal is not None
        
        # In sideways markets, should be more conservative
        if signal.signal == SignalType.HOLD:
            assert 0.4 <= signal.confidence <= 0.6  # Neutral confidence for sideways


class TestSwingTradingStrategyBoundaryConditions:
    """Test Swing Trading strategy boundary conditions"""
    
    def test_swing_strategy_extreme_volatility(self):
        """Test swing strategy with extreme volatility"""
        strategy = SwingTradingStrategy(swing_threshold=0.025)  # 2.5% threshold
        
        # Create extremely volatile data (>10% daily swings)
        volatile_prices = []
        base = 100
        for i in range(30):
            # Alternate between large up and down moves
            if i % 2 == 0:
                base *= 1.15  # 15% up
            else:
                base *= 0.85  # 15% down
            volatile_prices.append(base)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(volatile_prices), freq='D'),
            'open': volatile_prices,
            'high': [p * 1.03 for p in volatile_prices],
            'low': [p * 0.97 for p in volatile_prices],
            'close': volatile_prices,
            'volume': [2000] * len(volatile_prices)  # High volume
        })
        
        df_with_indicators = strategy.add_required_indicators(df)
        signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
        
        # Should handle extreme volatility (likely generate signals due to large moves)
        assert signal is not None
        assert 0 <= signal.confidence <= 1
        
        # With such large moves, should generate trading signals
        if signal.signal != SignalType.HOLD:
            assert signal.confidence >= 0.6  # High volatility should give confident signals
    
    def test_swing_strategy_volume_threshold_boundaries(self):
        """Test swing strategy with volume threshold boundary conditions"""
        strategies = [
            SwingTradingStrategy(volume_threshold=0.5),   # Very low volume threshold
            SwingTradingStrategy(volume_threshold=5.0),   # Very high volume threshold
            SwingTradingStrategy(volume_threshold=1.0),   # Standard threshold
        ]
        
        # Create data with varying volume
        prices = [100 + i for i in range(30)]  # Gradual uptrend
        volumes = [500 if i % 3 == 0 else 1000 for i in range(30)]  # Alternating volume
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='D'),
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        for strategy in strategies:
            df_with_indicators = strategy.add_required_indicators(df)
            signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
            
            # All volume thresholds should produce valid signals
            assert signal is not None
            assert isinstance(signal.signal, SignalType)
    
    def test_swing_strategy_minimum_movement_detection(self):
        """Test swing strategy with movements right at the threshold"""
        strategy = SwingTradingStrategy(swing_threshold=0.05)  # 5% threshold
        
        # Create movements exactly at the threshold
        threshold_prices = [100, 95.0, 99.75, 104.9, 100.1]  # Movements around 5%
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(threshold_prices), freq='D'),
            'open': threshold_prices,
            'high': [p * 1.01 for p in threshold_prices],
            'low': [p * 0.99 for p in threshold_prices],
            'close': threshold_prices,
            'volume': [1200] * len(threshold_prices)  # Above average volume
        })
        
        df_with_indicators = strategy.add_required_indicators(df)
        
        # Test sensitivity around the threshold
        signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
        assert signal is not None
        
        # Should be sensitive to movements near the threshold
        if 'pct_from_high' in df_with_indicators.columns:
            pct_from_high = df_with_indicators['pct_from_high'].iloc[-1]
            if not pd.isna(pct_from_high) and pct_from_high >= strategy.swing_threshold:
                # If we're at/above threshold, should consider a signal
                assert signal.signal in [SignalType.BUY, SignalType.STRONG_BUY, SignalType.HOLD]
    
    def test_swing_strategy_lookback_period_boundaries(self):
        """Test swing strategy with different lookback periods"""
        # Test extreme lookback periods
        strategies = [
            SwingTradingStrategy(swing_threshold=0.03, lookback_period=3),   # Very short lookback
            SwingTradingStrategy(swing_threshold=0.03, lookback_period=30),  # Very long lookback
            SwingTradingStrategy(swing_threshold=0.03, lookback_period=1),   # Minimum lookback
        ]
        
        # Create data with multiple peaks and troughs
        cyclical_prices = [100 + 10 * np.sin(i/5) for i in range(50)]  # Sine wave pattern
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(cyclical_prices), freq='D'),
            'open': cyclical_prices,
            'high': [p * 1.01 for p in cyclical_prices],
            'low': [p * 0.99 for p in cyclical_prices],
            'close': cyclical_prices,
            'volume': [1000] * len(cyclical_prices)
        })
        
        for strategy in strategies:
            df_with_indicators = strategy.add_required_indicators(df)
            signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
            
            # All lookback periods should produce valid results
            assert signal is not None
            assert 0 <= signal.confidence <= 1
            
            # Different lookback periods may produce different signals, but all should be valid
            assert isinstance(signal.signal, SignalType)


class TestPurePercentStrategyBoundaryConditions:
    """Test Pure Percent strategy boundary conditions"""
    
    def test_pure_percent_exact_threshold_hits(self):
        """Test Pure Percent strategy with exact threshold hits"""
        strategy = Pure5PercentStrategy(drop_threshold=0.05, rise_threshold=0.05)
        
        # Create data with exact 5% moves
        exact_threshold_prices = [
            100.0,  # Start
            95.0,   # Exactly -5% (should trigger buy)
            99.75,  # Small recovery
            100.0,  # Back to start
            105.0,  # Exactly +5% from recent low (should trigger sell)
            102.0   # Small pullback
        ]
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(exact_threshold_prices), freq='D'),
            'open': exact_threshold_prices,
            'high': [p * 1.001 for p in exact_threshold_prices],
            'low': [p * 0.999 for p in exact_threshold_prices],
            'close': exact_threshold_prices,
            'volume': [1000] * len(exact_threshold_prices)
        })
        
        df_with_indicators = strategy.add_required_indicators(df)
        
        # Test each point for threshold detection
        for i in range(2, len(df_with_indicators)):  # Start from index 2 to have history
            subset_df = df_with_indicators.iloc[:i+1].copy()
            signal = strategy.calculate_signal(subset_df, "BTC-USD")
            
            assert signal is not None
            assert 0 <= signal.confidence <= 1
    
    def test_pure_percent_micro_movements(self):
        """Test Pure Percent strategy with movements just below threshold"""
        strategy = Pure5PercentStrategy(drop_threshold=0.05, rise_threshold=0.05)
        
        # Create movements just below 5% threshold (4.9%)
        micro_movement_prices = [
            100.0,  # Start
            95.1,   # -4.9% (just below threshold)
            99.8,   # Small recovery
            100.0,  # Back to start
            104.9,  # +4.9% (just below threshold)
            102.0   # Small pullback
        ]
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(micro_movement_prices), freq='D'),
            'open': micro_movement_prices,
            'high': [p * 1.001 for p in micro_movement_prices],
            'low': [p * 0.999 for p in micro_movement_prices],
            'close': micro_movement_prices,
            'volume': [1000] * len(micro_movement_prices)
        })
        
        df_with_indicators = strategy.add_required_indicators(df)
        signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
        
        # Should mostly HOLD since we're below thresholds
        # But algorithm should handle these near-threshold conditions gracefully
        assert signal is not None
        assert signal.signal in [SignalType.HOLD, SignalType.BUY, SignalType.SELL]  # All valid
    
    def test_pure_percent_extreme_threshold_values(self):
        """Test Pure Percent strategy with extreme threshold values"""
        extreme_strategies = [
            Pure5PercentStrategy(drop_threshold=0.01, rise_threshold=0.01),  # Very sensitive (1%)
            Pure5PercentStrategy(drop_threshold=0.25, rise_threshold=0.25),  # Very insensitive (25%)
            Pure5PercentStrategy(drop_threshold=0.001, rise_threshold=0.001), # Ultra-sensitive (0.1%)
        ]
        
        # Normal market movements
        normal_prices = [100, 102, 98, 101, 97, 103, 96, 104]  # Various moves
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(normal_prices), freq='D'),
            'open': normal_prices,
            'high': [p * 1.01 for p in normal_prices],
            'low': [p * 0.99 for p in normal_prices],
            'close': normal_prices,
            'volume': [1000] * len(normal_prices)
        })
        
        for strategy in extreme_strategies:
            df_with_indicators = strategy.add_required_indicators(df)
            signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
            
            # All extreme thresholds should produce valid signals
            assert signal is not None
            assert 0 <= signal.confidence <= 1
            assert isinstance(signal.signal, SignalType)
    
    def test_pure_percent_lookback_boundary_conditions(self):
        """Test Pure Percent strategy with extreme lookback periods"""
        strategies = [
            Pure5PercentStrategy(lookback_days=1),   # Minimum lookback
            Pure5PercentStrategy(lookback_days=2),   # Very short
            Pure5PercentStrategy(lookback_days=30),  # Very long
        ]
        
        # Create diverse price history
        diverse_prices = [100 + 5 * np.sin(i/3) + np.random.normal(0, 1) for i in range(50)]
        np.random.seed(42)  # For reproducibility
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(diverse_prices), freq='D'),
            'open': diverse_prices,
            'high': [p * 1.02 for p in diverse_prices],
            'low': [p * 0.98 for p in diverse_prices],
            'close': diverse_prices,
            'volume': [1000] * len(diverse_prices)
        })
        
        for strategy in strategies:
            df_with_indicators = strategy.add_required_indicators(df)
            signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
            
            # All lookback periods should handle gracefully
            assert signal is not None
            assert 0 <= signal.confidence <= 1
            
            # Very short lookbacks might be more reactive
            # Very long lookbacks might be more stable
            # Both should produce valid signals


class TestCrossStrategyBoundaryConditions:
    """Test boundary conditions across all strategies"""
    
    def test_all_strategies_with_insufficient_data(self):
        """Test all strategies with various insufficient data scenarios"""
        strategies = [
            SwingTradingStrategy(),
            RSIStrategy(),
            MACDStrategy(),
            Pure5PercentStrategy(),
            DynamicPercentStrategy()
        ]
        
        # Various insufficient data scenarios
        insufficient_data_scenarios = [
            pd.DataFrame(),  # Empty DataFrame
            pd.DataFrame({'timestamp': [datetime.now()], 'close': [100]}),  # Single row
            pd.DataFrame({  # Missing required columns
                'timestamp': pd.date_range('2024-01-01', periods=5, freq='D'),
                'close': [100, 101, 102, 103, 104]  # Missing OHLCV columns
            }),
        ]
        
        for scenario in insufficient_data_scenarios:
            for strategy in strategies:
                # Should handle all insufficient data scenarios gracefully
                try:
                    if not scenario.empty and 'close' in scenario.columns:
                        # Add minimum required columns if close exists
                        if 'open' not in scenario.columns:
                            scenario['open'] = scenario['close']
                        if 'high' not in scenario.columns:
                            scenario['high'] = scenario['close'] * 1.01
                        if 'low' not in scenario.columns:
                            scenario['low'] = scenario['close'] * 0.99
                        if 'volume' not in scenario.columns:
                            scenario['volume'] = 1000
                    
                    signal = strategy.calculate_signal(scenario, "BTC-USD")
                    
                    # Should return HOLD with appropriate reason
                    assert signal.signal == SignalType.HOLD
                    assert "insufficient" in signal.reason.lower() or "missing" in signal.reason.lower()
                    
                except Exception as e:
                    # Some extreme cases might raise exceptions, which is acceptable
                    # as long as they're handled gracefully
                    assert "insufficient" in str(e).lower() or "missing" in str(e).lower() or "empty" in str(e).lower()
    
    def test_all_strategies_mathematical_stability(self):
        """Test mathematical stability across all strategies"""
        strategies = [
            SwingTradingStrategy(),
            RSIStrategy(),
            MACDStrategy(),
            Pure5PercentStrategy()
        ]
        
        # Create mathematically challenging data
        challenging_scenarios = [
            # Constant prices (division by zero scenarios)
            [100.0] * 30,
            # Very small prices
            [0.001 * (1 + 0.01 * np.sin(i)) for i in range(30)],
            # Very large prices  
            [1000000 * (1 + 0.01 * np.sin(i)) for i in range(30)],
            # High precision decimal prices
            [100.123456789 + 0.000000001 * i for i in range(30)]
        ]
        
        for scenario_prices in challenging_scenarios:
            df = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=len(scenario_prices), freq='D'),
                'open': scenario_prices,
                'high': [p * 1.001 for p in scenario_prices],
                'low': [p * 0.999 for p in scenario_prices],
                'close': scenario_prices,
                'volume': [1000] * len(scenario_prices)
            })
            
            for strategy in strategies:
                try:
                    df_with_indicators = strategy.add_required_indicators(df)
                    signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
                    
                    # Should produce mathematically stable results
                    assert signal is not None
                    assert np.isfinite(signal.price)
                    assert np.isfinite(signal.confidence)
                    assert 0 <= signal.confidence <= 1
                    
                    # Indicators should be finite
                    for key, value in signal.indicators.items():
                        if isinstance(value, (int, float)):
                            assert np.isfinite(value), f"Non-finite indicator {key}={value}"
                
                except Exception as e:
                    # If exceptions occur, they should be handled gracefully
                    # and should not be due to mathematical instability
                    assert "overflow" not in str(e).lower()
                    assert "underflow" not in str(e).lower()
                    assert "inf" not in str(e).lower()
    
    def test_all_strategies_confidence_bounds(self):
        """Test that all strategies maintain confidence within bounds"""
        strategies = [
            SwingTradingStrategy(),
            RSIStrategy(), 
            MACDStrategy(),
            Pure5PercentStrategy()
        ]
        
        # Create 100 random market scenarios
        np.random.seed(42)
        
        for _ in range(20):  # Test 20 random scenarios
            # Generate random walk
            returns = np.random.normal(0, 0.02, 40)  # 2% daily vol
            prices = [100]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            volumes = np.random.uniform(500, 2000, len(prices))
            
            df = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='D'),
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': prices,
                'volume': volumes
            })
            
            for strategy in strategies:
                df_with_indicators = strategy.add_required_indicators(df)
                signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
                
                # Confidence must always be in valid range
                assert 0 <= signal.confidence <= 1, \
                    f"{strategy.name} produced invalid confidence: {signal.confidence}"
                
                # Signal type must be valid
                assert isinstance(signal.signal, SignalType)
                
                # Price must be positive and finite
                assert signal.price > 0
                assert np.isfinite(signal.price)
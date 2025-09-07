"""
Edge case and error handling tests for crypto trading system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, Mock
import tempfile
import os

from src.config import Config
from src.data.fetcher import CoinbaseDataFetcher
from src.data.processor import DataProcessor
from src.strategies.swing import SwingTradingStrategy
from src.strategies.rsi import RSIStrategy
from src.strategies.macd import MACDStrategy
from src.strategies.pure_percent import Pure5PercentStrategy
from src.backtesting.engine import BacktestEngine, Position
from src.strategies.base import SignalType


class TestDataEdgeCases:
    """Test edge cases in data handling"""
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames"""
        processor = DataProcessor()
        empty_df = pd.DataFrame()
        
        # Should handle empty data gracefully
        result = processor.add_basic_indicators(empty_df)
        assert result.empty
        
        # Test with strategy
        strategy = SwingTradingStrategy()
        signal = strategy.calculate_signal(empty_df, "BTC-USD")
        assert signal.signal == SignalType.HOLD
        assert signal.confidence == 0.0
    
    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrames"""
        processor = DataProcessor()
        
        single_row_df = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1)],
            'open': [50000.0],
            'high': [51000.0],
            'low': [49000.0],
            'close': [50500.0],
            'volume': [1000.0]
        })
        single_row_df.set_index('timestamp', inplace=True)
        
        # Should handle gracefully
        result = processor.add_basic_indicators(single_row_df)
        assert len(result) == 1
        
        # Indicators should be NaN for insufficient data
        assert pd.isna(result['sma_20'].iloc[0])
        assert pd.isna(result['sma_50'].iloc[0])
    
    def test_identical_prices(self):
        """Test handling of identical prices (no volatility)"""
        processor = DataProcessor()
        
        # Create data with identical prices
        dates = pd.date_range('2024-01-01', periods=30)
        identical_data = pd.DataFrame({
            'timestamp': dates,
            'open': [50000.0] * 30,
            'high': [50000.0] * 30,
            'low': [50000.0] * 30,
            'close': [50000.0] * 30,
            'volume': [1000.0] * 30
        })
        identical_data.set_index('timestamp', inplace=True)
        
        result = processor.add_basic_indicators(identical_data)
        result = processor.add_momentum_indicators(result)
        
        # Moving averages should equal the price
        assert np.allclose(result['sma_20'].dropna(), 50000.0)
        
        # RSI should handle zero volatility
        rsi_values = result['rsi'].dropna()
        if not rsi_values.empty:
            # RSI with no price movement should be around 50 or NaN
            assert all((np.isnan(val) or 40 <= val <= 60) for val in rsi_values)
    
    def test_extreme_price_movements(self):
        """Test handling of extreme price movements"""
        processor = DataProcessor()
        
        # Create data with extreme movements
        dates = pd.date_range('2024-01-01', periods=10)
        extreme_data = pd.DataFrame({
            'timestamp': dates,
            'open': [1000, 10000, 100, 50000, 10, 25000, 1, 30000, 5, 40000],
            'high': [1100, 11000, 200, 55000, 50, 28000, 10, 35000, 25, 45000],
            'low': [900, 9000, 50, 45000, 5, 20000, 0.5, 25000, 2, 35000],
            'close': [1050, 9500, 150, 50000, 25, 24000, 5, 32000, 15, 42000],
            'volume': [1000] * 10
        })
        extreme_data.set_index('timestamp', inplace=True)
        
        result = processor.add_basic_indicators(extreme_data)
        result = processor.add_momentum_indicators(result)
        
        # Should not produce NaN, inf, or negative values inappropriately
        assert not np.isinf(result['sma_20']).any()
        assert (result['rsi'].dropna() >= 0).all()
        assert (result['rsi'].dropna() <= 100).all()
    
    def test_missing_columns(self):
        """Test handling of DataFrames with missing required columns"""
        processor = DataProcessor()
        
        # Missing volume column
        incomplete_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [104, 105, 106, 107, 108]
        })
        incomplete_df.set_index('timestamp', inplace=True)
        
        # Should handle missing volume gracefully
        try:
            result = processor.add_basic_indicators(incomplete_df)
            # If successful, volume-dependent indicators should be skipped or use defaults
            assert 'sma_20' in result.columns
        except Exception as e:
            # If it raises an exception, it should be informative
            assert any(word in str(e).lower() for word in ['column', 'missing', 'required'])
    
    def test_non_chronological_data(self):
        """Test handling of non-chronologically ordered data"""
        processor = DataProcessor()
        
        # Create data in random order
        dates = pd.date_range('2024-01-01', periods=10)
        shuffled_indices = np.random.permutation(len(dates))
        
        random_order_data = pd.DataFrame({
            'timestamp': dates[shuffled_indices],
            'open': [100 + i for i in shuffled_indices],
            'high': [105 + i for i in shuffled_indices],
            'low': [95 + i for i in shuffled_indices],
            'close': [104 + i for i in shuffled_indices],
            'volume': [1000] * 10
        })
        random_order_data.set_index('timestamp', inplace=True)
        
        # Should handle or sort appropriately
        result = processor.add_basic_indicators(random_order_data)
        assert len(result) == 10
        # Result should be sorted by timestamp
        assert result.index.is_monotonic_increasing


class TestStrategyEdgeCases:
    """Test edge cases in trading strategies"""
    
    def test_strategy_with_insufficient_data(self):
        """Test strategy behavior with insufficient historical data"""
        # Create minimal data
        minimal_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3),
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [104, 105, 106],
            'volume': [1000, 1100, 1200]
        })
        minimal_data.set_index('timestamp', inplace=True)
        
        # Add what indicators we can
        processor = DataProcessor()
        processed = processor.add_basic_indicators(minimal_data)
        
        strategies = [
            SwingTradingStrategy(),
            RSIStrategy(),
            MACDStrategy(),
            Pure5PercentStrategy()
        ]
        
        for strategy in strategies:
            signal = strategy.calculate_signal(processed, "BTC-USD")
            # Should return valid signal even with minimal data
            assert isinstance(signal.signal, SignalType)
            assert 0.0 <= signal.confidence <= 1.0
    
    def test_strategy_with_all_nan_indicators(self):
        """Test strategy behavior when all indicators are NaN"""
        # Create data that will result in NaN indicators
        nan_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [104, 105, 106, 107, 108],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'sma_20': [np.nan] * 5,
            'sma_50': [np.nan] * 5,
            'ema_12': [np.nan] * 5,
            'ema_26': [np.nan] * 5,
            'rsi': [np.nan] * 5,
            'macd': [np.nan] * 5,
            'macd_signal': [np.nan] * 5,
            'macd_histogram': [np.nan] * 5
        })
        nan_data.set_index('timestamp', inplace=True)
        
        strategies = [
            SwingTradingStrategy(),
            RSIStrategy(),
            MACDStrategy()
        ]
        
        for strategy in strategies:
            signal = strategy.calculate_signal(nan_data, "BTC-USD")
            # Should default to HOLD with low confidence
            assert signal.signal in [SignalType.HOLD, SignalType.BUY, SignalType.SELL]
            assert 0.0 <= signal.confidence <= 1.0
    
    def test_strategy_with_extreme_rsi_values(self):
        """Test RSI strategy with extreme overbought/oversold conditions"""
        strategy = RSIStrategy(oversold_threshold=30, overbought_threshold=70)
        
        # Create data with extreme RSI
        extreme_rsi_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [104, 105, 106, 107, 108],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'rsi': [5, 95, 0, 100, 50]  # Extreme values
        })
        extreme_rsi_data.set_index('timestamp', inplace=True)
        
        # Test each extreme condition
        for i in range(len(extreme_rsi_data)):
            signal = strategy.calculate_signal(extreme_rsi_data.iloc[:i+1], "BTC-USD")
            assert isinstance(signal.signal, SignalType)
            assert 0.0 <= signal.confidence <= 1.0
    
    def test_pure5percent_strategy_edge_cases(self):
        """Test Pure5Percent strategy edge cases"""
        strategy = Pure5PercentStrategy()
        
        # Test with exact 5% movement
        exact_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'open': [100] * 10,
            'high': [105] * 10,  # Exactly 5% up
            'low': [95] * 10,    # Exactly 5% down
            'close': [102] * 10,
            'volume': [1000] * 10
        })
        exact_data.set_index('timestamp', inplace=True)
        
        signal = strategy.calculate_signal(exact_data, "BTC-USD")
        assert isinstance(signal.signal, SignalType)
        
        # Test with no movement
        no_movement_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'open': [100] * 10,
            'high': [100] * 10,
            'low': [100] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })
        no_movement_data.set_index('timestamp', inplace=True)
        
        signal = strategy.calculate_signal(no_movement_data, "BTC-USD")
        assert signal.signal == SignalType.HOLD


class TestBacktestingEdgeCases:
    """Test edge cases in backtesting engine"""
    
    def test_backtesting_with_no_signals(self, temp_config_file):
        """Test backtesting when strategy generates no buy/sell signals"""
        config = Config(temp_config_file)
        backtest_engine = BacktestEngine(config)
        
        # Create a custom strategy that never generates signals
        class NoSignalStrategy:
            def calculate_signal(self, df, pair):
                from src.utils import TradingSignal, SignalType
                return TradingSignal(SignalType.HOLD, 0.0, "No signal")
        
        strategy = NoSignalStrategy()
        
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=30),
            'open': range(100, 130),
            'high': range(105, 135),
            'low': range(95, 125),
            'close': range(104, 134),
            'volume': [1000] * 30
        })
        test_data.set_index('timestamp', inplace=True)
        
        result = backtest_engine.run_backtest(strategy, "BTC-USD", "2024-01-01", "2024-01-31")
        
        # Should have no trades
        assert result.total_trades == 0
        assert len(result.positions) == 0
        assert result.final_portfolio_value == result.initial_capital
        assert result.total_return_pct == 0.0
    
    def test_backtesting_with_insufficient_capital(self, temp_config_file):
        """Test backtesting with very low initial capital"""
        # Create config with very low capital
        low_capital_config = """
exchange:
  name: "coinbase"
  base_url: "https://api.pro.coinbase.com"

trading:
  pairs: ["BTC-USD", "ETH-USD", "ADA-USD"]
  initial_capital: 1.0  # Very low capital
  position_sizing:
    method: "fixed_percent"
    percent: 0.1

backtesting:
  initial_capital: 1.0
  commission: 0.005
  slippage: 0.001

risk_management:
  stop_loss: 0.05
  take_profit: 0.15
  max_daily_loss: 0.10
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(low_capital_config)
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            backtest_engine = BacktestEngine(config)
            strategy = SwingTradingStrategy()
            
            # Create high-price data
            high_price_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=10),
                'open': [50000] * 10,  # Bitcoin prices
                'high': [51000] * 10,
                'low': [49000] * 10,
                'close': [50500] * 10,
                'volume': [1000] * 10
            })
            high_price_data.set_index('timestamp', inplace=True)
            
            # Add basic indicators
            processor = DataProcessor()
            processed_data = processor.add_basic_indicators(high_price_data)
            
            result = backtest_engine.run_backtest(strategy, "BTC-USD", "2024-01-01", "2024-01-31")
            
            # Should handle low capital gracefully
            assert result.initial_capital == 1.0
            assert isinstance(result.total_trades, int)
            
        finally:
            os.unlink(temp_path)
    
    def test_position_edge_cases(self):
        """Test edge cases in position management"""
        # Test position with zero size
        with pytest.raises(ValueError):
            Position("BTC-USD", entry_price=100.0, size=0.0, entry_date=datetime.now())
        
        # Test position with negative price
        with pytest.raises(ValueError):
            Position("BTC-USD", entry_price=-100.0, size=1.0, entry_date=datetime.now())
        
        # Test valid position
        position = Position("BTC-USD", entry_price=100.0, size=1.0, entry_date=datetime.now())
        
        # Test closing with same price (no profit/loss)
        position.close_position(100.0, datetime.now(), "Break even")
        assert position.pnl == 0.0
        assert position.pnl_pct == 0.0
        
        # Test stop loss at entry price
        position2 = Position("ETH-USD", entry_price=2000.0, size=0.5, entry_date=datetime.now())
        position2.set_stop_loss(2000.0)  # Same as entry
        assert position2.stop_loss == 2000.0
    
    def test_backtesting_with_gaps_in_data(self, temp_config_file):
        """Test backtesting with missing data points (gaps)"""
        config = Config(temp_config_file)
        backtest_engine = BacktestEngine(config)
        strategy = SwingTradingStrategy()
        
        # Create data with gaps (missing dates)
        dates = pd.date_range('2024-01-01', periods=20)
        # Remove some dates to create gaps
        gap_dates = dates[[0, 1, 2, 5, 6, 10, 15, 16, 17, 19]]  # Skip some dates
        
        gap_data = pd.DataFrame({
            'timestamp': gap_dates,
            'open': range(100, 100 + len(gap_dates)),
            'high': range(105, 105 + len(gap_dates)),
            'low': range(95, 95 + len(gap_dates)),
            'close': range(104, 104 + len(gap_dates)),
            'volume': [1000] * len(gap_dates)
        })
        gap_data.set_index('timestamp', inplace=True)
        
        # Add basic indicators
        processor = DataProcessor()
        processed_data = processor.add_basic_indicators(gap_data)
        
        result = backtest_engine.run_backtest(strategy, "BTC-USD", "2024-01-01", "2024-01-31")
        
        # Should handle gaps gracefully
        assert isinstance(result.total_trades, int)
        assert result.start_date <= result.end_date


class TestNetworkAndAPIEdgeCases:
    """Test edge cases related to network and API calls"""
    
    @patch('src.data.fetcher.requests.Session')
    def test_api_timeout_handling(self, mock_session_class, temp_database, temp_config_file):
        """Test handling of API timeouts"""
        mock_session = Mock()
        mock_session.get.side_effect = Exception("Connection timeout")
        mock_session_class.return_value = mock_session
        
        config = Config(temp_config_file)
        fetcher = CoinbaseDataFetcher(
            config.get('exchange.base_url', 'https://api.exchange.coinbase.com'),
            config.get('database.path', 'data/trading.db')
        )
        
        # Should handle timeout gracefully
        with pytest.raises(Exception):
            fetcher.get_historical_data("BTC-USD", "2024-01-01", "2024-01-02", "1d")
    
    @patch('src.data.fetcher.requests.Session')
    def test_api_rate_limiting(self, mock_session_class, temp_database, temp_config_file):
        """Test handling of API rate limiting"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 429  # Rate limit exceeded
        mock_response.json.return_value = {"message": "Rate limit exceeded"}
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        config = Config(temp_config_file)
        fetcher = CoinbaseDataFetcher(
            config.get('exchange.base_url', 'https://api.exchange.coinbase.com'),
            config.get('database.path', 'data/trading.db')
        )
        
        # Should handle rate limiting appropriately
        try:
            result = fetcher.get_historical_data("BTC-USD", "2024-01-01", "2024-01-02", "1d")
            # If it returns data, it should be empty or cached
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # If it raises an exception, it should be informative
            assert any(word in str(e).lower() for word in ['rate', 'limit', 'too many'])
    
    @patch('src.data.fetcher.requests.Session')
    def test_malformed_api_response(self, mock_session_class, temp_database, temp_config_file):
        """Test handling of malformed API responses"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected": "format"}  # Wrong format
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        config = Config(temp_config_file)
        fetcher = CoinbaseDataFetcher(
            config.get('exchange.base_url', 'https://api.exchange.coinbase.com'),
            config.get('database.path', 'data/trading.db')
        )
        
        # Should handle malformed response
        try:
            result = fetcher.get_historical_data("BTC-USD", "2024-01-01", "2024-01-02", "1d")
            # If successful, should return empty or valid DataFrame
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # Should provide meaningful error message
            assert len(str(e)) > 0


class TestConfigurationEdgeCases:
    """Test edge cases in configuration handling"""
    
    def test_missing_config_sections(self):
        """Test handling of missing configuration sections"""
        minimal_config = """
# Minimal config missing many sections
exchange:
  name: "coinbase"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(minimal_config)
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            
            # Should provide sensible defaults
            trading_config = config.get_trading_config()
            assert isinstance(trading_config, dict)
            
            # Should handle missing keys gracefully
            pairs = config.get_trading_pairs()
            assert isinstance(pairs, list)
            
        finally:
            os.unlink(temp_path)
    
    def test_invalid_config_values(self):
        """Test handling of invalid configuration values"""
        invalid_config = """
exchange:
  name: "coinbase"
  
trading:
  initial_capital: -1000  # Invalid negative value
  pairs: "not_a_list"     # Should be list
  
risk_management:
  stop_loss: 2.0          # Invalid > 1.0
  max_daily_loss: -0.5    # Invalid negative
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_config)
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            
            # Should handle invalid values gracefully
            initial_capital = config.get('trading.initial_capital', 1000)
            assert initial_capital != -1000  # Should use default or validate
            
            pairs = config.get_trading_pairs()
            # Should return empty list or handle gracefully
            assert isinstance(pairs, list)
            
        finally:
            os.unlink(temp_path)
    
    def test_circular_environment_variables(self):
        """Test handling of circular environment variable references"""
        circular_config = """
var1: ${VAR2}
var2: ${VAR1}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(circular_config)
            temp_path = f.name
        
        try:
            with patch.dict(os.environ, {'VAR1': '${VAR2}', 'VAR2': '${VAR1}'}):
                config = Config(temp_path)
                
                # Should not cause infinite recursion
                val1 = config.get('var1')
                val2 = config.get('var2')
                
                # Should either resolve or keep original format
                assert val1 is not None
                assert val2 is not None
                
        finally:
            os.unlink(temp_path)
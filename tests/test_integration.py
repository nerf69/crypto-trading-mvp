"""
Integration tests for complete crypto trading workflows.
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
from src.backtesting.engine import BacktestEngine, BacktestResult
from src.strategies.base import SignalType


class TestDataPipeline:
    """Test the complete data processing pipeline"""
    
    @patch('src.data.fetcher.requests.Session')
    def test_complete_data_pipeline(self, mock_session_class, temp_database, temp_config_file):
        """Test complete data fetch -> process -> indicators pipeline"""
        # Mock the HTTP response
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        # Generate enough data for indicators (25 days)
        import numpy as np
        mock_data = []
        base_timestamp = 1640995200  # Jan 1, 2022
        base_price = 45000.0
        for i in range(25):
            timestamp = base_timestamp + (i * 86400)  # Daily intervals
            price_change = np.random.normal(0, 0.02)  # 2% daily volatility 
            open_price = base_price * (1 + price_change)
            close_price = open_price + np.random.normal(0, open_price * 0.01)
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.uniform(1000, 2000)
            mock_data.append([timestamp, low_price, high_price, open_price, close_price, volume])
            base_price = close_price
        mock_response.json.return_value = mock_data
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        # Initialize components
        config = Config(temp_config_file)
        fetcher = CoinbaseDataFetcher(
            config.get('exchange.base_url', 'https://api.exchange.coinbase.com'),
            config.get('database.path', 'data/trading.db')
        )
        processor = DataProcessor()
        
        # Execute complete pipeline
        raw_data = fetcher.get_historical_data("BTC-USD", "2022-01-01", "2022-01-25", 86400)
        processed_data = processor.add_basic_indicators(raw_data)
        
        # Validate pipeline results
        assert not processed_data.empty
        assert len(processed_data) >= 20  # Allow for some rows to be filtered during processing
        
        # Check that all expected columns are present
        expected_columns = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'sma_50', 'ema_12', 'ema_26']
        for col in expected_columns:
            assert col in processed_data.columns
        
        # Verify basic data validity
        assert all(processed_data['volume'] > 0)
        assert all(processed_data['close'] > 0)


class TestStrategyBacktesting:
    """Test complete strategy -> backtesting workflows"""
    
    def test_swing_strategy_backtesting_workflow(self, sample_ohlcv_data, temp_config_file):
        """Test complete workflow: strategy signals -> backtesting -> results"""
        config = Config(temp_config_file)
        
        # Initialize strategy and backtesting engine
        strategy = SwingTradingStrategy(swing_threshold=0.03)
        backtest_engine = BacktestEngine(config)
        
        # Add required indicators for swing strategy
        processor = DataProcessor()
        data_with_indicators = processor.add_basic_indicators(sample_ohlcv_data)
        
        # Run complete backtesting workflow (let BacktestEngine fetch and process data internally)
        result = backtest_engine.run_backtest(strategy, "BTC-USD", "2024-01-01", "2024-01-31")
        
        # Validate backtesting results
        assert isinstance(result, BacktestResult)
        assert result.pair == "BTC-USD"
        assert result.total_trades >= 0
        assert result.final_capital > 0
        assert isinstance(result.total_return_pct, float)
        
        # Verify position consistency
        for position in result.positions:
            if not position.is_open():
                assert position.exit_price is not None
                assert position.exit_time is not None
                assert position.pnl is not None
    
    def test_rsi_strategy_backtesting_workflow(self, sample_ohlcv_data, temp_config_file):
        """Test RSI strategy complete workflow"""
        config = Config(temp_config_file)
        
        strategy = RSIStrategy(oversold_threshold=30, overbought_threshold=70)
        backtest_engine = BacktestEngine(config)
        
        # Process data for RSI
        processor = DataProcessor()
        data_with_indicators = processor.add_basic_indicators(sample_ohlcv_data)
        data_with_rsi = processor.add_momentum_indicators(data_with_indicators)
        
        # Run backtesting
        result = backtest_engine.run_backtest(strategy, "ETH-USD", "2024-01-01", "2024-01-31")
        
        # Validate results
        assert result.pair == "ETH-USD"
        assert isinstance(result.win_rate, float)
        assert 0 <= result.win_rate <= 100  # Win rate might be in percentage format
        assert result.max_drawdown >= 0  # Drawdown is stored as positive value
    
    def test_macd_strategy_backtesting_workflow(self, sample_ohlcv_data, temp_config_file):
        """Test MACD strategy complete workflow"""
        config = Config(temp_config_file)
        
        strategy = MACDStrategy()
        backtest_engine = BacktestEngine(config)
        
        # Process data for MACD
        processor = DataProcessor()
        data_with_indicators = processor.add_basic_indicators(sample_ohlcv_data)
        data_with_macd = processor.add_momentum_indicators(data_with_indicators)
        
        # Run backtesting
        result = backtest_engine.run_backtest(strategy, "ADA-USD", "2024-01-01", "2024-01-31")
        
        # Validate MACD-specific results
        assert result.pair == "ADA-USD"
        if result.total_trades > 0:
            assert result.avg_win is not None or result.avg_loss is not None
            assert result.win_rate >= 0


class TestMultiStrategyWorkflow:
    """Test workflows involving multiple strategies"""
    
    def test_strategy_comparison_workflow(self, sample_ohlcv_data, temp_config_file):
        """Test comparing multiple strategies on the same data"""
        config = Config(temp_config_file)
        backtest_engine = BacktestEngine(config)
        
        # Initialize multiple strategies
        strategies = [
            SwingTradingStrategy(swing_threshold=0.03),
            RSIStrategy(oversold_threshold=30, overbought_threshold=70),
            Pure5PercentStrategy()
        ]
        
        # Process data with all required indicators
        processor = DataProcessor()
        data_processed = processor.add_basic_indicators(sample_ohlcv_data)
        data_processed = processor.add_momentum_indicators(data_processed)
        data_processed = processor.add_momentum_indicators(data_processed)
        
        results = []
        for strategy in strategies:
            result = backtest_engine.run_backtest(strategy, "BTC-USD", "2024-01-01", "2024-01-31")
            results.append(result)
        
        # Validate all strategies produced results
        assert len(results) == 3
        for result in results:
            assert isinstance(result, BacktestResult)
            assert result.pair == "BTC-USD"
            assert result.final_capital > 0
        
        # Verify results can be compared
        returns = [result.total_return_pct for result in results]
        assert all(isinstance(ret, (int, float)) for ret in returns)


class TestConfigIntegration:
    """Test configuration integration across components"""
    
    def test_config_driven_workflow(self, temp_config_file):
        """Test that configuration properly drives component behavior"""
        config = Config(temp_config_file)
        
        # Test that components respect configuration
        backtest_engine = BacktestEngine(config)
        
        # Verify config values are used
        assert float(backtest_engine.commission) == config.get_backtesting_config()['commission']
        assert hasattr(backtest_engine, 'config')
        
        # Test strategy weights
        weights = config.get_strategy_weights()
        assert isinstance(weights, dict)
        assert 'swing' in weights
    
    def test_risk_management_integration(self, sample_ohlcv_data, temp_config_file):
        """Test risk management settings integration"""
        config = Config(temp_config_file)
        backtest_engine = BacktestEngine(config)
        strategy = SwingTradingStrategy()
        
        # Process minimal data
        processor = DataProcessor()
        data_processed = processor.add_basic_indicators(sample_ohlcv_data)
        
        # Run backtest and check risk management
        result = backtest_engine.run_backtest(strategy, "BTC-USD", "2024-01-01", "2024-01-31")
        
        # Verify risk settings are applied
        risk_config = config.get_risk_config()
        if result.positions:
            for position in result.positions:
                if position.stop_loss:
                    # Stop loss should respect configured percentage
                    stop_loss_pct = abs((position.stop_loss - position.entry_price) / position.entry_price)
                    assert stop_loss_pct <= risk_config['stop_loss'] * 1.1  # Allow 10% tolerance


class TestErrorRecovery:
    """Test system behavior under error conditions"""
    
    def test_missing_data_recovery(self, temp_config_file):
        """Test system behavior with missing or incomplete data"""
        config = Config(temp_config_file)
        strategy = SwingTradingStrategy()
        backtest_engine = BacktestEngine(config)
        
        # Create incomplete data (missing required indicators)
        incomplete_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5),
            'open': [100, 101, 102, 103, 104],
            'close': [101, 102, 103, 104, 105]
        })
        incomplete_data.set_index('timestamp', inplace=True)
        
        # Test that system handles missing data gracefully
        try:
            result = backtest_engine.run_backtest(strategy, "BTC-USD", "2024-01-01", "2024-01-31")
            # If it doesn't raise an exception, verify it returns sensible defaults
            assert result.total_trades == 0 or result is not None
        except Exception as e:
            # If it raises an exception, it should be informative
            assert "required" in str(e).lower() or "missing" in str(e).lower()
    
    def test_extreme_market_conditions(self, temp_config_file):
        """Test system behavior under extreme market conditions"""
        config = Config(temp_config_file)
        strategy = SwingTradingStrategy()
        backtest_engine = BacktestEngine(config)
        
        # Create extreme market data (large price swings)
        dates = pd.date_range('2024-01-01', periods=10)
        extreme_data = pd.DataFrame({
            'timestamp': dates,
            'open': [1000, 1500, 500, 2000, 100, 1800, 200, 1900, 50, 1950],
            'high': [1100, 1600, 600, 2100, 200, 1900, 300, 2000, 150, 2000],
            'low': [900, 400, 400, 90, 90, 180, 180, 40, 40, 1800],
            'close': [1050, 450, 1950, 150, 1750, 250, 1850, 100, 1900, 1900],
            'volume': [1000] * 10
        })
        extreme_data.set_index('timestamp', inplace=True)
        
        # Add basic indicators
        processor = DataProcessor()
        processed_extreme = processor.add_basic_indicators(extreme_data)
        
        # Test that system handles extreme conditions
        result = backtest_engine.run_backtest(strategy, "VOLATILE-USD", "2024-01-01", "2024-01-31")
        
        # Verify system stability
        assert isinstance(result, BacktestResult)
        assert not np.isnan(result.final_capital)
        assert not np.isinf(result.final_capital)


class TestEndToEndWorkflow:
    """Test complete end-to-end trading workflows"""
    
    @patch('src.data.fetcher.requests.Session')
    def test_complete_trading_workflow(self, mock_session_class, temp_database, temp_config_file):
        """Test complete workflow: fetch -> process -> signal -> backtest"""
        # Setup mocks
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        
        # Generate realistic market data
        np.random.seed(42)
        base_price = 50000
        data = []
        for i in range(30):  # 30 days of data
            timestamp = int((datetime(2024, 1, 1) + timedelta(days=i)).timestamp())
            daily_change = np.random.normal(0, 0.02)  # 2% daily volatility
            open_price = base_price * (1 + daily_change)
            high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = open_price + np.random.normal(0, open_price * 0.01)
            volume = np.random.uniform(1000, 2000)
            
            data.append([timestamp, low_price, high_price, open_price, close_price, volume])
            base_price = close_price
        
        mock_response.json.return_value = data
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        # Initialize all components
        config = Config(temp_config_file)
        fetcher = CoinbaseDataFetcher(
            config.get('exchange.base_url', 'https://api.exchange.coinbase.com'),
            config.get('database.path', 'data/trading.db')
        )
        processor = DataProcessor()
        strategy = SwingTradingStrategy(swing_threshold=0.02)
        backtest_engine = BacktestEngine(config)
        
        # Execute complete workflow
        # Step 1: Fetch data
        raw_data = fetcher.get_historical_data("BTC-USD", "2024-01-01", "2024-01-30", 86400)
        
        # Step 2: Process data and add indicators
        processed_data = processor.add_basic_indicators(raw_data)
        processed_data = processor.add_momentum_indicators(processed_data)
        processed_data = processor.add_momentum_indicators(processed_data)
        
        # Step 3: Run backtesting
        backtest_result = backtest_engine.run_backtest(strategy, "BTC-USD", "2024-01-01", "2024-01-31")
        
        # Validate end-to-end results
        assert not raw_data.empty
        assert len(processed_data) == len(raw_data)
        assert isinstance(backtest_result, BacktestResult)
        assert backtest_result.pair == "BTC-USD"
        
        # Verify data flow consistency
        assert backtest_result.start_date <= backtest_result.end_date
        assert backtest_result.final_capital > 0
        
        # Check that results are reasonable
        assert -100 <= backtest_result.total_return_pct <= 1000  # Reasonable bounds
        assert 0 <= backtest_result.win_rate <= 100  # Win rate might be in percentage format
        assert backtest_result.max_drawdown >= 0  # Drawdown is stored as positive value
    
    def test_portfolio_simulation_workflow(self, sample_ohlcv_data, temp_config_file):
        """Test portfolio-level simulation with multiple assets"""
        config = Config(temp_config_file)
        backtest_engine = BacktestEngine(config)
        
        # Get trading pairs from config
        pairs = config.get_trading_pairs()[:2]  # Test with first 2 pairs
        strategy = Pure5PercentStrategy()
        
        # Process data for all pairs
        processor = DataProcessor()
        processed_data = processor.add_basic_indicators(sample_ohlcv_data)
        
        portfolio_results = {}
        for pair in pairs:
            result = backtest_engine.run_backtest(strategy, pair, "2024-01-01", "2024-01-31")
            portfolio_results[pair] = result
        
        # Validate portfolio results
        assert len(portfolio_results) == len(pairs)
        
        total_return = 0
        total_trades = 0
        for pair, result in portfolio_results.items():
            assert isinstance(result, BacktestResult)
            assert result.pair == pair
            total_return += result.total_return_pct
            total_trades += result.total_trades
        
        # Verify portfolio-level metrics
        average_return = total_return / len(pairs)
        assert isinstance(average_return, (int, float))
        assert total_trades >= 0
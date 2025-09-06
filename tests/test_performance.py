"""
Performance and load tests for crypto trading system.
"""

import pytest
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from unittest.mock import patch, Mock
import tempfile
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import Config
from src.data.fetcher import CoinbaseDataFetcher
from src.data.processor import DataProcessor
from src.strategies.swing import SwingTradingStrategy
from src.strategies.rsi import RSIStrategy
from src.strategies.macd import MACDStrategy
from src.strategies.pure_percent import Pure5PercentStrategy
from src.backtesting.engine import BacktestEngine


class TestDataProcessingPerformance:
    """Test performance of data processing operations"""
    
    def test_large_dataset_processing_time(self):
        """Test processing time for large datasets"""
        processor = DataProcessor()
        
        # Generate large dataset (2 years of daily data)
        large_size = 730  # 2 years
        dates = pd.date_range('2022-01-01', periods=large_size)
        
        np.random.seed(42)
        base_price = 50000
        prices = []
        for i in range(large_size):
            daily_change = np.random.normal(0, 0.02)
            price = base_price * (1 + daily_change)
            prices.append(price)
            base_price = price
        
        large_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'volume': [np.random.uniform(1000, 5000) for _ in range(large_size)]
        })
        large_data.set_index('timestamp', inplace=True)
        
        # Test basic indicators performance
        start_time = time.time()
        result_basic = processor.add_basic_indicators(large_data)
        basic_time = time.time() - start_time
        
        # Should complete within reasonable time (< 5 seconds)
        assert basic_time < 5.0, f"Basic indicators took {basic_time:.2f}s for {large_size} records"
        assert len(result_basic) == large_size
        
        # Test oscillator indicators performance
        start_time = time.time()
        result_oscillator = processor.add_oscillator_indicators(result_basic)
        oscillator_time = time.time() - start_time
        
        assert oscillator_time < 3.0, f"Oscillator indicators took {oscillator_time:.2f}s"
        
        # Test momentum indicators performance
        start_time = time.time()
        result_momentum = processor.add_momentum_indicators(result_oscillator)
        momentum_time = time.time() - start_time
        
        assert momentum_time < 3.0, f"Momentum indicators took {momentum_time:.2f}s"
        
        total_time = basic_time + oscillator_time + momentum_time
        assert total_time < 10.0, f"Total processing took {total_time:.2f}s"
    
    def test_memory_usage_large_dataset(self):
        """Test memory efficiency with large datasets"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        processor = DataProcessor()
        
        # Generate very large dataset (5 years of hourly data)
        large_size = 5 * 365 * 24  # ~43,800 records
        dates = pd.date_range('2019-01-01', periods=large_size, freq='H')
        
        np.random.seed(42)
        large_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(40000, 60000, large_size),
            'high': np.random.uniform(40000, 60000, large_size),
            'low': np.random.uniform(40000, 60000, large_size),
            'close': np.random.uniform(40000, 60000, large_size),
            'volume': np.random.uniform(1000, 5000, large_size)
        })
        large_data.set_index('timestamp', inplace=True)
        
        # Process the large dataset
        result = processor.add_basic_indicators(large_data)
        result = processor.add_oscillator_indicators(result)
        result = processor.add_momentum_indicators(result)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500MB for this dataset)
        assert memory_increase < 500, f"Memory increased by {memory_increase:.2f}MB"
        
        # Cleanup
        del large_data, result
    
    def test_incremental_processing_performance(self):
        """Test performance of incremental data processing"""
        processor = DataProcessor()
        
        # Start with base dataset
        base_size = 100
        dates = pd.date_range('2024-01-01', periods=base_size)
        base_data = pd.DataFrame({
            'timestamp': dates,
            'open': range(100, 100 + base_size),
            'high': range(105, 105 + base_size),
            'low': range(95, 95 + base_size),
            'close': range(104, 104 + base_size),
            'volume': [1000] * base_size
        })
        base_data.set_index('timestamp', inplace=True)
        
        # Process base data
        processed_base = processor.add_basic_indicators(base_data)
        
        # Test incremental additions
        increment_times = []
        for i in range(10):  # Add 10 incremental updates
            new_row = pd.DataFrame({
                'timestamp': [dates[-1] + timedelta(days=i+1)],
                'open': [100 + base_size + i],
                'high': [105 + base_size + i],
                'low': [95 + base_size + i],
                'close': [104 + base_size + i],
                'volume': [1000]
            })
            new_row.set_index('timestamp', inplace=True)
            
            combined_data = pd.concat([processed_base, new_row])
            
            start_time = time.time()
            result = processor.add_basic_indicators(combined_data)
            increment_time = time.time() - start_time
            increment_times.append(increment_time)
        
        # Incremental updates should be fast (< 0.1s each)
        avg_increment_time = np.mean(increment_times)
        assert avg_increment_time < 0.1, f"Average incremental time: {avg_increment_time:.3f}s"


class TestStrategyPerformance:
    """Test performance of trading strategies"""
    
    def test_strategy_signal_calculation_speed(self):
        """Test speed of signal calculations across strategies"""
        # Generate test data
        test_size = 1000
        dates = pd.date_range('2022-01-01', periods=test_size)
        
        np.random.seed(42)
        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(40000, 60000, test_size),
            'high': np.random.uniform(40000, 60000, test_size),
            'low': np.random.uniform(40000, 60000, test_size),
            'close': np.random.uniform(40000, 60000, test_size),
            'volume': np.random.uniform(1000, 5000, test_size)
        })
        test_data.set_index('timestamp', inplace=True)
        
        # Add all indicators
        processor = DataProcessor()
        processed_data = processor.add_basic_indicators(test_data)
        processed_data = processor.add_oscillator_indicators(processed_data)
        processed_data = processor.add_momentum_indicators(processed_data)
        
        strategies = [
            ('Swing', SwingTradingStrategy()),
            ('RSI', RSIStrategy()),
            ('MACD', MACDStrategy()),
            ('Pure5%', Pure5PercentStrategy())
        ]
        
        for name, strategy in strategies:
            # Test multiple signal calculations
            times = []
            for i in range(50, min(test_size, 100), 5):  # Test on subsets
                subset = processed_data.iloc[:i]
                
                start_time = time.time()
                signal = strategy.calculate_signal(subset, "BTC-USD")
                calc_time = time.time() - start_time
                times.append(calc_time)
            
            avg_time = np.mean(times)
            max_time = np.max(times)
            
            # Each signal calculation should be very fast (< 0.01s average, < 0.05s max)
            assert avg_time < 0.01, f"{name} strategy avg time: {avg_time:.4f}s"
            assert max_time < 0.05, f"{name} strategy max time: {max_time:.4f}s"
    
    def test_concurrent_strategy_execution(self):
        """Test concurrent execution of multiple strategies"""
        # Generate test data
        test_size = 500
        dates = pd.date_range('2023-01-01', periods=test_size)
        
        np.random.seed(42)
        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(40000, 60000, test_size),
            'high': np.random.uniform(40000, 60000, test_size),
            'low': np.random.uniform(40000, 60000, test_size),
            'close': np.random.uniform(40000, 60000, test_size),
            'volume': np.random.uniform(1000, 5000, test_size)
        })
        test_data.set_index('timestamp', inplace=True)
        
        processor = DataProcessor()
        processed_data = processor.add_basic_indicators(test_data)
        processed_data = processor.add_oscillator_indicators(processed_data)
        processed_data = processor.add_momentum_indicators(processed_data)
        
        strategies = [
            SwingTradingStrategy(),
            RSIStrategy(),
            MACDStrategy(),
            Pure5PercentStrategy()
        ]
        
        def run_strategy(strategy, data, pair):
            """Run a strategy and return timing info"""
            start_time = time.time()
            signal = strategy.calculate_signal(data, pair)
            return time.time() - start_time, signal
        
        # Test sequential execution
        start_time = time.time()
        sequential_results = []
        for strategy in strategies:
            timing, signal = run_strategy(strategy, processed_data, "BTC-USD")
            sequential_results.append((timing, signal))
        sequential_time = time.time() - start_time
        
        # Test concurrent execution
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(run_strategy, strategy, processed_data, "BTC-USD")
                for strategy in strategies
            ]
            concurrent_results = [future.result() for future in as_completed(futures)]
        concurrent_time = time.time() - start_time
        
        # Concurrent execution should be faster or similar
        assert len(concurrent_results) == len(sequential_results)
        # Allow some overhead but expect speedup or at least not much slower
        assert concurrent_time <= sequential_time * 1.5


class TestBacktestingPerformance:
    """Test performance of backtesting operations"""
    
    def test_backtesting_large_dataset_performance(self, temp_config_file):
        """Test backtesting performance with large datasets"""
        config = Config(temp_config_file)
        backtest_engine = BacktestEngine(config)
        
        # Generate large dataset (1 year of daily data)
        test_size = 365
        dates = pd.date_range('2023-01-01', periods=test_size)
        
        np.random.seed(42)
        base_price = 50000
        prices = []
        for i in range(test_size):
            daily_change = np.random.normal(0, 0.02)  # 2% daily volatility
            price = base_price * (1 + daily_change)
            prices.append(price)
            base_price = price
        
        large_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.03 for p in prices],
            'low': [p * 0.97 for p in prices],
            'close': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'volume': [np.random.uniform(1000, 5000) for _ in range(test_size)]
        })
        large_data.set_index('timestamp', inplace=True)
        
        # Add indicators
        processor = DataProcessor()
        processed_data = processor.add_basic_indicators(large_data)
        processed_data = processor.add_oscillator_indicators(processed_data)
        
        strategies = [
            ('Swing', SwingTradingStrategy()),
            ('RSI', RSIStrategy()),
            ('MACD', MACDStrategy())
        ]
        
        for name, strategy in strategies:
            start_time = time.time()
            result = backtest_engine.run_backtest(strategy, processed_data, "BTC-USD")
            backtest_time = time.time() - start_time
            
            # Backtesting should complete within reasonable time (< 10s for 365 days)
            assert backtest_time < 10.0, f"{name} backtesting took {backtest_time:.2f}s"
            
            # Verify results are valid
            assert result.total_trades >= 0
            assert isinstance(result.final_portfolio_value, (int, float))
    
    def test_multiple_pair_backtesting_performance(self, temp_config_file):
        """Test performance when backtesting multiple pairs"""
        config = Config(temp_config_file)
        backtest_engine = BacktestEngine(config)
        strategy = SwingTradingStrategy()
        
        # Generate test data
        test_size = 200
        dates = pd.date_range('2023-01-01', periods=test_size)
        
        np.random.seed(42)
        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(40000, 60000, test_size),
            'high': np.random.uniform(40000, 60000, test_size),
            'low': np.random.uniform(40000, 60000, test_size),
            'close': np.random.uniform(40000, 60000, test_size),
            'volume': np.random.uniform(1000, 5000, test_size)
        })
        test_data.set_index('timestamp', inplace=True)
        
        processor = DataProcessor()
        processed_data = processor.add_basic_indicators(test_data)
        
        pairs = ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD"]
        
        # Test sequential backtesting
        start_time = time.time()
        sequential_results = {}
        for pair in pairs:
            result = backtest_engine.run_backtest(strategy, processed_data, pair)
            sequential_results[pair] = result
        sequential_time = time.time() - start_time
        
        # Test concurrent backtesting
        def run_backtest_for_pair(pair):
            return pair, backtest_engine.run_backtest(strategy, processed_data, pair)
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_backtest_for_pair, pair) for pair in pairs]
            concurrent_results = {pair: result for pair, result in [f.result() for f in futures]}
        concurrent_time = time.time() - start_time
        
        # Verify results
        assert len(sequential_results) == len(pairs)
        assert len(concurrent_results) == len(pairs)
        
        # Performance should be reasonable
        assert sequential_time < 30.0, f"Sequential backtesting took {sequential_time:.2f}s"
        assert concurrent_time < sequential_time * 0.8, f"Concurrent should be faster: {concurrent_time:.2f}s vs {sequential_time:.2f}s"


class TestDatabasePerformance:
    """Test database operation performance"""
    
    @patch('src.data.fetcher.requests.Session')
    def test_database_write_performance(self, mock_session_class, temp_database, temp_config_file):
        """Test database write performance with large datasets"""
        # Mock API response with large dataset
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        
        # Generate large mock response (1000 records)
        large_mock_data = []
        base_timestamp = int(datetime(2023, 1, 1).timestamp())
        for i in range(1000):
            timestamp = base_timestamp + (i * 86400)  # Daily intervals
            price = 50000 + np.random.normal(0, 1000)
            large_mock_data.append([
                timestamp, price, price * 1.02, price * 0.98, 
                price + np.random.normal(0, 500), np.random.uniform(1000, 5000)
            ])
        
        mock_response.json.return_value = large_mock_data
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        config = Config(temp_config_file)
        fetcher = CoinbaseDataFetcher(config)
        
        # Test database write performance
        start_time = time.time()
        result = fetcher.get_historical_data("BTC-USD", "2023-01-01", "2025-09-27", "1d")
        write_time = time.time() - start_time
        
        # Database writes should be efficient (< 5s for 1000 records)
        assert write_time < 5.0, f"Database write took {write_time:.2f}s"
        assert len(result) == 1000
        
        # Test subsequent read performance (should use cache)
        start_time = time.time()
        cached_result = fetcher.get_historical_data("BTC-USD", "2023-01-01", "2025-09-27", "1d")
        read_time = time.time() - start_time
        
        # Cached read should be very fast (< 0.5s)
        assert read_time < 0.5, f"Cached read took {read_time:.2f}s"
        assert len(cached_result) == len(result)
    
    def test_database_concurrent_access(self, temp_database, temp_config_file):
        """Test database performance under concurrent access"""
        config = Config(temp_config_file)
        
        def create_fetcher_and_query():
            """Create a fetcher and perform a query"""
            fetcher = CoinbaseDataFetcher(config)
            # This will create/access the database
            return fetcher
        
        # Test concurrent database access
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_fetcher_and_query) for _ in range(10)]
            results = [future.result() for future in as_completed(futures)]
        concurrent_time = time.time() - start_time
        
        # Should handle concurrent access gracefully
        assert len(results) == 10
        assert concurrent_time < 10.0, f"Concurrent database access took {concurrent_time:.2f}s"


class TestMemoryLeakDetection:
    """Test for memory leaks in long-running operations"""
    
    def test_repeated_backtesting_memory_stability(self, temp_config_file):
        """Test memory stability during repeated backtesting"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        config = Config(temp_config_file)
        backtest_engine = BacktestEngine(config)
        strategy = SwingTradingStrategy()
        
        # Generate test data once
        test_size = 100
        dates = pd.date_range('2023-01-01', periods=test_size)
        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': range(100, 100 + test_size),
            'high': range(105, 105 + test_size),
            'low': range(95, 95 + test_size),
            'close': range(104, 104 + test_size),
            'volume': [1000] * test_size
        })
        test_data.set_index('timestamp', inplace=True)
        
        processor = DataProcessor()
        processed_data = processor.add_basic_indicators(test_data)
        
        memory_measurements = []
        
        # Run many backtests
        for i in range(50):
            result = backtest_engine.run_backtest(strategy, processed_data, f"TEST-{i}")
            
            # Measure memory every 10 iterations
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_measurements.append(current_memory - initial_memory)
        
        # Memory should not continuously grow (no major leaks)
        # Allow some growth but not excessive
        final_memory_increase = memory_measurements[-1]
        assert final_memory_increase < 100, f"Memory increased by {final_memory_increase:.2f}MB"
        
        # Memory should stabilize (last measurement shouldn't be much higher than middle)
        if len(memory_measurements) >= 3:
            middle_memory = memory_measurements[len(memory_measurements)//2]
            memory_growth_rate = (final_memory_increase - middle_memory) / (len(memory_measurements)//2)
            assert memory_growth_rate < 5, f"Memory growing at {memory_growth_rate:.2f}MB per 10 iterations"
    
    def test_data_processing_memory_cleanup(self):
        """Test memory cleanup in data processing operations"""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        processor = DataProcessor()
        
        for i in range(20):
            # Generate data
            test_size = 500
            dates = pd.date_range('2023-01-01', periods=test_size)
            test_data = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.uniform(40000, 60000, test_size),
                'high': np.random.uniform(40000, 60000, test_size),
                'low': np.random.uniform(40000, 60000, test_size),
                'close': np.random.uniform(40000, 60000, test_size),
                'volume': np.random.uniform(1000, 5000, test_size)
            })
            test_data.set_index('timestamp', inplace=True)
            
            # Process data
            result = processor.add_basic_indicators(test_data)
            result = processor.add_oscillator_indicators(result)
            
            # Explicitly delete to test cleanup
            del test_data, result
            
            # Force garbage collection every 5 iterations
            if i % 5 == 0:
                gc.collect()
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal after cleanup
        assert memory_increase < 50, f"Memory increased by {memory_increase:.2f}MB after processing"


class TestScalabilityBounds:
    """Test system behavior at scale boundaries"""
    
    def test_maximum_dataset_size_handling(self):
        """Test handling of very large datasets"""
        processor = DataProcessor()
        
        # Test with maximum reasonable size (10 years of daily data)
        max_size = 10 * 365  # 3,650 records
        dates = pd.date_range('2014-01-01', periods=max_size)
        
        np.random.seed(42)
        max_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(1000, 60000, max_size),
            'high': np.random.uniform(1000, 60000, max_size),
            'low': np.random.uniform(1000, 60000, max_size),
            'close': np.random.uniform(1000, 60000, max_size),
            'volume': np.random.uniform(100, 10000, max_size)
        })
        max_data.set_index('timestamp', inplace=True)
        
        # Should handle large dataset without crashing
        start_time = time.time()
        try:
            result = processor.add_basic_indicators(max_data)
            result = processor.add_oscillator_indicators(result)
            processing_time = time.time() - start_time
            
            assert len(result) == max_size
            assert processing_time < 30.0, f"Processing {max_size} records took {processing_time:.2f}s"
            
        except MemoryError:
            pytest.skip(f"Insufficient memory to process {max_size} records")
        except Exception as e:
            pytest.fail(f"Failed to process large dataset: {e}")
    
    def test_concurrent_load_limits(self, temp_config_file):
        """Test system behavior under high concurrent load"""
        config = Config(temp_config_file)
        
        # Generate test data
        test_size = 100
        dates = pd.date_range('2023-01-01', periods=test_size)
        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': range(100, 100 + test_size),
            'high': range(105, 105 + test_size),
            'low': range(95, 95 + test_size),
            'close': range(104, 104 + test_size),
            'volume': [1000] * test_size
        })
        test_data.set_index('timestamp', inplace=True)
        
        processor = DataProcessor()
        processed_data = processor.add_basic_indicators(test_data)
        
        def run_concurrent_operations():
            """Run multiple operations concurrently"""
            backtest_engine = BacktestEngine(config)
            strategy = SwingTradingStrategy()
            return backtest_engine.run_backtest(strategy, processed_data, "BTC-USD")
        
        # Test with high concurrency
        max_workers = 10
        start_time = time.time()
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_concurrent_operations) for _ in range(20)]
                results = [future.result() for future in as_completed(futures)]
            
            concurrent_time = time.time() - start_time
            
            assert len(results) == 20
            assert all(result is not None for result in results)
            assert concurrent_time < 60.0, f"High concurrency test took {concurrent_time:.2f}s"
            
        except Exception as e:
            pytest.fail(f"System failed under high concurrent load: {e}")
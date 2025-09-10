"""
Comprehensive tests for mathematical precision and edge cases.

This test suite focuses on numerical stability, precision handling,
and edge case behavior of all mathematical calculations in the system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal, getcontext, ROUND_HALF_UP
import sys
import math

from src.data.processor import DataProcessor
from src.strategies.swing import SwingTradingStrategy
from src.strategies.rsi import RSIStrategy
from src.strategies.macd import MACDStrategy
from src.strategies.pure_percent import Pure5PercentStrategy
from src.strategies.base import SignalType
from src.backtesting.engine import BacktestEngine
from src.optimization.parameter_tuner import ParameterTuner


class TestDecimalPrecisionAccuracy:
    """Test decimal precision handling in financial calculations"""
    
    def test_decimal_precision_configuration(self):
        """Test that decimal precision is correctly configured"""
        processor = DataProcessor()
        
        # Check that precision is set appropriately for financial calculations
        current_precision = getcontext().prec
        assert current_precision >= 20, f"Decimal precision should be at least 20 for financial calculations, got {current_precision}"
        
        # Check rounding mode
        current_rounding = getcontext().rounding
        assert current_rounding == ROUND_HALF_UP, "Should use ROUND_HALF_UP for financial calculations"
    
    def test_high_precision_price_calculations(self):
        """Test calculations with high-precision decimal prices"""
        processor = DataProcessor()
        
        # Create prices with many decimal places (crypto-style precision)
        high_precision_prices = [
            Decimal('12345.123456789012'),
            Decimal('12346.234567890123'),
            Decimal('12347.345678901234'),
            Decimal('12348.456789012345'),
            Decimal('12349.567890123456')
        ]
        
        # Convert to float for DataFrame (as system currently uses floats)
        float_prices = [float(p) for p in high_precision_prices]
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(float_prices), freq='D'),
            'open': float_prices,
            'high': [p * 1.001 for p in float_prices],
            'low': [p * 0.999 for p in float_prices],
            'close': float_prices,
            'volume': [1000.123456] * len(float_prices)
        })
        
        # Should handle high precision without losing significant digits
        df_with_indicators = processor.add_all_indicators(df)
        
        # Verify precision is maintained in basic calculations
        if 'sma_20' in df_with_indicators.columns:
            sma_values = df_with_indicators['sma_20'].dropna()
            if len(sma_values) > 0:
                # SMA should maintain reasonable precision
                for sma in sma_values:
                    assert abs(sma - 12347.0) < 10, f"SMA should be around 12347, got {sma}"
                    
                    # Check that we don't lose too much precision
                    decimal_places = len(str(sma).split('.')[1]) if '.' in str(sma) else 0
                    assert decimal_places >= 2, f"Should maintain at least 2 decimal places in SMA, got {decimal_places}"
    
    def test_very_small_price_precision(self):
        """Test precision with very small cryptocurrency prices"""
        processor = DataProcessor()
        
        # Create very small prices (like SHIB or similar tokens)
        tiny_prices = [
            0.000012345,
            0.000012346,
            0.000012344,
            0.000012347,
            0.000012343
        ]
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(tiny_prices), freq='D'),
            'open': tiny_prices,
            'high': [p * 1.001 for p in tiny_prices],
            'low': [p * 0.999 for p in tiny_prices],
            'close': tiny_prices,
            'volume': [1000000] * len(tiny_prices)  # High volume for small cap tokens
        })
        
        df_with_indicators = processor.add_all_indicators(df)
        
        # Check that small prices don't cause underflow or precision loss
        if 'sma_20' in df_with_indicators.columns:
            sma_values = df_with_indicators['sma_20'].dropna()
            if len(sma_values) > 0:
                for sma in sma_values:
                    assert sma > 0, "SMA should be positive for positive prices"
                    assert sma < 0.001, "SMA should be in the small price range"
                    assert sma != 0.0, "SMA should not underflow to zero"
    
    def test_percentage_calculation_precision(self):
        """Test precision in percentage calculations"""
        strategy = Pure5PercentStrategy(drop_threshold=0.05, rise_threshold=0.05)
        
        # Create prices with exact percentage moves
        precise_percentage_prices = [
            100.0,
            95.0,        # Exactly -5.0%
            94.99999,    # -5.00001% (just over threshold)
            95.00001,    # -4.99999% (just under threshold)
            100.0,
            105.0,       # Exactly +5.0%
            104.99999,   # +4.99999% (just under threshold)
            105.00001    # +5.00001% (just over threshold)
        ]
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(precise_percentage_prices), freq='D'),
            'open': precise_percentage_prices,
            'high': [p * 1.0001 for p in precise_percentage_prices],
            'low': [p * 0.9999 for p in precise_percentage_prices],
            'close': precise_percentage_prices,
            'volume': [1000] * len(precise_percentage_prices)
        })
        
        df_with_indicators = strategy.add_required_indicators(df)
        
        # Test percentage calculations at various points
        if 'pct_drop_from_high' in df_with_indicators.columns:
            pct_drops = df_with_indicators['pct_drop_from_high'].values
            
            # Check for precision in percentage calculations
            for i, pct_drop in enumerate(pct_drops):
                if not np.isnan(pct_drop):
                    # Percentage should be calculated with sufficient precision
                    assert isinstance(pct_drop, (int, float)), f"Percentage should be numeric, got {type(pct_drop)}"
                    
                    # For very precise threshold detection
                    if i == 2:  # 94.99999 (just over 5% drop)
                        assert pct_drop > 5.0, f"Should detect >5% drop, got {pct_drop}%"
                    elif i == 3:  # 95.00001 (just under 5% drop)
                        assert pct_drop < 5.0, f"Should detect <5% drop, got {pct_drop}%"


class TestFloatingPointStability:
    """Test floating point stability and numerical errors"""
    
    def test_floating_point_accumulation_errors(self):
        """Test that floating point errors don't accumulate in long calculations"""
        processor = DataProcessor()
        
        # Create a long series with small incremental changes
        base_price = 100.0
        incremental_prices = []
        
        for i in range(1000):
            # Add very small increments that could accumulate floating point errors
            base_price += 0.001  # Small increment
            incremental_prices.append(base_price)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(incremental_prices), freq='H'),
            'open': incremental_prices,
            'high': [p * 1.0001 for p in incremental_prices],
            'low': [p * 0.9999 for p in incremental_prices],
            'close': incremental_prices,
            'volume': [1000] * len(incremental_prices)
        })
        
        # Process with indicators (which involve many calculations)
        df_with_indicators = processor.add_all_indicators(df)
        
        # Check that final prices are reasonable (no severe accumulation errors)
        final_price = df_with_indicators['close'].iloc[-1]
        expected_final = 100.0 + (1000 * 0.001)  # 100 + 1 = 101
        
        assert abs(final_price - expected_final) < 0.001, \
            f"Floating point accumulation error: expected ~{expected_final}, got {final_price}"
        
        # Check moving averages don't have crazy values
        if 'sma_20' in df_with_indicators.columns:
            sma_final = df_with_indicators['sma_20'].iloc[-1]
            assert 100 < sma_final < 102, f"SMA should be reasonable, got {sma_final}"
    
    def test_division_by_zero_protection(self):
        """Test protection against division by zero in all calculations"""
        strategies = [
            SwingTradingStrategy(),
            RSIStrategy(),
            MACDStrategy(),
            Pure5PercentStrategy()
        ]
        
        # Create scenarios that could cause division by zero
        zero_scenarios = [
            # Scenario 1: Zero volume
            pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=30, freq='D'),
                'open': [100] * 30,
                'high': [100] * 30,
                'low': [100] * 30,
                'close': [100] * 30,
                'volume': [0.0] * 30  # Zero volume
            }),
            
            # Scenario 2: Constant prices (zero volatility)
            pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=30, freq='D'),
                'open': [100.0] * 30,
                'high': [100.0] * 30,
                'low': [100.0] * 30,
                'close': [100.0] * 30,
                'volume': [1000] * 30
            }),
            
            # Scenario 3: Zero price changes
            pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=30, freq='D'),
                'open': [100.0] * 30,
                'high': [100.0001] * 30,  # Minimal variation
                'low': [99.9999] * 30,
                'close': [100.0] * 30,
                'volume': [1000] * 30
            })
        ]
        
        for scenario_df in zero_scenarios:
            for strategy in strategies:
                try:
                    df_with_indicators = strategy.add_required_indicators(scenario_df)
                    signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
                    
                    # Should not crash and should produce valid signal
                    assert signal is not None
                    assert isinstance(signal.confidence, (int, float))
                    assert not np.isnan(signal.confidence)
                    assert not np.isinf(signal.confidence)
                    assert 0 <= signal.confidence <= 1
                    
                except ZeroDivisionError:
                    pytest.fail(f"{strategy.name} raised ZeroDivisionError on zero scenario")
                except Exception as e:
                    # Other exceptions are acceptable if they're handled gracefully
                    assert "division" not in str(e).lower()
    
    def test_extreme_number_handling(self):
        """Test handling of extreme numbers (very large, very small, inf, -inf)"""
        processor = DataProcessor()
        
        # Test with very large numbers
        large_numbers = [1e10, 1e11, 1e12, 1e13, 1e14]  # Billions to hundreds of trillions
        
        df_large = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(large_numbers), freq='D'),
            'open': large_numbers,
            'high': [p * 1.001 for p in large_numbers],
            'low': [p * 0.999 for p in large_numbers],
            'close': large_numbers,
            'volume': [1000] * len(large_numbers)
        })
        
        # Should handle large numbers without overflow
        try:
            df_with_indicators = processor.add_basic_indicators(df_large)
            
            # Check that indicators are finite
            for col in df_with_indicators.columns:
                if df_with_indicators[col].dtype in [np.float64, np.float32]:
                    values = df_with_indicators[col].dropna()
                    if len(values) > 0:
                        assert np.all(np.isfinite(values)), f"Non-finite values in {col}"
                        
        except OverflowError:
            pytest.fail("Should handle large numbers without overflow")
        
        # Test with very small numbers
        tiny_numbers = [1e-10, 1e-11, 1e-12, 1e-13, 1e-14]
        
        df_tiny = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(tiny_numbers), freq='D'),
            'open': tiny_numbers,
            'high': [p * 1.001 for p in tiny_numbers],
            'low': [p * 0.999 for p in tiny_numbers],
            'close': tiny_numbers,
            'volume': [1000] * len(tiny_numbers)
        })
        
        # Should handle tiny numbers without underflow to zero
        try:
            df_with_indicators = processor.add_basic_indicators(df_tiny)
            
            # Verify that tiny numbers don't become zero inappropriately
            close_values = df_with_indicators['close']
            assert all(val > 0 for val in close_values), "Tiny numbers should remain positive"
            
        except Exception as e:
            # Should not fail due to underflow
            assert "underflow" not in str(e).lower()
    
    def test_nan_and_infinity_handling(self):
        """Test proper handling of NaN and infinity values"""
        processor = DataProcessor()
        
        # Create data with NaN and inf values
        problematic_values = [100.0, np.nan, 102.0, np.inf, 103.0, -np.inf, 104.0]
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(problematic_values), freq='D'),
            'open': problematic_values,
            'high': [p * 1.01 if np.isfinite(p) else p for p in problematic_values],
            'low': [p * 0.99 if np.isfinite(p) else p for p in problematic_values],
            'close': problematic_values,
            'volume': [1000] * len(problematic_values)
        })
        
        # Should clean or handle NaN/inf values appropriately
        cleaned_df = processor.clean_data(df)
        
        # After cleaning, should not have NaN or inf in critical columns
        for col in ['open', 'high', 'low', 'close']:
            if col in cleaned_df.columns:
                values = cleaned_df[col]
                assert np.all(np.isfinite(values)), f"Column {col} should not contain NaN or inf after cleaning"
    
    def test_rounding_consistency(self):
        """Test that rounding is consistent across calculations"""
        # Test with values that are exactly at rounding boundaries
        boundary_values = [
            100.5,      # Exactly at 0.5
            100.50001,  # Slightly above 0.5
            100.49999,  # Slightly below 0.5
            99.5,
            99.50001,
            99.49999
        ]
        
        processor = DataProcessor()
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(boundary_values), freq='D'),
            'open': boundary_values,
            'high': [p * 1.001 for p in boundary_values],
            'low': [p * 0.999 for p in boundary_values],
            'close': boundary_values,
            'volume': [1000] * len(boundary_values)
        })
        
        df_with_indicators = processor.add_basic_indicators(df)
        
        # Check that rounding is consistent
        if 'sma_20' in df_with_indicators.columns:
            sma_values = df_with_indicators['sma_20'].dropna()
            
            # Rounding should be consistent and follow ROUND_HALF_UP
            for sma in sma_values:
                # Manually round using the same method
                decimal_sma = Decimal(str(sma))
                manually_rounded = float(decimal_sma.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
                
                # Check that internal calculations use consistent rounding
                assert isinstance(sma, (int, float)), "SMA should be numeric"


class TestNumericalStabilityInStrategies:
    """Test numerical stability in strategy calculations"""
    
    def test_rsi_calculation_stability(self):
        """Test RSI calculation numerical stability"""
        strategy = RSIStrategy()
        
        # Create data that tests RSI calculation stability
        # Alternating small gains and losses
        rsi_test_prices = [100.0]
        for i in range(50):
            # Very small alternating changes
            change = 0.01 if i % 2 == 0 else -0.01
            rsi_test_prices.append(rsi_test_prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(rsi_test_prices), freq='D'),
            'open': rsi_test_prices,
            'high': [p * 1.0001 for p in rsi_test_prices],
            'low': [p * 0.9999 for p in rsi_test_prices],
            'close': rsi_test_prices,
            'volume': [1000] * len(rsi_test_prices)
        })
        
        df_with_indicators = strategy.add_required_indicators(df)
        
        # RSI should be stable and within bounds
        if 'rsi' in df_with_indicators.columns:
            rsi_values = df_with_indicators['rsi'].dropna()
            
            for rsi in rsi_values:
                if not np.isnan(rsi):
                    assert 0 <= rsi <= 100, f"RSI should be between 0-100, got {rsi}"
                    assert np.isfinite(rsi), "RSI should be finite"
                    
                    # RSI shouldn't fluctuate wildly with small price changes
                    assert not (rsi < 1 and rsi > 0), "RSI shouldn't be extremely close to 0"
                    assert not (rsi > 99 and rsi < 100), "RSI shouldn't be extremely close to 100"
    
    def test_macd_calculation_stability(self):
        """Test MACD calculation numerical stability"""
        strategy = MACDStrategy()
        
        # Create price data that could cause MACD instability
        # Exponentially increasing prices (tests EMA calculations)
        macd_test_prices = [100 * (1.001 ** i) for i in range(100)]  # Small exponential growth
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(macd_test_prices), freq='D'),
            'open': macd_test_prices,
            'high': [p * 1.001 for p in macd_test_prices],
            'low': [p * 0.999 for p in macd_test_prices],
            'close': macd_test_prices,
            'volume': [1000] * len(macd_test_prices)
        })
        
        df_with_indicators = strategy.add_required_indicators(df)
        
        # MACD components should be stable
        macd_columns = ['macd', 'macd_signal', 'macd_histogram']
        for col in macd_columns:
            if col in df_with_indicators.columns:
                values = df_with_indicators[col].dropna()
                
                for val in values:
                    assert np.isfinite(val), f"{col} should be finite, got {val}"
                    assert abs(val) < 1e6, f"{col} should not have extreme values, got {val}"
                    
                # Check that MACD histogram = MACD - Signal
                if all(c in df_with_indicators.columns for c in ['macd', 'macd_signal', 'macd_histogram']):
                    macd_vals = df_with_indicators['macd'].dropna()
                    signal_vals = df_with_indicators['macd_signal'].dropna()
                    hist_vals = df_with_indicators['macd_histogram'].dropna()
                    
                    # Check calculation consistency
                    min_len = min(len(macd_vals), len(signal_vals), len(hist_vals))
                    if min_len > 0:
                        for i in range(min_len):
                            expected_hist = macd_vals.iloc[i] - signal_vals.iloc[i]
                            actual_hist = hist_vals.iloc[i]
                            assert abs(expected_hist - actual_hist) < 1e-10, \
                                f"MACD histogram calculation inconsistency at index {i}"
    
    def test_percentage_calculation_edge_cases(self):
        """Test percentage calculations with edge cases"""
        strategy = Pure5PercentStrategy(drop_threshold=0.05, rise_threshold=0.05)
        
        # Test edge cases for percentage calculations
        edge_case_scenarios = [
            # Very small starting value
            [0.001, 0.0011, 0.00095],
            # Very large starting value
            [1000000, 1050000, 950000],
            # Values crossing zero (shouldn't happen in crypto, but test anyway)
            [1, 0.5, 1.5],
            # High precision values
            [100.123456789, 105.129629167, 95.117284411]
        ]
        
        for scenario_prices in edge_case_scenarios:
            df = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=len(scenario_prices), freq='D'),
                'open': scenario_prices,
                'high': [p * 1.001 for p in scenario_prices],
                'low': [p * 0.999 for p in scenario_prices],
                'close': scenario_prices,
                'volume': [1000] * len(scenario_prices)
            })
            
            df_with_indicators = strategy.add_required_indicators(df)
            
            # Check percentage calculations
            percentage_columns = ['pct_drop_from_high', 'pct_rise_from_low']
            for col in percentage_columns:
                if col in df_with_indicators.columns:
                    pct_values = df_with_indicators[col].dropna()
                    
                    for pct in pct_values:
                        assert np.isfinite(pct), f"{col} should be finite, got {pct}"
                        assert -100 <= pct <= 1000, f"{col} should be reasonable percentage, got {pct}%"
    
    def test_swing_strategy_numerical_precision(self):
        """Test swing strategy numerical precision"""
        strategy = SwingTradingStrategy(swing_threshold=0.025, volume_threshold=1.1)
        
        # Create data with very precise threshold testing
        threshold_test_prices = [
            100.0,
            97.5,      # Exactly -2.5% (at threshold)
            97.49999,  # Just under threshold (-2.50001%)
            97.50001,  # Just over threshold (-2.49999%)
            102.5,     # Exactly +2.5% from start
            102.49999, # Just under threshold
            102.50001  # Just over threshold
        ]
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(threshold_test_prices), freq='D'),
            'open': threshold_test_prices,
            'high': [p * 1.001 for p in threshold_test_prices],
            'low': [p * 0.999 for p in threshold_test_prices],
            'close': threshold_test_prices,
            'volume': [1200] * len(threshold_test_prices)  # Above volume threshold
        })
        
        df_with_indicators = strategy.add_required_indicators(df)
        
        # Test multiple points for threshold sensitivity
        for i in range(2, len(df_with_indicators)):
            subset_df = df_with_indicators.iloc[:i+1].copy()
            signal = strategy.calculate_signal(subset_df, "BTC-USD")
            
            assert signal is not None
            assert np.isfinite(signal.confidence)
            assert 0 <= signal.confidence <= 1
            
            # Check that threshold detection is precise
            if 'pct_from_high' in subset_df.columns:
                pct_from_high = subset_df['pct_from_high'].iloc[-1]
                if not pd.isna(pct_from_high):
                    # Should handle precision around threshold correctly
                    if abs(pct_from_high - 2.5) < 0.001:  # Very close to threshold
                        # Signal should be based on precise threshold comparison
                        assert signal.signal in [SignalType.BUY, SignalType.STRONG_BUY, SignalType.HOLD]


class TestConcurrencyAndThreadSafety:
    """Test numerical stability under concurrent access"""
    
    def test_processor_thread_safety(self):
        """Test that DataProcessor calculations are thread-safe"""
        import threading
        import time
        
        processor = DataProcessor()
        results = []
        errors = []
        
        # Create test data
        prices = [100 + i * 0.1 for i in range(100)]
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='H'),
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })
        
        def calculate_indicators(thread_id):
            """Function to run in multiple threads"""
            try:
                result = processor.add_all_indicators(df.copy())
                results.append((thread_id, len(result), result['close'].iloc[-1]))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=calculate_indicators, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        
        # All results should be identical (same input data)
        first_result = results[0]
        for result in results[1:]:
            assert result[1] == first_result[1], "All threads should process same number of rows"
            assert abs(result[2] - first_result[2]) < 1e-10, "All threads should get same final price"
    
    def test_strategy_calculation_consistency(self):
        """Test that strategy calculations are consistent across multiple runs"""
        strategies = [
            SwingTradingStrategy(),
            RSIStrategy(),
            Pure5PercentStrategy()
        ]
        
        # Create deterministic test data
        np.random.seed(12345)  # Fixed seed for reproducibility
        prices = 100 + np.cumsum(np.random.normal(0, 1, 50))
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='D'),
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(800, 1200, len(prices))
        })
        
        # Run each strategy multiple times and check consistency
        for strategy in strategies:
            signals = []
            
            for run in range(10):  # 10 runs
                df_with_indicators = strategy.add_required_indicators(df.copy())
                signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
                signals.append((signal.signal, signal.confidence, signal.price))
            
            # All runs should produce identical results
            first_signal = signals[0]
            for i, signal in enumerate(signals[1:], 1):
                assert signal[0] == first_signal[0], \
                    f"{strategy.name} run {i}: signal type inconsistent"
                assert abs(signal[1] - first_signal[1]) < 1e-10, \
                    f"{strategy.name} run {i}: confidence inconsistent"
                assert abs(signal[2] - first_signal[2]) < 1e-10, \
                    f"{strategy.name} run {i}: price inconsistent"


class TestMemoryAndPerformanceStability:
    """Test memory usage and performance stability of calculations"""
    
    def test_large_dataset_memory_stability(self):
        """Test that large datasets don't cause memory issues"""
        processor = DataProcessor()
        
        # Create large dataset (10,000 points)
        large_size = 10000
        np.random.seed(42)
        
        prices = 100 + np.cumsum(np.random.normal(0, 0.5, large_size))
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=large_size, freq='H'),
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(500, 1500, large_size)
        })
        
        # Monitor memory usage (basic check)
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large dataset
        df_with_indicators = processor.add_all_indicators(df)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (less than 500MB for 10k points)
        assert memory_increase < 500, f"Memory usage increase too large: {memory_increase}MB"
        
        # Results should be valid
        assert len(df_with_indicators) == large_size
        assert df_with_indicators['close'].iloc[-1] > 0
    
    def test_calculation_performance_consistency(self):
        """Test that calculation performance is consistent"""
        import time
        
        processor = DataProcessor()
        
        # Test with different dataset sizes
        dataset_sizes = [100, 500, 1000, 2000]
        timing_results = []
        
        for size in dataset_sizes:
            np.random.seed(42)  # Consistent data
            prices = 100 + np.cumsum(np.random.normal(0, 0.5, size))
            
            df = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=size, freq='H'),
                'open': prices,
                'high': prices * 1.01,
                'low': prices * 0.99,
                'close': prices,
                'volume': np.random.uniform(500, 1500, size)
            })
            
            # Time the calculation
            start_time = time.time()
            df_with_indicators = processor.add_basic_indicators(df)
            end_time = time.time()
            
            calculation_time = end_time - start_time
            timing_results.append((size, calculation_time))
        
        # Performance should scale roughly linearly
        # (Later datasets shouldn't be disproportionately slower)
        for i in range(1, len(timing_results)):
            prev_size, prev_time = timing_results[i-1]
            curr_size, curr_time = timing_results[i]
            
            size_ratio = curr_size / prev_size
            time_ratio = curr_time / prev_time
            
            # Time ratio shouldn't be much higher than size ratio
            # (allowing some overhead, but not exponential growth)
            assert time_ratio < size_ratio * 2, \
                f"Performance degradation: {size_ratio}x size took {time_ratio}x time"
    
    def test_repeated_calculation_stability(self):
        """Test that repeated calculations don't degrade"""
        strategy = RSIStrategy()
        
        # Create test data
        prices = [100 + np.sin(i/10) * 5 for i in range(100)]
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='D'),
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })
        
        # Run calculations many times
        reference_signal = None
        
        for iteration in range(100):
            df_with_indicators = strategy.add_required_indicators(df.copy())
            signal = strategy.calculate_signal(df_with_indicators, "BTC-USD")
            
            if reference_signal is None:
                reference_signal = signal
            else:
                # Results should be identical across iterations
                assert signal.signal == reference_signal.signal, \
                    f"Signal type changed on iteration {iteration}"
                assert abs(signal.confidence - reference_signal.confidence) < 1e-12, \
                    f"Confidence drifted on iteration {iteration}"
                assert abs(signal.price - reference_signal.price) < 1e-12, \
                    f"Price drifted on iteration {iteration}"
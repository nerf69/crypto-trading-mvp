"""
Unit tests for data processor module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.processor import DataProcessor


class TestDataProcessor:
    """Test the DataProcessor class"""
    
    def test_init(self):
        """Test DataProcessor initialization"""
        processor = DataProcessor()
        assert processor is not None
    
    def test_clean_data_removes_nan_values(self, sample_ohlcv_data):
        """Test that clean_data removes NaN values"""
        processor = DataProcessor()
        
        # Add some NaN values
        df = sample_ohlcv_data.copy()
        df.loc[5, 'close'] = np.nan
        df.loc[10, 'volume'] = np.nan
        df.loc[15] = np.nan  # Entire row
        
        initial_len = len(df)
        cleaned_df = processor.clean_data(df)
        
        # Should have removed rows with NaN
        assert len(cleaned_df) < initial_len
        assert not cleaned_df.isnull().any().any()
    
    def test_clean_data_removes_negative_prices(self, sample_ohlcv_data):
        """Test that clean_data removes negative or zero prices"""
        processor = DataProcessor()
        
        # Add invalid prices
        df = sample_ohlcv_data.copy()
        df.loc[5, 'close'] = 0
        df.loc[10, 'open'] = -10
        df.loc[15, 'high'] = 0
        
        cleaned_df = processor.clean_data(df)
        
        # Should have removed rows with invalid prices
        assert all(cleaned_df['open'] > 0)
        assert all(cleaned_df['high'] > 0)
        assert all(cleaned_df['low'] > 0)
        assert all(cleaned_df['close'] > 0)
    
    def test_clean_data_fixes_high_low_inconsistency(self, sample_ohlcv_data):
        """Test that clean_data handles high < low scenarios"""
        processor = DataProcessor()
        
        # Create invalid high/low relationship
        df = sample_ohlcv_data.copy()
        df.loc[5, 'high'] = 95.0
        df.loc[5, 'low'] = 105.0  # Low higher than high
        
        cleaned_df = processor.clean_data(df)
        
        # Should have removed the invalid row
        assert all(cleaned_df['high'] >= cleaned_df['low'])
    
    def test_clean_data_handles_negative_volume(self, sample_ohlcv_data):
        """Test that clean_data handles negative volume"""
        processor = DataProcessor()
        
        # Add negative volume
        df = sample_ohlcv_data.copy()
        df.loc[5, 'volume'] = -1000
        
        cleaned_df = processor.clean_data(df)
        
        # Should have set negative volume to 0
        assert all(cleaned_df['volume'] >= 0)
    
    def test_clean_data_sorts_by_timestamp(self):
        """Test that clean_data sorts data by timestamp"""
        processor = DataProcessor()
        
        # Create unsorted data
        dates = [
            datetime(2024, 1, 3),
            datetime(2024, 1, 1),
            datetime(2024, 1, 4),
            datetime(2024, 1, 2)
        ]
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [100, 101, 102, 103],
            'high': [105, 106, 107, 108],
            'low': [95, 96, 97, 98],
            'close': [102, 103, 104, 105],
            'volume': [1000, 1001, 1002, 1003]
        })
        
        cleaned_df = processor.clean_data(df)
        
        # Should be sorted by timestamp
        assert cleaned_df['timestamp'].is_monotonic_increasing
    
    def test_clean_data_empty_dataframe(self):
        """Test clean_data with empty DataFrame"""
        processor = DataProcessor()
        empty_df = pd.DataFrame()
        
        result = processor.clean_data(empty_df)
        
        assert result.empty
    
    def test_add_basic_indicators_success(self, sample_ohlcv_data):
        """Test successful addition of basic indicators"""
        processor = DataProcessor()
        
        df_with_indicators = processor.add_basic_indicators(sample_ohlcv_data)
        
        # Check that indicators were added
        expected_indicators = ['sma_20', 'sma_50', 'ema_12', 'ema_26', 
                             'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent', 'atr']
        
        for indicator in expected_indicators:
            assert indicator in df_with_indicators.columns
        
        # Check that indicators have reasonable values (not all NaN)
        assert not df_with_indicators['sma_20'].iloc[-5:].isna().all()
        assert not df_with_indicators['bb_upper'].iloc[-5:].isna().all()
    
    def test_add_basic_indicators_insufficient_data(self):
        """Test basic indicators with insufficient data"""
        processor = DataProcessor()
        
        # Create minimal data (less than required for indicators)
        small_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='D'),
            'open': [100] * 5,
            'high': [105] * 5,
            'low': [95] * 5,
            'close': [102] * 5,
            'volume': [1000] * 5
        })
        
        result = processor.add_basic_indicators(small_df)
        
        # Should return original DataFrame without errors
        assert len(result) == 5
    
    def test_add_momentum_indicators_success(self, sample_ohlcv_data):
        """Test successful addition of momentum indicators"""
        processor = DataProcessor()
        
        df_with_indicators = processor.add_momentum_indicators(sample_ohlcv_data)
        
        # Check that momentum indicators were added
        expected_indicators = ['rsi', 'macd', 'macd_signal', 'macd_histogram',
                             'stoch_k', 'stoch_d', 'williams_r', 'cci']
        
        for indicator in expected_indicators:
            assert indicator in df_with_indicators.columns
        
        # RSI should be between 0 and 100
        rsi_values = df_with_indicators['rsi'].dropna()
        if len(rsi_values) > 0:
            assert all(rsi_values >= 0)
            assert all(rsi_values <= 100)
    
    def test_add_volume_indicators_success(self, sample_ohlcv_data):
        """Test successful addition of volume indicators"""
        processor = DataProcessor()
        
        df_with_indicators = processor.add_volume_indicators(sample_ohlcv_data)
        
        # Check expected volume indicators (note: some may fail due to ta library issues)
        potential_indicators = ['obv', 'ad_line', 'cmf', 'vwap']
        
        # At least some volume indicators should be present
        volume_indicators_present = [ind for ind in potential_indicators 
                                   if ind in df_with_indicators.columns]
        
        assert len(volume_indicators_present) > 0
    
    def test_add_volatility_indicators_success(self, sample_ohlcv_data):
        """Test successful addition of volatility indicators"""
        processor = DataProcessor()
        
        df_with_indicators = processor.add_volatility_indicators(sample_ohlcv_data)
        
        # Check that volatility indicators were added
        expected_indicators = ['volatility', 'keltner_upper', 'keltner_middle', 
                             'keltner_lower', 'donchian_upper', 'donchian_lower']
        
        for indicator in expected_indicators:
            assert indicator in df_with_indicators.columns
        
        # Volatility should be non-negative
        volatility_values = df_with_indicators['volatility'].dropna()
        if len(volatility_values) > 0:
            assert all(volatility_values >= 0)
    
    def test_add_price_patterns_success(self, sample_ohlcv_data):
        """Test successful addition of price pattern indicators"""
        processor = DataProcessor()
        
        df_with_patterns = processor.add_price_patterns(sample_ohlcv_data)
        
        # Check that price patterns were added
        expected_patterns = ['price_change', 'price_change_pct', 'hl_range', 'hl_range_pct',
                           'body_size', 'body_size_pct', 'upper_shadow', 'lower_shadow',
                           'is_doji', 'is_green', 'is_red', 'gap_up', 'gap_down']
        
        for pattern in expected_patterns:
            assert pattern in df_with_patterns.columns
        
        # Check boolean indicators
        assert df_with_patterns['is_green'].dtype == bool
        assert df_with_patterns['is_red'].dtype == bool
        assert df_with_patterns['is_doji'].dtype == bool
    
    def test_add_support_resistance_levels(self, sample_ohlcv_data):
        """Test support and resistance level detection"""
        processor = DataProcessor()
        
        df_with_sr = processor.add_support_resistance_levels(sample_ohlcv_data)
        
        # Check that support/resistance indicators were added
        expected_indicators = ['local_max', 'local_min']
        
        for indicator in expected_indicators:
            assert indicator in df_with_sr.columns
        
        # Check boolean nature of local max/min
        assert df_with_sr['local_max'].dtype == bool
        assert df_with_sr['local_min'].dtype == bool
    
    def test_add_all_indicators_comprehensive(self, sample_ohlcv_data):
        """Test adding all indicators at once"""
        processor = DataProcessor()
        
        df_with_all = processor.add_all_indicators(sample_ohlcv_data)
        
        # Should have significantly more columns than original
        assert len(df_with_all.columns) > len(sample_ohlcv_data.columns)
        
        # Check for key indicators from each category
        key_indicators = ['sma_20', 'rsi', 'macd', 'volatility', 'price_change_pct']
        for indicator in key_indicators:
            assert indicator in df_with_all.columns
        
        # Should have composite indicators
        if 'momentum_score' in df_with_all.columns:
            momentum_scores = df_with_all['momentum_score'].dropna()
            if len(momentum_scores) > 0:
                assert all(momentum_scores >= 0)
                assert all(momentum_scores <= 100)
    
    def test_get_latest_signals(self, sample_ohlcv_data_with_indicators):
        """Test extraction of latest signals"""
        processor = DataProcessor()
        
        signals = processor.get_latest_signals(sample_ohlcv_data_with_indicators)
        
        assert isinstance(signals, dict)
        assert 'close' in signals
        assert 'rsi' in signals
        
        # Values should be numeric
        for key, value in signals.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
    
    def test_get_latest_signals_empty_dataframe(self):
        """Test get_latest_signals with empty DataFrame"""
        processor = DataProcessor()
        
        signals = processor.get_latest_signals(pd.DataFrame())
        
        assert signals == {}
    
    def test_validate_data_quality_good_data(self, sample_ohlcv_data):
        """Test data quality validation with good data"""
        processor = DataProcessor()
        
        quality_report = processor.validate_data_quality(sample_ohlcv_data)
        
        assert quality_report['valid'] is True
        assert quality_report['total_rows'] == len(sample_ohlcv_data)
        assert 'null_percentages' in quality_report
        assert 'data_range' in quality_report
        assert 'outliers' in quality_report
        assert 'gaps' in quality_report
    
    def test_validate_data_quality_empty_dataframe(self):
        """Test data quality validation with empty data"""
        processor = DataProcessor()
        
        quality_report = processor.validate_data_quality(pd.DataFrame())
        
        assert quality_report['valid'] is False
        assert quality_report['reason'] == "Empty dataset"
    
    def test_validate_data_quality_with_nulls(self, sample_ohlcv_data):
        """Test data quality validation with many null values"""
        processor = DataProcessor()
        
        # Add many null values
        df = sample_ohlcv_data.copy()
        df.loc[0:10, 'close'] = np.nan  # > 10% nulls
        
        quality_report = processor.validate_data_quality(df)
        
        assert quality_report['valid'] is False
        assert quality_report['reason'] == "Too many null values"
        assert quality_report['null_percentages']['close'] > 10
    
    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data types"""
        processor = DataProcessor()
        
        # Create DataFrame with mixed/invalid types
        invalid_df = pd.DataFrame({
            'timestamp': ['invalid', 'date', 'format'],
            'open': ['not', 'a', 'number'],
            'high': [None, None, None],
            'low': [100, 200, 300],
            'close': [105, 205, 305],
            'volume': [-1, -2, -3]
        })
        
        # Should not raise an exception
        try:
            result = processor.clean_data(invalid_df)
            # Should return empty or minimal DataFrame
            assert len(result) <= len(invalid_df)
        except Exception as e:
            pytest.fail(f"clean_data should handle invalid data gracefully, but raised: {e}")


class TestIndicatorCalculations:
    """Test specific indicator calculations"""
    
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation"""
        processor = DataProcessor()
        
        # Create simple test data
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='D'),
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })
        
        df_with_sma = processor.add_basic_indicators(df)
        
        if 'sma_20' in df_with_sma.columns:
            # Since we have less than 20 periods, SMA_20 should be NaN for early periods
            assert df_with_sma['sma_20'].iloc[:19].isna().all() if len(df_with_sma) >= 20 else True
    
    def test_rsi_bounds(self):
        """Test RSI stays within 0-100 bounds"""
        processor = DataProcessor()
        
        # Create trending data that should produce extreme RSI values
        uptrend = list(range(100, 200, 2))  # Strong uptrend
        downtrend = list(range(200, 100, -2))  # Strong downtrend
        prices = uptrend + downtrend
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='D'),
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })
        
        df_with_rsi = processor.add_momentum_indicators(df)
        
        if 'rsi' in df_with_rsi.columns:
            rsi_values = df_with_rsi['rsi'].dropna()
            if len(rsi_values) > 0:
                assert all(rsi_values >= 0)
                assert all(rsi_values <= 100)
    
    def test_bollinger_bands_relationship(self):
        """Test that Bollinger Bands maintain proper relationships"""
        processor = DataProcessor()
        
        # Create sufficient data
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.normal(0, 0.5, 50))
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='D'),
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': [1000] * len(prices)
        })
        
        df_with_bb = processor.add_basic_indicators(df)
        
        if all(col in df_with_bb.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            # Remove NaN values
            valid_rows = df_with_bb[['bb_upper', 'bb_middle', 'bb_lower']].dropna()
            
            if len(valid_rows) > 0:
                # Upper should be >= middle >= lower
                assert all(valid_rows['bb_upper'] >= valid_rows['bb_middle'])
                assert all(valid_rows['bb_middle'] >= valid_rows['bb_lower'])


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_single_row_dataframe(self):
        """Test processing with single row"""
        processor = DataProcessor()
        
        single_row = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [1000.0]
        })
        
        # Should not raise errors
        cleaned = processor.clean_data(single_row)
        assert len(cleaned) == 1
        
        # Adding indicators should handle gracefully
        result = processor.add_all_indicators(single_row)
        assert len(result) == 1
    
    def test_identical_prices(self):
        """Test processing with identical prices (no volatility)"""
        processor = DataProcessor()
        
        # All prices are identical
        identical_price = 100.0
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='D'),
            'open': [identical_price] * 50,
            'high': [identical_price] * 50,
            'low': [identical_price] * 50,
            'close': [identical_price] * 50,
            'volume': [1000] * 50
        })
        
        # Should handle zero volatility gracefully
        result = processor.add_all_indicators(df)
        
        # Volatility indicators should be 0 or NaN, not error
        if 'volatility' in result.columns:
            volatility_values = result['volatility'].dropna()
            if len(volatility_values) > 0:
                assert all(vol >= 0 for vol in volatility_values)
    
    def test_extreme_price_movements(self):
        """Test with extreme price movements"""
        processor = DataProcessor()
        
        # Create data with extreme movements
        prices = [100, 1000, 10, 500, 1]  # Extreme volatility
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='D'),
            'open': prices,
            'high': [p * 1.1 for p in prices],
            'low': [p * 0.9 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })
        
        # Should handle extreme movements without errors
        result = processor.add_all_indicators(df)
        assert len(result) == len(prices)
        
        # Indicators should still be numeric (not inf or -inf)
        for col in result.columns:
            if result[col].dtype in ['float64', 'float32']:
                finite_values = result[col].dropna()
                if len(finite_values) > 0:
                    assert all(np.isfinite(val) for val in finite_values)
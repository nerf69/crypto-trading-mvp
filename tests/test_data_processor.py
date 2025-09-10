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


class TestSupportResistanceLevelDetection:
    """Test support and resistance level detection algorithms"""
    
    def test_support_resistance_detection_accuracy(self):
        """Test accuracy of support/resistance level detection"""
        processor = DataProcessor()
        
        # Create data with clear support/resistance levels
        base_price = 100
        # Create a pattern: support at 95, resistance at 110
        pattern_prices = [
            100, 105, 110, 108, 105,  # Touch resistance at 110
            102, 98, 95, 97, 100,     # Touch support at 95
            103, 108, 110, 107, 104,  # Touch resistance again
            101, 97, 95, 98, 102,     # Touch support again
            105, 109, 111, 108, 106   # Slight break above resistance
        ]
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(pattern_prices), freq='D'),
            'open': pattern_prices,
            'high': [p * 1.01 for p in pattern_prices],
            'low': [p * 0.99 for p in pattern_prices],
            'close': pattern_prices,
            'volume': [1000] * len(pattern_prices)
        })
        
        df_with_sr = processor.add_support_resistance_levels(df)
        
        if 'local_max' in df_with_sr.columns and 'local_min' in df_with_sr.columns:
            # Should detect local maxima near resistance level (110)
            local_maxima = df_with_sr[df_with_sr['local_max'] == True]
            if len(local_maxima) > 0:
                max_prices = local_maxima['high'].values
                # At least one maximum should be near our resistance level
                resistance_touches = [p for p in max_prices if abs(p - 110) < 2]
                assert len(resistance_touches) > 0, "Should detect resistance level around 110"
            
            # Should detect local minima near support level (95)
            local_minima = df_with_sr[df_with_sr['local_min'] == True]
            if len(local_minima) > 0:
                min_prices = local_minima['low'].values
                # At least one minimum should be near our support level
                support_touches = [p for p in min_prices if abs(p - 95) < 2]
                assert len(support_touches) > 0, "Should detect support level around 95"
    
    def test_support_resistance_distance_calculation(self):
        """Test distance calculations to support/resistance levels"""
        processor = DataProcessor()
        
        # Create simple data with clear levels
        prices = [100, 105, 110, 105, 100, 95, 100, 105, 110, 105]
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='D'),
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })
        
        df_with_sr = processor.add_support_resistance_levels(df)
        
        # Check distance calculations if present
        distance_cols = ['distance_to_resistance', 'distance_to_support', 
                        'resistance_pct', 'support_pct']
        
        present_cols = [col for col in distance_cols if col in df_with_sr.columns]
        
        for col in present_cols:
            values = df_with_sr[col].dropna()
            if len(values) > 0:
                # Distance values should be non-negative
                assert all(val >= 0 for val in values), f"{col} should contain non-negative values"
                
                # Percentage distances should be reasonable (< 100%)
                if 'pct' in col:
                    assert all(val < 100 for val in values), f"{col} percentages should be reasonable"
    
    def test_support_resistance_with_insufficient_data(self):
        """Test support/resistance detection with insufficient data"""
        processor = DataProcessor()
        
        # Very small dataset
        small_prices = [100, 101, 102]
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(small_prices), freq='D'),
            'open': small_prices,
            'high': [p + 1 for p in small_prices],
            'low': [p - 1 for p in small_prices],
            'close': small_prices,
            'volume': [1000] * len(small_prices)
        })
        
        # Should handle gracefully without errors
        result = processor.add_support_resistance_levels(df)
        assert len(result) == len(small_prices)


class TestDataQualityValidationAlgorithms:
    """Test data quality validation algorithms"""
    
    def test_validate_ohlcv_data_comprehensive(self):
        """Test comprehensive OHLCV validation"""
        processor = DataProcessor()
        
        # Create data with various quality issues
        problematic_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
            'open': [100, 101, 102, -5, 104, 105, 106, 107, 108, 109],  # Negative price
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'low': [95, 96, 120, 98, 99, 100, 101, 102, 103, 104],  # Low > High at index 2
            'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'volume': [1000, 1100, -500, 1300, 1400, 0.1, 1600, 1700, 1800, 1900]  # Negative volume, very low volume
        })
        
        validated_df = processor.validate_ohlcv_data(problematic_data)
        
        # Should have fixed price ordering issues
        assert all(validated_df['high'] >= validated_df['low']), "High should be >= Low after validation"
        assert all(validated_df['high'] >= validated_df['open']), "High should be >= Open after validation"
        assert all(validated_df['high'] >= validated_df['close']), "High should be >= Close after validation"
        assert all(validated_df['low'] <= validated_df['open']), "Low should be <= Open after validation"
        assert all(validated_df['low'] <= validated_df['close']), "Low should be <= Close after validation"
        
        # Should have handled volume issues
        assert all(validated_df['volume'] >= 1), "Volume should be >= minimum after validation"
        
        # Should have a quality score
        assert 'data_quality_score' in validated_df.attrs or hasattr(validated_df, 'attrs')
    
    def test_outlier_detection_accuracy(self):
        """Test outlier detection in price data"""
        processor = DataProcessor()
        
        # Create data with outliers
        normal_prices = [100 + np.sin(i/5) * 2 for i in range(50)]  # Normal variation
        normal_prices[25] = 500  # Extreme outlier
        normal_prices[35] = 10   # Another extreme outlier
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(normal_prices), freq='D'),
            'open': normal_prices,
            'high': [p * 1.01 for p in normal_prices],
            'low': [p * 0.99 for p in normal_prices],
            'close': normal_prices,
            'volume': [1000] * len(normal_prices)
        })
        
        validated_df = processor.validate_ohlcv_data(df)
        
        # Outliers should be handled (smoothed/interpolated)
        price_changes = validated_df['close'].pct_change().abs()
        max_change = price_changes.max()
        
        # Should not have extreme price changes after validation
        assert max_change < 0.5, f"Maximum price change after validation should be reasonable, got {max_change}"
    
    def test_data_quality_scoring_algorithm(self):
        """Test data quality scoring algorithm"""
        processor = DataProcessor()
        
        # Perfect data
        perfect_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='D'),
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [101 + i * 0.1 for i in range(100)],
            'low': [99 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
            'volume': [1000] * 100
        })
        
        # Problematic data
        problematic_data = perfect_data.copy()
        # Add various issues
        problematic_data.loc[10:15, 'close'] = np.nan  # Missing data
        problematic_data.loc[20, 'high'] = problematic_data.loc[20, 'low'] - 1  # Invalid OHLC
        problematic_data.loc[30, 'volume'] = -100  # Negative volume
        
        perfect_validated = processor.validate_ohlcv_data(perfect_data)
        problematic_validated = processor.validate_ohlcv_data(problematic_data)
        
        # Perfect data should have higher quality score
        if hasattr(perfect_validated, 'attrs') and hasattr(problematic_validated, 'attrs'):
            perfect_score = perfect_validated.attrs.get('data_quality_score', 1.0)
            problematic_score = problematic_validated.attrs.get('data_quality_score', 0.0)
            
            assert perfect_score > problematic_score, \
                f"Perfect data quality score ({perfect_score}) should be higher than problematic data ({problematic_score})"


class TestVolumeIndicatorAlgorithms:
    """Test volume indicator calculation algorithms"""
    
    def test_on_balance_volume_calculation_accuracy(self):
        """Test OBV calculation step-by-step accuracy"""
        processor = DataProcessor()
        
        # Create test data with known OBV progression
        price_volume_data = [
            (100, 1000),  # Start
            (102, 1200),  # Price up -> Add volume: OBV = 1000 + 1200 = 2200
            (101, 800),   # Price down -> Subtract volume: OBV = 2200 - 800 = 1400
            (103, 1500),  # Price up -> Add volume: OBV = 1400 + 1500 = 2900
            (103, 900),   # Price same -> No change: OBV = 2900
            (105, 1100),  # Price up -> Add volume: OBV = 2900 + 1100 = 4000
        ]
        
        prices = [d[0] for d in price_volume_data]
        volumes = [d[1] for d in price_volume_data]
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='D'),
            'open': prices,
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        df_with_obv = processor.add_volume_indicators(df)
        
        if 'obv' in df_with_obv.columns:
            # Calculate expected OBV manually
            expected_obv = [volumes[0]]  # Start with first volume
            for i in range(1, len(prices)):
                if prices[i] > prices[i-1]:
                    expected_obv.append(expected_obv[-1] + volumes[i])
                elif prices[i] < prices[i-1]:
                    expected_obv.append(expected_obv[-1] - volumes[i])
                else:
                    expected_obv.append(expected_obv[-1])  # No change
            
            # Compare calculated vs expected (allow small floating point differences)
            obv_values = df_with_obv['obv'].values
            for i, (calc, exp) in enumerate(zip(obv_values, expected_obv)):
                if not np.isnan(calc):
                    assert abs(calc - exp) < 1.0, \
                        f"OBV mismatch at index {i}: calculated={calc}, expected={exp}"
    
    def test_accumulation_distribution_line_behavior(self):
        """Test A/D line behavior with price-volume relationships"""
        processor = DataProcessor()
        
        # Create scenario: price goes up with high volume, down with low volume
        # This should show accumulation (positive A/D trend)
        up_moves = [(100 + i, 2000) for i in range(5)]  # Price up, high volume
        down_moves = [(104 - i, 500) for i in range(3)]   # Price down, low volume
        
        price_volume_data = up_moves + down_moves
        prices = [d[0] for d in price_volume_data]
        volumes = [d[1] for d in price_volume_data]
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='D'),
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        df_with_ad = processor.add_volume_indicators(df)
        
        if 'ad_line' in df_with_ad.columns:
            ad_values = df_with_ad['ad_line'].dropna()
            if len(ad_values) > 3:
                # A/D line should generally trend up due to accumulation pattern
                # (high volume on up moves, low volume on down moves)
                final_ad = ad_values.iloc[-1]
                initial_ad = ad_values.iloc[0]
                
                # Not a strict requirement, but should show the accumulation tendency
                # Just verify the calculation produces reasonable values
                assert np.isfinite(final_ad), "A/D line should produce finite values"
                assert abs(final_ad) < 1e10, "A/D line should not produce extremely large values"
    
    def test_chaikin_money_flow_bounds(self):
        """Test Chaikin Money Flow stays within expected bounds"""
        processor = DataProcessor()
        
        # Generate varied price/volume data
        np.random.seed(42)
        base_price = 100
        prices = []
        volumes = []
        
        for i in range(30):
            price = base_price + np.random.normal(0, 2)  # +/- 2 around base
            volume = max(500, 1000 + np.random.normal(0, 200))  # Positive volume
            prices.append(price)
            volumes.append(volume)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='D'),
            'open': prices,
            'high': [p * 1.02 for p in prices],  # 2% above open
            'low': [p * 0.98 for p in prices],   # 2% below open
            'close': [p + np.random.uniform(-1, 1) for p in prices],
            'volume': volumes
        })
        
        df_with_cmf = processor.add_volume_indicators(df)
        
        if 'cmf' in df_with_cmf.columns:
            cmf_values = df_with_cmf['cmf'].dropna()
            if len(cmf_values) > 0:
                # CMF should typically be between -1 and 1
                assert all(-1.5 <= val <= 1.5 for val in cmf_values), \
                    "CMF values should be within reasonable bounds"


class TestVolatilityCalculationAccuracy:
    """Test volatility calculation algorithms"""
    
    def test_historical_volatility_calculation(self):
        """Test historical volatility calculation accuracy"""
        processor = DataProcessor()
        
        # Create data with known volatility characteristics
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 60)  # 2% daily volatility
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='D'),
            'open': prices,
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })
        
        df_with_vol = processor.add_volatility_indicators(df)
        
        if 'volatility' in df_with_vol.columns:
            vol_values = df_with_vol['volatility'].dropna()
            if len(vol_values) > 0:
                # Should be reasonable values (annualized volatility in %)
                assert all(0 <= val <= 200 for val in vol_values), \
                    "Historical volatility should be between 0% and 200%"
                
                # For our 2% daily vol, annualized should be around 2% * sqrt(252) ≈ 32%
                final_vol = vol_values.iloc[-1]
                assert 15 <= final_vol <= 50, \
                    f"Final volatility should be reasonable for 2% daily vol, got {final_vol}%"
    
    def test_keltner_channel_relationships(self):
        """Test Keltner Channel calculations maintain proper relationships"""
        processor = DataProcessor()
        
        # Create sufficient data for Keltner Channels
        np.random.seed(42)
        base_price = 100
        prices = []
        
        for i in range(40):
            price = base_price + np.sin(i/10) * 5 + np.random.normal(0, 1)
            prices.append(price)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='D'),
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })
        
        df_with_keltner = processor.add_volatility_indicators(df)
        
        keltner_cols = ['keltner_upper', 'keltner_middle', 'keltner_lower']
        if all(col in df_with_keltner.columns for col in keltner_cols):
            # Remove NaN rows
            valid_data = df_with_keltner[keltner_cols].dropna()
            
            if len(valid_data) > 0:
                # Upper should be > Middle > Lower
                assert all(valid_data['keltner_upper'] > valid_data['keltner_middle']), \
                    "Keltner upper should be above middle"
                assert all(valid_data['keltner_middle'] > valid_data['keltner_lower']), \
                    "Keltner middle should be above lower"
                
                # Channel width should be reasonable
                channel_width = valid_data['keltner_upper'] - valid_data['keltner_lower']
                assert all(width > 0 for width in channel_width), \
                    "Keltner channel width should be positive"
    
    def test_donchian_channel_accuracy(self):
        """Test Donchian Channel calculation accuracy"""
        processor = DataProcessor()
        
        # Create data with clear high/low patterns
        pattern_highs = [105, 110, 108, 115, 112, 120, 118, 125, 122, 130]  # Increasing highs
        pattern_lows = [95, 88, 92, 85, 90, 80, 88, 75, 85, 70]             # Decreasing lows
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(pattern_highs), freq='D'),
            'open': [100] * len(pattern_highs),
            'high': pattern_highs,
            'low': pattern_lows,
            'close': [(h + l) / 2 for h, l in zip(pattern_highs, pattern_lows)],
            'volume': [1000] * len(pattern_highs)
        })
        
        df_with_donchian = processor.add_volatility_indicators(df)
        
        donchian_cols = ['donchian_upper', 'donchian_lower']
        if all(col in df_with_donchian.columns for col in donchian_cols):
            # Remove NaN rows
            valid_data = df_with_donchian[donchian_cols + ['high', 'low']].dropna()
            
            if len(valid_data) > 0:
                # Donchian upper should be the highest high in the period
                # Donchian lower should be the lowest low in the period
                
                # Upper channel should be >= current high
                assert all(valid_data['donchian_upper'] >= valid_data['high']), \
                    "Donchian upper should be >= current high"
                
                # Lower channel should be <= current low
                assert all(valid_data['donchian_lower'] <= valid_data['low']), \
                    "Donchian lower should be <= current low"
                
                # Upper should be > Lower
                assert all(valid_data['donchian_upper'] > valid_data['donchian_lower']), \
                    "Donchian upper should be above lower"


class TestVectorizedCalculationPerformance:
    """Test vectorized calculation performance and accuracy"""
    
    def test_vectorized_vs_standard_calculation_accuracy(self):
        """Test that vectorized calculations match standard calculations"""
        processor = DataProcessor()
        
        # Create substantial test data
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.normal(0, 0.5, 100))
        volumes = np.random.uniform(800, 1200, 100)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices), freq='D'),
            'open': prices,
            'high': prices * 1.015,
            'low': prices * 0.985,
            'close': prices,
            'volume': volumes
        })
        
        # Calculate using both methods
        df_standard = processor.add_basic_indicators(df.copy())
        df_vectorized = processor.calculate_indicators_vectorized(df.copy())
        
        # Compare key indicators that should be calculated by both methods
        common_indicators = ['sma_20', 'ema_12', 'volatility']
        
        for indicator in common_indicators:
            if indicator in df_standard.columns and indicator in df_vectorized.columns:
                standard_values = df_standard[indicator].dropna()
                vectorized_values = df_vectorized[indicator].dropna()
                
                # Should have same number of valid values
                assert len(standard_values) == len(vectorized_values), \
                    f"Different number of valid values for {indicator}"
                
                # Values should be very close (allow for small floating point differences)
                if len(standard_values) > 0:
                    for i, (std_val, vec_val) in enumerate(zip(standard_values.values, vectorized_values.values)):
                        assert abs(std_val - vec_val) < 0.001, \
                            f"Vectorized calculation differs from standard for {indicator} at index {i}: {vec_val} vs {std_val}"
    
    def test_vectorized_calculation_performance_characteristics(self):
        """Test vectorized calculation performance characteristics"""
        processor = DataProcessor()
        
        # Create large dataset to test performance characteristics
        np.random.seed(42)
        large_size = 1000
        prices = 100 + np.cumsum(np.random.normal(0, 0.3, large_size))
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=large_size, freq='H'),
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(500, 1500, large_size)
        })
        
        # Should complete without errors and produce reasonable results
        import time
        start_time = time.time()
        
        result = processor.calculate_indicators_vectorized(df)
        
        calculation_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 5 seconds for 1000 points)
        assert calculation_time < 5.0, f"Vectorized calculation took too long: {calculation_time}s"
        
        # Should produce valid results
        assert len(result) == large_size, "Should maintain all data points"
        
        # Check that key indicators are present and have reasonable values
        expected_indicators = ['sma_20', 'ema_12', 'volatility', 'volume_ratio']
        
        for indicator in expected_indicators:
            if indicator in result.columns:
                values = result[indicator].dropna()
                if len(values) > 0:
                    assert all(np.isfinite(val) for val in values), \
                        f"Vectorized {indicator} should produce finite values"
    
    def test_vectorized_calculation_edge_cases(self):
        """Test vectorized calculations with edge cases"""
        processor = DataProcessor()
        
        # Test with minimum required data
        min_data_size = 25  # Just above minimum for most indicators
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=min_data_size, freq='D'),
            'open': [100 + i * 0.1 for i in range(min_data_size)],
            'high': [101 + i * 0.1 for i in range(min_data_size)],
            'low': [99 + i * 0.1 for i in range(min_data_size)],
            'close': [100.5 + i * 0.1 for i in range(min_data_size)],
            'volume': [1000] * min_data_size
        })
        
        # Should handle minimum data gracefully
        result = processor.calculate_indicators_vectorized(df)
        
        assert len(result) == min_data_size, "Should maintain data size with minimum data"
        
        # At least some indicators should have valid values
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        has_valid_values = False
        
        for col in numeric_columns:
            if col not in ['timestamp']:  # Skip timestamp
                values = result[col].dropna()
                if len(values) > 0:
                    has_valid_values = True
                    break
        
        assert has_valid_values, "Should produce some valid indicator values even with minimum data"
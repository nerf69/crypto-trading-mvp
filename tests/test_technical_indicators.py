"""
Comprehensive tests for technical indicator mathematical accuracy.

This test suite validates the mathematical correctness of all technical indicators
used in the trading strategies, ensuring they produce accurate results compared
to known reference implementations and manual calculations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
import math

from src.data.processor import DataProcessor


class TestRSICalculations:
    """Test RSI (Relative Strength Index) mathematical accuracy"""
    
    def test_rsi_manual_calculation_accuracy(self):
        """Test RSI calculation against manual computation"""
        processor = DataProcessor()
        
        # Create test data with known RSI values
        prices = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 
                 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 
                 46.22, 45.64, 46.21, 46.25, 46.23, 46.08, 47.43, 47.24]
        
        df = self._create_test_dataframe(prices)
        df_with_indicators = processor.add_momentum_indicators(df)
        
        # Calculate RSI manually for verification
        changes = np.diff(prices)
        gains = [change if change > 0 else 0 for change in changes]
        losses = [-change if change < 0 else 0 for change in changes]
        
        # For 14-period RSI
        avg_gain = np.mean(gains[:14])
        avg_loss = np.mean(losses[:14])
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            expected_rsi = 100 - (100 / (1 + rs))
        else:
            expected_rsi = 100
        
        # Verify calculation accuracy (within 1% tolerance for floating point)
        calculated_rsi = df_with_indicators.iloc[14]['rsi']  # 15th point (0-indexed)
        assert abs(calculated_rsi - expected_rsi) < 1.0, f"RSI mismatch: calculated={calculated_rsi}, expected={expected_rsi}"
    
    def test_rsi_boundary_conditions(self):
        """Test RSI behavior in boundary conditions"""
        processor = DataProcessor()
        
        # Test case 1: All prices increasing (RSI should approach 100)
        increasing_prices = [100 + i for i in range(30)]
        df_increasing = self._create_test_dataframe(increasing_prices)
        df_increasing = processor.add_momentum_indicators(df_increasing)
        
        final_rsi = df_increasing.iloc[-1]['rsi']
        assert 90 <= final_rsi <= 100, f"RSI for increasing prices should be near 100, got {final_rsi}"
        
        # Test case 2: All prices decreasing (RSI should approach 0)
        decreasing_prices = [100 - i for i in range(30)]
        df_decreasing = self._create_test_dataframe(decreasing_prices)
        df_decreasing = processor.add_momentum_indicators(df_decreasing)
        
        final_rsi = df_decreasing.iloc[-1]['rsi']
        assert 0 <= final_rsi <= 10, f"RSI for decreasing prices should be near 0, got {final_rsi}"
        
        # Test case 3: Constant prices (RSI should be around 50)
        constant_prices = [100] * 30
        df_constant = self._create_test_dataframe(constant_prices)
        df_constant = processor.add_momentum_indicators(df_constant)
        
        final_rsi = df_constant.iloc[-1]['rsi']
        # RSI for constant prices can be NaN or around 50 depending on implementation
        assert np.isnan(final_rsi) or 40 <= final_rsi <= 60, f"RSI for constant prices should be NaN or ~50, got {final_rsi}"
    
    def test_rsi_period_sensitivity(self):
        """Test RSI sensitivity to different periods"""
        # This would require modifying the processor to accept different RSI periods
        # For now, we test that the standard 14-period RSI is stable
        processor = DataProcessor()
        
        # Generate random walk data
        np.random.seed(42)
        prices = [100]
        for _ in range(50):
            change = np.random.normal(0, 1)
            prices.append(prices[-1] + change)
        
        df = self._create_test_dataframe(prices)
        df_with_indicators = processor.add_momentum_indicators(df)
        
        # Verify RSI values are within valid range
        rsi_values = df_with_indicators['rsi'].dropna()
        assert all(0 <= rsi <= 100 for rsi in rsi_values), "All RSI values must be between 0 and 100"
        
        # Verify RSI is not constant (should show some variation)
        assert len(rsi_values.unique()) > 1, "RSI should show variation across different price movements"
    
    def _create_test_dataframe(self, prices):
        """Helper to create test DataFrame"""
        dates = pd.date_range('2024-01-01', periods=len(prices))
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })


class TestMACDCalculations:
    """Test MACD (Moving Average Convergence Divergence) mathematical accuracy"""
    
    def test_macd_calculation_accuracy(self):
        """Test MACD calculation against manual computation"""
        processor = DataProcessor()
        
        # Create test data with known values
        prices = [22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29,
                 22.15, 22.39, 22.38, 22.61, 23.36, 24.05, 23.75, 23.83, 23.95, 23.63,
                 23.82, 23.87, 23.65, 23.19, 23.10, 23.33, 22.68, 23.10, 22.40, 22.17]
        
        df = self._create_test_dataframe(prices)
        df_with_indicators = processor.add_momentum_indicators(df)
        
        # Manual EMA calculation for verification
        close_series = pd.Series(prices)
        ema_12 = close_series.ewm(span=12, adjust=False).mean()
        ema_26 = close_series.ewm(span=26, adjust=False).mean()
        expected_macd = ema_12 - ema_26
        
        # Compare with calculated MACD (allow small floating point differences)
        calculated_macd = df_with_indicators['macd'].iloc[-1]
        expected_macd_value = expected_macd.iloc[-1]
        
        assert abs(calculated_macd - expected_macd_value) < 0.01, \
            f"MACD mismatch: calculated={calculated_macd}, expected={expected_macd_value}"
    
    def test_macd_signal_line_accuracy(self):
        """Test MACD signal line calculation"""
        processor = DataProcessor()
        
        # Generate more data for stable signal calculation
        np.random.seed(42)
        prices = [100]
        for _ in range(50):
            change = np.random.normal(0, 0.5)  # 0.5% daily volatility
            prices.append(prices[-1] * (1 + change/100))
        
        df = self._create_test_dataframe(prices)
        df_with_indicators = processor.add_momentum_indicators(df)
        
        # Verify signal line is EMA of MACD
        macd_values = df_with_indicators['macd'].dropna()
        signal_values = df_with_indicators['macd_signal'].dropna()
        
        # Signal should be smoother than MACD (less volatile)
        if len(macd_values) > 10 and len(signal_values) > 10:
            macd_volatility = macd_values.std()
            signal_volatility = signal_values.std()
            assert signal_volatility < macd_volatility, "Signal line should be smoother than MACD line"
    
    def test_macd_histogram_accuracy(self):
        """Test MACD histogram calculation"""
        processor = DataProcessor()
        
        prices = [50, 51, 49, 52, 48, 53, 47, 54, 46, 55, 45, 56, 44, 57, 43, 58, 42, 59, 41, 60]
        df = self._create_test_dataframe(prices)
        df_with_indicators = processor.add_momentum_indicators(df)
        
        # Verify histogram = MACD - Signal
        macd_values = df_with_indicators['macd']
        signal_values = df_with_indicators['macd_signal']
        histogram_values = df_with_indicators['macd_histogram']
        
        for i in range(len(df_with_indicators)):
            if not (pd.isna(macd_values.iloc[i]) or pd.isna(signal_values.iloc[i])):
                expected_histogram = macd_values.iloc[i] - signal_values.iloc[i]
                actual_histogram = histogram_values.iloc[i]
                assert abs(expected_histogram - actual_histogram) < 0.0001, \
                    f"Histogram calculation error at index {i}"
    
    def test_macd_crossover_detection(self):
        """Test MACD crossover detection accuracy"""
        processor = DataProcessor()
        
        # Create data with clear trend changes to generate crossovers
        prices = ([45] * 10 + [45 + i*0.5 for i in range(20)] + 
                 [55] * 10 + [55 - i*0.3 for i in range(20)])
        
        df = self._create_test_dataframe(prices)
        df = processor.add_momentum_indicators(df)
        
        # Add crossover detection (this should be added to strategies)
        df['macd_above_signal'] = df['macd'] > df['macd_signal']
        df['macd_crossover'] = df['macd_above_signal'].astype(int).diff()
        
        # Verify crossovers exist and are correctly identified
        bullish_crosses = (df['macd_crossover'] == 1).sum()
        bearish_crosses = (df['macd_crossover'] == -1).sum()
        
        assert bullish_crosses > 0, "Should detect bullish MACD crossovers in trending data"
        assert bearish_crosses > 0, "Should detect bearish MACD crossovers in trending data"
    
    def _create_test_dataframe(self, prices):
        """Helper to create test DataFrame"""
        dates = pd.date_range('2024-01-01', periods=len(prices))
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })


class TestBollingerBandCalculations:
    """Test Bollinger Bands mathematical accuracy"""
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation accuracy"""
        processor = DataProcessor()
        
        # Create test data
        prices = [20, 20.5, 19.8, 21.2, 20.9, 20.1, 22.3, 21.8, 20.5, 19.9,
                 21.5, 22.1, 20.8, 21.9, 22.5, 21.3, 20.7, 22.8, 23.1, 21.6,
                 20.9, 22.4, 23.5, 22.2, 21.8, 23.9, 24.2, 22.5, 21.9, 24.5]
        
        df = self._create_test_dataframe(prices)
        df_with_indicators = processor.add_basic_indicators(df)
        
        # Manual calculation for verification (20-period Bollinger Bands)
        window = 20
        for i in range(window-1, len(prices)):
            price_window = prices[i-window+1:i+1]
            sma = np.mean(price_window)
            std = np.std(price_window, ddof=0)  # Population standard deviation
            
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            # Verify calculations
            calculated_upper = df_with_indicators.iloc[i]['bb_upper']
            calculated_middle = df_with_indicators.iloc[i]['bb_middle']
            calculated_lower = df_with_indicators.iloc[i]['bb_lower']
            
            if not pd.isna(calculated_upper):
                assert abs(calculated_upper - upper_band) < 0.01, \
                    f"Upper band mismatch at index {i}: {calculated_upper} vs {upper_band}"
                assert abs(calculated_middle - sma) < 0.01, \
                    f"Middle band (SMA) mismatch at index {i}: {calculated_middle} vs {sma}"
                assert abs(calculated_lower - lower_band) < 0.01, \
                    f"Lower band mismatch at index {i}: {calculated_lower} vs {lower_band}"
    
    def test_bollinger_band_percentage(self):
        """Test Bollinger Band percentage calculation (%B)"""
        processor = DataProcessor()
        
        prices = [25, 26, 24, 27, 23, 28, 22, 29, 21, 30,
                 20, 31, 19, 32, 18, 33, 17, 34, 16, 35, 15]
        
        df = self._create_test_dataframe(prices)
        df_with_indicators = processor.add_basic_indicators(df)
        
        # Verify %B calculation: %B = (Price - Lower Band) / (Upper Band - Lower Band)
        for i in range(len(df_with_indicators)):
            price = df_with_indicators.iloc[i]['close']
            upper = df_with_indicators.iloc[i]['bb_upper']
            lower = df_with_indicators.iloc[i]['bb_lower']
            calculated_pct = df_with_indicators.iloc[i]['bb_percent']
            
            if not (pd.isna(upper) or pd.isna(lower) or upper == lower):
                expected_pct = (price - lower) / (upper - lower)
                assert abs(calculated_pct - expected_pct) < 0.01, \
                    f"%B calculation error at index {i}: {calculated_pct} vs {expected_pct}"
    
    def test_bollinger_band_width(self):
        """Test Bollinger Band width calculation"""
        processor = DataProcessor()
        
        # Create data with varying volatility
        low_vol_prices = [100 + 0.1*i + 0.05*np.sin(i/5) for i in range(30)]
        high_vol_prices = [100 + 2*i + 2*np.sin(i/2) for i in range(30)]
        
        df_low_vol = self._create_test_dataframe(low_vol_prices)
        df_high_vol = self._create_test_dataframe(high_vol_prices)
        
        df_low_vol = processor.add_basic_indicators(df_low_vol)
        df_high_vol = processor.add_basic_indicators(df_high_vol)
        
        # High volatility data should have wider bands
        low_vol_width = df_low_vol['bb_width'].iloc[-1]
        high_vol_width = df_high_vol['bb_width'].iloc[-1]
        
        if not (pd.isna(low_vol_width) or pd.isna(high_vol_width)):
            assert high_vol_width > low_vol_width, \
                "Higher volatility data should produce wider Bollinger Bands"
    
    def _create_test_dataframe(self, prices):
        """Helper to create test DataFrame"""
        dates = pd.date_range('2024-01-01', periods=len(prices))
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.015 for p in prices],
            'low': [p * 0.985 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })


class TestMovingAverageCalculations:
    """Test Simple and Exponential Moving Average calculations"""
    
    def test_simple_moving_average_accuracy(self):
        """Test SMA calculation accuracy"""
        processor = DataProcessor()
        
        prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        
        df = self._create_test_dataframe(prices)
        df_with_indicators = processor.add_basic_indicators(df)
        
        # Test 20-period SMA
        for i in range(19, len(prices)):  # Start from index 19 (20th element)
            expected_sma = np.mean(prices[i-19:i+1])
            calculated_sma = df_with_indicators.iloc[i]['sma_20']
            
            if not pd.isna(calculated_sma):
                assert abs(calculated_sma - expected_sma) < 0.0001, \
                    f"SMA calculation error at index {i}: {calculated_sma} vs {expected_sma}"
    
    def test_exponential_moving_average_accuracy(self):
        """Test EMA calculation accuracy"""
        processor = DataProcessor()
        
        prices = [22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29,
                 22.15, 22.39, 22.38, 22.61, 23.36, 24.05, 23.75, 23.83, 23.95, 23.63]
        
        df = self._create_test_dataframe(prices)
        df_with_indicators = processor.add_basic_indicators(df)
        
        # Manual EMA calculation (12-period)
        multiplier = 2 / (12 + 1)
        ema_values = [prices[0]]  # Start with first price
        
        for i in range(1, len(prices)):
            ema_value = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema_value)
        
        # Compare with calculated EMA
        for i in range(len(prices)):
            calculated_ema = df_with_indicators.iloc[i]['ema_12']
            if not pd.isna(calculated_ema):
                assert abs(calculated_ema - ema_values[i]) < 0.01, \
                    f"EMA calculation error at index {i}: {calculated_ema} vs {ema_values[i]}"
    
    def test_moving_average_lag_properties(self):
        """Test that EMAs respond faster than SMAs to price changes"""
        processor = DataProcessor()
        
        # Create step function: constant then sudden change
        prices = [50] * 20 + [60] * 20
        
        df = self._create_test_dataframe(prices)
        df_with_indicators = processor.add_basic_indicators(df)
        
        # Check response to price change at index 20
        change_index = 25  # A few periods after the change
        
        sma_20_value = df_with_indicators.iloc[change_index]['sma_20']
        ema_12_value = df_with_indicators.iloc[change_index]['ema_12']
        
        # EMA should be closer to new price (60) than SMA
        if not (pd.isna(sma_20_value) or pd.isna(ema_12_value)):
            assert abs(ema_12_value - 60) < abs(sma_20_value - 60), \
                "EMA should respond faster to price changes than SMA"
    
    def _create_test_dataframe(self, prices):
        """Helper to create test DataFrame"""
        dates = pd.date_range('2024-01-01', periods=len(prices))
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000] * len(prices)
        })


class TestATRCalculations:
    """Test Average True Range calculations"""
    
    def test_atr_calculation_accuracy(self):
        """Test ATR calculation against manual computation"""
        processor = DataProcessor()
        
        # Test data with clear high-low ranges
        ohlc_data = [
            (100, 105, 95, 102),   # open, high, low, close
            (102, 108, 99, 105),
            (105, 107, 101, 103),
            (103, 110, 98, 106),
            (106, 112, 104, 109),
            (109, 115, 107, 111),
            (111, 118, 109, 114),
            (114, 120, 112, 117),
            (117, 123, 115, 119),
            (119, 125, 117, 122),
            (122, 128, 120, 125),
            (125, 131, 123, 128),
            (128, 134, 126, 131),
            (131, 137, 129, 134),
            (134, 140, 132, 137)
        ]
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(ohlc_data)),
            'open': [d[0] for d in ohlc_data],
            'high': [d[1] for d in ohlc_data],
            'low': [d[2] for d in ohlc_data],
            'close': [d[3] for d in ohlc_data],
            'volume': [1000] * len(ohlc_data)
        })
        
        df_with_indicators = processor.add_basic_indicators(df)
        
        # Manual TR calculation
        true_ranges = []
        for i in range(len(ohlc_data)):
            high = ohlc_data[i][1]
            low = ohlc_data[i][2]
            close = ohlc_data[i][3]
            
            if i == 0:
                tr = high - low  # First period
            else:
                prev_close = ohlc_data[i-1][3]
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            
            true_ranges.append(tr)
        
        # ATR is typically 14-period moving average of TR
        window = 14
        for i in range(window-1, len(true_ranges)):
            expected_atr = np.mean(true_ranges[i-window+1:i+1])
            calculated_atr = df_with_indicators.iloc[i]['atr']
            
            if not pd.isna(calculated_atr):
                assert abs(calculated_atr - expected_atr) < 0.1, \
                    f"ATR calculation error at index {i}: {calculated_atr} vs {expected_atr}"
    
    def test_atr_volatility_sensitivity(self):
        """Test ATR sensitivity to market volatility"""
        processor = DataProcessor()
        
        # Low volatility scenario
        low_vol_data = [(100, 101, 99, 100.5) for _ in range(20)]
        
        # High volatility scenario
        high_vol_data = [(100, 110, 90, 105) for _ in range(20)]
        
        df_low_vol = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(low_vol_data)),
            'open': [d[0] for d in low_vol_data],
            'high': [d[1] for d in low_vol_data],
            'low': [d[2] for d in low_vol_data],
            'close': [d[3] for d in low_vol_data],
            'volume': [1000] * len(low_vol_data)
        })
        
        df_high_vol = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(high_vol_data)),
            'open': [d[0] for d in high_vol_data],
            'high': [d[1] for d in high_vol_data],
            'low': [d[2] for d in high_vol_data],
            'close': [d[3] for d in high_vol_data],
            'volume': [1000] * len(high_vol_data)
        })
        
        df_low_vol = processor.add_basic_indicators(df_low_vol)
        df_high_vol = processor.add_basic_indicators(df_high_vol)
        
        low_vol_atr = df_low_vol['atr'].iloc[-1]
        high_vol_atr = df_high_vol['atr'].iloc[-1]
        
        if not (pd.isna(low_vol_atr) or pd.isna(high_vol_atr)):
            assert high_vol_atr > low_vol_atr * 5, \
                "Higher volatility should produce significantly higher ATR values"


class TestVolumeIndicators:
    """Test volume-based indicator calculations"""
    
    def test_obv_calculation_accuracy(self):
        """Test On-Balance Volume calculation"""
        processor = DataProcessor()
        
        # Test data with clear price-volume relationships
        price_volume_data = [
            (100, 1000), (102, 1200), (101, 800), (103, 1500), (102, 900),
            (104, 1800), (103, 700), (105, 2000), (104, 600), (106, 2200)
        ]
        
        prices = [d[0] for d in price_volume_data]
        volumes = [d[1] for d in price_volume_data]
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(prices)),
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        df_with_indicators = processor.add_volume_indicators(df)
        
        # Manual OBV calculation
        obv_values = [volumes[0]]  # Start with first volume
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv_values.append(obv_values[-1] + volumes[i])
            elif prices[i] < prices[i-1]:
                obv_values.append(obv_values[-1] - volumes[i])
            else:
                obv_values.append(obv_values[-1])  # No change
        
        # Verify OBV calculation
        for i in range(len(prices)):
            calculated_obv = df_with_indicators.iloc[i]['obv']
            if not pd.isna(calculated_obv):
                assert abs(calculated_obv - obv_values[i]) < 0.1, \
                    f"OBV calculation error at index {i}: {calculated_obv} vs {obv_values[i]}"
    
    def test_volume_sma_accuracy(self):
        """Test Volume Simple Moving Average"""
        processor = DataProcessor()
        
        volumes = [1000, 1200, 800, 1500, 900, 1800, 700, 2000, 600, 2200,
                  1100, 1300, 850, 1600, 950, 1900, 750, 2100, 650, 2300, 1150]
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(volumes)),
            'open': [100] * len(volumes),
            'high': [101] * len(volumes),
            'low': [99] * len(volumes),
            'close': [100] * len(volumes),
            'volume': volumes
        })
        
        df_with_indicators = processor.add_volume_indicators(df)
        
        # Test 20-period volume SMA
        window = 20
        for i in range(window-1, len(volumes)):
            expected_vol_sma = np.mean(volumes[i-window+1:i+1])
            calculated_vol_sma = df_with_indicators.iloc[i]['volume_sma']
            
            if not pd.isna(calculated_vol_sma):
                assert abs(calculated_vol_sma - expected_vol_sma) < 0.1, \
                    f"Volume SMA error at index {i}: {calculated_vol_sma} vs {expected_vol_sma}"


class TestNumericalStability:
    """Test numerical stability and precision of calculations"""
    
    def test_extreme_price_values(self):
        """Test indicator calculations with extreme price values"""
        processor = DataProcessor()
        
        # Very small prices (crypto cents)
        small_prices = [0.000001 * (1 + 0.01 * np.sin(i/10)) for i in range(30)]
        
        # Very large prices (like BRK.A)
        large_prices = [500000 * (1 + 0.01 * np.sin(i/10)) for i in range(30)]
        
        for prices, name in [(small_prices, "small"), (large_prices, "large")]:
            df = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=len(prices)),
                'open': prices,
                'high': [p * 1.01 for p in prices],
                'low': [p * 0.99 for p in prices],
                'close': prices,
                'volume': [1000] * len(prices)
            })
            
            # Should not raise exceptions
            try:
                df_with_indicators = processor.add_all_indicators(df)
                
                # Check that we get finite values
                numeric_columns = df_with_indicators.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    finite_values = df_with_indicators[col].dropna()
                    if len(finite_values) > 0:
                        assert np.all(np.isfinite(finite_values)), \
                            f"Non-finite values found in {col} with {name} prices"
                        
            except Exception as e:
                pytest.fail(f"Indicator calculation failed with {name} prices: {e}")
    
    def test_division_by_zero_handling(self):
        """Test handling of division by zero scenarios"""
        processor = DataProcessor()
        
        # Constant prices (should not cause division by zero)
        constant_prices = [100] * 30
        
        # Prices with zero values (edge case)
        zero_prices = [100, 100, 0.0001, 100, 100] * 6  # Avoid actual zeros
        
        for prices, name in [(constant_prices, "constant"), (zero_prices, "near-zero")]:
            df = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=len(prices)),
                'open': prices,
                'high': [max(p * 1.01, 0.0001) for p in prices],
                'low': [max(p * 0.99, 0.0001) for p in prices],
                'close': prices,
                'volume': [max(1000, 1)] * len(prices)  # Ensure positive volume
            })
            
            # Should handle gracefully without exceptions
            try:
                df_with_indicators = processor.add_all_indicators(df)
                
                # RSI with constant prices should be NaN or reasonable value
                rsi_values = df_with_indicators['rsi'].dropna()
                if len(rsi_values) > 0:
                    assert all(np.isnan(val) or 0 <= val <= 100 for val in rsi_values), \
                        f"Invalid RSI values with {name} prices"
                        
            except Exception as e:
                pytest.fail(f"Indicator calculation failed with {name} prices: {e}")
    
    def test_floating_point_precision(self):
        """Test floating point precision in calculations"""
        processor = DataProcessor()
        
        # Prices with many decimal places
        precise_prices = [
            100.123456789, 100.234567891, 100.345678912, 100.456789123,
            100.567891234, 100.678912345, 100.789123456, 100.891234567,
            100.912345678, 100.123456789, 100.234567891, 100.345678912,
            100.456789123, 100.567891234, 100.678912345, 100.789123456
        ]
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=len(precise_prices)),
            'open': precise_prices,
            'high': [p * 1.001 for p in precise_prices],
            'low': [p * 0.999 for p in precise_prices],
            'close': precise_prices,
            'volume': [1000.123456] * len(precise_prices)
        })
        
        # Test with high precision settings
        original_precision = getcontext().prec
        try:
            getcontext().prec = 28  # High precision
            df_with_indicators = processor.add_basic_indicators(df)
            
            # Check that calculations maintain reasonable precision
            sma_values = df_with_indicators['sma_20'].dropna()
            if len(sma_values) > 0:
                # SMA should be close to input values (which are around 100)
                assert all(99 < val < 101 for val in sma_values), \
                    "SMA values should maintain precision around expected range"
                    
        finally:
            getcontext().prec = original_precision
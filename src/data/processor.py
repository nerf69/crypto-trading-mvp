import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, getcontext, ROUND_HALF_UP
import ta
from scipy import stats

from ..constants import (
    MIN_DATA_POINTS_FOR_INDICATORS, RSI_PERIOD, MACD_FAST_PERIOD,
    MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STD_DEV,
    MAX_PRICE_CHANGE_PCT, MIN_VOLUME, PRICE_PRECISION
)

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Data processing and technical indicator calculation
    """
    
    def __init__(self):
        # Set decimal precision for financial calculations
        getcontext().prec = PRICE_PRECISION + 2
        getcontext().rounding = ROUND_HALF_UP
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate OHLCV data
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        logger.debug("Cleaning data...")
        
        # Make a copy to avoid modifying original
        clean_df = df.copy()
        
        # Remove rows with NaN values
        initial_len = len(clean_df)
        clean_df = clean_df.dropna()
        
        if len(clean_df) < initial_len:
            logger.warning(f"Removed {initial_len - len(clean_df)} rows with NaN values")
        
        # Ensure prices are positive
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in clean_df.columns:
                negative_mask = clean_df[col] <= 0
                if negative_mask.any():
                    logger.warning(f"Found {negative_mask.sum()} non-positive {col} values, removing")
                    clean_df = clean_df[~negative_mask]
        
        # Ensure high >= low
        if 'high' in clean_df.columns and 'low' in clean_df.columns:
            invalid_mask = clean_df['high'] < clean_df['low']
            if invalid_mask.any():
                logger.warning(f"Found {invalid_mask.sum()} rows where high < low, removing")
                clean_df = clean_df[~invalid_mask]
        
        # Ensure volume is non-negative
        if 'volume' in clean_df.columns:
            negative_volume = clean_df['volume'] < 0
            if negative_volume.any():
                logger.warning(f"Found {negative_volume.sum()} negative volume values, setting to 0")
                clean_df.loc[negative_volume, 'volume'] = 0
        
        # Sort by timestamp
        if 'timestamp' in clean_df.columns:
            clean_df = clean_df.sort_values('timestamp').reset_index(drop=True)
        
        logger.debug(f"Data cleaned: {len(clean_df)} rows remaining")
        return clean_df
    
    def validate_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enterprise-grade OHLCV data validation with comprehensive checks
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Validated DataFrame with quality scores
        """
        if df.empty:
            return df
            
        logger.debug("Performing comprehensive OHLCV validation...")
        validated_df = df.copy()
        
        # Track validation issues
        validation_issues = 0
        
        # 1. Price ordering validation (Low ≤ Open,Close ≤ High)
        price_order_invalid = (
            (validated_df['low'] > validated_df['open']) |
            (validated_df['low'] > validated_df['close']) |
            (validated_df['high'] < validated_df['open']) |
            (validated_df['high'] < validated_df['close'])
        )
        
        if price_order_invalid.any():
            invalid_count = price_order_invalid.sum()
            validation_issues += invalid_count
            logger.warning(f"Found {invalid_count} rows with invalid OHLC price ordering")
            
            # Fix by adjusting high/low to accommodate open/close
            validated_df.loc[price_order_invalid, 'high'] = validated_df.loc[price_order_invalid, [
                'open', 'high', 'close']].max(axis=1)
            validated_df.loc[price_order_invalid, 'low'] = validated_df.loc[price_order_invalid, [
                'open', 'low', 'close']].min(axis=1)
        
        # 2. Volume validation
        if 'volume' in validated_df.columns:
            invalid_volume = validated_df['volume'] < MIN_VOLUME
            if invalid_volume.any():
                validation_issues += invalid_volume.sum()
                logger.warning(f"Found {invalid_volume.sum()} rows with volume below minimum {MIN_VOLUME}")
                validated_df.loc[invalid_volume, 'volume'] = MIN_VOLUME
        
        # 3. Outlier detection using price change percentage
        if len(validated_df) > 1:
            price_change_pct = validated_df['close'].pct_change().abs()
            outliers = price_change_pct > MAX_PRICE_CHANGE_PCT
            
            if outliers.any():
                outlier_count = outliers.sum()
                validation_issues += outlier_count
                logger.warning(f"Found {outlier_count} potential outliers with >50% price change")
                
                # Cap extreme price changes by interpolation
                validated_df.loc[outliers, 'close'] = validated_df['close'].interpolate()
        
        # 4. Add data quality score
        total_rows = len(validated_df)
        quality_score = max(0, (total_rows - validation_issues) / total_rows) if total_rows > 0 else 0
        validated_df.attrs['data_quality_score'] = quality_score
        
        if validation_issues > 0:
            logger.info(f"Data validation completed: {validation_issues} issues fixed, "
                       f"quality score: {quality_score:.3f}")
        else:
            logger.debug(f"Data validation passed: quality score: {quality_score:.3f}")
        
        return validated_df
    
    def add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators with enhanced precision"""
        if df.empty or len(df) < MIN_DATA_POINTS_FOR_INDICATORS:
            logger.warning("Insufficient data for indicator calculation")
            return df
        
        df = df.copy()
        logger.debug("Calculating basic indicators...")
        
        try:
            # Simple Moving Averages
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            
            # Exponential Moving Averages
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = bb.bollinger_wband()
            df['bb_percent'] = bb.bollinger_pband()
            
            # Average True Range (ATR)
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            
            logger.debug("Basic indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating basic indicators: {e}")
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators with enterprise constants"""
        if df.empty or len(df) < RSI_PERIOD:
            return df
        
        df = df.copy()
        logger.debug("Calculating momentum indicators with enhanced precision...")
        
        try:
            # RSI with configurable period
            df['rsi'] = ta.momentum.rsi(df['close'], window=RSI_PERIOD)
            
            # MACD with enterprise constants
            macd = ta.trend.MACD(df['close'], window_fast=MACD_FAST_PERIOD, 
                                window_slow=MACD_SLOW_PERIOD, window_sign=MACD_SIGNAL_PERIOD)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Williams %R
            df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
            
            # Commodity Channel Index (CCI)
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
            
            logger.debug("Momentum indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
        
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        if df.empty or len(df) < 20:
            return df
        
        df = df.copy()
        logger.debug("Calculating volume indicators...")
        
        try:
            # Volume Moving Average
            df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'], window=20)
            
            # On Balance Volume (OBV)
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            
            # Accumulation/Distribution Line
            df['ad_line'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
            
            # Chaikin Money Flow
            df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=20)
            
            # Volume Weighted Average Price (VWAP) - for intraday
            if len(df) > 50:  # Only calculate for sufficient data
                df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            
            logger.debug("Volume indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
        
        return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators"""
        if df.empty or len(df) < 20:
            return df
        
        df = df.copy()
        logger.debug("Calculating volatility indicators...")
        
        try:
            # Historical Volatility (20-period)
            df['volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
            
            # Keltner Channels
            keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=20)
            df['keltner_upper'] = keltner.keltner_channel_hband()
            df['keltner_middle'] = keltner.keltner_channel_mband()
            df['keltner_lower'] = keltner.keltner_channel_lband()
            
            # Donchian Channels
            donchian = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'], window=20)
            df['donchian_upper'] = donchian.donchian_channel_hband()
            df['donchian_lower'] = donchian.donchian_channel_lband()
            
            logger.debug("Volatility indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
        
        return df
    
    def add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern recognition"""
        if df.empty or len(df) < 5:
            return df
        
        df = df.copy()
        logger.debug("Calculating price patterns...")
        
        try:
            # Price change and percentage change
            df['price_change'] = df['close'] - df['close'].shift(1)
            df['price_change_pct'] = df['close'].pct_change() * 100
            
            # High-Low range
            df['hl_range'] = df['high'] - df['low']
            df['hl_range_pct'] = (df['hl_range'] / df['close']) * 100
            
            # Body size (for candlestick analysis)
            df['body_size'] = abs(df['close'] - df['open'])
            df['body_size_pct'] = (df['body_size'] / df['close']) * 100
            
            # Upper and lower shadows
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            
            # Doji pattern (small body relative to range)
            df['is_doji'] = df['body_size_pct'] < 0.1
            
            # Green/Red candles
            df['is_green'] = df['close'] > df['open']
            df['is_red'] = df['close'] < df['open']
            
            # Gaps
            df['gap_up'] = df['low'] > df['high'].shift(1)
            df['gap_down'] = df['high'] < df['low'].shift(1)
            
            logger.debug("Price patterns calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating price patterns: {e}")
        
        return df
    
    def add_support_resistance_levels(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add support and resistance level detection"""
        if df.empty or len(df) < window * 2:
            return df
        
        df = df.copy()
        logger.debug("Calculating support/resistance levels...")
        
        try:
            # Rolling local maxima and minima
            df['local_max'] = df['high'].rolling(window=window, center=True).apply(
                lambda x: x.iloc[window//2] == x.max() if len(x) == window else False
            )
            
            df['local_min'] = df['low'].rolling(window=window, center=True).apply(
                lambda x: x.iloc[window//2] == x.min() if len(x) == window else False
            )
            
            # Distance to recent support/resistance
            resistance_levels = df[df['local_max'] == True]['high'].tail(5).values
            support_levels = df[df['local_min'] == True]['low'].tail(5).values
            
            if len(resistance_levels) > 0:
                df['distance_to_resistance'] = df['close'].apply(
                    lambda x: min([abs(x - level) for level in resistance_levels])
                )
                df['resistance_pct'] = (df['distance_to_resistance'] / df['close']) * 100
            
            if len(support_levels) > 0:
                df['distance_to_support'] = df['close'].apply(
                    lambda x: min([abs(x - level) for level in support_levels])
                )
                df['support_pct'] = (df['distance_to_support'] / df['close']) * 100
            
            logger.debug("Support/resistance levels calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance levels: {e}")
        
        return df
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all available technical indicators"""
        logger.info("Adding all technical indicators...")
        
        # Clean the data first
        df = self.clean_data(df)
        
        if df.empty:
            logger.warning("No data to process after cleaning")
            return df
        
        # Add indicators in sequence
        df = self.add_basic_indicators(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volume_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_price_patterns(df)
        df = self.add_support_resistance_levels(df)
        
        # Calculate final derived indicators
        try:
            # Trend strength
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                df['trend_strength'] = (df['sma_20'] - df['sma_50']) / df['sma_50'] * 100
            
            # Momentum score (composite)
            momentum_cols = ['rsi', 'macd', 'stoch_k', 'williams_r']
            available_momentum = [col for col in momentum_cols if col in df.columns]
            
            if available_momentum:
                # Normalize indicators to 0-100 scale for composite score
                momentum_normalized = pd.DataFrame()
                
                if 'rsi' in df.columns:
                    momentum_normalized['rsi_norm'] = df['rsi']  # Already 0-100
                
                if 'stoch_k' in df.columns:
                    momentum_normalized['stoch_norm'] = df['stoch_k']  # Already 0-100
                
                if 'williams_r' in df.columns:
                    momentum_normalized['williams_norm'] = (df['williams_r'] + 100)  # Convert -100,0 to 0,100
                
                if 'macd' in df.columns:
                    # Normalize MACD to 0-100 using percentile ranks
                    momentum_normalized['macd_norm'] = df['macd'].rank(pct=True) * 100
                
                # Calculate composite momentum score
                df['momentum_score'] = momentum_normalized.mean(axis=1)
            
            logger.info(f"All indicators added successfully. DataFrame shape: {df.shape}")
            
        except Exception as e:
            logger.error(f"Error calculating derived indicators: {e}")
        
        return df
    
    def get_latest_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract the most recent indicator values for signal generation"""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        signals = {}
        
        # Extract key indicator values
        indicator_columns = [
            'close', 'rsi', 'macd', 'macd_signal', 'bb_percent', 'stoch_k',
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'atr', 'momentum_score',
            'trend_strength', 'volatility', 'price_change_pct'
        ]
        
        for col in indicator_columns:
            if col in df.columns and not pd.isna(latest[col]):
                signals[col] = float(latest[col])
        
        return signals
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate data quality and return quality metrics"""
        if df.empty:
            return {"valid": False, "reason": "Empty dataset"}
        
        quality_report = {
            "valid": True,
            "total_rows": len(df),
            "null_percentages": {},
            "data_range": {},
            "outliers": {},
            "gaps": 0
        }
        
        # Check for null values
        for col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            quality_report["null_percentages"][col] = round(null_pct, 2)
        
        # Check data ranges for price columns
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                quality_report["data_range"][col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean())
                }
        
        # Check for outliers (using price change)
        if 'price_change_pct' in df.columns:
            price_changes = df['price_change_pct'].dropna()
            if len(price_changes) > 0:
                q99 = price_changes.quantile(0.99)
                q1 = price_changes.quantile(0.01)
                outliers = ((price_changes > q99) | (price_changes < q1)).sum()
                quality_report["outliers"]["price_change"] = int(outliers)
        
        # Check for time gaps (if timestamp column exists)
        if 'timestamp' in df.columns and len(df) > 1:
            time_diffs = df['timestamp'].diff().dropna()
            expected_diff = time_diffs.median()
            gaps = (time_diffs > expected_diff * 2).sum()
            quality_report["gaps"] = int(gaps)
        
        # Determine overall validity
        max_null_threshold = 10  # 10% null values
        if any(pct > max_null_threshold for pct in quality_report["null_percentages"].values()):
            quality_report["valid"] = False
            quality_report["reason"] = "Too many null values"
        
        return quality_report
    
    def calculate_indicators_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation of all indicators for enhanced performance
        
        Uses numpy vectorization and efficient pandas operations to calculate
        multiple indicators simultaneously for better performance on large datasets.
        """
        if df.empty or len(df) < MIN_DATA_POINTS_FOR_INDICATORS:
            logger.warning("Insufficient data for vectorized indicator calculation")
            return df
            
        logger.debug("Starting vectorized indicator calculation...")
        start_time = pd.Timestamp.now()
        
        # Work with numpy arrays for maximum performance
        prices = df[['open', 'high', 'low', 'close', 'volume']].values
        close_prices = prices[:, 3]  # Close is column 3
        high_prices = prices[:, 1]   # High is column 1
        low_prices = prices[:, 2]    # Low is column 2
        volume = prices[:, 4]        # Volume is column 4
        
        result_df = df.copy()
        
        try:
            # Vectorized Simple Moving Averages
            result_df['sma_20'] = pd.Series(close_prices).rolling(window=20, min_periods=1).mean()
            result_df['sma_50'] = pd.Series(close_prices).rolling(window=50, min_periods=1).mean()
            
            # Vectorized Exponential Moving Averages
            close_series = pd.Series(close_prices)
            result_df['ema_12'] = close_series.ewm(span=12, adjust=False).mean()
            result_df['ema_26'] = close_series.ewm(span=26, adjust=False).mean()
            
            # Vectorized Price Changes and Returns
            result_df['price_change'] = np.diff(close_prices, prepend=close_prices[0])
            result_df['returns'] = np.concatenate([[0], np.diff(np.log(close_prices))])
            
            # Vectorized Volatility (rolling standard deviation of returns)
            returns_series = pd.Series(result_df['returns'])
            result_df['volatility'] = returns_series.rolling(window=20, min_periods=1).std()
            
            # Vectorized High-Low range calculations
            hl_range = high_prices - low_prices
            result_df['hl_range'] = hl_range
            result_df['hl_range_pct'] = (hl_range / close_prices) * 100
            
            # Vectorized Volume-based indicators
            volume_series = pd.Series(volume)
            result_df['volume_sma'] = volume_series.rolling(window=20, min_periods=1).mean()
            result_df['volume_ratio'] = volume / result_df['volume_sma'].fillna(volume.mean())
            
            # Performance logging
            calculation_time = (pd.Timestamp.now() - start_time).total_seconds()
            logger.info(f"Vectorized indicator calculation completed in {calculation_time:.3f}s "
                       f"for {len(df)} data points")
            
        except Exception as e:
            logger.error(f"Error in vectorized indicator calculation: {e}")
            
        return result_df
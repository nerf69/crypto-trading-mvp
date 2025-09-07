"""
Enterprise quantitative trading constants for crypto trading system.

This module centralizes all magic numbers and configuration constants
to improve code maintainability and reduce errors in financial calculations.
"""

from decimal import Decimal
from typing import Dict, List

# =============================================================================
# SIGNAL CONFIDENCE THRESHOLDS
# =============================================================================
MIN_SIGNAL_CONFIDENCE = 0.6  # Minimum confidence required to act on signals
HIGH_CONFIDENCE_THRESHOLD = 0.8  # High confidence for maximum position sizing
MEDIUM_CONFIDENCE_THRESHOLD = 0.6  # Medium confidence for moderate position sizing

# =============================================================================
# POSITION SIZING CONSTANTS
# =============================================================================
MAX_POSITION_SIZE = 1.0  # 100% of available capital for highest confidence signals
MEDIUM_POSITION_SIZE = 0.66  # 66% for medium confidence signals
MIN_POSITION_SIZE = 0.33  # 33% for low confidence signals

# =============================================================================
# RISK MANAGEMENT CONSTANTS
# =============================================================================
DEFAULT_STOP_LOSS_PCT = Decimal('0.05')  # 5% stop loss
DEFAULT_COMMISSION_RATE = Decimal('0.005')  # 0.5% commission per trade
DEFAULT_SLIPPAGE_PCT = Decimal('0.001')  # 0.1% slippage
MAX_PORTFOLIO_RISK = Decimal('0.20')  # Maximum 20% portfolio at risk
MAX_SINGLE_POSITION_RISK = Decimal('0.05')  # Maximum 5% risk per position

# =============================================================================
# COINBASE API CONSTANTS  
# =============================================================================
COINBASE_MAX_CANDLES_PER_REQUEST = 250  # Safe limit below Coinbase's 300 limit
COINBASE_RATE_LIMIT_DELAY = 0.1  # 100ms delay between requests
COINBASE_DEFAULT_TIMEOUT = 30  # 30 second request timeout
COINBASE_RETRY_ATTEMPTS = 3  # Number of retry attempts for failed requests

# Coinbase supported granularities (in seconds)
COINBASE_SUPPORTED_GRANULARITIES: List[int] = [60, 300, 900, 3600, 21600, 86400]

# Granularity mapping for readability
GRANULARITY_MAP: Dict[str, int] = {
    '1m': 60,
    '5m': 300,
    '15m': 900,
    '1h': 3600,
    '6h': 21600,
    '1d': 86400
}

# =============================================================================
# STRATEGY CONSTANTS
# =============================================================================
DEFAULT_SWING_THRESHOLD = 0.025  # 2.5% swing threshold
DEFAULT_VOLUME_THRESHOLD = 1.1  # Volume must be 1.1x average
DEFAULT_RSI_OVERSOLD = 30  # RSI oversold level
DEFAULT_RSI_OVERBOUGHT = 70  # RSI overbought level
DEFAULT_LOOKBACK_PERIOD = 10  # Default lookback period for calculations

# =============================================================================
# INDICATOR CALCULATION CONSTANTS
# =============================================================================
MIN_DATA_POINTS_FOR_INDICATORS = 20  # Minimum data points to calculate indicators
RSI_PERIOD = 14  # Standard RSI calculation period
MACD_FAST_PERIOD = 12  # MACD fast EMA period
MACD_SLOW_PERIOD = 26  # MACD slow EMA period
MACD_SIGNAL_PERIOD = 9  # MACD signal line period
BOLLINGER_PERIOD = 20  # Bollinger Bands period
BOLLINGER_STD_DEV = 2  # Bollinger Bands standard deviation multiplier

# =============================================================================
# DATA QUALITY CONSTANTS
# =============================================================================
MAX_PRICE_CHANGE_PCT = 0.5  # 50% maximum single-period price change (outlier detection)
MIN_VOLUME = 1.0  # Minimum volume to consider valid
MAX_SPREAD_PCT = 0.1  # 10% maximum bid-ask spread assumption
PRICE_PRECISION = 8  # Number of decimal places for price calculations

# =============================================================================
# PERFORMANCE METRICS CONSTANTS
# =============================================================================
RISK_FREE_RATE = Decimal('0.02')  # 2% annual risk-free rate for Sharpe calculation
TRADING_DAYS_PER_YEAR = 365  # Days per year for annualization
BENCHMARK_RETURN = Decimal('0.10')  # 10% annual benchmark return

# =============================================================================
# DATABASE CONSTANTS
# =============================================================================
DEFAULT_DB_PATH = "data/trading.db"
MAX_CACHE_AGE_DAYS = 30  # Maximum age for cached data
DB_CONNECTION_TIMEOUT = 30  # Database connection timeout in seconds

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================
MIN_PRICE = Decimal('0.000001')  # Minimum valid price
MAX_PRICE = Decimal('1000000')  # Maximum valid price (sanity check)
MAX_POSITION_COUNT = 10  # Maximum number of concurrent positions
MIN_TRADE_SIZE_USD = Decimal('10.00')  # Minimum trade size in USD

# =============================================================================
# LOGGING AND MONITORING
# =============================================================================
MAX_LOG_FILE_SIZE_MB = 100  # Maximum log file size in MB
LOG_RETENTION_DAYS = 30  # Number of days to retain logs
PERFORMANCE_SAMPLE_RATE = 0.1  # 10% sampling rate for performance monitoring
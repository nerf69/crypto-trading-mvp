"""
Pytest configuration and shared fixtures for the crypto trading application tests.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sqlite3
from pathlib import Path

# Add src to Python path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducible tests
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')  # 60 days for sufficient data
    
    # Generate realistic price data
    base_price = 100
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = []
    for i, date in enumerate(dates):
        price = prices[i]
        # Generate OHLC from price
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = price * (1 + np.random.normal(0, 0.005))
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': np.random.uniform(1000, 10000)
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_ohlcv_data_with_indicators(sample_ohlcv_data):
    """Generate sample OHLCV data with basic indicators calculated"""
    df = sample_ohlcv_data
    
    # Add basic indicators
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['rsi'] = 50 + np.random.normal(0, 15, len(df))  # Simplified RSI
    df['rsi'] = df['rsi'].clip(0, 100)
    
    # MACD components (simplified)
    df['macd'] = np.random.normal(0, 2, len(df))
    df['macd_signal'] = np.random.normal(0, 1.5, len(df))
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    return df


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file for testing"""
    config_content = """
# Test Configuration
exchange:
  name: "coinbase"
  base_url: "https://api.exchange.coinbase.com"

trading_pairs:
  - "BTC-USD"
  - "ETH-USD"
  - "SOL-USD"

trading:
  initial_capital: 1000
  max_positions: 3
  position_sizing:
    method: "dynamic"
    max_position_size: 1.0
    min_position_size: 0.33

risk:
  stop_loss: 0.05
  max_daily_loss: 0.10

backtesting:
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  initial_capital: 1000
  commission: 0.005

notifications:
  email:
    enabled: false
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    from_email: "test@example.com"
    to_email: "test@example.com"

strategy_weights:
  swing: 1.0
  rsi: 0.8
  macd: 0.7
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def temp_database():
    """Create a temporary SQLite database for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_db_path = f.name
    
    # Initialize database
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ohlcv_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(pair, timestamp)
        )
    ''')
    
    conn.commit()
    conn.close()
    
    yield temp_db_path
    
    # Cleanup
    os.unlink(temp_db_path)


@pytest.fixture
def mock_coinbase_response():
    """Mock successful Coinbase API response"""
    return [
        [1640995200, 47000.00, 48000.00, 46500.00, 47500.00, 1234.56],
        [1641081600, 47500.00, 48500.00, 47000.00, 48000.00, 2345.67],
        [1641168000, 48000.00, 49000.00, 47500.00, 48500.00, 3456.78],
    ]


@pytest.fixture
def mock_coinbase_ticker():
    """Mock Coinbase ticker response for current price"""
    return {
        "trade_id": 1234567,
        "price": "48000.00",
        "size": "0.1",
        "time": "2024-01-01T12:00:00.000000Z",
        "bid": "47990.00",
        "ask": "48010.00",
        "volume": "12345.67"
    }


@pytest.fixture
def mock_requests_session():
    """Create a mock requests session for testing API calls"""
    mock_session = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_session.get.return_value = mock_response
    return mock_session


@pytest.fixture(autouse=True)
def clean_config_cache():
    """Clean the global config cache before each test"""
    import src.config
    src.config._config_cache.clear()
    yield
    # Reset again after test
    src.config._config_cache.clear()


@pytest.fixture
def edge_case_data():
    """Generate data with edge cases for testing robustness"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    
    data = []
    for i, date in enumerate(dates):
        if i == 0:
            # Normal data point
            data.append({
                'timestamp': date,
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 102.0,
                'volume': 1000.0
            })
        elif i == 1:
            # Zero volume
            data.append({
                'timestamp': date,
                'open': 102.0,
                'high': 103.0,
                'low': 101.0,
                'close': 101.5,
                'volume': 0.0
            })
        elif i == 2:
            # High volatility
            data.append({
                'timestamp': date,
                'open': 101.5,
                'high': 150.0,  # 50% spike
                'low': 50.0,    # 50% drop
                'close': 100.0,
                'volume': 50000.0
            })
        elif i == 3:
            # Identical OHLC (no movement)
            price = 100.0
            data.append({
                'timestamp': date,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 1000.0
            })
        else:
            # Normal data with slight variations
            base = 100 + (i - 4) * 0.5
            data.append({
                'timestamp': date,
                'open': base,
                'high': base * 1.01,
                'low': base * 0.99,
                'close': base + np.random.normal(0, 0.5),
                'volume': 1000.0 + np.random.uniform(-100, 100)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def strategy_test_data():
    """Generate specific test data for strategy testing"""
    dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='D')
    
    # Create data that should trigger specific strategy signals
    data = []
    base_price = 100.0
    
    for i, date in enumerate(dates):
        if i < 5:
            # Initial steady period
            price = base_price + i * 0.1
        elif i < 10:
            # Downtrend (should trigger oversold)
            price = base_price - (i - 4) * 2
        elif i < 15:
            # Recovery (should trigger buy signals)
            price = base_price - 10 + (i - 9) * 1.5
        elif i < 20:
            # Uptrend (should trigger overbought)
            price = base_price + (i - 14) * 2
        else:
            # Sideways movement
            price = base_price + 10 + np.sin(i - 19) * 2
        
        # Generate OHLC
        high = price * 1.02
        low = price * 0.98
        open_price = price + np.random.normal(0, 0.5)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': 1000.0 + np.random.uniform(-200, 200)
        })
    
    return pd.DataFrame(data)


# Test markers for categorizing tests
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require API access"
    )
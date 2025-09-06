"""
Unit tests for data fetcher module.
"""

import pytest
import pandas as pd
import sqlite3
import tempfile
import os
import requests
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, call
import time

from src.data.fetcher import CoinbaseDataFetcher


class TestCoinbaseDataFetcher:
    """Test the CoinbaseDataFetcher class"""
    
    def test_init_with_default_parameters(self):
        """Test fetcher initialization with default parameters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'test.db')
            fetcher = CoinbaseDataFetcher(db_path=db_path)
            
            assert fetcher.base_url == "https://api.exchange.coinbase.com"
            assert fetcher.db_path == db_path
            assert fetcher.rate_limit_delay == 0.1
            assert os.path.exists(db_path)
    
    def test_init_with_custom_parameters(self):
        """Test fetcher initialization with custom parameters"""
        custom_url = "https://custom.api.url"
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'custom.db')
            fetcher = CoinbaseDataFetcher(base_url=custom_url, db_path=db_path)
            
            assert fetcher.base_url == custom_url
            assert fetcher.db_path == db_path
    
    def test_database_initialization(self, temp_database):
        """Test that database is properly initialized"""
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        
        # Check that table exists
        conn = sqlite3.connect(temp_database)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='ohlcv_data'
        """)
        
        assert cursor.fetchone() is not None
        conn.close()
    
    @patch('src.data.fetcher.requests.Session')
    def test_make_request_success(self, mock_session_class, temp_database):
        """Test successful API request"""
        # Setup mock
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data"}
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        fetcher.session = mock_session
        
        result = fetcher._make_request('/test-endpoint')
        
        assert result == {"test": "data"}
        mock_session.get.assert_called_once_with('https://api.exchange.coinbase.com/test-endpoint')
        mock_response.raise_for_status.assert_called_once()
    
    @patch('src.data.fetcher.requests.Session')
    def test_make_request_with_params(self, mock_session_class, temp_database):
        """Test API request with parameters"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        fetcher.session = mock_session
        
        params = {'start': '2024-01-01', 'end': '2024-01-02'}
        result = fetcher._make_request('/test', params=params)
        
        assert result == {"result": "success"}
        mock_session.get.assert_called_once_with(
            'https://api.exchange.coinbase.com/test', 
            params=params
        )
    
    @patch('src.data.fetcher.requests.Session')
    def test_make_request_http_error(self, mock_session_class, temp_database):
        """Test API request with HTTP error"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        fetcher.session = mock_session
        
        with pytest.raises(requests.exceptions.HTTPError):
            fetcher._make_request('/nonexistent')
    
    @patch('src.data.fetcher.requests.Session')
    def test_rate_limiting(self, mock_session_class, temp_database):
        """Test that rate limiting is enforced"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        fetcher.session = mock_session
        fetcher.rate_limit_delay = 0.01  # Small delay for testing
        
        start_time = time.time()
        fetcher._make_request('/test1')
        fetcher._make_request('/test2')
        end_time = time.time()
        
        # Should have waited at least the rate limit delay
        assert end_time - start_time >= fetcher.rate_limit_delay
    
    @patch('src.data.fetcher.CoinbaseDataFetcher._make_request')
    def test_get_current_price_success(self, mock_request, temp_database):
        """Test successful current price retrieval"""
        mock_request.return_value = {
            "trade_id": 12345,
            "price": "50000.00",
            "size": "0.1",
            "time": "2024-01-01T12:00:00Z"
        }
        
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        price = fetcher.get_current_price('BTC-USD')
        
        assert price == 50000.00
        mock_request.assert_called_once_with('/products/BTC-USD/ticker')
    
    @patch('src.data.fetcher.CoinbaseDataFetcher._make_request')
    def test_get_current_price_invalid_response(self, mock_request, temp_database):
        """Test current price with invalid response"""
        mock_request.return_value = {"invalid": "response"}
        
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        price = fetcher.get_current_price('BTC-USD')
        
        assert price is None
    
    @patch('src.data.fetcher.CoinbaseDataFetcher._make_request')
    def test_get_current_price_exception(self, mock_request, temp_database):
        """Test current price with API exception"""
        mock_request.side_effect = Exception("API Error")
        
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        price = fetcher.get_current_price('BTC-USD')
        
        assert price is None
    
    @patch('src.data.fetcher.CoinbaseDataFetcher._make_request')
    def test_get_historical_data_success(self, mock_request, temp_database, mock_coinbase_response):
        """Test successful historical data retrieval"""
        mock_request.return_value = mock_coinbase_response
        
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        df = fetcher.get_historical_data('BTC-USD', '2024-01-01', '2024-01-03')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'timestamp' in df.columns
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        
        # Check data types
        assert df['timestamp'].dtype.name.startswith('datetime')
        assert df['open'].dtype in ['float64', 'float32']
        assert df['close'].dtype in ['float64', 'float32']
    
    @patch('src.data.fetcher.CoinbaseDataFetcher._make_request')
    def test_get_historical_data_empty_response(self, mock_request, temp_database):
        """Test historical data with empty response"""
        mock_request.return_value = []
        
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        df = fetcher.get_historical_data('BTC-USD', '2024-01-01', '2024-01-02')
        
        assert df is None
    
    @patch('src.data.fetcher.CoinbaseDataFetcher._make_request')
    def test_get_historical_data_with_granularity(self, mock_request, temp_database, mock_coinbase_response):
        """Test historical data with custom granularity"""
        mock_request.return_value = mock_coinbase_response
        
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        df = fetcher.get_historical_data('BTC-USD', '2024-01-01', '2024-01-02', granularity=3600)
        
        # Check that granularity parameter was passed
        mock_request.assert_called_once()
        call_args = mock_request.call_args[1]  # kwargs
        assert 'params' in call_args
        assert call_args['params']['granularity'] == 3600
    
    def test_store_data_to_database(self, temp_database):
        """Test storing data to database"""
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        
        # Create sample data
        data = [
            [1640995200, 47000.00, 48000.00, 46500.00, 47500.00, 1234.56],
            [1641081600, 47500.00, 48500.00, 47000.00, 48000.00, 2345.67]
        ]
        
        fetcher._store_data('BTC-USD', data)
        
        # Verify data was stored
        conn = sqlite3.connect(temp_database)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT pair, timestamp, open, high, low, close, volume 
            FROM ohlcv_data 
            WHERE pair = 'BTC-USD'
            ORDER BY timestamp
        """)
        
        rows = cursor.fetchall()
        assert len(rows) == 2
        
        # Check first row
        assert rows[0][0] == 'BTC-USD'  # pair
        assert rows[0][1] == 1640995200  # timestamp
        assert rows[0][2] == 47000.00  # open
        assert rows[0][5] == 47500.00  # close
        
        conn.close()
    
    def test_store_data_duplicate_handling(self, temp_database):
        """Test handling of duplicate data in database"""
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        
        # Store initial data
        data = [[1640995200, 47000.00, 48000.00, 46500.00, 47500.00, 1234.56]]
        fetcher._store_data('BTC-USD', data)
        
        # Try to store the same data again (should not duplicate)
        fetcher._store_data('BTC-USD', data)
        
        # Verify only one row exists
        conn = sqlite3.connect(temp_database)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM ohlcv_data WHERE pair = 'BTC-USD'")
        count = cursor.fetchone()[0]
        assert count == 1
        
        conn.close()
    
    def test_get_cached_data(self, temp_database):
        """Test retrieving cached data from database"""
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        
        # Store some test data
        data = [
            [1640995200, 47000.00, 48000.00, 46500.00, 47500.00, 1234.56],
            [1641081600, 47500.00, 48500.00, 47000.00, 48000.00, 2345.67]
        ]
        fetcher._store_data('BTC-USD', data)
        
        # Retrieve cached data
        start_date = '2024-01-01'
        end_date = '2024-01-02'
        df = fetcher._get_cached_data('BTC-USD', start_date, end_date)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'timestamp' in df.columns
        assert df.iloc[0]['open'] == 47000.00
    
    def test_get_cached_data_empty(self, temp_database):
        """Test retrieving cached data when none exists"""
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        
        df = fetcher._get_cached_data('BTC-USD', '2024-01-01', '2024-01-02')
        
        assert df is None or df.empty
    
    @patch('src.data.fetcher.CoinbaseDataFetcher.get_historical_data')
    def test_get_latest_data(self, mock_get_historical, temp_database):
        """Test getting latest data with specified periods"""
        # Mock return data
        mock_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='D'),
            'open': [100] * 50,
            'high': [105] * 50,
            'low': [95] * 50,
            'close': [102] * 50,
            'volume': [1000] * 50
        })
        mock_get_historical.return_value = mock_df
        
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        result = fetcher.get_latest_data('BTC-USD', periods=50)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 50
        mock_get_historical.assert_called_once()
    
    def test_convert_granularity_to_seconds(self, temp_database):
        """Test granularity conversion to seconds"""
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        
        # Test valid granularities
        assert fetcher._convert_granularity_to_seconds(60) == 60
        assert fetcher._convert_granularity_to_seconds(300) == 300
        assert fetcher._convert_granularity_to_seconds(86400) == 86400
        
        # Test invalid granularity (should default to 86400)
        assert fetcher._convert_granularity_to_seconds(12345) == 86400
    
    def test_format_coinbase_timestamp(self, temp_database):
        """Test timestamp formatting for Coinbase API"""
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        
        # Test date string formatting
        result = fetcher._format_timestamp('2024-01-01')
        assert result == '2024-01-01T00:00:00Z'
        
        # Test datetime object formatting
        dt = datetime(2024, 1, 15, 12, 30, 45)
        result = fetcher._format_timestamp(dt)
        assert result == '2024-01-15T12:30:45Z'


class TestDataValidation:
    """Test data validation and cleaning"""
    
    @patch('src.data.fetcher.CoinbaseDataFetcher._make_request')
    def test_data_cleaning_removes_invalid_entries(self, mock_request, temp_database):
        """Test that invalid data entries are cleaned"""
        # Mock data with some invalid entries
        invalid_data = [
            [1640995200, 47000.00, 48000.00, 46500.00, 47500.00, 1234.56],  # Valid
            [1641081600, 0, 0, 0, 0, 0],  # Invalid (zero prices)
            [1641168000, 48000.00, 49000.00, 47500.00, 48500.00, -100],  # Invalid (negative volume)
            [1641254400, 48500.00, 49500.00, 48000.00, 49000.00, 2345.67],  # Valid
        ]
        
        mock_request.return_value = invalid_data
        
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        df = fetcher.get_historical_data('BTC-USD', '2024-01-01', '2024-01-04')
        
        # Should only have 2 valid entries
        assert len(df) == 2
        assert all(df['open'] > 0)
        assert all(df['close'] > 0)
        assert all(df['volume'] >= 0)
    
    def test_invalid_date_format_handling(self, temp_database):
        """Test handling of invalid date formats"""
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        
        # Test with invalid date format
        with pytest.raises(ValueError):
            fetcher._format_timestamp('invalid-date-format')
    
    @patch('src.data.fetcher.CoinbaseDataFetcher._make_request')
    def test_malformed_api_response(self, mock_request, temp_database):
        """Test handling of malformed API responses"""
        # Mock malformed response
        mock_request.return_value = [
            [1640995200, "invalid", "price", "data"],  # Too few fields, invalid types
            ["invalid_timestamp", 47000.00, 48000.00],  # Invalid timestamp, too few fields
        ]
        
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        df = fetcher.get_historical_data('BTC-USD', '2024-01-01', '2024-01-02')
        
        # Should handle gracefully and return None or empty DataFrame
        assert df is None or df.empty


class TestCacheManagement:
    """Test data caching functionality"""
    
    def test_cache_data_density_check(self, temp_database):
        """Test cache data density validation"""
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        
        # Store sparse data (should be considered insufficient)
        sparse_data = [
            [1640995200, 47000.00, 48000.00, 46500.00, 47500.00, 1234.56],
            # Missing many data points
            [1641254400, 48000.00, 49000.00, 47500.00, 48500.00, 2345.67],
        ]
        fetcher._store_data('BTC-USD', sparse_data)
        
        # Should detect insufficient data density
        start_date = '2024-01-01'
        end_date = '2024-01-04'  # 4 days but only 2 data points
        
        sufficient_data = fetcher._is_cache_sufficient('BTC-USD', start_date, end_date, 86400)
        assert not sufficient_data
    
    def test_cache_time_range_validation(self, temp_database):
        """Test cache time range validation"""
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        
        # Store data for specific date range
        data = []
        for i in range(5):
            timestamp = 1640995200 + (i * 86400)  # Daily data
            data.append([timestamp, 47000 + i*100, 48000 + i*100, 46500 + i*100, 47500 + i*100, 1234.56])
        
        fetcher._store_data('BTC-USD', data)
        
        # Test range that's fully covered
        assert fetcher._is_cache_sufficient('BTC-USD', '2024-01-01', '2024-01-05', 86400)
        
        # Test range that extends beyond cached data
        assert not fetcher._is_cache_sufficient('BTC-USD', '2023-12-31', '2024-01-06', 86400)
    
    def test_database_connection_handling(self, temp_database):
        """Test proper database connection handling"""
        fetcher = CoinbaseDataFetcher(db_path=temp_database)
        
        # Test that multiple operations work correctly
        data1 = [[1640995200, 47000.00, 48000.00, 46500.00, 47500.00, 1234.56]]
        data2 = [[1641081600, 47500.00, 48500.00, 47000.00, 48000.00, 2345.67]]
        
        fetcher._store_data('BTC-USD', data1)
        fetcher._store_data('ETH-USD', data2)
        
        # Verify both datasets stored correctly
        df1 = fetcher._get_cached_data('BTC-USD', '2024-01-01', '2024-01-02')
        df2 = fetcher._get_cached_data('ETH-USD', '2024-01-01', '2024-01-02')
        
        assert len(df1) == 1
        assert len(df2) == 1
        assert df1.iloc[0]['close'] != df2.iloc[0]['close']
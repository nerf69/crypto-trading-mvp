import requests
import time
import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class CoinbaseDataFetcher:
    """
    Coinbase Pro API data fetcher with rate limiting and caching
    """
    
    def __init__(self, base_url: str = "https://api.exchange.coinbase.com", db_path: str = "data/trading.db"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.db_path = db_path
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        
        # Create database directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for caching data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for OHLCV data
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
        
        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pair_timestamp ON ohlcv_data(pair, timestamp)')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def _rate_limit(self):
        """Enforce rate limiting between API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None, retries: int = 3) -> Optional[Dict]:
        """
        Make HTTP request with exponential backoff retry logic
        """
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = (2 ** attempt) * 1.0  # Exponential backoff
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"API request failed: {response.status_code} - {response.text}")
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                wait_time = (2 ** attempt) * 1.0
                logger.error(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
                
                if attempt < retries - 1:
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All retry attempts failed for {url}")
                    return None
        
        return None
    
    def get_products(self) -> List[Dict]:
        """Get list of available trading products"""
        logger.info("Fetching available products...")
        data = self._make_request("/products")
        
        if data:
            logger.info(f"Retrieved {len(data)} products")
            return data
        
        logger.error("Failed to fetch products")
        return []
    
    def get_current_price(self, pair: str) -> Optional[float]:
        """Get current ticker price for a pair"""
        logger.debug(f"Fetching current price for {pair}")
        data = self._make_request(f"/products/{pair}/ticker")
        
        if data and 'price' in data:
            price = float(data['price'])
            logger.debug(f"{pair} current price: ${price}")
            return price
        
        logger.error(f"Failed to fetch current price for {pair}")
        return None
    
    def get_historical_data(self, pair: str, start_date: str, end_date: str, 
                          granularity: int = 300) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data for a trading pair
        
        Args:
            pair: Trading pair (e.g., 'BTC-USD')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format  
            granularity: Time slice in seconds (300 = 5min, 3600 = 1hr, 86400 = 1day)
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        logger.info(f"Fetching historical data for {pair} from {start_date} to {end_date}")
        
        # Check cache first
        cached_data = self._get_cached_data(pair, start_date, end_date)
        if cached_data is not None and len(cached_data) > 0:
            logger.info(f"Using cached data for {pair}")
            return cached_data
        
        # Convert dates to ISO format
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').isoformat() + 'Z'
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').isoformat() + 'Z'
        
        params = {
            'start': start_dt,
            'end': end_dt,
            'granularity': granularity
        }
        
        data = self._make_request(f"/products/{pair}/candles", params)
        
        if not data:
            logger.error(f"Failed to fetch historical data for {pair}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
        
        if df.empty:
            logger.warning(f"No data returned for {pair}")
            return None
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Ensure proper data types
        price_columns = ['open', 'high', 'low', 'close']
        df[price_columns] = df[price_columns].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        logger.info(f"Retrieved {len(df)} data points for {pair}")
        
        # Cache the data
        self._cache_data(pair, df)
        
        return df
    
    def _get_cached_data(self, pair: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
            
            query = '''
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_data 
                WHERE pair = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            '''
            
            df = pd.read_sql_query(query, conn, params=(pair, start_ts, end_ts))
            conn.close()
            
            if df.empty:
                return None
            
            # Convert timestamp back to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            logger.debug(f"Retrieved {len(df)} cached data points for {pair}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving cached data: {e}")
            return None
    
    def _cache_data(self, pair: str, df: pd.DataFrame):
        """Cache OHLCV data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Prepare data for insertion
            data_to_insert = []
            for _, row in df.iterrows():
                data_to_insert.append((
                    pair,
                    int(row['timestamp'].timestamp()),
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume']
                ))
            
            # Insert or ignore (handle duplicates)
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR IGNORE INTO ohlcv_data 
                (pair, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', data_to_insert)
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Cached {len(data_to_insert)} data points for {pair}")
            
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    def get_latest_data(self, pair: str, periods: int = 100) -> Optional[pd.DataFrame]:
        """
        Get the most recent OHLCV data points for a pair
        
        Args:
            pair: Trading pair
            periods: Number of recent periods to fetch
            
        Returns:
            DataFrame with recent OHLCV data
        """
        logger.debug(f"Fetching latest {periods} data points for {pair}")
        
        params = {'granularity': 300}  # 5-minute candles
        data = self._make_request(f"/products/{pair}/candles", params)
        
        if not data:
            logger.error(f"Failed to fetch latest data for {pair}")
            return None
        
        # Take only the requested number of periods
        data = data[:periods]
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
        
        if df.empty:
            return None
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Ensure proper data types
        price_columns = ['open', 'high', 'low', 'close']
        df[price_columns] = df[price_columns].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        logger.debug(f"Retrieved {len(df)} latest data points for {pair}")
        return df
    
    def get_multiple_pairs_data(self, pairs: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple trading pairs
        
        Args:
            pairs: List of trading pairs
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary mapping pair names to DataFrames
        """
        logger.info(f"Fetching data for {len(pairs)} pairs")
        
        results = {}
        for i, pair in enumerate(pairs):
            logger.info(f"Processing {pair} ({i+1}/{len(pairs)})")
            
            df = self.get_historical_data(pair, start_date, end_date)
            if df is not None:
                results[pair] = df
            else:
                logger.warning(f"No data retrieved for {pair}")
            
            # Add delay between requests for different pairs
            if i < len(pairs) - 1:
                time.sleep(0.5)
        
        logger.info(f"Successfully retrieved data for {len(results)} pairs")
        return results
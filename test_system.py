#!/usr/bin/env python3
"""
Crypto Trading MVP - System Validation Test Script

This script tests all major components of the crypto trading system:
- Configuration loading
- Coinbase API data fetching
- Technical indicator calculation
- Trading strategy signal generation
- Backtesting engine functionality

Usage:
    python test_system.py
"""

import sys
import traceback
from datetime import datetime, timedelta

# Add src to path
sys.path.append('.')

def test_configuration():
    """Test configuration loading"""
    print("🔧 Testing Configuration Loading...")
    try:
        from src.config import get_config
        
        config = get_config()
        trading_pairs = config.get_trading_pairs()
        initial_capital = config.get('trading.initial_capital')
        
        assert len(trading_pairs) > 0, "No trading pairs configured"
        assert initial_capital > 0, "Invalid initial capital"
        
        print(f"   ✅ Configuration loaded: {len(trading_pairs)} pairs, ${initial_capital} capital")
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_data_fetcher():
    """Test Coinbase API data fetching"""
    print("📊 Testing Data Fetcher...")
    try:
        from src.data.fetcher import CoinbaseDataFetcher
        
        fetcher = CoinbaseDataFetcher()
        
        # Test current price
        price = fetcher.get_current_price('ETH-USD')
        assert price and price > 0, "Invalid price data"
        
        # Test latest data
        latest_df = fetcher.get_latest_data('ETH-USD', periods=50)
        assert latest_df is not None and not latest_df.empty, "No historical data"
        assert len(latest_df) > 0, "Empty historical data"
        
        print(f"   ✅ Data fetcher working: ETH-USD at ${price:,.2f}, {len(latest_df)} data points")
        return True
        
    except Exception as e:
        print(f"   ❌ Data fetcher test failed: {e}")
        return False

def test_indicators():
    """Test technical indicator calculation"""
    print("📈 Testing Technical Indicators...")
    try:
        from src.data.fetcher import CoinbaseDataFetcher
        from src.data.processor import DataProcessor
        
        fetcher = CoinbaseDataFetcher()
        processor = DataProcessor()
        
        # Get data
        df = fetcher.get_latest_data('ETH-USD', periods=100)
        assert df is not None and not df.empty, "No data for indicators"
        
        # Add indicators
        df_with_indicators = processor.add_all_indicators(df)
        
        # Check key indicators exist
        required_indicators = ['rsi', 'macd', 'macd_signal', 'sma_20', 'bb_upper']
        for indicator in required_indicators:
            assert indicator in df_with_indicators.columns, f"Missing indicator: {indicator}"
        
        print(f"   ✅ Indicators calculated: {len(df_with_indicators.columns)} columns")
        return True
        
    except Exception as e:
        print(f"   ❌ Technical indicators test failed: {e}")
        return False

def test_strategies():
    """Test trading strategy signal generation"""
    print("🎯 Testing Trading Strategies...")
    try:
        from src.data.fetcher import CoinbaseDataFetcher
        from src.data.processor import DataProcessor
        from src.strategies.swing import SwingTradingStrategy
        from src.strategies.rsi import RSIStrategy
        from src.strategies.macd import MACDStrategy
        
        # Prepare data
        fetcher = CoinbaseDataFetcher()
        processor = DataProcessor()
        df = fetcher.get_latest_data('ETH-USD', periods=100)
        df = processor.add_all_indicators(df)
        
        strategies = [
            ('Swing', SwingTradingStrategy()),
            ('RSI', RSIStrategy()),
            ('MACD', MACDStrategy())
        ]
        
        signals = {}
        for name, strategy in strategies:
            df = strategy.add_required_indicators(df)
            signal = strategy.calculate_signal(df, 'ETH-USD')
            
            # Validate signal
            assert signal.signal is not None, f"No signal from {name} strategy"
            assert 0 <= signal.confidence <= 1, f"Invalid confidence from {name} strategy"
            assert signal.price > 0, f"Invalid price from {name} strategy"
            
            signals[name] = signal
        
        print(f"   ✅ Strategy signals: ", end="")
        for name, signal in signals.items():
            print(f"{name}={signal.signal.value}({signal.confidence:.2f}) ", end="")
        print()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Strategy test failed: {e}")
        return False

def test_backtesting():
    """Test backtesting engine"""
    print("🔄 Testing Backtesting Engine...")
    try:
        from src.backtesting.engine import BacktestEngine
        from src.strategies.swing import SwingTradingStrategy
        
        engine = BacktestEngine()
        strategy = SwingTradingStrategy()
        
        # Short backtest period
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        result = engine.run_backtest(
            strategy=strategy,
            pair='ETH-USD',
            start_date=start_date,
            end_date=end_date,
            initial_capital=1000
        )
        
        # Validate result
        assert result is not None, "Backtest returned None"
        assert result.initial_capital > 0, "Invalid initial capital"
        assert result.final_capital >= 0, "Invalid final capital"
        assert result.total_trades >= 0, "Invalid trade count"
        
        print(f"   ✅ Backtest completed: {result.total_trades} trades, {result.total_return_pct:.2f}% return")
        return True
        
    except Exception as e:
        print(f"   ❌ Backtesting test failed: {e}")
        return False

def test_database_persistence():
    """Test data caching and persistence"""
    print("🗄️  Testing Database Persistence...")
    try:
        from src.data.fetcher import CoinbaseDataFetcher
        import os
        
        # Check if database is created
        fetcher = CoinbaseDataFetcher()
        fetcher.get_latest_data('ETH-USD', periods=10)  # This should create/use database
        
        # Check if database file exists
        db_path = "data/trading.db"
        assert os.path.exists(db_path), "Database file not created"
        
        print(f"   ✅ Database persistence working: {db_path} exists")
        return True
        
    except Exception as e:
        print(f"   ❌ Database persistence test failed: {e}")
        return False

def run_all_tests():
    """Run all system tests"""
    print("🚀 Starting Crypto Trading MVP System Tests\n")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Data Fetcher", test_data_fetcher), 
        ("Technical Indicators", test_indicators),
        ("Trading Strategies", test_strategies),
        ("Backtesting Engine", test_backtesting),
        ("Database Persistence", test_database_persistence)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ {test_name} test crashed: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ ALL TESTS PASSED! The crypto trading system is ready to use.")
        print("\nNext steps:")
        print("1. Copy .env.template to .env and add your credentials")
        print("2. Run longer backtests with historical data")
        print("3. Implement performance metrics and dashboard")
        print("4. Set up paper trading for live testing")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
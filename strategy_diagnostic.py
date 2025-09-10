#!/usr/bin/env python3
"""
Strategy Diagnostic Tool
Analyzes why strategies are not generating trades and examines market data
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from src.backtesting.engine import BacktestEngine
from src.strategies.pure_percent import Pure5PercentStrategy
from src.strategies.rsi import RSIStrategy
from src.strategies.macd import MACDStrategy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_market_data(pair: str, start_date: str, end_date: str):
    """Analyze market data characteristics"""
    
    engine = BacktestEngine()
    
    # Fetch data
    df = engine.data_fetcher.get_historical_data(pair, start_date, end_date, granularity=86400)
    
    if df is None or df.empty:
        print(f"❌ No data available for {pair}")
        return None
        
    print(f"\n📊 MARKET DATA ANALYSIS - {pair}")
    print("=" * 50)
    
    # Basic statistics
    print(f"Data points: {len(df)}")
    print(f"Date range: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()
    df['abs_daily_return'] = df['daily_return'].abs()
    
    # Price movement statistics
    print(f"\nPrice Movement Statistics:")
    print(f"Average daily return: {df['daily_return'].mean()*100:.3f}%")
    print(f"Daily volatility: {df['daily_return'].std()*100:.3f}%")
    print(f"Max single-day gain: {df['daily_return'].max()*100:.2f}%")
    print(f"Max single-day loss: {df['daily_return'].min()*100:.2f}%")
    
    # Movement threshold analysis
    print(f"\nDaily Movements Analysis:")
    movements_2pct = (df['abs_daily_return'] >= 0.02).sum()
    movements_3pct = (df['abs_daily_return'] >= 0.03).sum()
    movements_5pct = (df['abs_daily_return'] >= 0.05).sum()
    
    print(f"Days with ≥2% movement: {movements_2pct} ({movements_2pct/len(df)*100:.1f}%)")
    print(f"Days with ≥3% movement: {movements_3pct} ({movements_3pct/len(df)*100:.1f}%)")
    print(f"Days with ≥5% movement: {movements_5pct} ({movements_5pct/len(df)*100:.1f}%)")
    
    return df

def test_strategy_signals(pair: str, start_date: str, end_date: str):
    """Test individual strategy signal generation"""
    
    engine = BacktestEngine()
    
    # Fetch and prepare data
    df = engine._fetch_and_prepare_data(pair, start_date, end_date)
    if df is None:
        return
        
    print(f"\n🧪 STRATEGY SIGNAL TESTING - {pair}")
    print("=" * 50)
    
    strategies_to_test = [
        ("Pure 2%", Pure5PercentStrategy(drop_threshold=0.02, rise_threshold=0.02)),
        ("Pure 5%", Pure5PercentStrategy(drop_threshold=0.05, rise_threshold=0.05)),
        ("RSI Standard", RSIStrategy()),
    ]
    
    for name, strategy in strategies_to_test:
        print(f"\n🔍 Testing {name}:")
        
        try:
            # Add required indicators
            df_with_indicators = strategy.add_required_indicators(df.copy())
            
            # Test signal generation on last 20 data points
            signals_generated = []
            hold_reasons = []
            
            for i in range(max(50, len(df_with_indicators) - 20), len(df_with_indicators)):
                if i >= len(df_with_indicators):
                    continue
                    
                test_df = df_with_indicators.iloc[:i+1]
                signal = strategy.calculate_signal(test_df, pair)
                
                signals_generated.append(signal.signal.value)
                if signal.signal.value == 'HOLD':
                    hold_reasons.append(signal.reason)
            
            # Count signal types
            signal_counts = {}
            for sig in signals_generated:
                signal_counts[sig] = signal_counts.get(sig, 0) + 1
            
            print(f"  Signal distribution: {signal_counts}")
            
            if hold_reasons:
                # Show most common hold reasons
                reason_counts = {}
                for reason in hold_reasons[-5:]:  # Last 5 hold reasons
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                
                print(f"  Recent hold reasons:")
                for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {reason}")
            
            # Check if any buy/sell signals were generated
            non_hold_signals = [s for s in signals_generated if s != 'HOLD']
            if non_hold_signals:
                print(f"  ✅ Generated {len(non_hold_signals)} trading signals!")
            else:
                print(f"  ⚠️ No trading signals generated - all HOLD")
                
        except Exception as e:
            print(f"  ❌ Error testing {name}: {str(e)}")

def analyze_pure_strategy_logic(pair: str, start_date: str, end_date: str):
    """Deep dive into Pure Percentage Strategy logic"""
    
    engine = BacktestEngine()
    df = engine._fetch_and_prepare_data(pair, start_date, end_date)
    
    if df is None or len(df) < 10:
        return
    
    print(f"\n🔎 PURE STRATEGY DEEP ANALYSIS - {pair}")
    print("=" * 50)
    
    strategy = Pure5PercentStrategy(drop_threshold=0.02, rise_threshold=0.02)
    df_with_indicators = strategy.add_required_indicators(df.copy())
    
    # Look at recent 20 data points
    recent_df = df_with_indicators.tail(20).copy()
    
    print(f"Recent price movements (last 20 days):")
    print(f"{'Date':<12} {'Close':<8} {'Daily%':<8} {'3D%':<8} {'5D%':<8}")
    print("-" * 50)
    
    for i, row in recent_df.iterrows():
        daily_pct = row.get('daily_return', 0) * 100
        pct_3d = row.get('pct_change_3d', 0)
        pct_5d = row.get('pct_change_5d', 0)
        
        print(f"{str(row['timestamp'])[:10]:<12} ${row['close']:<7.2f} {daily_pct:+6.2f}% {pct_3d:+6.2f}% {pct_5d:+6.2f}%")
    
    # Test signal generation on the last few points
    print(f"\nSignal generation test:")
    for i in range(max(0, len(df_with_indicators) - 5), len(df_with_indicators)):
        test_df = df_with_indicators.iloc[:i+1]
        signal = strategy.calculate_signal(test_df, pair)
        
        current_price = test_df.iloc[-1]['close']
        pct_3d = test_df.iloc[-1].get('pct_change_3d', 0)
        pct_5d = test_df.iloc[-1].get('pct_change_5d', 0)
        
        print(f"  Day {i}: ${current_price:.2f}, 3D: {pct_3d:+.2f}%, 5D: {pct_5d:+.2f}% -> {signal.signal.value} ({signal.reason})")

def main():
    """Run comprehensive diagnostics"""
    
    pairs = ['ETH-USD', 'SOL-USD']
    start_date = '2024-01-01'
    end_date = '2024-09-01'
    
    print("🚀 STRATEGY DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    print(f"Period: {start_date} to {end_date}")
    print(f"Pairs: {', '.join(pairs)}")
    
    for pair in pairs:
        print(f"\n{'='*60}")
        print(f"ANALYZING {pair}")
        print(f"{'='*60}")
        
        # Analyze market data
        df = analyze_market_data(pair, start_date, end_date)
        
        if df is not None:
            # Test strategy signals
            test_strategy_signals(pair, start_date, end_date)
            
            # Deep dive into Pure strategy
            analyze_pure_strategy_logic(pair, start_date, end_date)
    
    print(f"\n✅ DIAGNOSTIC COMPLETE")

if __name__ == "__main__":
    main()
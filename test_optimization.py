#!/usr/bin/env python3
"""
Quick Optimization Test
Tests the parameter optimization on a smaller scale
"""

import sys
sys.path.append('.')

from src.optimization.parameter_tuner import ParameterTuner
from src.backtesting.engine import BacktestEngine
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def quick_optimization_test():
    """Run a quick optimization test on one pair"""
    
    engine = BacktestEngine()
    tuner = ParameterTuner(engine)
    
    # Test on just one pair with recent data
    pairs = ['ETH-USD']
    start_date = '2024-07-01'  # 2 months of data
    end_date = '2024-09-01'
    initial_capital = 1000
    
    print("🧪 QUICK OPTIMIZATION TEST")
    print("=" * 50)
    print(f"Testing on: {pairs[0]}")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 50)
    
    # Test RSI optimization (smallest parameter space)
    print("🔍 Testing RSI Strategy Optimization...")
    rsi_results = tuner.optimize_rsi_strategy(pairs, start_date, end_date, initial_capital)
    
    if pairs[0] in rsi_results and rsi_results[pairs[0]]:
        best_rsi = rsi_results[pairs[0]][0]  # Best result
        print(f"\n✅ Best RSI Result:")
        print(f"   Oversold: {best_rsi['oversold_threshold']}")
        print(f"   Overbought: {best_rsi['overbought_threshold']}")
        print(f"   Period: {best_rsi['rsi_period']}")
        print(f"   Return: {best_rsi['total_return_pct']:.2f}%")
        print(f"   Win Rate: {best_rsi['win_rate']:.1f}%")
        print(f"   Score: {best_rsi['score']:.2f}")
        
        # Show top 3 results
        print(f"\n📊 Top 3 RSI Configurations:")
        for i, result in enumerate(rsi_results[pairs[0]][:3], 1):
            print(f"   {i}. {result['oversold_threshold']}/{result['overbought_threshold']}/{result['rsi_period']} "
                  f"-> {result['total_return_pct']:.2f}% (score: {result['score']:.2f})")
    else:
        print("❌ No valid RSI results found")
    
    # Test Swing optimization
    print(f"\n🔍 Testing Swing Strategy Optimization...")
    swing_results = tuner.optimize_swing_strategy(pairs, start_date, end_date, initial_capital)
    
    if pairs[0] in swing_results and swing_results[pairs[0]]:
        best_swing = swing_results[pairs[0]][0]  # Best result
        print(f"\n✅ Best Swing Result:")
        print(f"   Swing Threshold: {best_swing['swing_threshold']}")
        print(f"   Volume Threshold: {best_swing['volume_threshold']}")
        print(f"   Lookback Period: {best_swing['lookback_period']}")
        print(f"   Return: {best_swing['total_return_pct']:.2f}%")
        print(f"   Win Rate: {best_swing['win_rate']:.1f}%")
        print(f"   Score: {best_swing['score']:.2f}")
        
        # Show top 3 results
        print(f"\n📊 Top 3 Swing Configurations:")
        for i, result in enumerate(swing_results[pairs[0]][:3], 1):
            print(f"   {i}. {result['swing_threshold']:.3f}/{result['volume_threshold']:.1f}/{result['lookback_period']} "
                  f"-> {result['total_return_pct']:.2f}% (score: {result['score']:.2f})")
    else:
        print("❌ No valid Swing results found")
    
    print("\n🎯 OPTIMIZATION TEST COMPLETE!")
    print("=" * 50)
    
    return rsi_results, swing_results

if __name__ == "__main__":
    try:
        rsi_results, swing_results = quick_optimization_test()
    except Exception as e:
        print(f"💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
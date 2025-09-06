#!/usr/bin/env python3
"""
Analyze Current Strategy Performance
Tests the existing 2.5% swing strategy to understand baseline performance
"""

import sys
sys.path.append('.')

from src.backtesting.engine import BacktestEngine
from src.strategies.swing import SwingTradingStrategy
from src.strategies.rsi import RSIStrategy
from src.strategies.macd import MACDStrategy
from src.config import get_config

def analyze_strategies():
    """Analyze current strategies performance"""
    
    engine = BacktestEngine()
    config = get_config()
    
    # Test parameters
    pairs = ['SOL-USD', 'ETH-USD']  # Test on 2 major pairs
    start_date = '2024-07-01'  # 2 months of data
    end_date = '2024-09-01'
    initial_capital = 1000
    
    strategies = [
        ('Swing 2.5%', SwingTradingStrategy()),
        ('RSI', RSIStrategy()),
        ('MACD', MACDStrategy())
    ]
    
    print("=" * 80)
    print("CURRENT STRATEGY PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital}")
    print()
    
    for pair in pairs:
        print(f"\n🪙 TESTING ON {pair}")
        print("-" * 60)
        
        for strategy_name, strategy in strategies:
            try:
                result = engine.run_backtest(strategy, pair, start_date, end_date, initial_capital)
                
                print(f"\n{strategy_name} Strategy:")
                print(f"  📈 Total Return: ${result.total_return:.2f} ({result.total_return_pct:.2f}%)")
                print(f"  📊 Total Trades: {result.total_trades}")
                print(f"  🎯 Win Rate: {result.win_rate:.1f}%")
                print(f"  💰 Avg Win: ${result.avg_win:.2f}")
                print(f"  📉 Avg Loss: ${result.avg_loss:.2f}")
                print(f"  ⚠️  Max Drawdown: {result.max_drawdown_pct:.2f}%")
                print(f"  📊 Sharpe Ratio: {result.sharpe_ratio:.3f}")
                
            except Exception as e:
                print(f"\n{strategy_name} Strategy: ❌ FAILED - {e}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    analyze_strategies()
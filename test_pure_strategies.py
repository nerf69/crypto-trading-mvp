#!/usr/bin/env python3
"""
Test Pure Percentage Strategies
Compares the new Pure 5% strategy with existing strategies and different percentage variants
"""

import sys
sys.path.append('.')

from src.backtesting.engine import BacktestEngine
from src.strategies.swing import SwingTradingStrategy
from src.strategies.rsi import RSIStrategy
from src.strategies.macd import MACDStrategy
from src.strategies.pure_percent import Pure5PercentStrategy, DynamicPercentStrategy
from src.config import get_config

def test_pure_strategies():
    """Test pure percentage strategies vs existing ones"""
    
    engine = BacktestEngine()
    config = get_config()
    
    # Test parameters
    pairs = ['SOL-USD', 'ETH-USD', 'AVAX-USD']  # Test on 3 major pairs
    start_date = '2024-06-01'  # 3 months of data
    end_date = '2024-09-01'
    initial_capital = 1000
    
    # Define all strategies to test
    strategies = [
        # Existing strategies
        ('Swing 2.5%', SwingTradingStrategy()),
        ('RSI', RSIStrategy()),
        ('MACD', MACDStrategy()),
        
        # New pure percentage strategies
        ('Pure 5%', Pure5PercentStrategy(drop_threshold=0.05, rise_threshold=0.05)),
        ('Pure 3%', Pure5PercentStrategy(drop_threshold=0.03, rise_threshold=0.03)),
        ('Pure 7%', Pure5PercentStrategy(drop_threshold=0.07, rise_threshold=0.07)),
        ('Pure 10%', Pure5PercentStrategy(drop_threshold=0.10, rise_threshold=0.10)),
        
        # Dynamic strategies
        ('Dynamic 3%', DynamicPercentStrategy(drop_threshold=0.03, rise_threshold=0.03)),
        ('Dynamic 5%', DynamicPercentStrategy(drop_threshold=0.05, rise_threshold=0.05)),
        ('Adaptive 5%', DynamicPercentStrategy(drop_threshold=0.05, rise_threshold=0.05, adapt_to_volatility=True)),
    ]
    
    print("=" * 100)
    print("PURE PERCENTAGE STRATEGY COMPARISON")
    print("=" * 100)
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital}")
    print()
    
    # Store all results for final comparison
    all_results = {}
    
    for pair in pairs:
        print(f"\n🪙 TESTING ON {pair}")
        print("-" * 80)
        
        pair_results = {}
        
        for strategy_name, strategy in strategies:
            try:
                result = engine.run_backtest(strategy, pair, start_date, end_date, initial_capital)
                pair_results[strategy_name] = result
                
                print(f"\n{strategy_name:12} | "
                      f"Return: {result.total_return_pct:6.2f}% | "
                      f"Trades: {result.total_trades:2d} | "
                      f"Win Rate: {result.win_rate:5.1f}% | "
                      f"Drawdown: {result.max_drawdown_pct:5.1f}% | "
                      f"Sharpe: {result.sharpe_ratio:5.2f}")
                
            except Exception as e:
                print(f"\n{strategy_name:12} | ❌ FAILED - {str(e)[:50]}...")
        
        all_results[pair] = pair_results
    
    # Summary analysis
    print("\n" + "=" * 100)
    print("OVERALL PERFORMANCE SUMMARY")
    print("=" * 100)
    
    # Calculate average performance across all pairs
    strategy_averages = {}
    
    for strategy_name, _ in strategies:
        returns = []
        win_rates = []
        sharpe_ratios = []
        total_trades = []
        
        for pair in pairs:
            if pair in all_results and strategy_name in all_results[pair]:
                result = all_results[pair][strategy_name]
                returns.append(result.total_return_pct)
                win_rates.append(result.win_rate)
                sharpe_ratios.append(result.sharpe_ratio)
                total_trades.append(result.total_trades)
        
        if returns:  # Only if we have results
            strategy_averages[strategy_name] = {
                'avg_return': sum(returns) / len(returns),
                'avg_win_rate': sum(win_rates) / len(win_rates),
                'avg_sharpe': sum(sharpe_ratios) / len(sharpe_ratios),
                'total_trades': sum(total_trades),
                'num_pairs': len(returns)
            }
    
    # Sort by average return
    sorted_strategies = sorted(strategy_averages.items(), 
                             key=lambda x: x[1]['avg_return'], reverse=True)
    
    print(f"\n{'Strategy':15} | {'Avg Return':10} | {'Avg Win Rate':11} | {'Total Trades':12} | {'Avg Sharpe':10}")
    print("-" * 85)
    
    for strategy_name, metrics in sorted_strategies:
        print(f"{strategy_name:15} | "
              f"{metrics['avg_return']:8.2f}% | "
              f"{metrics['avg_win_rate']:9.1f}% | "
              f"{metrics['total_trades']:10d} | "
              f"{metrics['avg_sharpe']:8.2f}")
    
    # Find best strategy
    if sorted_strategies:
        best_strategy = sorted_strategies[0]
        print(f"\n🏆 BEST PERFORMING STRATEGY: {best_strategy[0]}")
        print(f"   Average Return: {best_strategy[1]['avg_return']:.2f}%")
        print(f"   Average Win Rate: {best_strategy[1]['avg_win_rate']:.1f}%")
        print(f"   Average Sharpe Ratio: {best_strategy[1]['avg_sharpe']:.2f}")
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    
    # Recommendations
    print("\n📋 STRATEGY RECOMMENDATIONS:")
    
    pure_strategies = [(name, metrics) for name, metrics in sorted_strategies 
                      if 'Pure' in name or 'Dynamic' in name or 'Adaptive' in name]
    
    if pure_strategies:
        best_pure = pure_strategies[0]
        print(f"• Best Pure Strategy: {best_pure[0]} ({best_pure[1]['avg_return']:.2f}% avg return)")
    
    # Find most consistent (highest win rate)
    most_consistent = max(sorted_strategies, key=lambda x: x[1]['avg_win_rate'])
    print(f"• Most Consistent: {most_consistent[0]} ({most_consistent[1]['avg_win_rate']:.1f}% win rate)")
    
    # Find most active (most trades)
    most_active = max(sorted_strategies, key=lambda x: x[1]['total_trades'])
    print(f"• Most Active: {most_active[0]} ({most_active[1]['total_trades']} total trades)")

if __name__ == "__main__":
    test_pure_strategies()
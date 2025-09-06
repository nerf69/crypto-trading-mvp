#!/usr/bin/env python3
"""
Comprehensive Strategy Optimization
Uses the parameter tuner to find optimal settings for all strategies
"""

import sys
sys.path.append('.')

from src.optimization.parameter_tuner import ParameterTuner
from src.backtesting.engine import BacktestEngine
from src.config import get_config
import logging

# Set up logging to see optimization progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_optimization():
    """Run comprehensive optimization across all strategies"""
    
    # Initialize components
    config = get_config()
    engine = BacktestEngine()
    tuner = ParameterTuner(engine)
    
    # Test parameters - use a good period with enough data
    pairs = ['ETH-USD', 'SOL-USD', 'AVAX-USD']  # Focus on 3 major pairs
    start_date = '2024-04-01'  # 5 months of data
    end_date = '2024-09-01'
    initial_capital = 1000
    
    print("🚀 STARTING COMPREHENSIVE STRATEGY OPTIMIZATION")
    print("=" * 80)
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"🪙 Trading Pairs: {', '.join(pairs)}")
    print(f"💰 Initial Capital: ${initial_capital}")
    print(f"🔄 This will test hundreds of parameter combinations...")
    print("⏱️  Expected time: 5-15 minutes")
    print("=" * 80)
    
    # Run the comprehensive optimization
    results = tuner.run_comprehensive_optimization(
        pairs=pairs,
        start_date=start_date, 
        end_date=end_date,
        initial_capital=initial_capital
    )
    
    # Save results
    tuner.save_optimization_results(results)
    
    # Create optimal strategies for live trading
    optimal_strategies = tuner.create_optimal_strategies(results)
    
    print("\n" + "🏆 OPTIMIZATION COMPLETE - FINAL RECOMMENDATIONS")
    print("=" * 80)
    
    for pair, strategy_info in optimal_strategies.items():
        strategy = strategy_info['strategy']
        print(f"\n📈 {pair}: {strategy.name}")
        print(f"   Expected Return: {strategy_info['expected_return']:.2f}%")
        print(f"   Win Rate: {strategy_info['win_rate']:.1f}%")
        print(f"   Max Drawdown: {strategy_info['max_drawdown']:.1f}%")
        print(f"   Optimization Score: {strategy_info['score']:.2f}")
    
    # Show strategy type distribution
    strategy_types = {}
    for pair, strategy_info in optimal_strategies.items():
        strategy_name = strategy_info['strategy'].name
        if 'RSI' in strategy_name:
            strategy_type = 'RSI'
        elif 'Swing' in strategy_name:
            strategy_type = 'Swing'
        elif 'Pure' in strategy_name:
            strategy_type = 'Pure Percent'
        else:
            strategy_type = 'Other'
            
        strategy_types[strategy_type] = strategy_types.get(strategy_type, 0) + 1
    
    print(f"\n📊 STRATEGY TYPE DISTRIBUTION:")
    for strategy_type, count in strategy_types.items():
        print(f"   {strategy_type}: {count} pair(s)")
    
    # Performance summary
    avg_return = sum(s['expected_return'] for s in optimal_strategies.values()) / len(optimal_strategies)
    avg_win_rate = sum(s['win_rate'] for s in optimal_strategies.values()) / len(optimal_strategies)
    avg_drawdown = sum(s['max_drawdown'] for s in optimal_strategies.values()) / len(optimal_strategies)
    
    print(f"\n📈 OVERALL OPTIMIZED PERFORMANCE:")
    print(f"   Average Expected Return: {avg_return:.2f}%")
    print(f"   Average Win Rate: {avg_win_rate:.1f}%")
    print(f"   Average Max Drawdown: {avg_drawdown:.1f}%")
    
    print(f"\n💡 KEY INSIGHTS:")
    
    # Find best performing strategy type
    best_strategy_type = max(strategy_types.keys(), 
                           key=lambda st: sum(s['expected_return'] for p, s in optimal_strategies.items() 
                                            if st in s['strategy'].name))
    print(f"   • Best strategy type overall: {best_strategy_type}")
    
    # Find most consistent (lowest drawdown)
    most_consistent_pair = min(optimal_strategies.keys(),
                              key=lambda p: optimal_strategies[p]['max_drawdown'])
    print(f"   • Most consistent pair: {most_consistent_pair} "
          f"({optimal_strategies[most_consistent_pair]['max_drawdown']:.1f}% max drawdown)")
    
    # Find highest return
    highest_return_pair = max(optimal_strategies.keys(),
                             key=lambda p: optimal_strategies[p]['expected_return'])
    print(f"   • Highest return pair: {highest_return_pair} "
          f"({optimal_strategies[highest_return_pair]['expected_return']:.2f}% expected return)")
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Review the optimization results JSON file for detailed parameters")
    print("   2. Implement the optimal strategies for paper trading")
    print("   3. Monitor performance and re-optimize quarterly")
    print("   4. Consider combining strategies for portfolio approach")
    print("=" * 80)
    
    return results, optimal_strategies

if __name__ == "__main__":
    try:
        results, strategies = run_optimization()
    except KeyboardInterrupt:
        print("\n❌ Optimization interrupted by user")
    except Exception as e:
        print(f"\n💥 Optimization failed: {e}")
        import traceback
        traceback.print_exc()
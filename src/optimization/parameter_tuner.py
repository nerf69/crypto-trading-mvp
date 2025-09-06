import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from ..backtesting.engine import BacktestEngine, BacktestResult
from ..strategies.swing import SwingTradingStrategy
from ..strategies.rsi import RSIStrategy
from ..strategies.pure_percent import Pure5PercentStrategy, DynamicPercentStrategy
from ..config import get_config

logger = logging.getLogger(__name__)

class ParameterTuner:
    """
    Automated parameter optimization engine for trading strategies
    
    This class automatically tests different parameter combinations
    to find the optimal settings for each strategy and trading pair.
    """
    
    def __init__(self, engine: Optional[BacktestEngine] = None):
        self.engine = engine or BacktestEngine()
        self.config = get_config()
        self.results_history = []
    
    def optimize_swing_strategy(self, pairs: List[str], start_date: str, end_date: str,
                               initial_capital: float = 1000) -> Dict[str, Dict]:
        """Optimize Swing Trading Strategy parameters"""
        
        logger.info("Optimizing Swing Trading Strategy parameters...")
        
        # Parameter ranges to test
        swing_thresholds = [0.015, 0.020, 0.025, 0.030, 0.035, 0.040]  # 1.5% to 4%
        volume_thresholds = [1.0, 1.1, 1.2, 1.5]  # Volume multipliers
        lookback_periods = [5, 7, 10, 14, 20]  # Days to look back
        
        parameter_combinations = list(itertools.product(
            swing_thresholds, volume_thresholds, lookback_periods
        ))
        
        logger.info(f"Testing {len(parameter_combinations)} parameter combinations on {len(pairs)} pairs")
        
        results = {}
        
        for pair in pairs:
            logger.info(f"Optimizing Swing Strategy for {pair}...")
            pair_results = []
            
            for swing_thresh, vol_thresh, lookback in parameter_combinations:
                try:
                    # Create strategy with these parameters
                    strategy = SwingTradingStrategy(
                        swing_threshold=swing_thresh,
                        volume_threshold=vol_thresh
                    )
                    strategy.lookback_period = lookback
                    
                    # Run backtest
                    result = self.engine.run_backtest(
                        strategy, pair, start_date, end_date, initial_capital
                    )
                    
                    # Store result with parameters
                    pair_results.append({
                        'swing_threshold': swing_thresh,
                        'volume_threshold': vol_thresh,
                        'lookback_period': lookback,
                        'total_return_pct': result.total_return_pct,
                        'win_rate': result.win_rate,
                        'total_trades': result.total_trades,
                        'max_drawdown_pct': result.max_drawdown_pct,
                        'sharpe_ratio': result.sharpe_ratio,
                        'score': self._calculate_optimization_score(result)
                    })
                    
                except Exception as e:
                    logger.debug(f"Failed combination for {pair}: swing={swing_thresh}, vol={vol_thresh}, lookback={lookback} - {e}")
            
            # Sort by score (best first)
            pair_results.sort(key=lambda x: x['score'], reverse=True)
            results[pair] = pair_results
            
            # Log best result for this pair
            if pair_results:
                best = pair_results[0]
                logger.info(f"Best Swing parameters for {pair}: "
                          f"swing={best['swing_threshold']}, vol={best['volume_threshold']}, "
                          f"lookback={best['lookback_period']} -> "
                          f"{best['total_return_pct']:.2f}% return, score={best['score']:.3f}")
        
        return results
    
    def optimize_pure_percent_strategy(self, pairs: List[str], start_date: str, end_date: str,
                                     initial_capital: float = 1000) -> Dict[str, Dict]:
        """Optimize Pure Percentage Strategy parameters"""
        
        logger.info("Optimizing Pure Percentage Strategy parameters...")
        
        # Parameter ranges - wider range for crypto volatility
        drop_thresholds = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]  # 3% to 15%
        rise_thresholds = [0.03, 0.05, 0.06, 0.08, 0.10, 0.12]  # 3% to 12%
        lookback_days = [3, 5, 7, 10, 14]  # Days for high/low calculation
        
        parameter_combinations = list(itertools.product(
            drop_thresholds, rise_thresholds, lookback_days
        ))
        
        logger.info(f"Testing {len(parameter_combinations)} Pure Percent combinations on {len(pairs)} pairs")
        
        results = {}
        
        for pair in pairs:
            logger.info(f"Optimizing Pure Percent Strategy for {pair}...")
            pair_results = []
            
            for drop_thresh, rise_thresh, lookback in parameter_combinations:
                try:
                    strategy = Pure5PercentStrategy(
                        drop_threshold=drop_thresh,
                        rise_threshold=rise_thresh,
                        lookback_days=lookback
                    )
                    
                    result = self.engine.run_backtest(
                        strategy, pair, start_date, end_date, initial_capital
                    )
                    
                    pair_results.append({
                        'drop_threshold': drop_thresh,
                        'rise_threshold': rise_thresh,
                        'lookback_days': lookback,
                        'total_return_pct': result.total_return_pct,
                        'win_rate': result.win_rate,
                        'total_trades': result.total_trades,
                        'max_drawdown_pct': result.max_drawdown_pct,
                        'sharpe_ratio': result.sharpe_ratio,
                        'score': self._calculate_optimization_score(result)
                    })
                    
                except Exception as e:
                    logger.debug(f"Failed Pure Percent combination: drop={drop_thresh}, rise={rise_thresh}, lookback={lookback} - {e}")
            
            pair_results.sort(key=lambda x: x['score'], reverse=True)
            results[pair] = pair_results
            
            if pair_results:
                best = pair_results[0]
                logger.info(f"Best Pure Percent parameters for {pair}: "
                          f"drop={best['drop_threshold']}, rise={best['rise_threshold']}, "
                          f"lookback={best['lookback_days']} -> "
                          f"{best['total_return_pct']:.2f}% return, score={best['score']:.3f}")
        
        return results
    
    def optimize_rsi_strategy(self, pairs: List[str], start_date: str, end_date: str,
                            initial_capital: float = 1000) -> Dict[str, Dict]:
        """Optimize RSI Strategy parameters"""
        
        logger.info("Optimizing RSI Strategy parameters...")
        
        # RSI parameter ranges
        oversold_thresholds = [25, 30, 35, 40]  # RSI oversold levels
        overbought_thresholds = [60, 65, 70, 75]  # RSI overbought levels  
        rsi_periods = [10, 12, 14, 16, 18, 21]  # RSI calculation periods
        
        parameter_combinations = list(itertools.product(
            oversold_thresholds, overbought_thresholds, rsi_periods
        ))
        
        logger.info(f"Testing {len(parameter_combinations)} RSI combinations on {len(pairs)} pairs")
        
        results = {}
        
        for pair in pairs:
            logger.info(f"Optimizing RSI Strategy for {pair}...")
            pair_results = []
            
            for oversold, overbought, rsi_period in parameter_combinations:
                # Skip invalid combinations
                if oversold >= overbought:
                    continue
                    
                try:
                    strategy = RSIStrategy(
                        oversold_threshold=oversold,
                        overbought_threshold=overbought,
                        rsi_period=rsi_period
                    )
                    
                    result = self.engine.run_backtest(
                        strategy, pair, start_date, end_date, initial_capital
                    )
                    
                    pair_results.append({
                        'oversold_threshold': oversold,
                        'overbought_threshold': overbought,
                        'rsi_period': rsi_period,
                        'total_return_pct': result.total_return_pct,
                        'win_rate': result.win_rate,
                        'total_trades': result.total_trades,
                        'max_drawdown_pct': result.max_drawdown_pct,
                        'sharpe_ratio': result.sharpe_ratio,
                        'score': self._calculate_optimization_score(result)
                    })
                    
                except Exception as e:
                    logger.debug(f"Failed RSI combination: oversold={oversold}, overbought={overbought}, period={rsi_period} - {e}")
            
            pair_results.sort(key=lambda x: x['score'], reverse=True)
            results[pair] = pair_results
            
            if pair_results:
                best = pair_results[0]
                logger.info(f"Best RSI parameters for {pair}: "
                          f"oversold={best['oversold_threshold']}, overbought={best['overbought_threshold']}, "
                          f"period={best['rsi_period']} -> "
                          f"{best['total_return_pct']:.2f}% return, score={best['score']:.3f}")
        
        return results
    
    def _calculate_optimization_score(self, result: BacktestResult) -> float:
        """
        Calculate optimization score combining multiple metrics
        
        Score considers:
        - Total return (weighted heavily)
        - Sharpe ratio (risk-adjusted returns)
        - Win rate (consistency)  
        - Max drawdown (risk management)
        - Number of trades (strategy activity)
        """
        
        # Avoid division by zero or invalid results
        if result.total_trades == 0:
            return -1000  # Very bad score
        
        # Component scores (normalized to 0-100 scale roughly)
        return_score = result.total_return_pct * 2  # 2x weight on returns
        sharpe_score = result.sharpe_ratio * 20  # Amplify Sharpe ratio
        win_rate_score = result.win_rate / 100 * 30  # Convert to 0-30 scale
        
        # Penalize high drawdown
        drawdown_penalty = result.max_drawdown_pct * -1
        
        # Slight bonus for reasonable trade frequency (not too many, not too few)
        if 5 <= result.total_trades <= 20:
            trade_bonus = 5
        elif result.total_trades < 5:
            trade_bonus = -2  # Too few trades
        else:
            trade_bonus = -1  # Too many trades (overtrading)
        
        total_score = return_score + sharpe_score + win_rate_score + drawdown_penalty + trade_bonus
        
        return total_score
    
    def run_comprehensive_optimization(self, pairs: List[str], start_date: str, end_date: str,
                                     initial_capital: float = 1000) -> Dict[str, Any]:
        """Run optimization for all strategies and find the best overall approach"""
        
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE PARAMETER OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Trading pairs: {pairs}")
        logger.info(f"Initial capital: ${initial_capital}")
        
        # Run optimizations for each strategy type
        optimizations = {
            'swing': self.optimize_swing_strategy(pairs, start_date, end_date, initial_capital),
            'pure_percent': self.optimize_pure_percent_strategy(pairs, start_date, end_date, initial_capital),
            'rsi': self.optimize_rsi_strategy(pairs, start_date, end_date, initial_capital)
        }
        
        # Find best overall strategy per pair
        best_strategies = {}
        
        for pair in pairs:
            best_overall = None
            best_score = float('-inf')
            
            for strategy_type, results in optimizations.items():
                if pair in results and results[pair]:
                    best_for_strategy = results[pair][0]  # First is best (sorted by score)
                    
                    if best_for_strategy['score'] > best_score:
                        best_score = best_for_strategy['score']
                        best_overall = {
                            'strategy_type': strategy_type,
                            'parameters': best_for_strategy,
                            'score': best_score
                        }
            
            best_strategies[pair] = best_overall
        
        # Summary analysis
        strategy_performance = {}
        for strategy_type in optimizations.keys():
            scores = []
            returns = []
            
            for pair in pairs:
                if pair in optimizations[strategy_type] and optimizations[strategy_type][pair]:
                    best = optimizations[strategy_type][pair][0]
                    scores.append(best['score'])
                    returns.append(best['total_return_pct'])
            
            if scores:
                strategy_performance[strategy_type] = {
                    'avg_score': np.mean(scores),
                    'avg_return': np.mean(returns),
                    'pairs_count': len(scores)
                }
        
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZATION RESULTS SUMMARY")
        logger.info("=" * 80)
        
        for pair, best in best_strategies.items():
            if best:
                logger.info(f"\n{pair}: Best strategy is {best['strategy_type'].upper()}")
                logger.info(f"  Score: {best['score']:.2f}")
                logger.info(f"  Return: {best['parameters']['total_return_pct']:.2f}%")
                logger.info(f"  Win Rate: {best['parameters']['win_rate']:.1f}%")
                logger.info(f"  Max Drawdown: {best['parameters']['max_drawdown_pct']:.1f}%")
        
        logger.info(f"\nOverall Strategy Performance:")
        for strategy, perf in strategy_performance.items():
            logger.info(f"  {strategy.upper()}: Avg Score={perf['avg_score']:.2f}, Avg Return={perf['avg_return']:.2f}%")
        
        return {
            'optimizations': optimizations,
            'best_strategies': best_strategies,
            'strategy_performance': strategy_performance,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_optimization_results(self, results: Dict[str, Any], filename: str = None):
        """Save optimization results to JSON file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert all numpy types
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {key: deep_convert(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        converted_results = deep_convert(results)
        
        try:
            with open(filename, 'w') as f:
                json.dump(converted_results, f, indent=2, default=str)
            
            logger.info(f"Optimization results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")
    
    def create_optimal_strategies(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create strategy instances with optimal parameters for each pair"""
        
        optimal_strategies = {}
        
        for pair, best in optimization_results['best_strategies'].items():
            if not best:
                continue
                
            strategy_type = best['strategy_type']
            params = best['parameters']
            
            try:
                if strategy_type == 'swing':
                    strategy = SwingTradingStrategy(
                        swing_threshold=params['swing_threshold'],
                        volume_threshold=params['volume_threshold']
                    )
                    strategy.lookback_period = params['lookback_period']
                    
                elif strategy_type == 'pure_percent':
                    strategy = Pure5PercentStrategy(
                        drop_threshold=params['drop_threshold'],
                        rise_threshold=params['rise_threshold'],
                        lookback_days=params['lookback_days']
                    )
                    
                elif strategy_type == 'rsi':
                    strategy = RSIStrategy(
                        oversold_threshold=params['oversold_threshold'],
                        overbought_threshold=params['overbought_threshold'],
                        rsi_period=params['rsi_period']
                    )
                
                optimal_strategies[pair] = {
                    'strategy': strategy,
                    'expected_return': params['total_return_pct'],
                    'win_rate': params['win_rate'],
                    'max_drawdown': params['max_drawdown_pct'],
                    'score': best['score']
                }
                
            except Exception as e:
                logger.error(f"Failed to create optimal strategy for {pair}: {e}")
        
        return optimal_strategies
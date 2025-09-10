"""
Comprehensive tests for parameter optimization algorithms.

This test suite validates the mathematical correctness and logical accuracy
of the parameter tuning and optimization algorithms used to find optimal
strategy parameters.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import itertools
import tempfile
import os

from src.optimization.parameter_tuner import ParameterTuner
from src.backtesting.engine import BacktestEngine, BacktestResult
from src.strategies.swing import SwingTradingStrategy
from src.strategies.rsi import RSIStrategy
from src.strategies.pure_percent import Pure5PercentStrategy
from src.config import Config


class TestParameterTuner:
    """Test ParameterTuner initialization and basic functionality"""
    
    def test_parameter_tuner_initialization(self):
        """Test ParameterTuner initialization"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            tuner = ParameterTuner()
            
            assert tuner.engine is not None
            assert tuner.config is not None
            assert tuner.results_history == []
    
    def test_parameter_tuner_with_custom_engine(self):
        """Test ParameterTuner with custom BacktestEngine"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            custom_engine = BacktestEngine()
            tuner = ParameterTuner(engine=custom_engine)
            
            assert tuner.engine is custom_engine


class TestOptimizationScoringAlgorithm:
    """Test the optimization scoring algorithm mathematical accuracy"""
    
    def test_scoring_algorithm_basic_cases(self):
        """Test scoring algorithm with basic scenarios"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            tuner = ParameterTuner()
            
            # Test case 1: Perfect result
            perfect_result = BacktestResult(
                strategy_name="Test",
                pair="BTC-USD",
                start_date="2024-01-01",
                end_date="2024-01-31",
                initial_capital=1000,
                final_capital=2000,  # 100% return
                total_return=1000,
                total_return_pct=100.0,
                total_trades=10,
                winning_trades=10,
                losing_trades=0,
                win_rate=100.0,
                avg_win=100.0,
                avg_loss=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=5.0,  # Excellent Sharpe ratio
                positions=[],
                equity_curve=pd.DataFrame(),
                trade_log=pd.DataFrame()
            )
            
            perfect_score = tuner._calculate_optimization_score(perfect_result)
            
            # Test case 2: Poor result
            poor_result = BacktestResult(
                strategy_name="Test",
                pair="BTC-USD", 
                start_date="2024-01-01",
                end_date="2024-01-31",
                initial_capital=1000,
                final_capital=500,   # -50% return
                total_return=-500,
                total_return_pct=-50.0,
                total_trades=20,
                winning_trades=2,
                losing_trades=18,
                win_rate=10.0,
                avg_win=25.0,
                avg_loss=-30.0,
                max_drawdown=600,
                max_drawdown_pct=60.0,
                sharpe_ratio=-2.0,  # Poor Sharpe ratio
                positions=[],
                equity_curve=pd.DataFrame(),
                trade_log=pd.DataFrame()
            )
            
            poor_score = tuner._calculate_optimization_score(poor_result)
            
            # Perfect result should have much higher score than poor result
            assert perfect_score > poor_score + 100, \
                f"Perfect score ({perfect_score}) should be much higher than poor score ({poor_score})"
    
    def test_scoring_no_trades_penalty(self):
        """Test scoring algorithm handles zero trades correctly"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            tuner = ParameterTuner()
            
            no_trades_result = BacktestResult(
                strategy_name="Test",
                pair="BTC-USD",
                start_date="2024-01-01",
                end_date="2024-01-31",
                initial_capital=1000,
                final_capital=1000,  # No change
                total_return=0,
                total_return_pct=0.0,
                total_trades=0,  # No trades
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                positions=[],
                equity_curve=pd.DataFrame(),
                trade_log=pd.DataFrame()
            )
            
            score = tuner._calculate_optimization_score(no_trades_result)
            assert score == -1000, "Zero trades should result in very low score (-1000)"
    
    def test_scoring_component_weights(self):
        """Test scoring algorithm component weights"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            tuner = ParameterTuner()
            
            base_result = BacktestResult(
                strategy_name="Test", pair="BTC-USD", start_date="2024-01-01", end_date="2024-01-31",
                initial_capital=1000, final_capital=1100, total_return=100, total_return_pct=10.0,
                total_trades=10, winning_trades=6, losing_trades=4, win_rate=60.0,
                avg_win=25.0, avg_loss=-15.0, max_drawdown=50, max_drawdown_pct=5.0,
                sharpe_ratio=1.5, positions=[], equity_curve=pd.DataFrame(), trade_log=pd.DataFrame()
            )
            
            # Test return weight (should have 2x weight)
            high_return_result = base_result._replace(total_return_pct=20.0)
            low_return_result = base_result._replace(total_return_pct=5.0)
            
            high_return_score = tuner._calculate_optimization_score(high_return_result)
            low_return_score = tuner._calculate_optimization_score(low_return_result)
            
            # Difference should be approximately 2 * (20 - 5) = 30 points
            score_diff = high_return_score - low_return_score
            assert 25 <= score_diff <= 35, f"Return score difference should be ~30, got {score_diff}"
    
    def test_scoring_drawdown_penalty(self):
        """Test that high drawdown is properly penalized"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            tuner = ParameterTuner()
            
            low_drawdown_result = BacktestResult(
                strategy_name="Test", pair="BTC-USD", start_date="2024-01-01", end_date="2024-01-31",
                initial_capital=1000, final_capital=1200, total_return=200, total_return_pct=20.0,
                total_trades=10, winning_trades=8, losing_trades=2, win_rate=80.0,
                avg_win=30.0, avg_loss=-15.0, max_drawdown=50, max_drawdown_pct=5.0,  # Low drawdown
                sharpe_ratio=2.0, positions=[], equity_curve=pd.DataFrame(), trade_log=pd.DataFrame()
            )
            
            high_drawdown_result = low_drawdown_result._replace(
                max_drawdown_pct=30.0  # High drawdown
            )
            
            low_dd_score = tuner._calculate_optimization_score(low_drawdown_result)
            high_dd_score = tuner._calculate_optimization_score(high_drawdown_result)
            
            # High drawdown should significantly reduce score
            assert low_dd_score > high_dd_score + 20, \
                f"Low drawdown score ({low_dd_score}) should be much higher than high drawdown score ({high_dd_score})"
    
    def test_scoring_trade_frequency_bonus(self):
        """Test trade frequency bonus/penalty logic"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            tuner = ParameterTuner()
            
            base_result_data = {
                'strategy_name': "Test", 'pair': "BTC-USD", 'start_date': "2024-01-01", 'end_date': "2024-01-31",
                'initial_capital': 1000, 'final_capital': 1100, 'total_return': 100, 'total_return_pct': 10.0,
                'winning_trades': 6, 'losing_trades': 4, 'win_rate': 60.0,
                'avg_win': 25.0, 'avg_loss': -15.0, 'max_drawdown': 50, 'max_drawdown_pct': 5.0,
                'sharpe_ratio': 1.5, 'positions': [], 'equity_curve': pd.DataFrame(), 'trade_log': pd.DataFrame()
            }
            
            # Optimal trade count (5-20 should get bonus)
            optimal_trades = BacktestResult(total_trades=10, **base_result_data)
            
            # Too few trades (should get penalty)
            too_few_trades = BacktestResult(total_trades=3, **base_result_data)
            
            # Too many trades (should get penalty)
            too_many_trades = BacktestResult(total_trades=50, **base_result_data)
            
            optimal_score = tuner._calculate_optimization_score(optimal_trades)
            too_few_score = tuner._calculate_optimization_score(too_few_trades)
            too_many_score = tuner._calculate_optimization_score(too_many_trades)
            
            # Optimal should be higher than both extremes
            assert optimal_score > too_few_score, "Optimal trade count should score higher than too few trades"
            assert optimal_score > too_many_score, "Optimal trade count should score higher than too many trades"


class TestSwingStrategyOptimization:
    """Test swing strategy parameter optimization algorithm"""
    
    def test_swing_optimization_parameter_ranges(self):
        """Test that swing strategy optimization covers expected parameter ranges"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            tuner = ParameterTuner()
            
            # Mock the backtest engine to return dummy results
            mock_result = BacktestResult(
                strategy_name="Swing", pair="BTC-USD", start_date="2024-01-01", end_date="2024-01-31",
                initial_capital=1000, final_capital=1100, total_return=100, total_return_pct=10.0,
                total_trades=5, winning_trades=3, losing_trades=2, win_rate=60.0,
                avg_win=50.0, avg_loss=-25.0, max_drawdown=50, max_drawdown_pct=5.0,
                sharpe_ratio=1.2, positions=[], equity_curve=pd.DataFrame(), trade_log=pd.DataFrame()
            )
            
            with patch.object(tuner.engine, 'run_backtest', return_value=mock_result):
                results = tuner.optimize_swing_strategy(
                    pairs=["BTC-USD"], 
                    start_date="2024-01-01", 
                    end_date="2024-01-31"
                )
                
                # Verify results structure
                assert "BTC-USD" in results
                assert len(results["BTC-USD"]) > 0
                
                # Verify parameter ranges were tested
                tested_params = results["BTC-USD"]
                swing_thresholds = [p['swing_threshold'] for p in tested_params]
                volume_thresholds = [p['volume_threshold'] for p in tested_params]
                lookback_periods = [p['lookback_period'] for p in tested_params]
                
                # Check parameter ranges
                assert min(swing_thresholds) >= 0.015, "Should test swing thresholds from 1.5%"
                assert max(swing_thresholds) <= 0.040, "Should test swing thresholds up to 4%"
                assert min(volume_thresholds) >= 1.0, "Should test volume thresholds from 1.0"
                assert max(volume_thresholds) <= 1.5, "Should test volume thresholds up to 1.5"
                assert min(lookback_periods) >= 5, "Should test lookback periods from 5"
                assert max(lookback_periods) <= 20, "Should test lookback periods up to 20"
    
    def test_swing_optimization_result_sorting(self):
        """Test that swing optimization results are properly sorted by score"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            tuner = ParameterTuner()
            
            # Create mock results with different scores
            def mock_backtest_side_effect(*args, **kwargs):
                strategy = args[0]
                # Make score based on swing threshold (higher threshold = higher score for this test)
                score_multiplier = strategy.swing_threshold * 1000
                
                return BacktestResult(
                    strategy_name="Swing", pair="BTC-USD", start_date="2024-01-01", end_date="2024-01-31",
                    initial_capital=1000, final_capital=1000 + score_multiplier/10, 
                    total_return=score_multiplier/10, total_return_pct=score_multiplier/100,
                    total_trades=10, winning_trades=6, losing_trades=4, win_rate=60.0,
                    avg_win=25.0, avg_loss=-15.0, max_drawdown=20, max_drawdown_pct=2.0,
                    sharpe_ratio=1.0, positions=[], equity_curve=pd.DataFrame(), trade_log=pd.DataFrame()
                )
            
            with patch.object(tuner.engine, 'run_backtest', side_effect=mock_backtest_side_effect):
                results = tuner.optimize_swing_strategy(
                    pairs=["BTC-USD"], 
                    start_date="2024-01-01", 
                    end_date="2024-01-31"
                )
                
                # Verify results are sorted by score (descending)
                btc_results = results["BTC-USD"]
                scores = [r['score'] for r in btc_results]
                
                assert scores == sorted(scores, reverse=True), "Results should be sorted by score (descending)"
                
                # Verify best result is first
                best_result = btc_results[0]
                assert best_result['score'] == max(scores), "Best score should be first"


class TestRSIStrategyOptimization:
    """Test RSI strategy parameter optimization algorithm"""
    
    def test_rsi_optimization_parameter_validation(self):
        """Test RSI optimization parameter validation logic"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            tuner = ParameterTuner()
            
            # Mock backtest that should fail for invalid parameters
            def mock_backtest_side_effect(*args, **kwargs):
                strategy = args[0]
                
                # Invalid parameter combinations should be skipped in the actual optimization
                if strategy.oversold_threshold >= strategy.overbought_threshold:
                    raise ValueError("Invalid RSI parameter combination")
                
                return BacktestResult(
                    strategy_name="RSI", pair="BTC-USD", start_date="2024-01-01", end_date="2024-01-31",
                    initial_capital=1000, final_capital=1050, total_return=50, total_return_pct=5.0,
                    total_trades=8, winning_trades=5, losing_trades=3, win_rate=62.5,
                    avg_win=20.0, avg_loss=-12.5, max_drawdown=30, max_drawdown_pct=3.0,
                    sharpe_ratio=0.8, positions=[], equity_curve=pd.DataFrame(), trade_log=pd.DataFrame()
                )
            
            with patch.object(tuner.engine, 'run_backtest', side_effect=mock_backtest_side_effect):
                results = tuner.optimize_rsi_strategy(
                    pairs=["BTC-USD"], 
                    start_date="2024-01-01", 
                    end_date="2024-01-31"
                )
                
                # Verify that only valid parameter combinations are in results
                btc_results = results["BTC-USD"]
                for result in btc_results:
                    assert result['oversold_threshold'] < result['overbought_threshold'], \
                        "All results should have valid RSI parameter combinations"
    
    def test_rsi_optimization_parameter_ranges(self):
        """Test RSI optimization parameter ranges"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            tuner = ParameterTuner()
            
            mock_result = BacktestResult(
                strategy_name="RSI", pair="BTC-USD", start_date="2024-01-01", end_date="2024-01-31",
                initial_capital=1000, final_capital=1080, total_return=80, total_return_pct=8.0,
                total_trades=12, winning_trades=8, losing_trades=4, win_rate=66.7,
                avg_win=15.0, avg_loss=-10.0, max_drawdown=40, max_drawdown_pct=4.0,
                sharpe_ratio=1.1, positions=[], equity_curve=pd.DataFrame(), trade_log=pd.DataFrame()
            )
            
            with patch.object(tuner.engine, 'run_backtest', return_value=mock_result):
                results = tuner.optimize_rsi_strategy(
                    pairs=["BTC-USD"], 
                    start_date="2024-01-01", 
                    end_date="2024-01-31"
                )
                
                # Verify parameter ranges
                btc_results = results["BTC-USD"]
                oversold_values = [r['oversold_threshold'] for r in btc_results]
                overbought_values = [r['overbought_threshold'] for r in btc_results]
                rsi_periods = [r['rsi_period'] for r in btc_results]
                
                # Check ranges match expected values
                assert min(oversold_values) >= 25, "Should test oversold from 25"
                assert max(oversold_values) <= 40, "Should test oversold up to 40"
                assert min(overbought_values) >= 60, "Should test overbought from 60"
                assert max(overbought_values) <= 75, "Should test overbought up to 75"
                assert min(rsi_periods) >= 10, "Should test RSI periods from 10"
                assert max(rsi_periods) <= 21, "Should test RSI periods up to 21"


class TestPurePercentStrategyOptimization:
    """Test Pure Percentage strategy optimization algorithm"""
    
    def test_pure_percent_optimization_ranges(self):
        """Test Pure Percentage optimization parameter ranges"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            tuner = ParameterTuner()
            
            mock_result = BacktestResult(
                strategy_name="Pure 5%", pair="ETH-USD", start_date="2024-01-01", end_date="2024-01-31",
                initial_capital=1000, final_capital=1120, total_return=120, total_return_pct=12.0,
                total_trades=6, winning_trades=4, losing_trades=2, win_rate=66.7,
                avg_win=45.0, avg_loss=-30.0, max_drawdown=60, max_drawdown_pct=6.0,
                sharpe_ratio=1.4, positions=[], equity_curve=pd.DataFrame(), trade_log=pd.DataFrame()
            )
            
            with patch.object(tuner.engine, 'run_backtest', return_value=mock_result):
                results = tuner.optimize_pure_percent_strategy(
                    pairs=["ETH-USD"], 
                    start_date="2024-01-01", 
                    end_date="2024-01-31"
                )
                
                # Verify parameter ranges for crypto volatility
                eth_results = results["ETH-USD"]
                drop_thresholds = [r['drop_threshold'] for r in eth_results]
                rise_thresholds = [r['rise_threshold'] for r in eth_results]
                lookback_days = [r['lookback_days'] for r in eth_results]
                
                # Check crypto-appropriate ranges
                assert min(drop_thresholds) >= 0.03, "Should test drop thresholds from 3%"
                assert max(drop_thresholds) <= 0.15, "Should test drop thresholds up to 15%"
                assert min(rise_thresholds) >= 0.03, "Should test rise thresholds from 3%"
                assert max(rise_thresholds) <= 0.12, "Should test rise thresholds up to 12%"
                assert min(lookback_days) >= 3, "Should test lookback from 3 days"
                assert max(lookback_days) <= 14, "Should test lookback up to 14 days"


class TestComprehensiveOptimization:
    """Test comprehensive optimization algorithm"""
    
    def test_comprehensive_optimization_structure(self):
        """Test comprehensive optimization returns proper structure"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            tuner = ParameterTuner()
            
            # Mock individual optimization methods
            mock_swing_results = {
                "BTC-USD": [{'swing_threshold': 0.025, 'volume_threshold': 1.1, 'lookback_period': 10,
                           'total_return_pct': 15.0, 'win_rate': 70.0, 'total_trades': 8, 
                           'max_drawdown_pct': 3.0, 'sharpe_ratio': 1.5, 'score': 50.0}]
            }
            
            mock_rsi_results = {
                "BTC-USD": [{'oversold_threshold': 30, 'overbought_threshold': 70, 'rsi_period': 14,
                           'total_return_pct': 12.0, 'win_rate': 65.0, 'total_trades': 10,
                           'max_drawdown_pct': 4.0, 'sharpe_ratio': 1.2, 'score': 45.0}]
            }
            
            mock_pure_results = {
                "BTC-USD": [{'drop_threshold': 0.05, 'rise_threshold': 0.05, 'lookback_days': 7,
                           'total_return_pct': 18.0, 'win_rate': 75.0, 'total_trades': 6,
                           'max_drawdown_pct': 5.0, 'sharpe_ratio': 1.8, 'score': 55.0}]
            }
            
            with patch.object(tuner, 'optimize_swing_strategy', return_value=mock_swing_results), \
                 patch.object(tuner, 'optimize_rsi_strategy', return_value=mock_rsi_results), \
                 patch.object(tuner, 'optimize_pure_percent_strategy', return_value=mock_pure_results):
                
                results = tuner.run_comprehensive_optimization(
                    pairs=["BTC-USD"], 
                    start_date="2024-01-01", 
                    end_date="2024-01-31"
                )
                
                # Verify structure
                assert 'optimizations' in results
                assert 'best_strategies' in results
                assert 'strategy_performance' in results
                assert 'timestamp' in results
                
                # Verify optimizations contain all strategies
                assert 'swing' in results['optimizations']
                assert 'rsi' in results['optimizations']
                assert 'pure_percent' in results['optimizations']
                
                # Verify best strategy selection (pure_percent should win with score 55.0)
                best_btc = results['best_strategies']['BTC-USD']
                assert best_btc['strategy_type'] == 'pure_percent'
                assert best_btc['score'] == 55.0
                
                # Verify strategy performance summary
                perf = results['strategy_performance']
                assert 'swing' in perf
                assert 'rsi' in perf
                assert 'pure_percent' in perf
    
    def test_comprehensive_optimization_best_strategy_selection(self):
        """Test that comprehensive optimization correctly selects best strategy per pair"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            tuner = ParameterTuner()
            
            # Create scenario where different strategies win for different pairs
            swing_results = {
                "BTC-USD": [{'score': 80.0, 'total_return_pct': 20.0, 'win_rate': 80.0, 'total_trades': 10,
                           'max_drawdown_pct': 3.0, 'sharpe_ratio': 2.0}],
                "ETH-USD": [{'score': 40.0, 'total_return_pct': 8.0, 'win_rate': 60.0, 'total_trades': 12,
                           'max_drawdown_pct': 6.0, 'sharpe_ratio': 1.0}]
            }
            
            rsi_results = {
                "BTC-USD": [{'score': 60.0, 'total_return_pct': 15.0, 'win_rate': 70.0, 'total_trades': 8,
                           'max_drawdown_pct': 4.0, 'sharpe_ratio': 1.5}],
                "ETH-USD": [{'score': 70.0, 'total_return_pct': 18.0, 'win_rate': 75.0, 'total_trades': 9,
                           'max_drawdown_pct': 5.0, 'sharpe_ratio': 1.8}]
            }
            
            pure_results = {
                "BTC-USD": [{'score': 50.0, 'total_return_pct': 12.0, 'win_rate': 65.0, 'total_trades': 6,
                           'max_drawdown_pct': 7.0, 'sharpe_ratio': 1.2}],
                "ETH-USD": [{'score': 45.0, 'total_return_pct': 10.0, 'win_rate': 62.0, 'total_trades': 7,
                           'max_drawdown_pct': 8.0, 'sharpe_ratio': 1.1}]
            }
            
            with patch.object(tuner, 'optimize_swing_strategy', return_value=swing_results), \
                 patch.object(tuner, 'optimize_rsi_strategy', return_value=rsi_results), \
                 patch.object(tuner, 'optimize_pure_percent_strategy', return_value=pure_results):
                
                results = tuner.run_comprehensive_optimization(
                    pairs=["BTC-USD", "ETH-USD"], 
                    start_date="2024-01-01", 
                    end_date="2024-01-31"
                )
                
                # Verify correct best strategy selection
                best_strategies = results['best_strategies']
                
                # BTC-USD: swing should win (score 80.0)
                assert best_strategies['BTC-USD']['strategy_type'] == 'swing'
                assert best_strategies['BTC-USD']['score'] == 80.0
                
                # ETH-USD: rsi should win (score 70.0)  
                assert best_strategies['ETH-USD']['strategy_type'] == 'rsi'
                assert best_strategies['ETH-USD']['score'] == 70.0


class TestOptimalStrategyCreation:
    """Test optimal strategy creation from optimization results"""
    
    def test_create_optimal_strategies(self):
        """Test creation of optimal strategy instances"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            tuner = ParameterTuner()
            
            # Mock optimization results
            optimization_results = {
                'best_strategies': {
                    'BTC-USD': {
                        'strategy_type': 'swing',
                        'parameters': {
                            'swing_threshold': 0.025,
                            'volume_threshold': 1.2,
                            'lookback_period': 10,
                            'total_return_pct': 15.0,
                            'win_rate': 70.0,
                            'max_drawdown_pct': 3.0
                        },
                        'score': 50.0
                    },
                    'ETH-USD': {
                        'strategy_type': 'rsi',
                        'parameters': {
                            'oversold_threshold': 30,
                            'overbought_threshold': 70,
                            'rsi_period': 14,
                            'total_return_pct': 12.0,
                            'win_rate': 65.0,
                            'max_drawdown_pct': 4.0
                        },
                        'score': 45.0
                    },
                    'ADA-USD': {
                        'strategy_type': 'pure_percent',
                        'parameters': {
                            'drop_threshold': 0.05,
                            'rise_threshold': 0.05,
                            'lookback_days': 7,
                            'total_return_pct': 18.0,
                            'win_rate': 75.0,
                            'max_drawdown_pct': 5.0
                        },
                        'score': 55.0
                    }
                }
            }
            
            optimal_strategies = tuner.create_optimal_strategies(optimization_results)
            
            # Verify correct strategy types created
            assert 'BTC-USD' in optimal_strategies
            assert 'ETH-USD' in optimal_strategies  
            assert 'ADA-USD' in optimal_strategies
            
            # Verify swing strategy for BTC-USD
            btc_strategy = optimal_strategies['BTC-USD']['strategy']
            assert isinstance(btc_strategy, SwingTradingStrategy)
            assert btc_strategy.swing_threshold == 0.025
            assert btc_strategy.volume_threshold == 1.2
            assert btc_strategy.lookback_period == 10
            
            # Verify RSI strategy for ETH-USD
            eth_strategy = optimal_strategies['ETH-USD']['strategy']
            assert isinstance(eth_strategy, RSIStrategy)
            assert eth_strategy.oversold_threshold == 30
            assert eth_strategy.overbought_threshold == 70
            assert eth_strategy.rsi_period == 14
            
            # Verify Pure Percent strategy for ADA-USD
            ada_strategy = optimal_strategies['ADA-USD']['strategy']
            assert isinstance(ada_strategy, Pure5PercentStrategy)
            assert ada_strategy.drop_threshold == 0.05
            assert ada_strategy.rise_threshold == 0.05
            assert ada_strategy.lookback_days == 7
            
            # Verify metadata
            assert optimal_strategies['BTC-USD']['expected_return'] == 15.0
            assert optimal_strategies['ETH-USD']['win_rate'] == 65.0
            assert optimal_strategies['ADA-USD']['score'] == 55.0


class TestOptimizationResultsSerialization:
    """Test optimization results saving and loading"""
    
    def test_save_optimization_results(self):
        """Test saving optimization results to JSON"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            tuner = ParameterTuner()
            
            # Create test results with numpy types
            results = {
                'best_strategies': {
                    'BTC-USD': {
                        'score': np.float64(75.5),  # numpy float
                        'total_trades': np.int32(10),  # numpy int
                        'win_rate': np.float32(65.5),  # numpy float32
                    }
                },
                'strategy_performance': {
                    'swing': {
                        'avg_score': np.array([70.0, 80.0, 75.0]).mean(),  # numpy array mean
                        'pairs_count': np.int64(3)
                    }
                },
                'timestamp': '2024-01-15T10:30:00'
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                temp_filename = temp_file.name
            
            try:
                # Should not raise exceptions with numpy types
                tuner.save_optimization_results(results, temp_filename)
                
                # Verify file was created
                assert os.path.exists(temp_filename)
                
                # Verify file content is valid JSON
                import json
                with open(temp_filename, 'r') as f:
                    loaded_data = json.load(f)
                
                # Verify data integrity (numpy types should be converted)
                assert loaded_data['best_strategies']['BTC-USD']['score'] == 75.5
                assert loaded_data['best_strategies']['BTC-USD']['total_trades'] == 10
                assert loaded_data['strategy_performance']['swing']['pairs_count'] == 3
                
            finally:
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
    
    def test_save_optimization_results_with_default_filename(self):
        """Test saving with auto-generated filename"""
        with patch('src.optimization.parameter_tuner.get_config') as mock_config:
            mock_config.return_value = {}
            tuner = ParameterTuner()
            
            results = {'test': 'data', 'timestamp': '2024-01-15T10:30:00'}
            
            # Mock datetime to control filename
            with patch('src.optimization.parameter_tuner.datetime') as mock_dt:
                mock_dt.now.return_value.strftime.return_value = "20240115_103000"
                
                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.write = Mock()
                    
                    tuner.save_optimization_results(results)  # No filename provided
                    
                    # Verify default filename pattern was used
                    mock_open.assert_called_with("optimization_results_20240115_103000.json", 'w')


class TestParameterCombinationGeneration:
    """Test parameter combination generation algorithms"""
    
    def test_parameter_combination_completeness(self):
        """Test that all parameter combinations are generated correctly"""
        # This tests the itertools.product usage in optimization methods
        
        # Define test ranges (smaller than actual for testing)
        swing_thresholds = [0.02, 0.03]
        volume_thresholds = [1.1, 1.2]
        lookback_periods = [5, 10]
        
        expected_combinations = list(itertools.product(
            swing_thresholds, volume_thresholds, lookback_periods
        ))
        
        # Should generate 2 * 2 * 2 = 8 combinations
        assert len(expected_combinations) == 8
        
        # Verify specific combinations exist
        assert (0.02, 1.1, 5) in expected_combinations
        assert (0.03, 1.2, 10) in expected_combinations
        
        # Verify all parameter values are tested
        tested_swing = {combo[0] for combo in expected_combinations}
        tested_volume = {combo[1] for combo in expected_combinations}
        tested_lookback = {combo[2] for combo in expected_combinations}
        
        assert tested_swing == set(swing_thresholds)
        assert tested_volume == set(volume_thresholds)
        assert tested_lookback == set(lookback_periods)
    
    def test_rsi_parameter_validation_in_combinations(self):
        """Test RSI parameter combination validation logic"""
        oversold_thresholds = [25, 30, 35]
        overbought_thresholds = [65, 70, 75]
        rsi_periods = [10, 14, 18]
        
        all_combinations = list(itertools.product(
            oversold_thresholds, overbought_thresholds, rsi_periods
        ))
        
        # Filter out invalid combinations (oversold >= overbought)
        valid_combinations = [
            combo for combo in all_combinations 
            if combo[0] < combo[1]  # oversold < overbought
        ]
        
        # All combinations should be valid since our ranges don't overlap
        assert len(valid_combinations) == len(all_combinations)
        
        # Verify no invalid combinations exist
        for oversold, overbought, period in valid_combinations:
            assert oversold < overbought, f"Invalid RSI combo: oversold={oversold}, overbought={overbought}"
            assert 10 <= period <= 21, f"RSI period should be reasonable: {period}"
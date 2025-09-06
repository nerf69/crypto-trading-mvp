"""
Unit tests for backtesting engine.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.backtesting.engine import BacktestEngine, BacktestResult, Position
from src.strategies.base import Strategy, TradingSignal, SignalType
from src.strategies.swing import SwingTradingStrategy


class TestBacktestResult:
    """Test BacktestResult class"""
    
    def test_backtest_result_creation(self):
        """Test BacktestResult initialization"""
        positions = []
        equity_curve = pd.DataFrame()
        trade_log = pd.DataFrame()
        
        result = BacktestResult(
            strategy_name="Test Strategy",
            pair="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-01-31",
            initial_capital=1000.0,
            final_capital=1100.0,
            total_return=100.0,
            total_return_pct=10.0,
            total_trades=5,
            winning_trades=3,
            losing_trades=2,
            win_rate=60.0,
            avg_win=50.0,
            avg_loss=-25.0,
            max_drawdown=50.0,
            max_drawdown_pct=5.0,
            sharpe_ratio=1.2,
            positions=positions,
            equity_curve=equity_curve,
            trade_log=trade_log
        )
        
        assert result.strategy_name == "Test Strategy"
        assert result.total_return_pct == 10.0
        assert result.win_rate == 60.0
        assert result.sharpe_ratio == 1.2
    
    def test_backtest_result_to_dict(self):
        """Test BacktestResult to_dict conversion"""
        # Create sample position
        position = Position(
            pair="BTC-USD",
            entry_price=100.0,
            entry_time=datetime(2024, 1, 1),
            size=1.0,
            signal_type=SignalType.BUY,
            strategy_name="Test",
            confidence=0.8,
            stop_loss=95.0,
            take_profit=110.0
        )
        position.close_position(105.0, datetime(2024, 1, 2), "Take profit")
        
        # Create sample equity curve
        equity_curve = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'equity': [1000, 1100]
        })
        
        result = BacktestResult(
            strategy_name="Test Strategy",
            pair="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-01-31",
            initial_capital=1000.0,
            final_capital=1100.0,
            total_return=100.0,
            total_return_pct=10.0,
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            win_rate=100.0,
            avg_win=100.0,
            avg_loss=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=2.0,
            positions=[position],
            equity_curve=equity_curve,
            trade_log=pd.DataFrame()
        )
        
        result_dict = result.to_dict()
        
        assert 'metrics' in result_dict
        assert 'trades' in result_dict
        assert 'equity_curve' in result_dict
        assert result_dict['metrics']['total_return'] == 10.0
        assert len(result_dict['trades']) == 1
        assert len(result_dict['equity_curve']) == 2


class TestPosition:
    """Test Position class"""
    
    def test_position_creation(self):
        """Test Position initialization"""
        position = Position(
            pair="BTC-USD",
            entry_price=100.0,
            entry_time=datetime(2024, 1, 1),
            size=1.0,
            signal_type=SignalType.BUY,
            strategy_name="Test Strategy",
            confidence=0.8,
            stop_loss=95.0,
            take_profit=110.0
        )
        
        assert position.pair == "BTC-USD"
        assert position.entry_price == 100.0
        assert position.size == 1.0
        assert position.signal_type == SignalType.BUY
        assert position.is_open() is True
        assert position.pnl is None
    
    def test_position_close_profitable(self):
        """Test closing a profitable position"""
        position = Position(
            pair="BTC-USD",
            entry_price=100.0,
            entry_time=datetime(2024, 1, 1),
            size=1.0,
            signal_type=SignalType.BUY,
            strategy_name="Test",
            confidence=0.8,
            stop_loss=95.0,
            take_profit=110.0
        )
        
        exit_price = 105.0
        exit_time = datetime(2024, 1, 2)
        
        position.close_position(exit_price, exit_time, "Take profit")
        
        assert position.is_open() is False
        assert position.exit_price == exit_price
        assert position.exit_time == exit_time
        assert position.exit_reason == "Take profit"
        assert position.pnl == 5.0  # (105 - 100) * 1
        assert position.pnl_pct == 5.0  # 5% gain
    
    def test_position_close_loss(self):
        """Test closing a losing position"""
        position = Position(
            pair="BTC-USD",
            entry_price=100.0,
            entry_time=datetime(2024, 1, 1),
            size=1.0,
            signal_type=SignalType.BUY,
            strategy_name="Test",
            confidence=0.8,
            stop_loss=95.0,
            take_profit=110.0
        )
        
        exit_price = 95.0
        position.close_position(exit_price, datetime(2024, 1, 2), "Stop loss")
        
        assert position.pnl == -5.0  # (95 - 100) * 1
        assert position.pnl_pct == -5.0  # 5% loss


class TestBacktestEngine:
    """Test BacktestEngine class"""
    
    def test_engine_initialization(self):
        """Test BacktestEngine initialization"""
        engine = BacktestEngine()
        
        assert engine.config is not None
        assert engine.data_fetcher is not None
        assert engine.data_processor is not None
        assert engine.commission == 0.005  # Default commission
    
    @patch('src.backtesting.engine.BacktestEngine._fetch_and_prepare_data')
    def test_run_backtest_no_data(self, mock_fetch_data):
        """Test backtest with no data available"""
        mock_fetch_data.return_value = None
        
        engine = BacktestEngine()
        strategy = SwingTradingStrategy()
        
        result = engine.run_backtest(strategy, "BTC-USD", "2024-01-01", "2024-01-31")
        
        assert result.total_trades == 0
        assert result.total_return == 0.0
        assert result.final_capital == result.initial_capital
    
    @patch('src.backtesting.engine.BacktestEngine._fetch_and_prepare_data')
    def test_run_backtest_with_data(self, mock_fetch_data, sample_ohlcv_data):
        """Test successful backtest run"""
        mock_fetch_data.return_value = sample_ohlcv_data
        
        engine = BacktestEngine()
        strategy = SwingTradingStrategy()
        
        result = engine.run_backtest(strategy, "BTC-USD", "2024-01-01", "2024-01-31", 1000)
        
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == strategy.name
        assert result.pair == "BTC-USD"
        assert result.initial_capital == 1000
        assert result.final_capital >= 0
        assert result.total_trades >= 0
    
    def test_calculate_optimal_granularity_short_period(self):
        """Test granularity calculation for short time periods"""
        engine = BacktestEngine()
        
        # 1 day period
        granularity, desc = engine._calculate_optimal_granularity("2024-01-01", "2024-01-02")
        assert granularity == 300  # 5-minute
        assert desc == "5-minute"
    
    def test_calculate_optimal_granularity_long_period(self):
        """Test granularity calculation for long time periods"""
        engine = BacktestEngine()
        
        # 100 day period
        granularity, desc = engine._calculate_optimal_granularity("2024-01-01", "2024-04-10")
        assert granularity == 86400  # 1-day
        assert desc == "1-day"
    
    def test_calculate_optimal_granularity_medium_period(self):
        """Test granularity calculation for medium time periods"""
        engine = BacktestEngine()
        
        # 10 day period
        granularity, desc = engine._calculate_optimal_granularity("2024-01-01", "2024-01-11")
        assert granularity in [3600, 21600]  # 1-hour or 6-hour
    
    def test_simulate_trading_basic_flow(self, sample_ohlcv_data):
        """Test basic trading simulation flow"""
        engine = BacktestEngine()
        
        # Create a simple test strategy
        class TestStrategy(Strategy):
            def calculate_signal(self, df, pair):
                # Always generate buy signal for testing
                return TradingSignal(
                    timestamp=df.iloc[-1]['timestamp'],
                    pair=pair,
                    signal=SignalType.BUY,
                    price=df.iloc[-1]['close'],
                    confidence=0.8,
                    strategy_name="Test",
                    reason="Test signal",
                    indicators={}
                )
        
        strategy = TestStrategy("Test Strategy")
        result = engine._simulate_trading(strategy, sample_ohlcv_data, "BTC-USD", 
                                        "2024-01-01", "2024-01-31", 1000)
        
        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0
        assert len(result.positions) >= 0
        assert not result.equity_curve.empty
    
    def test_check_exit_conditions_stop_loss(self):
        """Test stop loss exit condition"""
        engine = BacktestEngine()
        
        # Long position with stop loss
        position = Position(
            pair="BTC-USD",
            entry_price=100.0,
            entry_time=datetime(2024, 1, 1),
            size=1.0,
            signal_type=SignalType.BUY,
            strategy_name="Test",
            confidence=0.8,
            stop_loss=95.0,
            take_profit=110.0
        )
        
        # Price hits stop loss
        exit_reason = engine._check_exit_conditions(position, 94.0, datetime(2024, 1, 2))
        assert exit_reason == "Stop loss"
        
        # Price above stop loss
        exit_reason = engine._check_exit_conditions(position, 96.0, datetime(2024, 1, 2))
        assert exit_reason is None
    
    def test_check_exit_conditions_take_profit(self):
        """Test take profit exit condition"""
        engine = BacktestEngine()
        
        position = Position(
            pair="BTC-USD",
            entry_price=100.0,
            entry_time=datetime(2024, 1, 1),
            size=1.0,
            signal_type=SignalType.BUY,
            strategy_name="Test",
            confidence=0.8,
            stop_loss=95.0,
            take_profit=110.0
        )
        
        # Price hits take profit
        exit_reason = engine._check_exit_conditions(position, 111.0, datetime(2024, 1, 2))
        assert exit_reason == "Take profit"
        
        # Price below take profit
        exit_reason = engine._check_exit_conditions(position, 109.0, datetime(2024, 1, 2))
        assert exit_reason is None
    
    def test_calculate_position_size_dynamic(self):
        """Test dynamic position sizing"""
        engine = BacktestEngine()
        
        signal = TradingSignal(
            timestamp=datetime.now(),
            pair="BTC-USD",
            signal=SignalType.BUY,
            price=100.0,
            confidence=0.8,  # High confidence
            strategy_name="Test",
            reason="Test",
            indicators={}
        )
        
        position_sizing = {
            'method': 'dynamic',
            'max_position_size': 1.0,
            'min_position_size': 0.33
        }
        
        size = engine._calculate_position_size(signal, 1000, position_sizing)
        
        # High confidence should result in larger position
        assert size > 0
        expected_value = 1000 * 1.0  # max_position_size for high confidence
        expected_size = expected_value / 100.0  # divided by price
        assert abs(size - expected_size) < 0.01
    
    def test_calculate_position_size_fixed(self):
        """Test fixed position sizing"""
        engine = BacktestEngine()
        
        signal = TradingSignal(
            timestamp=datetime.now(),
            pair="BTC-USD",
            signal=SignalType.BUY,
            price=100.0,
            confidence=0.6,
            strategy_name="Test",
            reason="Test",
            indicators={}
        )
        
        position_sizing = {
            'method': 'fixed',
            'max_position_size': 0.5,
            'min_position_size': 0.33
        }
        
        size = engine._calculate_position_size(signal, 1000, position_sizing)
        
        # Fixed method should use max_position_size regardless of confidence
        expected_value = 1000 * 0.5
        expected_size = expected_value / 100.0
        assert abs(size - expected_size) < 0.01
    
    def test_create_position(self):
        """Test position creation"""
        engine = BacktestEngine()
        
        signal = TradingSignal(
            timestamp=datetime.now(),
            pair="BTC-USD",
            signal=SignalType.BUY,
            price=100.0,
            confidence=0.8,
            strategy_name="Test Strategy",
            reason="Test signal",
            indicators={}
        )
        
        strategy = SwingTradingStrategy()  # Has stop loss and take profit methods
        
        position = engine._create_position(signal, 100.0, datetime.now(), 1.0, strategy)
        
        assert isinstance(position, Position)
        assert position.pair == "BTC-USD"
        assert position.entry_price == 100.0
        assert position.size == 1.0
        assert position.signal_type == SignalType.BUY
        assert position.stop_loss < 100.0  # Stop loss should be below entry for long
        assert position.take_profit > 100.0  # Take profit should be above entry for long
    
    def test_calculate_position_value_long(self):
        """Test position value calculation for long position"""
        engine = BacktestEngine()
        
        position = Position(
            pair="BTC-USD",
            entry_price=100.0,
            entry_time=datetime(2024, 1, 1),
            size=1.0,
            signal_type=SignalType.BUY,
            strategy_name="Test",
            confidence=0.8,
            stop_loss=95.0,
            take_profit=110.0
        )
        
        # Current price above entry
        current_value = engine._calculate_position_value(position, 105.0)
        assert current_value == 5.0  # (105 - 100) * 1
        
        # Current price below entry
        current_value = engine._calculate_position_value(position, 95.0)
        assert current_value == -5.0  # (95 - 100) * 1
    
    def test_create_trade_log(self):
        """Test trade log creation"""
        engine = BacktestEngine()
        
        position1 = Position(
            pair="BTC-USD",
            entry_price=100.0,
            entry_time=datetime(2024, 1, 1),
            size=1.0,
            signal_type=SignalType.BUY,
            strategy_name="Test",
            confidence=0.8,
            stop_loss=95.0,
            take_profit=110.0
        )
        position1.close_position(105.0, datetime(2024, 1, 2), "Take profit")
        
        position2 = Position(
            pair="ETH-USD",
            entry_price=200.0,
            entry_time=datetime(2024, 1, 3),
            size=0.5,
            signal_type=SignalType.BUY,
            strategy_name="Test",
            confidence=0.7,
            stop_loss=190.0,
            take_profit=220.0
        )
        # Leave position2 open
        
        positions = [position1, position2]
        trade_log = engine._create_trade_log(positions)
        
        # Should only include closed positions
        assert len(trade_log) == 1
        assert trade_log.iloc[0]['pair'] == 'BTC-USD'
        assert trade_log.iloc[0]['pnl'] == 5.0
    
    def test_run_multi_pair_backtest(self):
        """Test multi-pair backtesting"""
        engine = BacktestEngine()
        
        with patch.object(engine, 'run_backtest') as mock_run_backtest:
            # Mock successful backtest results
            mock_result = BacktestResult(
                strategy_name="Test",
                pair="BTC-USD",
                start_date="2024-01-01",
                end_date="2024-01-31",
                initial_capital=1000,
                final_capital=1100,
                total_return=100,
                total_return_pct=10.0,
                total_trades=5,
                winning_trades=3,
                losing_trades=2,
                win_rate=60.0,
                avg_win=50.0,
                avg_loss=-25.0,
                max_drawdown=50.0,
                max_drawdown_pct=5.0,
                sharpe_ratio=1.2,
                positions=[],
                equity_curve=pd.DataFrame(),
                trade_log=pd.DataFrame()
            )
            mock_run_backtest.return_value = mock_result
            
            strategy = SwingTradingStrategy()
            pairs = ["BTC-USD", "ETH-USD"]
            
            results = engine.run_multi_pair_backtest(strategy, pairs, "2024-01-01", "2024-01-31")
            
            assert len(results) == 2
            assert "BTC-USD" in results
            assert "ETH-USD" in results
            assert mock_run_backtest.call_count == 2
    
    def test_compare_strategies(self):
        """Test strategy comparison"""
        engine = BacktestEngine()
        
        with patch.object(engine, 'run_backtest') as mock_run_backtest:
            mock_result = BacktestResult(
                strategy_name="Test",
                pair="BTC-USD",
                start_date="2024-01-01",
                end_date="2024-01-31",
                initial_capital=1000,
                final_capital=1100,
                total_return=100,
                total_return_pct=10.0,
                total_trades=5,
                winning_trades=3,
                losing_trades=2,
                win_rate=60.0,
                avg_win=50.0,
                avg_loss=-25.0,
                max_drawdown=50.0,
                max_drawdown_pct=5.0,
                sharpe_ratio=1.2,
                positions=[],
                equity_curve=pd.DataFrame(),
                trade_log=pd.DataFrame()
            )
            mock_run_backtest.return_value = mock_result
            
            strategies = [SwingTradingStrategy(), SwingTradingStrategy()]  # Two strategies
            
            results = engine.compare_strategies(strategies, "BTC-USD", "2024-01-01", "2024-01-31")
            
            assert len(results) == 2
            assert mock_run_backtest.call_count == 2


class TestBacktestEngineEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_data_handling(self, sample_ohlcv_data):
        """Test handling of empty data"""
        engine = BacktestEngine()
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        
        class TestStrategy(Strategy):
            def calculate_signal(self, df, pair):
                return TradingSignal(
                    timestamp=datetime.now(),
                    pair=pair,
                    signal=SignalType.HOLD,
                    price=100.0,
                    confidence=0.5,
                    strategy_name="Test",
                    reason="Test",
                    indicators={}
                )
        
        strategy = TestStrategy("Test")
        result = engine._simulate_trading(strategy, empty_df, "BTC-USD", 
                                        "2024-01-01", "2024-01-31", 1000)
        
        # Should handle gracefully
        assert result.total_trades == 0
        assert result.final_capital == result.initial_capital
    
    def test_insufficient_warmup_data(self):
        """Test with insufficient data for warmup period"""
        engine = BacktestEngine()
        
        # Very small dataset
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='D'),
            'open': [100] * 5,
            'high': [105] * 5,
            'low': [95] * 5,
            'close': [102] * 5,
            'volume': [1000] * 5
        })
        
        class TestStrategy(Strategy):
            def calculate_signal(self, df, pair):
                return TradingSignal(
                    timestamp=df.iloc[-1]['timestamp'],
                    pair=pair,
                    signal=SignalType.BUY,
                    price=df.iloc[-1]['close'],
                    confidence=0.8,
                    strategy_name="Test",
                    reason="Test",
                    indicators={}
                )
        
        strategy = TestStrategy("Test")
        result = engine._simulate_trading(strategy, small_data, "BTC-USD", 
                                        "2024-01-01", "2024-01-05", 1000)
        
        # Should handle small dataset without errors
        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0
    
    def test_extreme_price_movements(self):
        """Test with extreme price movements"""
        engine = BacktestEngine()
        
        # Create data with extreme price swings
        extreme_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
            'open': [100, 200, 50, 300, 25, 400, 10, 500, 5, 600],
            'high': [110, 220, 60, 330, 35, 440, 20, 550, 15, 660],
            'low': [90, 180, 40, 270, 15, 360, 1, 450, 1, 540],
            'close': [105, 190, 45, 290, 20, 390, 8, 490, 3, 590],
            'volume': [1000] * 10
        })
        
        class TestStrategy(Strategy):
            def calculate_signal(self, df, pair):
                return TradingSignal(
                    timestamp=df.iloc[-1]['timestamp'],
                    pair=pair,
                    signal=SignalType.HOLD,
                    price=df.iloc[-1]['close'],
                    confidence=0.5,
                    strategy_name="Test",
                    reason="Test",
                    indicators={}
                )
        
        strategy = TestStrategy("Test")
        result = engine._simulate_trading(strategy, extreme_data, "BTC-USD", 
                                        "2024-01-01", "2024-01-10", 1000)
        
        # Should handle extreme movements without crashing
        assert isinstance(result, BacktestResult)
        assert np.isfinite(result.final_capital)
        assert np.isfinite(result.total_return)
    
    def test_zero_commission_edge_case(self):
        """Test backtesting with zero commission"""
        # Test with custom config that has zero commission
        with patch('src.backtesting.engine.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.get.return_value = 0.0  # Zero commission
            mock_get_config.return_value = mock_config
            
            engine = BacktestEngine()
            assert engine.commission == 0.0
    
    def test_max_positions_limit(self, sample_ohlcv_data):
        """Test that max positions limit is respected"""
        engine = BacktestEngine()
        
        # Strategy that always generates buy signals
        class AlwaysBuyStrategy(Strategy):
            def calculate_signal(self, df, pair):
                return TradingSignal(
                    timestamp=df.iloc[-1]['timestamp'],
                    pair=pair,
                    signal=SignalType.BUY,
                    price=df.iloc[-1]['close'],
                    confidence=0.8,
                    strategy_name="Always Buy",
                    reason="Always buying",
                    indicators={}
                )
        
        strategy = AlwaysBuyStrategy("Always Buy")
        
        # Mock config to set max_positions to 2
        with patch.object(engine, 'config') as mock_config:
            mock_config.get.return_value = 2  # Max 2 positions
            
            result = engine._simulate_trading(strategy, sample_ohlcv_data, "BTC-USD", 
                                            "2024-01-01", "2024-01-31", 1000)
            
            # Should respect position limits (this is hard to test directly without 
            # inspecting intermediate state, but the simulation should complete)
            assert isinstance(result, BacktestResult)
    
    def test_signal_generation_exceptions(self, sample_ohlcv_data):
        """Test handling of strategy exceptions during signal generation"""
        engine = BacktestEngine()
        
        # Strategy that raises exceptions
        class FaultyStrategy(Strategy):
            def __init__(self):
                super().__init__("Faulty Strategy")
                self.call_count = 0
            
            def calculate_signal(self, df, pair):
                self.call_count += 1
                if self.call_count > 2:  # Fail after 2 calls
                    raise ValueError("Strategy error")
                return TradingSignal(
                    timestamp=df.iloc[-1]['timestamp'],
                    pair=pair,
                    signal=SignalType.HOLD,
                    price=df.iloc[-1]['close'],
                    confidence=0.5,
                    strategy_name="Faulty",
                    reason="Test",
                    indicators={}
                )
        
        strategy = FaultyStrategy()
        result = engine._simulate_trading(strategy, sample_ohlcv_data, "BTC-USD", 
                                        "2024-01-01", "2024-01-31", 1000)
        
        # Should handle strategy exceptions gracefully
        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0  # Some trades might have executed before errors
    
    def test_create_empty_result(self):
        """Test creation of empty result when no data available"""
        engine = BacktestEngine()
        
        empty_result = engine._create_empty_result("Test Strategy", "BTC-USD", 
                                                 "2024-01-01", "2024-01-31", 1000)
        
        assert empty_result.strategy_name == "Test Strategy"
        assert empty_result.pair == "BTC-USD"
        assert empty_result.total_trades == 0
        assert empty_result.total_return == 0.0
        assert empty_result.final_capital == empty_result.initial_capital
        assert empty_result.positions == []
        assert empty_result.equity_curve.empty
        assert empty_result.trade_log.empty
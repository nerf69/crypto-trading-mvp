import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from ..strategies.base import Strategy, TradingSignal, SignalType
from ..data.fetcher import CoinbaseDataFetcher
from ..data.processor import DataProcessor
from ..config import get_config

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a trading position"""
    pair: str
    entry_price: float
    entry_time: datetime
    size: float
    signal_type: SignalType
    strategy_name: str
    confidence: float
    stop_loss: float
    take_profit: float
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    def is_open(self) -> bool:
        return self.exit_price is None
    
    def close_position(self, exit_price: float, exit_time: datetime, reason: str):
        """Close the position and calculate P&L"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        
        # Calculate P&L
        if self.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            self.pnl = (exit_price - self.entry_price) * self.size
            self.pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
        else:  # SELL or STRONG_SELL
            self.pnl = (self.entry_price - exit_price) * self.size
            self.pnl_pct = ((self.entry_price - exit_price) / self.entry_price) * 100

@dataclass  
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    pair: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    positions: List[Position]
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame

class BacktestEngine:
    """
    Backtesting engine for trading strategies
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.data_fetcher = CoinbaseDataFetcher()
        self.data_processor = DataProcessor()
        
        # Backtesting parameters
        self.commission = self.config.get('backtesting.commission', 0.005)  # 0.5% per trade
        self.slippage = 0.001  # 0.1% slippage
        
        logger.info(f"Backtesting engine initialized with {self.commission*100}% commission")
    
    def run_backtest(self, strategy: Strategy, pair: str, start_date: str, end_date: str,
                    initial_capital: float = None) -> BacktestResult:
        """
        Run backtest for a single strategy and trading pair
        
        Args:
            strategy: Trading strategy to test
            pair: Trading pair (e.g., 'SOL-USD')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            initial_capital: Initial capital amount
            
        Returns:
            BacktestResult object with performance metrics
        """
        logger.info(f"Starting backtest: {strategy.name} on {pair} from {start_date} to {end_date}")
        
        if initial_capital is None:
            initial_capital = self.config.get('backtesting.initial_capital', 1000)
        
        # Fetch historical data
        df = self._fetch_and_prepare_data(pair, start_date, end_date)
        if df is None or df.empty:
            logger.error(f"No data available for {pair}")
            return self._create_empty_result(strategy.name, pair, start_date, end_date, initial_capital)
        
        # Add strategy-specific indicators
        df = strategy.add_required_indicators(df)
        
        # Add all technical indicators
        df = self.data_processor.add_all_indicators(df)
        
        # Run the backtesting simulation
        result = self._simulate_trading(strategy, df, pair, start_date, end_date, initial_capital)
        
        logger.info(f"Backtest completed: {result.total_trades} trades, "
                   f"{result.total_return_pct:.2f}% return, {result.win_rate:.1f}% win rate")
        
        return result
    
    def run_multi_pair_backtest(self, strategy: Strategy, pairs: List[str], 
                              start_date: str, end_date: str, 
                              initial_capital: float = None) -> Dict[str, BacktestResult]:
        """Run backtest on multiple trading pairs"""
        logger.info(f"Running multi-pair backtest: {strategy.name} on {len(pairs)} pairs")
        
        results = {}
        for pair in pairs:
            try:
                result = self.run_backtest(strategy, pair, start_date, end_date, initial_capital)
                results[pair] = result
            except Exception as e:
                logger.error(f"Error backtesting {pair}: {e}")
                
        return results
    
    def compare_strategies(self, strategies: List[Strategy], pair: str, 
                          start_date: str, end_date: str,
                          initial_capital: float = None) -> Dict[str, BacktestResult]:
        """Compare multiple strategies on the same pair"""
        logger.info(f"Comparing {len(strategies)} strategies on {pair}")
        
        results = {}
        for strategy in strategies:
            try:
                result = self.run_backtest(strategy, pair, start_date, end_date, initial_capital)
                results[strategy.name] = result
            except Exception as e:
                logger.error(f"Error backtesting {strategy.name}: {e}")
                
        return results
    
    def _fetch_and_prepare_data(self, pair: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch and prepare historical data for backtesting"""
        try:
            df = self.data_fetcher.get_historical_data(pair, start_date, end_date, granularity=300)  # 5-min data
            
            if df is None or df.empty:
                logger.warning(f"No data returned for {pair}")
                return None
            
            # Clean and validate data
            df = self.data_processor.clean_data(df)
            
            if df.empty:
                logger.warning(f"No data remaining after cleaning for {pair}")
                return None
            
            logger.debug(f"Prepared {len(df)} data points for {pair}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {e}")
            return None
    
    def _simulate_trading(self, strategy: Strategy, df: pd.DataFrame, pair: str,
                         start_date: str, end_date: str, initial_capital: float) -> BacktestResult:
        """Simulate trading with the given strategy"""
        
        positions: List[Position] = []
        open_positions: List[Position] = []
        equity_curve = []
        current_capital = initial_capital
        max_capital = initial_capital
        max_drawdown = 0.0
        
        # Risk management parameters
        max_positions = self.config.get('trading.max_positions', 3)
        position_sizing = self.config.get('trading.position_sizing', {})
        
        logger.debug(f"Starting simulation with ${initial_capital:.2f} capital")
        
        for i in range(len(df)):
            current_row = df.iloc[i]
            current_time = current_row['timestamp']
            current_price = current_row['close']
            
            # Check stop losses and take profits for open positions
            for position in open_positions.copy():
                exit_reason = self._check_exit_conditions(position, current_price, current_time)
                if exit_reason:
                    position.close_position(current_price, current_time, exit_reason)
                    open_positions.remove(position)
                    
                    # Update capital
                    current_capital += position.pnl
                    logger.debug(f"Closed position: {position.pair} {position.signal_type.value} "
                               f"P&L: ${position.pnl:.2f} ({position.pnl_pct:.2f}%)")
            
            # Generate trading signal (need sufficient historical data)
            if i >= 50:  # Wait for indicators to stabilize
                try:
                    signal = strategy.calculate_signal(df.iloc[:i+1], pair)
                    
                    # Check if we should enter a new position
                    if (signal.signal in [SignalType.BUY, SignalType.STRONG_BUY, SignalType.SELL, SignalType.STRONG_SELL] and
                        len(open_positions) < max_positions and
                        signal.confidence >= 0.6):  # Minimum confidence threshold
                        
                        position_size = self._calculate_position_size(
                            signal, current_capital, position_sizing
                        )
                        
                        if position_size > 0:
                            # Create new position
                            position = self._create_position(
                                signal, current_price, current_time, position_size, strategy
                            )
                            
                            positions.append(position)
                            open_positions.append(position)
                            
                            # Update capital (subtract position value + commission)
                            trade_value = position_size * current_price
                            commission_cost = trade_value * self.commission
                            current_capital -= commission_cost
                            
                            logger.debug(f"Opened position: {pair} {signal.signal.value} "
                                       f"at ${current_price:.2f}, size: {position_size:.4f}, "
                                       f"confidence: {signal.confidence:.2f}")
                
                except Exception as e:
                    logger.debug(f"Error generating signal at index {i}: {e}")
            
            # Track equity curve
            position_value = sum(self._calculate_position_value(pos, current_price) for pos in open_positions)
            total_equity = current_capital + position_value
            
            equity_curve.append({
                'timestamp': current_time,
                'equity': total_equity,
                'cash': current_capital,
                'positions_value': position_value,
                'open_positions': len(open_positions)
            })
            
            # Track maximum drawdown
            if total_equity > max_capital:
                max_capital = total_equity
            
            drawdown = (max_capital - total_equity) / max_capital * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Close any remaining open positions at the end
        final_price = df.iloc[-1]['close']
        final_time = df.iloc[-1]['timestamp']
        
        for position in open_positions:
            position.close_position(final_price, final_time, "End of backtest")
            current_capital += position.pnl
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(equity_curve)
        
        # Create trade log DataFrame
        trade_log = self._create_trade_log(positions)
        
        # Calculate final metrics
        final_capital = current_capital
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # Calculate trade statistics
        closed_positions = [pos for pos in positions if not pos.is_open()]
        total_trades = len(closed_positions)
        winning_trades = len([pos for pos in closed_positions if pos.pnl > 0])
        losing_trades = len([pos for pos in closed_positions if pos.pnl < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = np.mean([pos.pnl for pos in closed_positions if pos.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([pos.pnl for pos in closed_positions if pos.pnl < 0]) if losing_trades > 0 else 0
        
        # Calculate Sharpe ratio
        if len(equity_df) > 1:
            returns = equity_df['equity'].pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        return BacktestResult(
            strategy_name=strategy.name,
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            positions=positions,
            equity_curve=equity_df,
            trade_log=trade_log
        )
    
    def _check_exit_conditions(self, position: Position, current_price: float, 
                             current_time: datetime) -> Optional[str]:
        """Check if position should be closed based on stop loss or take profit"""
        
        if position.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            # Long position
            if current_price <= position.stop_loss:
                return "Stop loss"
            elif current_price >= position.take_profit:
                return "Take profit"
        else:
            # Short position (sell signals)
            if current_price >= position.stop_loss:
                return "Stop loss"  
            elif current_price <= position.take_profit:
                return "Take profit"
        
        return None
    
    def _calculate_position_size(self, signal: TradingSignal, available_capital: float,
                               position_sizing: Dict) -> float:
        """Calculate position size based on signal confidence and available capital"""
        
        sizing_method = position_sizing.get('method', 'dynamic')
        max_position_size = position_sizing.get('max_position_size', 1.0)
        min_position_size = position_sizing.get('min_position_size', 0.33)
        
        if sizing_method == 'fixed':
            allocation_pct = max_position_size
        else:  # dynamic
            # Size based on confidence
            if signal.confidence >= 0.8:
                allocation_pct = max_position_size
            elif signal.confidence >= 0.6:
                allocation_pct = (min_position_size + max_position_size) / 2
            else:
                allocation_pct = min_position_size
        
        # Calculate position size in terms of quantity
        position_value = available_capital * allocation_pct
        position_size = position_value / signal.price
        
        return position_size
    
    def _create_position(self, signal: TradingSignal, current_price: float, 
                        current_time: datetime, position_size: float,
                        strategy: Strategy) -> Position:
        """Create a new trading position"""
        
        # Calculate stop loss and take profit levels
        if hasattr(strategy, 'get_stop_loss_level'):
            stop_loss = strategy.get_stop_loss_level(current_price, signal.signal)
        else:
            stop_loss_pct = 0.05  # Default 5%
            if signal.signal in [SignalType.BUY, SignalType.STRONG_BUY]:
                stop_loss = current_price * (1 - stop_loss_pct)
            else:
                stop_loss = current_price * (1 + stop_loss_pct)
        
        if hasattr(strategy, 'get_take_profit_level'):
            take_profit = strategy.get_take_profit_level(current_price, signal.signal, signal.confidence)
        else:
            profit_pct = 0.06  # Default 6%
            if signal.signal in [SignalType.BUY, SignalType.STRONG_BUY]:
                take_profit = current_price * (1 + profit_pct)
            else:
                take_profit = current_price * (1 - profit_pct)
        
        return Position(
            pair=signal.pair,
            entry_price=current_price,
            entry_time=current_time,
            size=position_size,
            signal_type=signal.signal,
            strategy_name=strategy.name,
            confidence=signal.confidence,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def _calculate_position_value(self, position: Position, current_price: float) -> float:
        """Calculate current value of an open position"""
        if position.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return (current_price - position.entry_price) * position.size
        else:
            return (position.entry_price - current_price) * position.size
    
    def _create_trade_log(self, positions: List[Position]) -> pd.DataFrame:
        """Create a DataFrame with detailed trade log"""
        trades = []
        
        for position in positions:
            if not position.is_open():  # Only include closed positions
                trades.append({
                    'pair': position.pair,
                    'strategy': position.strategy_name,
                    'signal_type': position.signal_type.value,
                    'entry_time': position.entry_time,
                    'exit_time': position.exit_time,
                    'entry_price': position.entry_price,
                    'exit_price': position.exit_price,
                    'size': position.size,
                    'confidence': position.confidence,
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit,
                    'exit_reason': position.exit_reason,
                    'pnl': position.pnl,
                    'pnl_pct': position.pnl_pct,
                    'duration': position.exit_time - position.entry_time
                })
        
        return pd.DataFrame(trades)
    
    def _create_empty_result(self, strategy_name: str, pair: str, start_date: str, 
                           end_date: str, initial_capital: float) -> BacktestResult:
        """Create an empty result when no data is available"""
        return BacktestResult(
            strategy_name=strategy_name,
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=initial_capital,
            total_return=0.0,
            total_return_pct=0.0,
            total_trades=0,
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
    
    def save_results(self, results: Dict[str, BacktestResult], output_dir: str = "backtest_results"):
        """Save backtest results to files"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for name, result in results.items():
            safe_name = name.replace(' ', '_').replace('/', '-')
            
            # Save trade log
            if not result.trade_log.empty:
                result.trade_log.to_csv(f"{output_dir}/{safe_name}_trades.csv", index=False)
            
            # Save equity curve
            if not result.equity_curve.empty:
                result.equity_curve.to_csv(f"{output_dir}/{safe_name}_equity.csv", index=False)
            
            # Save summary metrics
            summary = {
                'strategy_name': result.strategy_name,
                'pair': result.pair,
                'start_date': result.start_date,
                'end_date': result.end_date,
                'initial_capital': result.initial_capital,
                'final_capital': result.final_capital,
                'total_return': result.total_return,
                'total_return_pct': result.total_return_pct,
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'win_rate': result.win_rate,
                'avg_win': result.avg_win,
                'avg_loss': result.avg_loss,
                'max_drawdown': result.max_drawdown,
                'max_drawdown_pct': result.max_drawdown_pct,
                'sharpe_ratio': result.sharpe_ratio
            }
            
            with open(f"{output_dir}/{safe_name}_summary.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Backtest results saved to {output_dir}/")
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from decimal import Decimal, getcontext, ROUND_HALF_UP
import json

from ..strategies.base import Strategy, TradingSignal, SignalType
from ..data.fetcher import CoinbaseDataFetcher
from ..data.processor import DataProcessor
from ..config import get_config
from ..constants import (
    MIN_SIGNAL_CONFIDENCE, DEFAULT_COMMISSION_RATE, DEFAULT_SLIPPAGE_PCT,
    MAX_PORTFOLIO_RISK, COINBASE_MAX_CANDLES_PER_REQUEST, MAX_POSITION_COUNT,
    PRICE_PRECISION, MIN_TRADE_SIZE_USD
)

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
        
        # Calculate P&L for spot trading (long positions only)
        # P&L = (exit_price - entry_price) * size
        self.pnl = (exit_price - self.entry_price) * self.size
        self.pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100

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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert BacktestResult to dictionary format compatible with dashboard"""
        # Convert positions to dict format
        trades_data = []
        for pos in self.positions:
            if not pos.is_open():
                trades_data.append({
                    'entry_time': pos.entry_time.strftime('%Y-%m-%d %H:%M:%S') if pos.entry_time else None,
                    'exit_time': pos.exit_time.strftime('%Y-%m-%d %H:%M:%S') if pos.exit_time else None,
                    'signal': pos.signal_type.value,
                    'entry_price': pos.entry_price,
                    'exit_price': pos.exit_price,
                    'return': pos.pnl_pct,
                    'pnl': pos.pnl,
                    'exit_reason': pos.exit_reason
                })
        
        # Convert equity curve to dict format
        equity_data = []
        if not self.equity_curve.empty:
            for _, row in self.equity_curve.iterrows():
                equity_data.append({
                    'date': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if 'timestamp' in row else None,
                    'value': row.get('equity', 0)
                })
        
        return {
            'metrics': {
                'total_return': self.total_return_pct,
                'win_rate': self.win_rate,
                'max_drawdown': self.max_drawdown_pct,
                'sharpe_ratio': self.sharpe_ratio,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'initial_capital': self.initial_capital,
                'final_capital': self.final_capital,
                'total_return_absolute': self.total_return
            },
            'trades': trades_data,
            'equity_curve': equity_data
        }
    
    def get(self, key: str, default=None):
        """Dictionary-like get method for backward compatibility"""
        result_dict = self.to_dict()
        return result_dict.get(key, default)

class BacktestEngine:
    """
    Backtesting engine for trading strategies
    """
    
    def __init__(self, config=None):
        # Set decimal precision for financial calculations
        getcontext().prec = PRICE_PRECISION + 2  # Extra precision for intermediate calculations
        getcontext().rounding = ROUND_HALF_UP
        
        self.config = config or get_config()
        self.data_fetcher = CoinbaseDataFetcher(
            self.config.get('exchange.base_url', 'https://api.exchange.coinbase.com'),
            self.config.get('database.path', 'data/trading.db')
        )
        self.data_processor = DataProcessor()
        
        # Backtesting parameters with proper decimal handling
        commission_config = self.config.get('backtesting.commission', float(DEFAULT_COMMISSION_RATE))
        self.commission = Decimal(str(commission_config))
        self.slippage = DEFAULT_SLIPPAGE_PCT
        self.max_positions = MAX_POSITION_COUNT
        self.min_trade_size = MIN_TRADE_SIZE_USD
        
        logger.info(f"Backtesting engine initialized with {float(self.commission)*100:.3f}% commission, "
                   f"{float(self.slippage)*100:.3f}% slippage")
    
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
    
    def _calculate_optimal_granularity(self, start_date: str, end_date: str) -> Tuple[int, str]:
        """
        Calculate optimal granularity based on date range to respect Coinbase API limits
        Uses only Coinbase-supported granularities: 60s, 300s, 900s, 3600s, 21600s, 86400s
        
        Returns:
            Tuple of (granularity_seconds, description)
        """
        from datetime import datetime
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end_dt - start_dt).days
        
        # Coinbase allows max 300 data points per request
        # Use safe buffer to prevent API limit errors
        max_points = COINBASE_MAX_CANDLES_PER_REQUEST
        
        if days <= 1:
            # Up to 1 day: 5-minute granularity (288 points max)
            return 300, "5-minute"
        elif days <= 7:
            # Up to 1 week: 1-hour granularity (168 points max)
            return 3600, "1-hour"
        elif days <= 42:
            # Up to 6 weeks: 6-hour granularity (168 points max)
            return 21600, "6-hour"
        else:
            # More than 6 weeks: 1-day granularity (max ~200 points for 200 days)
            return 86400, "1-day"
    
    def _fetch_and_prepare_data(self, pair: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch and prepare historical data for backtesting with proper validation"""
        try:
            # Calculate optimal granularity for the date range
            granularity, granularity_desc = self._calculate_optimal_granularity(start_date, end_date)
            
            logger.info(f"Using {granularity_desc} granularity for {pair} backtesting ({start_date} to {end_date})")
            
            df = self.data_fetcher.get_historical_data(pair, start_date, end_date, granularity=granularity)
            
            if df is None or df.empty:
                logger.warning(f"No data returned for {pair}")
                return None
            
            logger.info(f"Fetched {len(df)} raw data points for {pair} from {start_date} to {end_date} with {granularity_desc} resolution")
            
            # Clean and validate data
            df = self.data_processor.clean_data(df)
            
            if df.empty:
                logger.warning(f"No data remaining after cleaning for {pair}")
                return None
            
            # Validate minimum data requirements for backtesting (adaptive based on granularity)
            granularity_desc = "1-day" if granularity == 86400 else "1-hour" if granularity == 3600 else "5-minute"
            
            # Adaptive minimum based on granularity and date range
            from datetime import datetime
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            days_requested = (end_dt - start_dt).days
            
            if granularity >= 86400:  # Daily or larger
                min_required_points = max(30, days_requested // 3)  # At least 30 points, or 1/3 of days
            elif granularity >= 3600:  # Hourly
                min_required_points = max(50, days_requested * 4)  # More points for hourly data
            else:  # 5-minute or smaller
                min_required_points = 100  # Original requirement for high-frequency data
                
            if len(df) < min_required_points:
                logger.warning(f"Insufficient data for backtesting {pair} with {granularity_desc} granularity. "
                             f"Got {len(df)} points, need at least {min_required_points}. "
                             f"Consider using a longer date range.")
                
                if days_requested < 7:
                    logger.info(f"Recommendation: Try a longer date range (current: {days_requested} days)")
                elif granularity >= 86400 and days_requested < 90:
                    logger.info(f"Recommendation: For daily data, try at least 90 days for better results")
                
                return None
            
            # Check data continuity - look for large gaps based on granularity
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff()
            
            # Define expected gap threshold based on granularity
            if granularity <= 300:  # 5-minute or smaller
                gap_threshold = pd.Timedelta(hours=1)
            elif granularity <= 3600:  # up to 1 hour
                gap_threshold = pd.Timedelta(hours=6)
            elif granularity <= 21600:  # up to 6 hours
                gap_threshold = pd.Timedelta(days=1)
            else:  # daily or larger
                gap_threshold = pd.Timedelta(days=3)
            
            # Find gaps larger than expected
            large_gaps = time_diffs[time_diffs > gap_threshold]
            
            if len(large_gaps) > len(df) * 0.1:  # More than 10% gaps
                logger.warning(f"Data for {pair} has significant gaps: {len(large_gaps)} large gaps found with {granularity_desc} granularity")
            
            logger.info(f"Prepared {len(df)} data points for {pair} backtesting "
                       f"(from {df['timestamp'].min().date()} to {df['timestamp'].max().date()})")
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
        
        logger.info(f"Starting simulation with ${initial_capital:.2f} capital on {len(df)} data points")
        # Larger warmup period for daily data to allow indicators to stabilize
        warmup_period = min(50, len(df) // 3)  # Use 33% of data or 50 points, whichever is smaller
        logger.info(f"Signal generation will start after {warmup_period} warmup periods")
        
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
            if i >= warmup_period:  # Wait for indicators to stabilize
                try:
                    signal = strategy.calculate_signal(df.iloc[:i+1], pair)
                    
                    # Log first signal generation and debug info
                    if i == warmup_period:
                        logger.info(f"Signal generation started at period {i+1}/{len(df)} for {pair}")
                        # Debug: Show available indicators
                        available_indicators = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                        logger.debug(f"Available indicators: {available_indicators}")
                    
                    # Debug: Log signal details every 10 periods or for non-hold signals
                    if (i % 10 == 0 and i > warmup_period) or signal.signal != SignalType.HOLD:
                        logger.debug(f"Period {i+1}: {signal.signal.value} signal - {signal.reason} (confidence: {signal.confidence:.2f})")
                    
                    # Handle BUY signals - enter new long positions with proper validation
                    if (signal.signal in [SignalType.BUY, SignalType.STRONG_BUY] and
                        len(open_positions) < self.max_positions and
                        signal.confidence >= MIN_SIGNAL_CONFIDENCE):
                        
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
                    
                    # Handle SELL signals - close existing long positions only
                    elif (signal.signal in [SignalType.SELL, SignalType.STRONG_SELL] and
                          signal.confidence >= MIN_SIGNAL_CONFIDENCE):
                        
                        # Find and close any open long positions for this pair
                        positions_to_close = [pos for pos in open_positions 
                                           if pos.pair == pair and 
                                              pos.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]]
                        
                        for position in positions_to_close:
                            position.close_position(current_price, current_time, f"Strategy sell signal ({signal.signal.value})")
                            open_positions.remove(position)
                            
                            # Update capital
                            current_capital += position.pnl
                            logger.debug(f"Closed long position: {position.pair} due to {signal.signal.value} signal "
                                       f"P&L: ${position.pnl:.2f} ({position.pnl_pct:.2f}%)")
                
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
        
        # Validation and debugging summary
        logger.info(f"Backtest completed for {strategy.name} on {pair}")
        logger.info(f"Total positions opened: {len(positions)}")
        logger.info(f"Data points processed: {len(df)}")
        logger.info(f"Warmup period: {warmup_period} points")
        logger.info(f"Signal generation periods: {len(df) - warmup_period}")
        
        # Debug: Check if indicators are present in final data
        final_data = df.iloc[-1]
        key_indicators = ['rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50']
        missing_indicators = [ind for ind in key_indicators if ind not in df.columns]
        if missing_indicators:
            logger.warning(f"Missing key indicators: {missing_indicators}")
        else:
            logger.debug(f"Final indicator values: {dict(final_data[key_indicators])}")
        
        # Debug: Signal distribution summary
        signal_counts = {'BUY': 0, 'STRONG_BUY': 0, 'SELL': 0, 'STRONG_SELL': 0, 'HOLD': 0}
        for pos in positions:
            if pos.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                signal_counts['BUY'] += 1
                if pos.signal_type == SignalType.STRONG_BUY:
                    signal_counts['STRONG_BUY'] += 1
        logger.debug(f"Signal distribution: {signal_counts}")
        
        # Calculate final metrics
        final_capital = current_capital
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # Calculate trade statistics
        closed_positions = [pos for pos in positions if not pos.is_open()]
        total_trades = len(closed_positions)
        winning_trades = len([pos for pos in closed_positions if pos.pnl > 0])
        losing_trades = len([pos for pos in closed_positions if pos.pnl < 0])
        win_rate = float(winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        avg_win = np.mean([pos.pnl for pos in closed_positions if pos.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([pos.pnl for pos in closed_positions if pos.pnl < 0]) if losing_trades > 0 else 0
        
        # Calculate enhanced risk-adjusted performance metrics
        risk_metrics = self._calculate_risk_adjusted_metrics(equity_df, initial_capital, total_return)
        sharpe_ratio = risk_metrics['sharpe_ratio']
        
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
        """
        Calculate position size with comprehensive risk validation and proper decimal handling
        """
        if available_capital <= 0:
            logger.warning("No available capital for new positions")
            return 0
        
        # Convert to Decimal for precise financial calculations
        available_capital_decimal = Decimal(str(available_capital))
        signal_price_decimal = Decimal(str(signal.price))
        
        sizing_method = position_sizing.get('method', 'dynamic')
        max_position_size = min(position_sizing.get('max_position_size', MAX_POSITION_SIZE), MAX_POSITION_SIZE)
        min_position_size = max(position_sizing.get('min_position_size', MIN_POSITION_SIZE), Decimal('0.01'))
        
        if sizing_method == 'fixed':
            allocation_pct = Decimal(str(max_position_size))
        else:  # dynamic sizing based on confidence with enterprise constants
            if signal.confidence >= HIGH_CONFIDENCE_THRESHOLD:
                allocation_pct = Decimal(str(max_position_size))
            elif signal.confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
                allocation_pct = Decimal(str((min_position_size + max_position_size) / 2))
            else:
                allocation_pct = Decimal(str(min_position_size))
        
        # Calculate position value with risk controls
        position_value = available_capital_decimal * allocation_pct
        
        # Apply maximum single position risk limit
        max_risk_value = available_capital_decimal * MAX_SINGLE_POSITION_RISK
        if position_value > max_risk_value:
            position_value = max_risk_value
            logger.info(f"Position size capped by risk limit: ${float(max_risk_value):.2f}")
        
        # Check minimum trade size requirement
        if position_value < self.min_trade_size:
            logger.debug(f"Position value ${float(position_value):.2f} below minimum ${float(self.min_trade_size):.2f}")
            return 0
        
        # Calculate position size in terms of quantity
        position_size = position_value / signal_price_decimal
        
        return float(position_size)
    
    def _create_position(self, signal: TradingSignal, current_price: float, 
                        current_time: datetime, position_size: float,
                        strategy: Strategy) -> Position:
        """Create a new trading position"""
        
        # Calculate stop loss and take profit levels
        if hasattr(strategy, 'get_stop_loss_level'):
            stop_loss = strategy.get_stop_loss_level(current_price, signal.signal)
        else:
            stop_loss_pct = float(DEFAULT_STOP_LOSS_PCT)
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
        """Calculate current value of an open long position (spot trading only)"""
        # For spot trading, we only have long positions (BUY signals)
        # Position value = unrealized P&L = (current_price - entry_price) * size
        if position.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return (current_price - position.entry_price) * position.size
        else:
            # This shouldn't happen in spot trading, but handle gracefully
            logger.warning(f"Unexpected position type {position.signal_type} in spot trading")
            return 0.0
    
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
    
    def _calculate_risk_adjusted_metrics(self, equity_df: pd.DataFrame, 
                                       initial_capital: float, total_return: float) -> Dict[str, float]:
        """
        Calculate comprehensive risk-adjusted performance metrics for enterprise trading
        
        Returns:
            Dictionary containing Sharpe, Sortino, Calmar ratios and other risk metrics
        """
        from ..constants import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
        
        if len(equity_df) < 2:
            return {
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'max_consecutive_losses': 0,
                'profit_factor': 0,
                'value_at_risk_5pct': 0
            }
        
        # Calculate returns
        equity_values = equity_df['equity'].values
        returns = np.diff(equity_values) / equity_values[:-1]
        
        # Basic statistics
        mean_return = np.mean(returns) * TRADING_DAYS_PER_YEAR  # Annualized
        return_std = np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR)  # Annualized
        
        # Sharpe Ratio
        risk_free = float(RISK_FREE_RATE)
        sharpe_ratio = (mean_return - risk_free) / return_std if return_std > 0 else 0
        
        # Sortino Ratio (uses downside deviation instead of total volatility)
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) * np.sqrt(TRADING_DAYS_PER_YEAR) if len(negative_returns) > 1 else return_std
        sortino_ratio = (mean_return - risk_free) / downside_std if downside_std > 0 else 0
        
        # Maximum Drawdown (already calculated, but let's be precise)
        peak = np.maximum.accumulate(equity_values)
        drawdown = (peak - equity_values) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Calmar Ratio (annual return / max drawdown)
        annual_return = (equity_values[-1] / equity_values[0]) ** (TRADING_DAYS_PER_YEAR / len(equity_values)) - 1
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # Maximum Consecutive Losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        # Profit Factor (gross profit / gross loss)
        gross_profit = np.sum(returns[returns > 0]) * initial_capital
        gross_loss = abs(np.sum(returns[returns < 0])) * initial_capital
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Value at Risk (5th percentile of daily returns)
        value_at_risk_5pct = np.percentile(returns, 5) * initial_capital if len(returns) > 0 else 0
        
        return {
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'max_consecutive_losses': int(max_consecutive_losses),
            'profit_factor': float(profit_factor),
            'value_at_risk_5pct': float(value_at_risk_5pct)
        }
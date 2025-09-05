# Crypto Trading MVP Template

## Project Overview
Build a desktop crypto trading application with backtesting capabilities, multiple trading strategies, and real-time monitoring for Coinbase.

## Project Structure
```
crypto-trading-mvp/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration and constants
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py      # Coinbase API data fetching
│   │   └── processor.py    # Data cleaning and preparation
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py         # Base strategy class
│   │   ├── swing.py        # 5% swing trading strategy
│   │   ├── rsi.py          # RSI strategy
│   │   ├── macd.py         # MACD strategy
│   │   ├── bollinger.py    # Bollinger Bands strategy
│   │   ├── support_resistance.py  # Support/Resistance strategy
│   │   └── combined.py     # Multi-strategy combiner
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── engine.py       # Backtesting engine
│   │   └── metrics.py      # Performance metrics calculator
│   ├── trading/
│   │   ├── __init__.py
│   │   ├── portfolio.py    # Portfolio management
│   │   ├── risk_manager.py # Risk management and position sizing
│   │   └── signal_generator.py  # Signal generation and ranking
│   ├── notifications/
│   │   ├── __init__.py
│   │   └── email_alerts.py # Email notification system
│   └── dashboard/
│       ├── __init__.py
│       └── app.py          # Dash web dashboard
├── tests/
│   └── test_strategies.py
├── requirements.txt
├── config.yaml
├── run_backtest.py
├── run_live.py
└── README.md
```

## Requirements.txt
```txt
# Data and API
requests==2.31.0
pandas==2.1.3
numpy==1.24.3

# Technical Analysis
ta==0.11.0
scipy==1.11.4

# Visualization and Dashboard
plotly==5.18.0
dash==2.14.1
dash-bootstrap-components==1.5.0

# Utilities
pyyaml==6.0.1
python-dotenv==1.0.0
schedule==1.2.0

# Testing
pytest==7.4.3
pytest-mock==3.12.0

# Data storage
sqlalchemy==2.0.23
sqlite3  # Built-in
```

## Core Implementation Templates

### 1. Configuration (config.yaml)
```yaml
# Coinbase Configuration
exchange:
  name: "coinbase"
  base_url: "https://api.exchange.coinbase.com"
  
# Trading Pairs - Focus on non-BTC majors
trading_pairs:
  - "SOL-USD"
  - "ETH-USD"
  - "AVAX-USD"
  - "MATIC-USD"
  - "LINK-USD"
  - "UNI-USD"
  - "ATOM-USD"
  - "DOT-USD"

# Trading Configuration  
trading:
  initial_capital: 1000
  max_positions: 3
  position_sizing:
    method: "dynamic"  # dynamic or fixed
    max_position_size: 1.0  # 100% for best signals
    min_position_size: 0.33  # 33% for weaker signals
  update_interval: 300  # 5 minutes in seconds

# Risk Management
risk:
  stop_loss: 0.05  # 5%
  use_trailing_stop: false
  max_daily_loss: 0.10  # 10% daily loss limit

# Backtesting
backtesting:
  start_date: "2024-03-01"  # 6 months ago
  end_date: "2024-09-01"
  initial_capital: 100
  commission: 0.005  # 0.5% per trade

# Notifications
notifications:
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    from_email: "your_email@gmail.com"
    to_email: "your_email@gmail.com"
    # Password should be in .env file

# Strategy Weights (for combined strategy)
strategy_weights:
  swing: 1.0
  rsi: 0.8
  macd: 0.7
  bollinger: 0.6
  support_resistance: 0.9
```

### 2. Base Strategy Class (src/strategies/base.py)
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd
from datetime import datetime

class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    timestamp: datetime
    pair: str
    signal: SignalType
    price: float
    confidence: float  # 0-1 score
    strategy_name: str
    reason: str
    indicators: dict  # Store relevant indicator values
    
    def position_size_recommendation(self) -> float:
        """Recommend position size based on confidence"""
        if self.confidence >= 0.8:
            return 1.0  # 100% of available capital
        elif self.confidence >= 0.6:
            return 0.66  # 66%
        else:
            return 0.33  # 33%

class Strategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def calculate_signal(self, df: pd.DataFrame, pair: str) -> TradingSignal:
        """
        Calculate trading signal based on data
        
        Args:
            df: DataFrame with OHLCV data and indicators
            pair: Trading pair symbol
            
        Returns:
            TradingSignal object
        """
        pass
    
    def add_required_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add any indicators required by this strategy
        Override in subclasses to add specific indicators
        """
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that required data is present"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in df.columns for col in required_columns)
```

### 3. Backtesting Engine Template (src/backtesting/engine.py)
```python
"""
Backtesting Engine Implementation TODO:

1. BacktestEngine class should:
   - Load historical data for specified period
   - Run strategies on historical data
   - Track portfolio value over time
   - Execute trades based on signals
   - Apply commission and slippage
   - Respect position limits and risk management rules
   - Generate detailed trade log

2. Key methods needed:
   - run_backtest(strategy, start_date, end_date, initial_capital)
   - execute_trade(signal, portfolio)
   - calculate_position_size(signal, portfolio, risk_params)
   - apply_stop_loss(position, current_price)
   - get_performance_metrics()

3. Track for each trade:
   - Entry/exit times and prices
   - Position size
   - P&L
   - Strategy used
   - Signal confidence

4. Support backtesting modes:
   - Single strategy
   - Multiple strategies (ensemble)
   - Strategy comparison
   - Walk-forward analysis
"""

class BacktestEngine:
    def __init__(self, config):
        # TODO: Initialize with configuration
        pass
    
    def run_backtest(self, strategy, pair, start_date, end_date):
        """
        Main backtesting loop
        Should iterate through historical data day by day
        Generate signals and execute trades
        Track portfolio performance
        """
        # TODO: Implement backtesting logic
        pass
```

### 4. Performance Metrics (src/backtesting/metrics.py)
```python
"""
Performance Metrics Calculator TODO:

Calculate and return these metrics:
1. Total Return (%)
2. Annualized Return (%)
3. Sharpe Ratio
4. Sortino Ratio
5. Maximum Drawdown (%)
6. Win Rate (%)
7. Average Win/Loss
8. Profit Factor
9. Number of Trades
10. Average Trade Duration
11. Best Trade
12. Worst Trade
13. Consecutive Wins/Losses
14. Risk/Reward Ratio
15. Calmar Ratio

Output format:
- Summary statistics dictionary
- Detailed trade log DataFrame
- Equity curve data for plotting
"""

class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(trades_df, equity_curve):
        # TODO: Implement all metrics calculations
        pass
```

### 5. Dashboard Template (src/dashboard/app.py)
```python
"""
Dashboard Implementation TODO:

Create a Dash application with these components:

1. Header Section:
   - Current Portfolio Value
   - Today's P&L
   - Active Positions Count

2. Main Charts:
   - Portfolio value over time (line chart)
   - Individual trade markers on chart
   - Volume bars

3. Positions Table:
   - Current positions with live P&L
   - Entry price, current price, change %
   - Stop loss levels
   - Position size

4. Signals Table:
   - Latest signals from all strategies
   - Confidence scores
   - Recommended actions

5. Performance Metrics:
   - Win rate gauge
   - Sharpe ratio
   - Maximum drawdown
   - Recent trades list

6. Strategy Performance:
   - Comparison chart of different strategies
   - Individual strategy metrics

Update interval: 60 seconds for live data

Use plotly.graph_objects for interactive charts
Use dash_bootstrap_components for styling
"""

import dash
from dash import dcc, html, dash_table
import plotly.graph_objects as go

def create_dashboard():
    # TODO: Implement dashboard
    pass
```

### 6. Email Alerts (src/notifications/email_alerts.py)
```python
"""
Email Alert System TODO:

Send email notifications for:
1. New BUY signals (with confidence > 0.7)
2. New SELL signals for existing positions
3. Stop loss triggered
4. Daily summary (end of day)
5. Significant drawdown warning (>5%)

Email should include:
- Signal details (pair, price, confidence)
- Current portfolio status
- Recommended action
- Link to dashboard (if applicable)

Use HTML email format for better formatting
Include error handling for failed sends
Queue system for rate limiting
"""

class EmailNotifier:
    def send_trade_signal(self, signal):
        # TODO: Implement
        pass
    
    def send_daily_summary(self, portfolio, trades):
        # TODO: Implement  
        pass
```

### 7. Main Execution Scripts

#### run_backtest.py
```python
"""
Backtesting Script TODO:

1. Load configuration
2. Fetch historical data for all pairs
3. Run backtests for:
   - Each individual strategy
   - Combined strategy
   - Different time periods
4. Compare strategy performance
5. Generate report with:
   - Performance metrics table
   - Equity curves chart
   - Trade distribution analysis
   - Best/worst performing pairs
   - Optimal strategy weights
6. Save results to CSV/JSON
7. Create visualizations

Command line arguments:
--strategy: specific strategy or "all"
--start-date: backtest start date
--end-date: backtest end date  
--pairs: specific pairs or "all"
--export: export format (csv, json, html)
"""

def main():
    # TODO: Implement backtesting workflow
    pass

if __name__ == "__main__":
    main()
```

#### run_live.py
```python
"""
Live Trading Script TODO:

1. Initialize all components
2. Start dashboard in separate thread
3. Main loop:
   - Fetch latest data every 5 minutes
   - Calculate indicators
   - Generate signals from all strategies
   - Combine signals with weights
   - Check risk management rules
   - Send email alerts for signals
   - Update dashboard
   - Log all actions
4. Graceful shutdown handling

Command line arguments:
--paper: paper trading mode (default)
--dashboard: enable/disable dashboard
--strategies: comma-separated list of strategies to use

Safety features:
- Confirm before starting
- Daily loss limit check
- Position limit enforcement
- Error recovery
- State persistence
"""

def main():
    # TODO: Implement live trading workflow
    pass

if __name__ == "__main__":
    main()
```

## Implementation Priority Order

1. **Phase 1 - Data Foundation**
   - Implement Coinbase data fetcher
   - Add technical indicators calculation
   - Create data storage/caching system

2. **Phase 2 - Strategy Implementation**
   - Implement basic 5% swing strategy
   - Add RSI strategy
   - Add MACD strategy
   - Implement strategy combination logic

3. **Phase 3 - Backtesting**
   - Build backtesting engine
   - Implement performance metrics
   - Create comparison tools

4. **Phase 4 - Live System**
   - Implement portfolio manager
   - Add risk management
   - Create signal generator

5. **Phase 5 - Monitoring**
   - Build dashboard
   - Add email notifications
   - Implement logging system

## Testing Checklist

- [ ] Data fetching works for all pairs
- [ ] Indicators calculate correctly
- [ ] Each strategy generates valid signals
- [ ] Backtesting produces accurate results
- [ ] Risk management limits are enforced
- [ ] Dashboard updates in real-time
- [ ] Email notifications send properly
- [ ] System handles API errors gracefully
- [ ] Portfolio calculations are accurate
- [ ] Stop losses trigger correctly

## Additional Notes for Implementation

1. **API Rate Limiting**: Implement exponential backoff for Coinbase API
2. **Data Persistence**: Use SQLite for storing historical data and trades
3. **Error Handling**: Wrap all API calls in try-except blocks
4. **Logging**: Use rotating file handler for logs
5. **Configuration**: Support both YAML and environment variables
6. **Testing**: Write unit tests for each strategy
7. **Documentation**: Add docstrings to all functions
8. **Performance**: Cache indicator calculations where possible
9. **Security**: Never commit API keys or passwords
10. **Modularity**: Keep strategies independent and pluggable

## Example Usage

```bash
# Run backtest on all strategies
python run_backtest.py --strategy all --start-date 2024-03-01 --end-date 2024-09-01

# Run specific strategy backtest
python run_backtest.py --strategy rsi --pairs SOL-USD,ETH-USD

# Start live monitoring (paper trading)
python run_live.py --paper --dashboard --strategies swing,rsi,macd

# Run tests
pytest tests/
```

## Next Steps
1. Copy this template to your local machine
2. Set up Python virtual environment
3. Install requirements
4. Create .env file with email credentials
5. Use Claude Code to implement each module
6. Test with historical data first
7. Run paper trading before real trading

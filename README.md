# Crypto Trading MVP

A comprehensive cryptocurrency trading application with backtesting capabilities, multiple trading strategies, and real-time monitoring for Coinbase.

## 🚀 Features

- **Multiple Trading Strategies**: 5% Swing Trading, RSI-based, and MACD strategies
- **Advanced Backtesting**: Historical performance analysis with realistic trading simulation  
- **Technical Indicators**: 20+ indicators including RSI, MACD, Bollinger Bands, support/resistance
- **Risk Management**: Stop losses, position sizing, and portfolio limits
- **Data Caching**: SQLite caching for efficient historical data retrieval
- **Comprehensive Configuration**: YAML-based config with environment variable support

## 📊 Supported Trading Pairs

- SOL-USD, ETH-USD, AVAX-USD, MATIC-USD
- LINK-USD, UNI-USD, ATOM-USD, DOT-USD

## 🏗️ Architecture

```
src/
├── config.py              # Configuration management
├── data/
│   ├── fetcher.py         # Coinbase API client with rate limiting
│   └── processor.py       # Technical indicator calculation
├── strategies/
│   ├── base.py           # Base strategy framework
│   ├── swing.py          # 5% swing trading strategy
│   ├── rsi.py            # RSI-based strategy
│   └── macd.py           # MACD strategy
├── backtesting/
│   ├── engine.py         # Backtesting simulation engine
│   └── metrics.py        # Performance metrics (planned)
└── dashboard/            # Web dashboard (planned)
```

## ⚡ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/crypto-trading-mvp.git
cd crypto-trading-mvp

# Create virtual environment
python -m venv trading_env
source trading_env/bin/activate  # On Windows: trading_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your credentials (for live trading)
# Edit config.yaml to adjust trading parameters
```

### 3. Run Backtests

```bash
# Backtest individual strategy (when script is ready)
python run_backtest.py --strategy swing --pairs SOL-USD,ETH-USD --start-date 2024-03-01

# Compare all strategies
python run_backtest.py --strategy all --pairs all
```

## 📈 Strategy Overview

### 5% Swing Trading Strategy
- Buys on 5%+ drops from recent highs with RSI confirmation
- Sells on 5%+ rises from recent lows with volume confirmation
- Uses dynamic position sizing based on signal confidence

### RSI Strategy  
- Oversold/overbought signals with momentum confirmation
- Bullish/bearish divergence detection
- Stochastic confirmation for additional validation

### MACD Strategy
- MACD line and signal line crossovers
- Zero-line crosses and histogram analysis
- Momentum and divergence confirmation

## 🛠️ Current Implementation Status

✅ **Phase 1 - Data Foundation**
- Coinbase API integration with rate limiting
- Technical indicator calculation (20+ indicators)
- Data caching and validation

✅ **Phase 2 - Strategy Framework**  
- Base strategy class with signal generation
- Three complete trading strategies
- Confidence-based position sizing

✅ **Phase 3 - Backtesting Engine**
- Complete trading simulation
- Position management with stop losses
- Performance tracking and equity curves

🔄 **Phase 4 - Live Trading** (In Progress)
- Portfolio management system
- Risk management implementation  
- Live trading execution

📋 **Phase 5 - Monitoring** (Planned)
- Web dashboard with real-time charts
- Email notification system
- Performance monitoring

## 🔧 Configuration

Key configuration options in `config.yaml`:

```yaml
# Trading Configuration
trading:
  initial_capital: 1000
  max_positions: 3
  position_sizing:
    method: "dynamic"
    max_position_size: 1.0  # 100% for strong signals
    min_position_size: 0.33 # 33% for weak signals

# Risk Management
risk:
  stop_loss: 0.05  # 5%
  max_daily_loss: 0.10  # 10%

# Strategy Weights
strategy_weights:
  swing: 1.0
  rsi: 0.8
  macd: 0.7
```

## 🔒 Security & Safety

- **Paper Trading Mode**: Test strategies without real money
- **Environment Variables**: Sensitive credentials stored in `.env`
- **Rate Limiting**: Respect API limits with exponential backoff
- **Risk Limits**: Built-in stop losses and position limits
- **No Credential Harvesting**: Defensive security implementation only

## 📊 Performance Metrics

The backtesting engine calculates comprehensive metrics:
- Total Return & Annualized Return
- Sharpe Ratio & Sortino Ratio  
- Maximum Drawdown
- Win Rate & Average Win/Loss
- Trade Distribution Analysis

## 🤝 Contributing

This is an educational/research project for algorithmic trading. Contributions welcome for:
- Additional trading strategies
- Performance optimizations
- Bug fixes and testing
- Documentation improvements

## ⚠️ Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. The authors are not responsible for any financial losses. Always test strategies thoroughly with paper trading before using real funds.

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Related Resources

- [Coinbase Pro API Documentation](https://docs.pro.coinbase.com/)
- [Technical Analysis Library](https://github.com/bukosabino/ta)  
- [Algorithmic Trading Best Practices](https://www.quantstart.com/)
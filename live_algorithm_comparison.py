#!/usr/bin/env python3
"""
Live Algorithm Performance Comparison
Compares all identified best algorithms on live ETH-USD and SOL-USD data
to determine definitive win rates and profitability rankings.
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

from src.backtesting.engine import BacktestEngine, BacktestResult
from src.strategies.swing import SwingTradingStrategy
from src.strategies.rsi import RSIStrategy
from src.strategies.macd import MACDStrategy
from src.strategies.pure_percent import Pure5PercentStrategy, DynamicPercentStrategy
from src.config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlgorithmComparator:
    """Comprehensive algorithm comparison on live market data"""
    
    def __init__(self):
        self.config = get_config()
        self.engine = BacktestEngine()
        self.results = {}
        
    def define_test_strategies(self) -> List[Tuple[str, object]]:
        """Define all strategies to test based on previous analysis"""
        
        strategies = [
            # Pure Percentage Strategies (Winner from previous analysis) - Focus on most promising
            ('Pure 2%', Pure5PercentStrategy(drop_threshold=0.02, rise_threshold=0.02)),
            ('Pure 3%', Pure5PercentStrategy(drop_threshold=0.03, rise_threshold=0.03)),
            ('Pure 5%', Pure5PercentStrategy(drop_threshold=0.05, rise_threshold=0.05)),
            
            # Dynamic Percentage Strategy - Best variant
            ('Dynamic 3%', DynamicPercentStrategy(drop_threshold=0.03, rise_threshold=0.03)),
            
            # Technical Indicator Based Strategies - Core performers
            ('MACD Standard', MACDStrategy()),
            ('RSI Standard', RSIStrategy(oversold_threshold=30, overbought_threshold=70)),
            
            # Swing Trading - Standard variant
            ('Swing 2.5%', SwingTradingStrategy()),
        ]
        
        return strategies
    
    def run_comprehensive_comparison(self, pairs: List[str], start_date: str, end_date: str, 
                                   initial_capital: float = 1000) -> Dict:
        """Run comprehensive comparison across all algorithms"""
        
        logger.info(f"🚀 Starting comprehensive algorithm comparison")
        logger.info(f"📅 Period: {start_date} to {end_date}")
        logger.info(f"🪙 Trading Pairs: {', '.join(pairs)}")
        logger.info(f"💰 Initial Capital: ${initial_capital}")
        
        strategies = self.define_test_strategies()
        all_results = {}
        
        total_tests = len(strategies) * len(pairs)
        current_test = 0
        
        for pair in pairs:
            logger.info(f"\n📈 Testing {pair}...")
            pair_results = {}
            
            for strategy_name, strategy in strategies:
                current_test += 1
                progress = (current_test / total_tests) * 100
                
                logger.info(f"[{current_test}/{total_tests}] ({progress:.1f}%) Testing {strategy_name} on {pair}")
                
                try:
                    result = self.engine.run_backtest(
                        strategy=strategy,
                        pair=pair,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=initial_capital
                    )
                    
                    if result:
                        pair_results[strategy_name] = result
                        logger.info(f"   ✅ {strategy_name}: {result.total_return_pct:.2f}% return, {result.win_rate:.1f}% win rate")
                    else:
                        logger.warning(f"   ⚠️ {strategy_name}: No result (insufficient data)")
                        
                except Exception as e:
                    logger.error(f"   ❌ {strategy_name}: Error - {str(e)}")
                    continue
            
            all_results[pair] = pair_results
        
        self.results = all_results
        return all_results
    
    def calculate_performance_metrics(self, results: Dict) -> pd.DataFrame:
        """Calculate comprehensive performance metrics for all strategies"""
        
        metrics_data = []
        
        for pair, pair_results in results.items():
            for strategy_name, result in pair_results.items():
                if isinstance(result, BacktestResult):
                    
                    # Calculate additional metrics
                    profit_factor = (result.avg_win * result.winning_trades) / (abs(result.avg_loss) * result.losing_trades) if result.losing_trades > 0 else float('inf')
                    
                    # Risk-adjusted return
                    risk_adjusted_return = result.total_return_pct / max(result.max_drawdown_pct, 1) if result.max_drawdown_pct > 0 else result.total_return_pct
                    
                    metrics = {
                        'Pair': pair,
                        'Strategy': strategy_name,
                        'Total_Return_%': result.total_return_pct,
                        'Win_Rate_%': result.win_rate,
                        'Total_Trades': result.total_trades,
                        'Winning_Trades': result.winning_trades,
                        'Losing_Trades': result.losing_trades,
                        'Avg_Win_%': result.avg_win,
                        'Avg_Loss_%': result.avg_loss,
                        'Max_Drawdown_%': result.max_drawdown_pct,
                        'Sharpe_Ratio': result.sharpe_ratio,
                        'Profit_Factor': profit_factor,
                        'Risk_Adjusted_Return': risk_adjusted_return,
                        'Final_Capital': result.final_capital
                    }
                    
                    metrics_data.append(metrics)
        
        return pd.DataFrame(metrics_data)
    
    def rank_strategies(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Rank strategies using comprehensive scoring system"""
        
        if metrics_df.empty:
            logger.warning("No metrics data to rank")
            return pd.DataFrame()
        
        # Create separate DataFrames for each pair
        pair_rankings = []
        
        for pair in metrics_df['Pair'].unique():
            pair_df = metrics_df[metrics_df['Pair'] == pair].copy()
            
            if len(pair_df) == 0:
                continue
            
            # Normalize metrics for scoring (0-1 scale)
            pair_df['Win_Rate_Score'] = pair_df['Win_Rate_%'] / 100  # Already 0-100
            pair_df['Return_Score'] = (pair_df['Total_Return_%'] - pair_df['Total_Return_%'].min()) / (pair_df['Total_Return_%'].max() - pair_df['Total_Return_%'].min()) if pair_df['Total_Return_%'].max() != pair_df['Total_Return_%'].min() else 0.5
            pair_df['Sharpe_Score'] = (pair_df['Sharpe_Ratio'] - pair_df['Sharpe_Ratio'].min()) / (pair_df['Sharpe_Ratio'].max() - pair_df['Sharpe_Ratio'].min()) if pair_df['Sharpe_Ratio'].max() != pair_df['Sharpe_Ratio'].min() else 0.5
            
            # Drawdown score (lower is better)
            pair_df['Drawdown_Score'] = 1 - ((pair_df['Max_Drawdown_%'] - pair_df['Max_Drawdown_%'].min()) / (pair_df['Max_Drawdown_%'].max() - pair_df['Max_Drawdown_%'].min())) if pair_df['Max_Drawdown_%'].max() != pair_df['Max_Drawdown_%'].min() else 0.5
            
            # Composite score: Win Rate (40%) + Return (30%) + Sharpe (20%) + Drawdown (10%)
            pair_df['Composite_Score'] = (
                pair_df['Win_Rate_Score'] * 0.40 +
                pair_df['Return_Score'] * 0.30 +
                pair_df['Sharpe_Score'] * 0.20 +
                pair_df['Drawdown_Score'] * 0.10
            )
            
            # Rank within pair
            pair_df['Rank'] = pair_df['Composite_Score'].rank(ascending=False, method='dense')
            
            pair_rankings.append(pair_df)
        
        return pd.concat(pair_rankings, ignore_index=True) if pair_rankings else pd.DataFrame()
    
    def generate_summary_report(self, rankings_df: pd.DataFrame) -> str:
        """Generate comprehensive summary report"""
        
        if rankings_df.empty:
            return "No data available for report generation."
        
        report = []
        report.append("🏆 LIVE ALGORITHM PERFORMANCE COMPARISON REPORT")
        report.append("=" * 80)
        
        # Overall best performers across both pairs
        overall_ranking = rankings_df.groupby('Strategy').agg({
            'Composite_Score': 'mean',
            'Total_Return_%': 'mean',
            'Win_Rate_%': 'mean',
            'Max_Drawdown_%': 'mean',
            'Sharpe_Ratio': 'mean',
            'Total_Trades': 'sum'
        }).sort_values('Composite_Score', ascending=False)
        
        report.append("\n🎯 TOP PERFORMING ALGORITHMS (Combined ETH-USD & SOL-USD)")
        report.append("-" * 60)
        
        for i, (strategy, row) in enumerate(overall_ranking.head(5).iterrows(), 1):
            report.append(f"{i:2d}. {strategy:<20} Score: {row['Composite_Score']:.3f}")
            report.append(f"    Return: {row['Total_Return_%']:+6.2f}% | Win Rate: {row['Win_Rate_%']:5.1f}% | Drawdown: {row['Max_Drawdown_%']:5.1f}% | Trades: {row['Total_Trades']:3.0f}")
        
        # Individual pair performance
        for pair in ['ETH-USD', 'SOL-USD']:
            pair_data = rankings_df[rankings_df['Pair'] == pair].sort_values('Rank')
            if len(pair_data) > 0:
                report.append(f"\n📈 {pair} PERFORMANCE RANKING")
                report.append("-" * 40)
                
                for _, row in pair_data.head(5).iterrows():
                    rank = int(row['Rank'])
                    strategy = row['Strategy']
                    return_pct = row['Total_Return_%']
                    win_rate = row['Win_Rate_%']
                    drawdown = row['Max_Drawdown_%']
                    score = row['Composite_Score']
                    
                    report.append(f"{rank:2d}. {strategy:<20} {return_pct:+6.2f}% ({win_rate:4.1f}% WR) DD:{drawdown:5.1f}% Score:{score:.3f}")
        
        # Strategy type analysis
        strategy_types = {}
        for strategy in rankings_df['Strategy'].unique():
            if 'Pure' in strategy:
                strategy_type = 'Pure Percentage'
            elif 'Dynamic' in strategy or 'Adaptive' in strategy:
                strategy_type = 'Dynamic Percentage'
            elif 'MACD' in strategy:
                strategy_type = 'MACD'
            elif 'RSI' in strategy:
                strategy_type = 'RSI'
            elif 'Swing' in strategy:
                strategy_type = 'Swing Trading'
            else:
                strategy_type = 'Other'
            
            if strategy_type not in strategy_types:
                strategy_types[strategy_type] = []
            strategy_types[strategy_type].append(strategy)
        
        report.append(f"\n📊 STRATEGY TYPE PERFORMANCE")
        report.append("-" * 30)
        
        for strategy_type, strategies in strategy_types.items():
            type_data = rankings_df[rankings_df['Strategy'].isin(strategies)]
            avg_score = type_data['Composite_Score'].mean()
            avg_return = type_data['Total_Return_%'].mean()
            avg_win_rate = type_data['Win_Rate_%'].mean()
            
            report.append(f"{strategy_type:<20} Avg Score: {avg_score:.3f} | Return: {avg_return:+6.2f}% | WR: {avg_win_rate:5.1f}%")
        
        return "\n".join(report)
    
    def save_results(self, rankings_df: pd.DataFrame, filename: str = None):
        """Save results to CSV and JSON files"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"algorithm_comparison_{timestamp}"
        
        # Save detailed CSV
        csv_filename = f"{filename}.csv"
        rankings_df.to_csv(csv_filename, index=False)
        logger.info(f"📊 Detailed results saved to {csv_filename}")
        
        # Save summary JSON
        if not rankings_df.empty:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_strategies_tested': len(rankings_df['Strategy'].unique()),
                'total_pairs_tested': len(rankings_df['Pair'].unique()),
                'best_overall_strategy': rankings_df.groupby('Strategy')['Composite_Score'].mean().idxmax(),
                'best_eth_strategy': rankings_df[rankings_df['Pair'] == 'ETH-USD'].loc[rankings_df['Rank'] == 1, 'Strategy'].iloc[0] if len(rankings_df[rankings_df['Pair'] == 'ETH-USD']) > 0 else None,
                'best_sol_strategy': rankings_df[rankings_df['Pair'] == 'SOL-USD'].loc[rankings_df['Rank'] == 1, 'Strategy'].iloc[0] if len(rankings_df[rankings_df['Pair'] == 'SOL-USD']) > 0 else None,
            }
            
            json_filename = f"{filename}_summary.json"
            with open(json_filename, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"📄 Summary saved to {json_filename}")

def main():
    """Main execution function"""
    
    # Initialize comparator
    comparator = AlgorithmComparator()
    
    # Test parameters - Use 2024 data for better volatility and trading signals
    pairs = ['ETH-USD', 'SOL-USD']
    end_date = '2024-09-01'
    start_date = '2024-01-01'  # Full year 2024 for better market volatility
    initial_capital = 1000
    
    print("🚀 STARTING COMPREHENSIVE LIVE ALGORITHM COMPARISON")
    print("=" * 80)
    print(f"📅 Testing Period: {start_date} to {end_date}")
    print(f"🪙 Trading Pairs: {', '.join(pairs)}")
    print(f"💰 Initial Capital: ${initial_capital}")
    print(f"🧪 Strategies to Test: {len(comparator.define_test_strategies())}")
    print(f"⏱️  Expected Runtime: 10-20 minutes")
    print("=" * 80)
    
    try:
        # Run comprehensive comparison
        results = comparator.run_comprehensive_comparison(
            pairs=pairs,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        if not results:
            print("❌ No results obtained. Check data availability and strategy implementations.")
            return
        
        # Calculate performance metrics
        metrics_df = comparator.calculate_performance_metrics(results)
        
        if metrics_df.empty:
            print("❌ No performance metrics calculated. Check backtest results.")
            return
        
        # Rank strategies
        rankings_df = comparator.rank_strategies(metrics_df)
        
        # Generate and display report
        report = comparator.generate_summary_report(rankings_df)
        print(report)
        
        # Save results
        comparator.save_results(rankings_df)
        
        print("\n✅ ANALYSIS COMPLETE!")
        print("📊 Check the generated CSV and JSON files for detailed results.")
        
    except Exception as e:
        logger.error(f"❌ Error during comparison: {str(e)}")
        raise

if __name__ == "__main__":
    main()
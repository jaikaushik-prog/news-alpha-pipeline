#!/usr/bin/env python
"""
03 - Event Study Analysis Script

Performs the core event study analysis:
1. Load processed speech data
2. Build sector portfolios
3. Detect first sector mentions
4. Calculate Cumulative Abnormal Returns (CAR)
5. Run statistical tests
6. Visualize results

Usage:
    python scripts/03_event_study.py [--year FISCAL_YEAR] [--save-plots]
    
Examples:
    python scripts/03_event_study.py                    # Run all years
    python scripts/03_event_study.py --year 2024-25     # Single year
    python scripts/03_event_study.py --save-plots       # Save visualizations
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import yaml

# Import VIX signal module
try:
    from src.market.option_iv import get_pre_budget_vix_signal, interpret_vix_for_strategy
    VIX_AVAILABLE = True
except ImportError:
    VIX_AVAILABLE = False


def load_processed_speeches(speech_dir, target_year=None):
    """Load processed speech sentences."""
    sentences_by_year = {}
    
    if speech_dir.exists():
        speech_files = list(speech_dir.glob('*.parquet'))
        print(f"Found {len(speech_files)} processed speech files")
        
        for f in speech_files:
            if 'demo' in f.stem:
                continue
            fy = f.stem.replace('_sentences', '').replace('_', '-')
            
            if target_year and fy != target_year:
                continue
                
            sentences_by_year[fy] = pd.read_parquet(f)
            print(f"  {fy}: {len(sentences_by_year[fy])} sentences")
    else:
        print("No processed speech data found. Run 02_nlp_processing.py first.")
    
    return sentences_by_year


def load_sector_returns(project_root, budget_date, sectors_config):
    """Load stock data and build sector portfolio returns and volume."""
    from src.ingestion import load_single_stock
    from src.market import clean_stock_data
    
    sample_sectors = list(sectors_config['sectors'].keys())  # ALL sectors
    sector_data = {}
    sector_volumes = {}
    
    for sector_key in sample_sectors:
        sector_info = sectors_config['sectors'].get(sector_key, {})
        stocks = sector_info.get('stocks', [])  # Use ALL stocks in config
        
        stock_returns = []
        stock_volumes = []
        
        for symbol in stocks:
            df = load_single_stock(symbol)
            if df.empty:
                continue
            
            # Filter to budget day
            df = df[df.index.date == budget_date]
            
            if len(df) > 0:
                df = clean_stock_data(df)
                if 'return' in df.columns:
                    stock_returns.append(df['return'].rename(symbol))
                if 'volume' in df.columns:
                    stock_volumes.append(df['volume'].rename(symbol))
        
        if stock_returns:
            # Returns
            sector_df = pd.concat(stock_returns, axis=1)
            sector_df['portfolio_return'] = sector_df.mean(axis=1)
            sector_data[sector_key] = sector_df
            
            # Volume
            if stock_volumes:
                vol_df = pd.concat(stock_volumes, axis=1)
                total_vol = vol_df.sum(axis=1)
                # Calculate Relative Volume (RVOL) vs 60-min moving average
                # Fill initial NaNs with mean to avoid empty starts
                rolling_avg = total_vol.rolling(window=12, min_periods=1).mean()
                rvol = total_vol / rolling_avg.replace(0, 1) # Avoid div by zero
                sector_volumes[sector_key] = rvol
            
            print(f"  {sector_key}: {len(stock_returns)} stocks, {len(sector_df)} bars")
    
    if sector_data:
        sector_returns = pd.DataFrame({
            sector: data['portfolio_return'] for sector, data in sector_data.items()
        })
        sector_rvol = pd.DataFrame(sector_volumes)
        return sector_returns, sector_rvol
    
    return pd.DataFrame(), pd.DataFrame()


def calculate_abnormal_returns(sector_returns, market_return):
    """Calculate market-adjusted abnormal returns."""
    abnormal = sector_returns.sub(market_return, axis=0)
    return abnormal


def run_statistical_tests(event_df):
    """Run significance tests on CAR values."""
    print("\nCAR Significance Tests:")
    print("=" * 60)
    
    results = []
    
    for col in ['car_5m', 'car_15m', 'car_30m', 'car_60m']:
        if col not in event_df.columns:
            continue
            
        values = event_df[col].dropna()
        if len(values) < 2:
            continue
            
        mean = values.mean()
        std = values.std()
        n = len(values)
        t_stat = mean / (std / np.sqrt(n)) if std > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-1)) if n > 1 else 1
        
        print(f"\n{col}:")
        print(f"  Mean CAR: {mean*100:.3f}%")
        print(f"  Std Dev: {std*100:.3f}%")
        print(f"  t-stat: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant at 5%: {'Yes' if p_value < 0.05 else 'No'}")
        
        results.append({
            'window': col,
            'mean_car': mean,
            'std_car': std,
            'n': n,
            't_stat': t_stat,
            'p_value': p_value,
            'significant_5pct': p_value < 0.05
        })
    
    return pd.DataFrame(results)


    return pd.DataFrame(results)


def analyze_strategies(event_df):
    """Compare Momentum vs Contrarian strategies."""
    print("\nStrategy Performance Simulation (60-min horizon):")
    print("=" * 60)
    
    if 'car_60m' not in event_df.columns or 'sentiment' not in event_df.columns:
        print("Missing data for strategy analysis")
        return

    # Filter critical events
    df = event_df.copy()
    
    # 1. Momentum: Buy if Sentiment > 0, Short if < 0
    # PnL = sign(sentiment) * returns
    df['momentum_pnl'] = np.sign(df['sentiment']) * df['car_60m']
    
    # 2. Contrarian: Short if Sentiment > 0, Buy if < 0
    df['contrarian_pnl'] = -1 * df['momentum_pnl']
    
    strategies = {
        'Momentum (Follow)': df['momentum_pnl'],
        'Contrarian (Fade)': df['contrarian_pnl']
    }
    
    results = []
    
    for name, pnl in strategies.items():
        total_ret = pnl.sum()
        mean_ret = pnl.mean()
        win_rate = (pnl > 0).mean()
        sharpe = (mean_ret / pnl.std()) * np.sqrt(len(pnl)) if pnl.std() > 0 else 0
        
        print(f"\n{name}:")
        print(f"  Total Return: {total_ret*100:.3f}%")
        print(f"  Avg Trade:    {mean_ret*100:.3f}%")
        print(f"  Win Rate:     {win_rate*100:.1f}%")
        print(f"  Sharpe Ratio: {sharpe:.3f}")
        
        results.append({'strategy': name, 'total_ret': total_ret, 'sharpe': sharpe})
        
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Event Study Analysis')
    parser.add_argument('--year', type=str, help='Analyze specific fiscal year (e.g., 2024-25)')
    parser.add_argument('--save-plots', action='store_true', help='Save visualizations')
    args = parser.parse_args()
    
    print("=" * 60)
    print("EVENT STUDY ANALYSIS")
    print("=" * 60)
    print(f"\nProject root: {project_root}")
    
    # Create output directory
    output_dir = project_root / 'outputs' / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config_dir = project_root / 'config'
    
    with open(config_dir / 'sectors.yaml', 'r', encoding='utf-8') as f:
        sectors_config = yaml.safe_load(f)
    
    with open(config_dir / 'event_dates.yaml', 'r', encoding='utf-8') as f:
        event_dates_config = yaml.safe_load(f)
    
    # =========================================================================
    # 1. Load Processed Speech Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. LOADING PROCESSED SPEECH DATA")
    print("=" * 60)
    
    speech_dir = project_root / 'data' / 'intermediate' / 'speech_text'
    sentences_by_year = load_processed_speeches(speech_dir, args.year)
    
    if not sentences_by_year:
        print("No speech data available. Exiting.")
        return
    
    # =========================================================================
    # 2. Detect First Sector Mentions
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. DETECTING FIRST SECTOR MENTIONS")
    print("=" * 60)
    
    from src.events import detect_all_sector_mentions, calculate_cumulative_attention
    
    mentions_by_year = {}
    
    for fy, sentences_df in sentences_by_year.items():
        mentions = detect_all_sector_mentions(sentences_df, threshold=0.3)
        mentions_by_year[fy] = mentions
        
        print(f"\n{fy}: Detected {len(mentions)} sector first mentions")
        if not mentions.empty:
            print(mentions[['sector', 'sentence_position', 'cumulative_attention']].head(10).to_string())
    
    # =========================================================================
    # 3. Load Market Data and Calculate Returns
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. LOADING MARKET DATA")
    print("=" * 60)
    
    all_event_results = []
    
    for fy in sentences_by_year.keys():
        budget_info = event_dates_config['budget_events'].get(fy, {})
        if not budget_info.get('date'):
            print(f"No date found for {fy}, skipping...")
            continue
        
        budget_date_str = budget_info.get('date')
        budget_date = datetime.strptime(budget_date_str, '%Y-%m-%d').date()
        print(f"\nLoading market data for {fy} ({budget_date})...")
        
        # =====================================================================
        # Fetch Pre-Budget VIX Signal
        # =====================================================================
        vix_signals = {}
        if VIX_AVAILABLE:
            try:
                vix_signals = get_pre_budget_vix_signal(budget_date_str, lookback_days=10)
                if vix_signals:
                    print(f"  VIX Pre-Event: {vix_signals.get('vix_pre_event', 'N/A'):.2f}")
                    print(f"  VIX Regime: {vix_signals.get('vix_regime', 'N/A')}")
                    print(f"  Strategy Hint: {interpret_vix_for_strategy(vix_signals)[:80]}...")
            except Exception as e:
                print(f"  VIX fetch failed: {e}")
        
        sector_returns, sector_rvol = load_sector_returns(project_root, budget_date, sectors_config)
        
        if sector_returns.empty:
            print(f"  No market data available for {budget_date}")
            continue
        
        print(f"  Sector returns shape: {sector_returns.shape}")
        print(f"  Time range: {sector_returns.index.min()} to {sector_returns.index.max()}")
        
        # =====================================================================
        # 4. Calculate Abnormal Returns
        # =====================================================================
        market_return = sector_returns.mean(axis=1)
        abnormal_returns = calculate_abnormal_returns(sector_returns, market_return)
        
        # =====================================================================
        # 5. Event Study Around First Mentions
        # =====================================================================
        from src.utils.time_utils import IST
        
        mentions = mentions_by_year.get(fy, pd.DataFrame())
        
        if mentions.empty:
            continue
        
        speech_start_time = budget_info.get('speech_start', '11:00')
        speech_start = datetime.combine(budget_date, datetime.strptime(speech_start_time, '%H:%M').time())
        speech_start = IST.localize(speech_start)
        
        for _, mention in mentions.iterrows():
            sector = mention['sector']
            
            # Estimate event time based on position
            position = mention.get('sentence_position', 0)
            total_sentences = len(sentences_by_year[fy])
            progress = position / max(total_sentences, 1)
            event_time = speech_start + timedelta(minutes=progress * 90)
            
            if sector not in abnormal_returns.columns:
                continue
            
            # Calculate CAR for different windows
            sector_ar = abnormal_returns[sector]
            sector_vol = sector_rvol[sector] if sector in sector_rvol.columns else pd.Series(dtype=float)
            
            # Find data after event time
            post_event_ar = sector_ar[sector_ar.index >= event_time]
            post_event_vol = sector_vol[sector_vol.index >= event_time] if not sector_vol.empty else pd.Series(dtype=float)
            
            rvol_val = post_event_vol.iloc[0] if len(post_event_vol) > 0 else 1.0
            
            result = {
                'fiscal_year': fy,
                'sector': sector,
                'mention_position': position,
                'event_time': event_time,
                'sentiment': mention.get('avg_sentiment', 0),
                'attention': mention.get('cumulative_attention', 0),
                'rvol': rvol_val,
                'weighted_attention': mention.get('cumulative_attention', 0) * rvol_val,
                # Pre-budget VIX signals
                'vix_pre_event': vix_signals.get('vix_pre_event', None),
                'vix_regime': vix_signals.get('vix_regime', 'unknown'),
                'vix_zscore': vix_signals.get('vix_zscore', None)
            }
            
            if len(post_event_ar) >= 1:
                result['car_5m'] = post_event_ar.iloc[:1].sum()
            if len(post_event_ar) >= 3:
                result['car_15m'] = post_event_ar.iloc[:3].sum()
            if len(post_event_ar) >= 6:
                result['car_30m'] = post_event_ar.iloc[:6].sum()
            if len(post_event_ar) >= 12:
                result['car_60m'] = post_event_ar.iloc[:12].sum()
            
            all_event_results.append(result)
    
    # =========================================================================
    # 6. Statistical Significance Tests
    # =========================================================================
    if all_event_results:
        event_df = pd.DataFrame(all_event_results)
        print("\n" + "=" * 60)
        print(f"EVENT STUDY RESULTS: {len(event_df)} events")
        print("=" * 60)
        
        if not event_df.empty:
            print("\nEvent Results Summary:")
            car_cols = [c for c in event_df.columns if c.startswith('car_')]
            display_cols = ['fiscal_year', 'sector', 'attention', 'rvol', 'weighted_attention'] + car_cols
            # Filter cols that exist
            display_cols = [c for c in display_cols if c in event_df.columns]
            print(event_df[display_cols].to_string())
            
            # Run statistical tests
            stats_df = run_statistical_tests(event_df)
            
            # Run strategy simulation
            analyze_strategies(event_df)
            
            # Save results
            event_df.to_csv(output_dir / 'event_study_results.csv', index=False)
            print(f"\nSaved results to {output_dir / 'event_study_results.csv'}")
            
            if not stats_df.empty:
                stats_df.to_csv(output_dir / 'event_study_statistics.csv', index=False)
                print(f"Saved statistics to {output_dir / 'event_study_statistics.csv'}")
    else:
        print("\nNo event results to analyze.")
    
    print("\n" + "=" * 60)
    print("EVENT STUDY ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

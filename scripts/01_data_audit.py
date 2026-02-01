#!/usr/bin/env python
"""
01 - Data Audit Script

Performs initial data exploration and quality checks for the Budget Speech Impact Analysis project.

Contents:
1. Market Data Overview
2. Budget Speech PDF Overview
3. Data Quality Checks
4. Coverage Analysis
5. Sample Visualizations

Usage:
    python scripts/01_data_audit.py
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("DATA AUDIT - Budget Speech Impact Analysis")
    print("=" * 60)
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass
    pd.set_option('display.max_columns', 50)
    
    print(f"\nProject root: {project_root}")
    
    # =========================================================================
    # 1. Market Data Overview
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. MARKET DATA OVERVIEW")
    print("=" * 60)
    
    from src.ingestion import get_available_stocks, load_single_stock, validate_data_quality
    
    # Get available stocks
    stocks = get_available_stocks()
    print(f"\nTotal stocks available: {len(stocks)}")
    print(f"\nFirst 20 stocks: {stocks[:20]}")
    print(f"\nLast 20 stocks: {stocks[-20:]}")
    
    # Load sample stock (RELIANCE)
    sample_stock = 'RELIANCE'
    df = load_single_stock(sample_stock)
    
    if not df.empty:
        print(f"\n{sample_stock} Data Shape: {df.shape}")
        print(f"Date Range: {df.index.min()} to {df.index.max()}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSample Data:")
        print(df.head(10))
        
        # Data quality check for sample stock
        quality = validate_data_quality(df, sample_stock)
        print("\nData Quality Report:")
        for key, value in quality.items():
            print(f"  {key}: {value}")
    
    # Check data quality for multiple stocks
    sample_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'SBIN', 'TATAMOTORS']
    
    quality_reports = []
    for symbol in sample_stocks:
        stock_df = load_single_stock(symbol)
        if not stock_df.empty:
            q = validate_data_quality(stock_df, symbol)
            q['date_start'] = q['date_range'][0] if q['date_range'][0] else None
            q['date_end'] = q['date_range'][1] if q['date_range'][1] else None
            del q['date_range']
            quality_reports.append(q)
    
    if quality_reports:
        quality_df = pd.DataFrame(quality_reports)
        print("\nMulti-stock Quality Report:")
        print(quality_df)
    
    # =========================================================================
    # 2. Budget Speech PDF Overview
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. BUDGET SPEECH PDF OVERVIEW")
    print("=" * 60)
    
    from src.ingestion import get_available_speeches, load_event_dates, get_budget_dates
    
    # Get available speech PDFs
    speeches = get_available_speeches()
    print(f"\nAvailable Budget Speech PDFs: {len(speeches)}")
    for speech in speeches:
        print(f"  - {speech}")
    
    # Load event dates configuration
    event_dates = load_event_dates()
    print("\nBudget Event Schedule:")
    print("=" * 60)
    
    for fy, info in event_dates.items():
        print(f"\n{fy}:")
        print(f"  Date: {info.get('date')}")
        print(f"  Speech: {info.get('speech_start')} - {info.get('speech_end')}")
        print(f"  Finance Minister: {info.get('finance_minister')}")
    
    # Budget dates list
    budget_dates = get_budget_dates()
    print("\nBudget Presentation Dates:")
    for date in budget_dates:
        print(f"  {date.strftime('%Y-%m-%d (%A)')}")
    
    # =========================================================================
    # 3. Sector Configuration Overview
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. SECTOR CONFIGURATION OVERVIEW")
    print("=" * 60)
    
    from src.ingestion import load_config
    
    # Load sector configuration
    config = load_config()
    sectors_config = config.get('sectors', {}).get('sectors', {})
    
    sector_summary = []
    for sector_key, sector_info in sectors_config.items():
        stocks_list = sector_info.get('stocks', [])
        keywords = sector_info.get('keywords', [])
        name = sector_info.get('name', sector_key)
        
        sector_summary.append({
            'sector_key': sector_key,
            'sector_name': name,
            'n_stocks': len(stocks_list),
            'n_keywords': len(keywords),
            'sample_stocks': stocks_list[:5] if stocks_list else [],
            'sample_keywords': keywords[:5] if keywords else []
        })
    
    sector_df = pd.DataFrame(sector_summary)
    print("\nSector Overview:")
    print(sector_df[['sector_name', 'n_stocks', 'n_keywords']])
    
    print(f"\nTotal stocks across all sectors: {sector_df['n_stocks'].sum()}")
    print(f"Average stocks per sector: {sector_df['n_stocks'].mean():.1f}")
    
    # =========================================================================
    # 4. Market Data Coverage for Budget Days
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. MARKET DATA COVERAGE FOR BUDGET DAYS")
    print("=" * 60)
    
    from src.ingestion import get_budget_day_data
    
    # Check data availability on budget days
    budget_dates = get_budget_dates()
    
    coverage_by_date = []
    for budget_date in budget_dates:
        date_obj = budget_date.date()
        
        available_count = 0
        sample_bars = 0
        
        for symbol in stocks[:50]:  # Check first 50 stocks
            day_data = get_budget_day_data(symbol, date_obj)
            if not day_data.empty:
                available_count += 1
                sample_bars = max(sample_bars, len(day_data))
        
        coverage_by_date.append({
            'budget_date': date_obj,
            'stocks_available': available_count,
            'max_bars': sample_bars,
            'coverage_pct': available_count / 50 * 100
        })
    
    coverage_df = pd.DataFrame(coverage_by_date)
    print("\nBudget Day Data Coverage:")
    print(coverage_df)
    
    # =========================================================================
    # 5. Summary Statistics
    # =========================================================================
    print("\n" + "=" * 60)
    print("DATA AUDIT SUMMARY")
    print("=" * 60)
    print(f"\nMarket Data:")
    print(f"  - Total stocks: {len(stocks)}")
    print(f"  - Data frequency: 5-minute bars")
    print(f"  - Approximate date range: Feb 2015 - Jan 2026")
    
    print(f"\nBudget Speeches:")
    print(f"  - Total PDFs: {len(speeches)}")
    print(f"  - Years covered: {len(event_dates)}")
    
    print(f"\nSector Configuration:")
    print(f"  - Number of sectors: {len(sectors_config)}")
    print(f"  - Total stocks mapped: {sector_df['n_stocks'].sum()}")
    print(f"  - Total keywords: {sector_df['n_keywords'].sum()}")
    
    print(f"\nNext Steps:")
    print("  1. Run NLP pipeline on budget speech PDFs")
    print("  2. Build sector portfolios")
    print("  3. Align speech events with price data")
    print("  4. Run event study analysis")
    
    print("\n" + "=" * 60)
    print("DATA AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

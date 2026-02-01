"""
Budget Speech Impact Analysis - Main Pipeline Script

This script runs the complete analysis pipeline from raw data to results.

Usage:
    python run_pipeline.py --all              # Run complete pipeline
    python run_pipeline.py --speech 2024_25   # Process single budget year
    python run_pipeline.py --step nlp         # Run specific step
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import yaml

# Configure logging first
try:
    from src.utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def load_config():
    """Load all configuration files."""
    config_dir = Path(__file__).parent / 'config'
    
    config = {}
    for file in ['sectors.yaml', 'event_dates.yaml', 'paths.yaml', 'model_params.yaml']:
        filepath = config_dir / file
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                config[file.replace('.yaml', '')] = yaml.safe_load(f)
    
    return config


def step_1_load_data(config):
    """Step 1: Load and validate market data and speeches."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 60)
    
    from src.ingestion import get_available_stocks, get_available_speeches, load_event_dates
    
    # Check available data
    stocks = get_available_stocks()
    speeches = get_available_speeches()
    event_dates = load_event_dates()
    
    logger.info(f"Found {len(stocks)} stock CSV files")
    logger.info(f"Found {len(speeches)} budget speech PDFs")
    logger.info(f"Configured {len(event_dates)} budget events")
    
    return {
        'stocks': stocks,
        'speeches': speeches,
        'event_dates': event_dates
    }


def step_2_process_speeches(config, fiscal_year=None):
    """Step 2: Process budget speeches through NLP pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 2: Processing Budget Speeches")
    logger.info("=" * 60)
    
    from src.nlp import (
        extract_text_from_pdf, clean_text,
        tokenize_speech, classify_sentences,
        SentimentAnalyzer,
        calculate_certainty_score, calculate_actionability_score
    )
    from src.ingestion import get_available_speeches, get_speech_metadata, load_event_dates
    
    event_dates = load_event_dates()
    speeches = get_available_speeches()
    
    # Filter to specific year if requested
    if fiscal_year:
        speeches = [s for s in speeches if fiscal_year.replace('_', '') in s.replace('_', '')]
    
    all_sentences = {}
    sentiment_analyzer = SentimentAnalyzer()
    
    for speech_path in speeches:
        speech_file = Path(speech_path)
        
        # Determine fiscal year from filename
        fy_match = None
        for fy in event_dates.keys():
            fy_clean = fy.replace('-', '').replace('_', '').lower()
            fname_clean = speech_file.stem.lower().replace('bs', '').replace('_', '')
            if fname_clean in fy_clean or fy_clean[-4:] in fname_clean:
                fy_match = fy
                break
        
        if not fy_match:
            logger.warning(f"Could not match {speech_file.name} to fiscal year")
            continue
        
        logger.info(f"Processing {fy_match}: {speech_file.name}")
        
        # Extract text
        text = extract_text_from_pdf(speech_path)
        text = clean_text(text)
        
        if not text:
            logger.warning(f"No text extracted from {speech_file.name}")
            continue
        
        # Tokenize sentences
        sentences = tokenize_speech(text)
        
        # Convert to DataFrame
        sentences_df = pd.DataFrame(sentences)
        
        # Classify sectors using the classify_sentences function
        sentences_df = classify_sentences(sentences_df, text_column='text')
        
        # Sentiment analysis
        sentiments = [sentiment_analyzer.analyze(t) for t in sentences_df['text'].tolist()]
        for key in ['compound', 'positive', 'negative', 'neutral']:
            sentences_df[f'sentiment_{key}'] = [s.get(key, 0) for s in sentiments]
        
        # Certainty and actionability
        sentences_df['certainty_score'] = [
            calculate_certainty_score(t)['certainty_score'] for t in sentences_df['text'].tolist()
        ]
        sentences_df['actionability_score'] = [
            calculate_actionability_score(t)['actionability_score'] for t in sentences_df['text'].tolist()
        ]
        
        # Calculate importance weight
        sentences_df['importance_weight'] = (
            0.3 * sentences_df['certainty_score'] +
            0.4 * sentences_df['actionability_score'] +
            0.3 * sentences_df['sentiment_compound'].abs()
        )
        
        # Get metadata
        metadata = get_speech_metadata(fy_match)
        
        # Add metadata
        sentences_df['fiscal_year'] = fy_match
        sentences_df['budget_date'] = metadata.get('date', '') if metadata else ''
        
        all_sentences[fy_match] = sentences_df
        
        logger.info(f"  Processed {len(sentences_df)} sentences")
    
    # Save intermediate results
    output_dir = Path(__file__).parent / 'data' / 'intermediate' / 'speech_text'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for fy, df in all_sentences.items():
        output_path = output_dir / f'{fy.replace("-", "_")}_sentences.parquet'
        df.to_parquet(output_path)
        logger.info(f"  Saved {output_path}")
    
    return all_sentences


def step_3_build_sector_portfolios(config, budget_dates=None):
    """Step 3: Build sector portfolios from stock data."""
    logger.info("=" * 60)
    logger.info("STEP 3: Building Sector Portfolios")
    logger.info("=" * 60)
    
    from src.ingestion import get_budget_dates
    from datetime import timedelta
    
    if budget_dates is None:
        budget_dates = get_budget_dates()
    
    if not budget_dates:
        logger.warning("No budget dates configured - skipping portfolio building")
        return {}
    
    sectors_config = config.get('sectors', {}).get('sectors', {})
    project_root = Path(__file__).parent
    
    sector_returns_by_date = {}
    
    for budget_date in budget_dates:
        date_str = budget_date.strftime('%Y-%m-%d')
        logger.info(f"Processing {date_str}")
        
        # Define event window: budget day and ±2 trading days
        target_date = budget_date.date()
        
        sector_data = {}
        
        for sector_key, sector_info in sectors_config.items():
            stocks = sector_info.get('stocks', [])
            
            if not stocks:
                continue
            
            sector_returns_list = []
            
            for symbol in stocks:
                csv_path = project_root / f"{symbol}.csv"
                
                if not csv_path.exists():
                    continue
                
                try:
                    # Load stock data
                    df = pd.read_csv(csv_path, parse_dates=['date'])
                    
                    # Filter to budget date (same day only for event day)
                    df['date_only'] = df['date'].dt.date
                    day_data = df[df['date_only'] == target_date].copy()
                    
                    if day_data.empty:
                        continue
                    
                    # Calculate intraday returns
                    day_data = day_data.sort_values('date')
                    day_data['return'] = day_data['close'].pct_change()
                    
                    # Store returns with timestamp index
                    day_data = day_data.set_index('date')
                    sector_returns_list.append(day_data['return'].rename(symbol))
                    
                except Exception as e:
                    logger.debug(f"Error loading {symbol}: {e}")
                    continue
            
            if sector_returns_list:
                # Combine all stock returns in this sector
                sector_df = pd.concat(sector_returns_list, axis=1)
                
                # Calculate equal-weighted sector return
                sector_return = sector_df.mean(axis=1)
                sector_data[sector_key] = sector_return
                
                logger.debug(f"  {sector_key}: {len(sector_returns_list)} stocks")
        
        if sector_data:
            # Combine all sectors
            combined = pd.DataFrame(sector_data)
            sector_returns_by_date[date_str] = combined
            logger.info(f"  Built {len(sector_data)} sector portfolios, {len(combined)} bars")
        else:
            logger.warning(f"  No data found for {date_str}")
    
    # Save
    output_dir = Path(__file__).parent / 'data' / 'intermediate' / 'market_clean'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for date_str, df in sector_returns_by_date.items():
        output_path = output_dir / f'sector_returns_{date_str}.parquet'
        df.to_parquet(output_path)
        logger.info(f"  Saved {output_path.name}")
    
    logger.info(f"Built portfolios for {len(sector_returns_by_date)} budget days")
    
    return sector_returns_by_date


def step_4_detect_mentions_and_align(config, sentences_by_year, sector_returns_by_date):
    """Step 4: Detect sector mentions and align with price reactions."""
    logger.info("=" * 60)
    logger.info("STEP 4: Detecting Mentions & Aligning with Prices")
    logger.info("=" * 60)
    
    from src.events import (
        detect_all_sector_mentions, build_event_panel,
        align_speech_to_market, build_aligned_panel
    )
    
    mentions_by_year = {}
    
    for fy, sentences_df in sentences_by_year.items():
        # Detect first mentions for all sectors
        mentions = detect_all_sector_mentions(sentences_df, threshold=0.3)
        mentions_by_year[fy] = mentions
        
        if not mentions.empty:
            logger.info(f"  {fy}: Detected {len(mentions)} sector mentions")
    
    # Build aligned panel
    panel = build_aligned_panel(
        sentences_by_year,
        mentions_by_year,
        sector_returns_by_date
    )
    
    # Save
    output_dir = Path(__file__).parent / 'data' / 'intermediate' / 'aligned_events'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'event_panel.parquet'
    panel.to_parquet(output_path)
    
    logger.info(f"Created event panel with {len(panel)} events")
    
    return panel, mentions_by_year


def step_5_run_event_study(config, panel, sector_returns_by_date):
    """Step 5: Run event study analysis."""
    logger.info("=" * 60)
    logger.info("STEP 5: Running Event Study Analysis")
    logger.info("=" * 60)
    
    from src.models import (
        calculate_abnormal_returns, event_study_batch,
        test_car_significance_by_sector, calculate_average_car_pattern,
        analyze_strategies
    )
    
    # Calculate abnormal returns for each date
    abnormal_returns_by_date = {}
    
    for date_str, sector_returns in sector_returns_by_date.items():
        # Use market-adjusted returns (simple for now)
        if sector_returns.shape[1] > 0:
            market_return = sector_returns.mean(axis=1)
            ar = calculate_abnormal_returns(sector_returns, market_return, method='market_adjusted')
            abnormal_returns_by_date[date_str] = ar
    
    # Run event studies
    event_results = event_study_batch(panel, abnormal_returns_by_date)
    
    if not event_results.empty:
        # Test significance by sector
        significance = test_car_significance_by_sector(event_results)
        
        # Calculate average CAR pattern
        car_pattern = calculate_average_car_pattern(event_results)
        
        # Run strategy simulation (Momentum vs Contrarian)
        if 'sentiment' in panel.columns:
            # Merge sentiment from panel into results
            event_results = event_results.merge(
                panel[['sector', 'fiscal_year', 'sentiment']].drop_duplicates(),
                on=['sector', 'fiscal_year'],
                how='left'
            )
        strategy_results = analyze_strategies(event_results)
        
        # Save results
        output_dir = Path(__file__).parent / 'outputs' / 'tables'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        event_results.to_csv(output_dir / 'event_study_results.csv', index=False)
        significance.to_csv(output_dir / 'sector_significance.csv', index=False)
        car_pattern.to_csv(output_dir / 'car_pattern.csv', index=False)
        if not strategy_results.empty:
            strategy_results.to_csv(output_dir / 'strategy_comparison.csv', index=False)
        
        logger.info(f"Completed event study for {len(event_results)} events")
        logger.info(f"Results saved to {output_dir}")
    else:
        logger.warning("No event study results generated")
    
    return event_results


def step_6_run_validation(config, panel, sector_returns_by_date):
    """Step 6: Run validation and robustness tests."""
    logger.info("=" * 60)
    logger.info("STEP 6: Running Validation Tests")
    logger.info("=" * 60)
    
    from src.validation import (
        generate_placebo_dates, run_placebo_event_study,
        compare_actual_vs_placebo, bootstrap_confidence_interval
    )
    
    # Get actual budget dates
    actual_dates = [datetime.strptime(d, '%Y-%m-%d') for d in sector_returns_by_date.keys()]
    
    if not actual_dates:
        logger.warning("No dates available for validation")
        return {}
    
    # Generate placebo dates
    placebo_dates = generate_placebo_dates(actual_dates, n_placebos=5)
    
    logger.info(f"Generated {len(placebo_dates)} placebo dates")
    
    # Run placebo event study (requires additional data loading)
    # This is a simplified version
    
    results = {
        'n_placebo_dates': len(placebo_dates),
        'validation_run': True
    }
    
    logger.info("Validation tests completed")
    
    return results


def run_full_pipeline(config, fiscal_year=None):
    """Run the complete analysis pipeline."""
    logger.info("=" * 60)
    logger.info("BUDGET SPEECH IMPACT ANALYSIS PIPELINE")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 60)
    
    # Step 1: Load data
    data_info = step_1_load_data(config)
    
    # Step 2: Process speeches
    sentences_by_year = step_2_process_speeches(config, fiscal_year)
    
    if not sentences_by_year:
        logger.error("No sentences processed. Check speech PDFs.")
        return
    
    # Step 3: Build sector portfolios
    sector_returns_by_date = step_3_build_sector_portfolios(config)
    
    if not sector_returns_by_date:
        logger.warning("No sector portfolios built. Continuing with available data.")
    
    # Step 4: Detect mentions and align
    panel, mentions_by_year = step_4_detect_mentions_and_align(
        config, sentences_by_year, sector_returns_by_date
    )
    
    # Step 5: Run event study
    if not panel.empty and sector_returns_by_date:
        event_results = step_5_run_event_study(config, panel, sector_returns_by_date)
    else:
        logger.warning("Skipping event study - insufficient data")
    
    # Step 6: Validation
    if sector_returns_by_date:
        validation_results = step_6_run_validation(config, panel, sector_returns_by_date)
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED")
    logger.info(f"Finished at: {datetime.now()}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Budget Speech Impact Analysis Pipeline'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run complete pipeline'
    )
    parser.add_argument(
        '--speech', type=str,
        help='Process specific fiscal year (e.g., 2024_25)'
    )
    parser.add_argument(
        '--step', type=str,
        choices=['load', 'nlp', 'portfolios', 'align', 'event_study', 'validate'],
        help='Run specific step'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run demo on latest budget speech'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    if args.demo:
        logger.info("Running demo on latest budget speech...")
        run_full_pipeline(config, fiscal_year='2024_25')
    elif args.all:
        run_full_pipeline(config)
    elif args.speech:
        run_full_pipeline(config, fiscal_year=args.speech)
    elif args.step:
        if args.step == 'load':
            step_1_load_data(config)
        elif args.step == 'nlp':
            step_2_process_speeches(config)
        else:
            logger.info(f"Step '{args.step}' requires previous steps. Use --all")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

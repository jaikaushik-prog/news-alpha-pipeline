#!/usr/bin/env python
"""
02 - NLP Processing Pipeline Script

Processes budget speech PDFs through the NLP pipeline:
1. Extract text from PDFs
2. Tokenize into sentences
3. Classify sectors (soft probabilities)
4. Analyze sentiment
5. Score certainty and actionability

Usage:
    python scripts/02_nlp_processing.py [--year FISCAL_YEAR]
    
Examples:
    python scripts/02_nlp_processing.py              # Process all speeches
    python scripts/02_nlp_processing.py --year 2024-25  # Process single year
"""

import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import yaml

# Download NLTK data
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


def process_single_speech(speech_file, fiscal_year, event_info, sectors_config):
    """Process a single budget speech PDF."""
    from src.nlp import (
        extract_text_from_pdf, clean_text, tokenize_speech,
        classify_sentences, analyze_sentiment,
        calculate_certainty_score, calculate_actionability_score
    )
    
    print(f"\nProcessing: {speech_file.name} ({fiscal_year})")
    
    # 1. Extract text
    print("  Extracting text...")
    raw_text = extract_text_from_pdf(str(speech_file))
    print(f"    Extracted {len(raw_text)} characters")
    
    # 2. Tokenize
    print("  Tokenizing...")
    sentences = tokenize_speech(raw_text)
    sentences_df = pd.DataFrame(sentences)
    print(f"    {len(sentences_df)} sentences")
    
    # 3. Classify sectors
    print("  Classifying sectors...")
    sentences_df = classify_sentences(sentences_df)
    prob_columns = [c for c in sentences_df.columns if c.startswith('prob_')]
    print(f"    Added {len(prob_columns)} sector probability columns")
    
    # 4. Sentiment analysis (using FinBERT + VADER combined)
    print("  Analyzing sentiment (FinBERT + VADER)...")
    sentiments = [analyze_sentiment(t, method='combined') for t in sentences_df['text'].tolist()]
    # Keys from analyze_sentiment are: sentiment_compound, sentiment_pos, sentiment_neg, sentiment_neu
    sentences_df['sentiment_compound'] = [s.get('sentiment_compound', 0) for s in sentiments]
    sentences_df['sentiment_positive'] = [s.get('sentiment_pos', 0) for s in sentiments]
    sentences_df['sentiment_negative'] = [s.get('sentiment_neg', 0) for s in sentiments]
    sentences_df['sentiment_neutral'] = [s.get('sentiment_neu', 0) for s in sentiments]
    # FinBERT scores
    sentences_df['finbert_compound'] = [s.get('finbert_compound', 0) for s in sentiments]
    
    # 5. Certainty and actionability
    print("  Scoring certainty/actionability...")
    sentences_df['certainty_score'] = [
        calculate_certainty_score(t)['certainty_score'] 
        for t in sentences_df['text'].tolist()
    ]
    sentences_df['actionability_score'] = [
        calculate_actionability_score(t)['actionability_score'] 
        for t in sentences_df['text'].tolist()
    ]
    
    # 6. Importance weight
    sentences_df['importance_weight'] = (
        0.3 * sentences_df['certainty_score'] +
        0.4 * sentences_df['actionability_score'] +
        0.3 * sentences_df['sentiment_compound'].abs()
    )
    
    # Add metadata
    sentences_df['fiscal_year'] = fiscal_year
    sentences_df['budget_date'] = event_info.get('date')
    
    return sentences_df


def main():
    parser = argparse.ArgumentParser(description='NLP Processing Pipeline')
    parser.add_argument('--year', type=str, help='Process specific fiscal year (e.g., 2024-25)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("NLP PROCESSING PIPELINE")
    print("=" * 60)
    print(f"\nProject root: {project_root}")
    
    # Load configuration
    config_dir = project_root / 'config'
    
    with open(config_dir / 'sectors.yaml', 'r', encoding='utf-8') as f:
        sectors_config = yaml.safe_load(f)
    
    with open(config_dir / 'event_dates.yaml', 'r', encoding='utf-8') as f:
        event_dates = yaml.safe_load(f)
    
    print(f"Loaded {len(sectors_config['sectors'])} sector definitions")
    print(f"Loaded {len(event_dates['budget_events'])} budget events")
    
    # Find speech PDFs
    speech_files = list(project_root.glob('*.pdf'))
    print(f"\nFound {len(speech_files)} budget speech PDFs")
    
    # Create output directory
    output_dir = project_root / 'data' / 'intermediate' / 'speech_text'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map fiscal years to PDFs
    fy_to_pdf = {}
    for fy, info in event_dates['budget_events'].items():
        pdf_name = info.get('pdf_file', '')
        for speech_file in speech_files:
            if pdf_name in speech_file.name or fy.replace('-', '') in speech_file.name.replace('_', ''):
                fy_to_pdf[fy] = (speech_file, info)
                break
    
    # Filter by year if specified
    if args.year:
        if args.year in fy_to_pdf:
            fy_to_pdf = {args.year: fy_to_pdf[args.year]}
        else:
            print(f"Error: Fiscal year {args.year} not found")
            return
    
    # Process each speech
    all_results = []
    for fy, (speech_file, event_info) in fy_to_pdf.items():
        try:
            sentences_df = process_single_speech(speech_file, fy, event_info, sectors_config)
            
            # Save
            output_path = output_dir / f'{fy.replace("-", "_")}_sentences.parquet'
            sentences_df.to_parquet(output_path)
            print(f"  Saved: {output_path.name}")
            
            # Also save as CSV
            csv_path = output_dir / f'{fy.replace("-", "_")}_sentences.csv'
            sentences_df.to_csv(csv_path, index=False)
            
            all_results.append({
                'fiscal_year': fy,
                'sentences': len(sentences_df),
                'output_file': str(output_path)
            })
            
        except Exception as e:
            print(f"  Error processing {fy}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("NLP PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nProcessed {len(all_results)} budget speeches")
    
    for result in all_results:
        print(f"  {result['fiscal_year']}: {result['sentences']} sentences")
    
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

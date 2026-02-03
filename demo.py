
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

print("=" * 60)
print("BUDGET SPEECH IMPACT ANALYSIS - QUICK DEMO")
print("=" * 60)

# Download NLTK data
print("\n[1/6] Setting up NLP resources...")
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Load config
print("\n[2/6] Loading configuration...")
import yaml

config_dir = Path(__file__).parent / 'config'

with open(config_dir / 'sectors.yaml', 'r', encoding='utf-8') as f:
    sectors_config = yaml.safe_load(f)

with open(config_dir / 'event_dates.yaml', 'r', encoding='utf-8') as f:
    event_dates = yaml.safe_load(f)

print(f"  Loaded {len(sectors_config['sectors'])} sectors")
print(f"  Loaded {len(event_dates['budget_events'])} budget events")

# Find latest budget speech
print("\n[3/6] Finding and processing budget speech PDF...")
project_root = Path(__file__).parent

# Try to find 2024-25 or 2025-26 budget
speech_files = list(project_root.glob('bs2024*.pdf')) + list(project_root.glob('bs2025*.pdf'))
if not speech_files:
    speech_files = list(project_root.glob('*.pdf'))

if not speech_files:
    print("  ERROR: No budget speech PDFs found!")
    sys.exit(1)

speech_file = speech_files[-1]
print(f"  Processing: {speech_file.name}")

# Extract text
from src.nlp import extract_text_from_pdf, clean_text
raw_text = extract_text_from_pdf(str(speech_file))
cleaned_text = clean_text(raw_text)
print(f"  Extracted {len(cleaned_text):,} characters")

# Tokenize
from src.nlp import tokenize_speech
sentences = tokenize_speech(cleaned_text)
print(f"  Tokenized into {len(sentences)} sentences")

# Convert to DataFrame
sentences_df = pd.DataFrame(sentences)

# Classify sectors
print("\n[4/6] Classifying sectors and analyzing sentiment...")
from src.nlp import classify_sentences

# Use the classify_sentences function which adds prob_ columns
sentences_df = classify_sentences(sentences_df, text_column='text')

prob_cols = [c for c in sentences_df.columns if c.startswith('prob_')]
print(f"  Classified {len(prob_cols)} sectors")

# Sentiment
from src.nlp import SentimentAnalyzer
analyzer = SentimentAnalyzer()

sentiments = [analyzer.analyze(text) for text in sentences_df['text'].tolist()]
sentences_df['sentiment_compound'] = [s.get('compound', 0) for s in sentiments]

print(f"  Average sentiment: {sentences_df['sentiment_compound'].mean():.3f}")

# Detect first mentions
print("\n[5/6] Detecting sector first mentions...")
from src.events import detect_all_sector_mentions

# Set importance weight
sentences_df['importance_weight'] = 1.0

mentions = detect_all_sector_mentions(sentences_df, threshold=0.2)
print(f"  Detected {len(mentions)} sector mentions")

if not mentions.empty:
    print("\n  First Mentions Order:")
    print("  " + "-" * 50)
    for i, (_, row) in enumerate(mentions.head(12).iterrows()):
        sector = row['sector']
        pos = row.get('sentence_position', 0)
        attn = row.get('cumulative_attention', 0)
        print(f"  {i+1}. {sector:<20} (position: {pos:.0f}, attention: {attn:.2f})")

# Calculate sector attention
print("\n[6/6] Calculating sector attention distribution...")
prob_columns = [c for c in sentences_df.columns if c.startswith('prob_')]

sector_attention = {}
for col in prob_columns:
    sector = col.replace('prob_', '')
    sector_attention[sector] = sentences_df[col].sum()

# Sort by attention
attention_sorted = sorted(sector_attention.items(), key=lambda x: x[1], reverse=True)

print("\n  Top 10 Sectors by Attention:")
print("  " + "-" * 50)
max_attn = max(sector_attention.values()) if sector_attention else 1
for i, (sector, attention) in enumerate(attention_sorted[:10]):
    bar = "█" * int(attention / max_attn * 30) if max_attn > 0 else ""
    print(f"  {i+1}. {sector:<20} {attention:.2f} {bar}")

# Save results
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

output_dir = project_root / 'data' / 'intermediate' / 'speech_text'
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / 'demo_sentences.parquet'
sentences_df.to_parquet(output_file)
print(f"  Saved to: {output_file}")

# Also save mentions
mentions_file = output_dir / 'demo_mentions.csv'
if not mentions.empty:
    mentions.to_csv(mentions_file, index=False)
    print(f"  Mentions saved to: {mentions_file}")

print("\n" + "=" * 60)
print("DEMO COMPLETE!")
print("=" * 60)
print("\nNext steps:")
print("  1. Open notebooks/02_nlp_processing.ipynb for detailed analysis")
print("  2. Open notebooks/03_event_study.ipynb for market reaction analysis")
print("  3. Run: python run_pipeline.py --all for full analysis")

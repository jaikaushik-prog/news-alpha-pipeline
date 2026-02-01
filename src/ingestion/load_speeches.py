"""
Budget speech loading module.

Loads Union Budget PDF files for text extraction.
"""

from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import yaml

from ..utils.logging import get_logger

logger = get_logger(__name__)


def get_speeches_dir() -> Path:
    """
    Get the speeches directory path.
    
    Returns
    -------
    Path
        Path to speeches directory
    """
    # Budget PDFs are currently in the base data directory
    return Path(__file__).parents[2]


def get_available_speeches() -> List[str]:
    """
    Get list of available budget speech PDF files.
    
    Returns
    -------
    list
        List of PDF filenames
    """
    speeches_dir = get_speeches_dir()
    pdf_files = list(speeches_dir.glob("bs*.pdf"))
    
    logger.info(f"Found {len(pdf_files)} budget speech PDFs")
    
    return sorted([f.name for f in pdf_files])


def get_speech_path(fiscal_year: str) -> Optional[Path]:
    """
    Get the path to a specific budget speech PDF.
    
    Parameters
    ----------
    fiscal_year : str
        Fiscal year (e.g., '2024-25' or '2024_25')
        
    Returns
    -------
    Path or None
        Path to PDF file, or None if not found
    """
    speeches_dir = get_speeches_dir()
    
    # Try different filename patterns
    patterns = [
        f"bs{fiscal_year.replace('-', '')}.pdf",
        f"bs{fiscal_year.replace('-', '_')}.pdf",
        f"bs{fiscal_year}.pdf",
    ]
    
    for pattern in patterns:
        path = speeches_dir / pattern
        if path.exists():
            return path
    
    # Also try without century (e.g., bs201415 vs bs20142015)
    fy_short = fiscal_year.replace('-', '').replace('_', '')
    if len(fy_short) == 8:  # e.g., 20142015
        fy_short = fy_short[:4] + fy_short[6:]  # e.g., 201415
    
    path = speeches_dir / f"bs{fy_short}.pdf"
    if path.exists():
        return path
    
    logger.warning(f"Speech PDF not found for fiscal year: {fiscal_year}")
    return None


def load_event_dates() -> Dict:
    """
    Load budget event dates from configuration.
    
    Returns
    -------
    dict
        Budget event information keyed by fiscal year
    """
    config_path = Path(__file__).parents[2] / "config" / "event_dates.yaml"
    
    if not config_path.exists():
        logger.error(f"Event dates config not found: {config_path}")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config.get('budget_events', {})


def get_budget_dates() -> List[datetime]:
    """
    Get all budget presentation dates.
    
    Returns
    -------
    list
        List of budget dates as datetime objects
    """
    events = load_event_dates()
    
    dates = []
    for fy, info in events.items():
        if 'date' in info:
            dates.append(datetime.strptime(info['date'], '%Y-%m-%d'))
    
    return sorted(dates)


def get_speech_metadata(fiscal_year: str) -> Optional[Dict]:
    """
    Get metadata for a specific budget speech.
    
    Parameters
    ----------
    fiscal_year : str
        Fiscal year (e.g., '2024-25')
        
    Returns
    -------
    dict or None
        Speech metadata including date, times, finance minister, etc.
    """
    events = load_event_dates()
    return events.get(fiscal_year)


def map_pdf_to_fiscal_year() -> Dict[str, str]:
    """
    Create mapping from PDF filename to fiscal year.
    
    Returns
    -------
    dict
        Mapping of PDF filename to fiscal year
    """
    events = load_event_dates()
    
    mapping = {}
    for fy, info in events.items():
        if 'pdf_file' in info:
            mapping[info['pdf_file']] = fy
    
    return mapping


def get_speech_duration_minutes(fiscal_year: str) -> float:
    """
    Calculate speech duration in minutes for a fiscal year.
    
    Parameters
    ----------
    fiscal_year : str
        Fiscal year
        
    Returns
    -------
    float
        Duration in minutes
    """
    metadata = get_speech_metadata(fiscal_year)
    
    if not metadata:
        return 120.0  # Default 2 hours
    
    start = metadata.get('speech_start', '11:00')
    end = metadata.get('speech_end', '13:00')
    
    start_time = datetime.strptime(start, '%H:%M')
    end_time = datetime.strptime(end, '%H:%M')
    
    duration = (end_time - start_time).total_seconds() / 60
    
    return duration

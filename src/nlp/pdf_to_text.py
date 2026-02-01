"""
PDF to text extraction module.

Converts Union Budget PDF files to clean text using pdfplumber.
Falls back to PyMuPDF if pdfplumber fails.
"""

from pathlib import Path
from typing import Optional, List
import re

from ..utils.logging import get_logger

logger = get_logger(__name__)


def extract_text_pdfplumber(pdf_path: Path) -> str:
    """
    Extract text from PDF using pdfplumber.
    
    Parameters
    ----------
    pdf_path : Path
        Path to PDF file
        
    Returns
    -------
    str
        Extracted text
    """
    try:
        import pdfplumber
        
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
        
        return "\n\n".join(text_parts)
    
    except ImportError:
        logger.warning("pdfplumber not installed, trying PyMuPDF")
        return ""
    except Exception as e:
        logger.error(f"pdfplumber extraction failed: {e}")
        return ""


def extract_text_pymupdf(pdf_path: Path) -> str:
    """
    Extract text from PDF using PyMuPDF (fitz).
    
    Parameters
    ----------
    pdf_path : Path
        Path to PDF file
        
    Returns
    -------
    str
        Extracted text
    """
    try:
        import fitz  # PyMuPDF
        
        text_parts = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text:
                text_parts.append(text)
        
        doc.close()
        return "\n\n".join(text_parts)
    
    except ImportError:
        logger.warning("PyMuPDF not installed")
        return ""
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")
        return ""


def clean_text(text: str) -> str:
    """
    Clean extracted text from PDF.
    
    Parameters
    ----------
    text : str
        Raw extracted text
        
    Returns
    -------
    str
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Fix common OCR issues
    text = text.replace('Rs.', '₹')
    text = text.replace('Rs ', '₹')
    text = re.sub(r'(\d),(\d{3})', r'\1,\2', text)  # Preserve number formatting
    
    # Remove page numbers (common patterns)
    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'\nPage \d+\n', '\n', text)
    
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def extract_text_from_pdf(pdf_path: Path, clean: bool = True) -> str:
    """
    Extract text from a PDF file.
    
    Tries pdfplumber first, then falls back to PyMuPDF.
    
    Parameters
    ----------
    pdf_path : Path
        Path to PDF file
    clean : bool
        Whether to clean the extracted text
        
    Returns
    -------
    str
        Extracted text
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return ""
    
    logger.info(f"Extracting text from: {pdf_path.name}")
    
    # Try pdfplumber first
    text = extract_text_pdfplumber(pdf_path)
    
    # Fall back to PyMuPDF if pdfplumber fails or returns empty
    if not text.strip():
        logger.info("Trying PyMuPDF as fallback...")
        text = extract_text_pymupdf(pdf_path)
    
    if not text.strip():
        logger.error(f"Failed to extract text from: {pdf_path.name}")
        return ""
    
    # Clean if requested
    if clean:
        text = clean_text(text)
    
    logger.info(f"Extracted {len(text)} characters from {pdf_path.name}")
    
    return text


def extract_all_speeches(speeches_dir: Path, output_dir: Path) -> dict:
    """
    Extract text from all budget speech PDFs.
    
    Parameters
    ----------
    speeches_dir : Path
        Directory containing PDF files
    output_dir : Path
        Directory to save extracted text files
        
    Returns
    -------
    dict
        Mapping of fiscal year to extracted text
    """
    speeches_dir = Path(speeches_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    pdf_files = list(speeches_dir.glob("bs*.pdf"))
    
    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        
        if text:
            # Save to text file
            txt_path = output_dir / f"{pdf_path.stem}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            results[pdf_path.stem] = text
            logger.info(f"Saved extracted text to: {txt_path.name}")
    
    return results

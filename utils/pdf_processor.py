"""utils/pdf_processor.py — PDF text extraction and chunking."""

import logging
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import streamlit as st

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_file) -> List[Dict[str, Any]]:
    """Extract page-level text with metadata from a single uploaded PDF."""
    pages    = []
    filename = pdf_file.name
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text")
            if text.strip():
                pages.append({
                    "text":        text,
                    "page_number": page_num + 1,
                    "filename":    filename,
                })
        doc.close()
        logger.info(f"Extracted {len(pages)} pages from '{filename}'")
    except Exception as e:
        logger.error(f"Error processing '{filename}': {e}")
        st.error(f"Error processing {filename}: {e}")
    return pages


def extract_text_from_multiple_pdfs(pdf_files) -> List[Dict[str, Any]]:
    """Extract text from multiple uploaded PDFs."""
    all_pages = []
    for f in pdf_files:
        all_pages.extend(extract_text_from_pdf(f))
    return all_pages


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int  = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Split page texts into overlapping chunks, preserving metadata.
    Recommended: chunk_size=800–1200, chunk_overlap=150–250.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks   = []
    chunk_id = 0
    for page in pages:
        for split in splitter.split_text(page["text"]):
            if split.strip():
                chunks.append({
                    "text":        split,
                    "page_number": page["page_number"],
                    "filename":    page["filename"],
                    "chunk_id":    chunk_id,
                })
                chunk_id += 1
    logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks

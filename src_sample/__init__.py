"""
PDF to Q&A Pipeline - Convert PDF documents to question-answer pairs
"""

from .models import Document, QAPair
from .pdf_processor import PDFProcessor
from .llm_processor import LLMProcessor
from .pipeline import PDFToQAPipeline

__version__ = "0.1.0"
__all__ = ["Document", "QAPair", "PDFProcessor", "LLMProcessor", "PDFToQAPipeline"]

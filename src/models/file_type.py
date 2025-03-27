from __future__ import annotations

from enum import Enum


class FileType(str, Enum):
    """Enum for file types"""

    PDF = 'pdf'
    DOCX = 'docx'
    DOC = 'doc'
    XLSX = 'xlsx'
    XLS = 'xls'
    CSV = 'csv'
    TXT = 'txt'
    HTML = 'html'

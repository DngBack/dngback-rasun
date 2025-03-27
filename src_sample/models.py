from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class QAPair(BaseModel):
    """Model for a single question-answer pair"""

    question: str = Field(..., description="The question text")
    answer: str = Field(..., description="The answer text")
    context: Optional[str] = Field(
        None, description="Optional context from the source document"
    )
    source_page: Optional[int] = Field(
        None, description="Page number where the content was found"
    )
    metadata: Optional[dict] = Field(
        default_factory=dict, description="Additional metadata"
    )


class Document(BaseModel):
    """Model for a processed document"""

    file_name: str = Field(..., description="Name of the source PDF file")
    processed_at: datetime = Field(
        default_factory=datetime.now, description="When the document was processed"
    )
    qa_pairs: List[QAPair] = Field(
        default_factory=list, description="List of question-answer pairs extracted"
    )
    total_pages: Optional[int] = Field(
        None, description="Total number of pages in the document"
    )
    processing_metadata: Optional[dict] = Field(
        default_factory=dict, description="Metadata about the processing"
    )

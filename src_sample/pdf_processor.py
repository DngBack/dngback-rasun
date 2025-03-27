from pathlib import Path
from typing import List, Dict, Any
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
from .models import Document


class PDFProcessor:
    def __init__(self, api_key: str):
        """Initialize the PDF processor with LlamaParser API key"""
        self.parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",  # Use enum value
        )

    def process_file(self, file_path: Path) -> Document:
        """Process a single PDF file and extract its content"""
        try:
            # Set up file extractor
            file_extractor: Dict[str, BaseReader] = {".pdf": self.parser}

            # Use SimpleDirectoryReader to parse the file
            documents = SimpleDirectoryReader(
                input_files=[str(file_path)], file_extractor=file_extractor
            ).load_data()

            if not documents:
                raise Exception(f"No content extracted from {file_path}")

            # Get the first document
            doc = documents[0]

            # Extract text content and metadata
            text_content = doc.text
            metadata = {
                "total_pages": getattr(doc.metadata, "total_pages", None),
                "title": getattr(doc.metadata, "title", None),
                "author": getattr(doc.metadata, "author", None),
                "creation_date": getattr(doc.metadata, "creation_date", None),
                "text_content": text_content,
            }

            # Create document object
            document = Document(
                file_name=file_path.name,
                total_pages=metadata["total_pages"],
                processing_metadata=metadata,
            )

            return document

        except Exception as e:
            raise Exception(f"Error processing file {file_path}: {str(e)}")

    def process_directory(self, directory_path: Path) -> List[Document]:
        """Process all PDF files in a directory"""
        documents = []
        pdf_files = list(directory_path.glob("**/*.pdf"))

        for pdf_file in pdf_files:
            try:
                document = self.process_file(pdf_file)
                documents.append(document)
            except Exception as e:
                print(f"Failed to process {pdf_file}: {str(e)}")
                continue

        return documents

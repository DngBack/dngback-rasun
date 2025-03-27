from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
from .pdf_processor import PDFProcessor
from .llm_processor import LLMProcessor
from .models import Document, QAPair


class PDFToQAPipeline:
    def __init__(self, llama_api_key: str, openai_api_key: str):
        """Initialize the pipeline with necessary API keys"""
        self.pdf_processor = PDFProcessor(llama_api_key)
        self.llm_processor = LLMProcessor(openai_api_key)

    def process_directory(self, input_dir: Path, output_file: Path) -> None:
        """Process all PDFs in a directory and save Q&A pairs to CSV"""
        # Process PDFs
        print("Processing PDF files...")
        documents = self.pdf_processor.process_directory(input_dir)

        # Generate Q&A pairs
        print("Generating Q&A pairs...")
        processed_documents = []
        for doc in tqdm(documents):
            try:
                processed_doc = self.llm_processor.process_document(doc)
                processed_documents.append(processed_doc)
            except Exception as e:
                print(f"Failed to process document {doc.file_name}: {str(e)}")
                continue

        # Convert to DataFrame
        print("Converting to DataFrame...")
        qa_data = []
        for doc in processed_documents:
            for qa in doc.qa_pairs:
                qa_data.append(
                    {
                        "file_name": doc.file_name,
                        "question": qa.question,
                        "answer": qa.answer,
                        "context": qa.context,
                        "source_page": qa.source_page,
                        "processed_at": doc.processed_at,
                        **qa.metadata,
                    }
                )

        # Save to CSV
        print(f"Saving to {output_file}...")
        df = pd.DataFrame(qa_data)
        df.to_csv(output_file, index=False)
        print(f"Successfully processed {len(processed_documents)} documents")
        print(f"Generated {len(qa_data)} Q&A pairs")

from pathlib import Path
from typing import List
from ...parser.pdf_processor import PDFProcessor
from ...models.base import Document
from ...llm.processor import LLMProcessor


class DomainKnowledgeGenerator:
    def __init__(self, llama_api_key: str, openai_api_key: str):
        """Initialize the domain knowledge generator"""
        self.pdf_processor = PDFProcessor(llama_api_key)
        self.llm_processor = LLMProcessor(openai_api_key)

    def generate_qa_pairs(self, input_dir: Path, output_file: Path) -> None:
        """Generate Q&A pairs from domain knowledge documents"""
        # Process PDFs
        print("Processing PDF files...")
        documents = self.pdf_processor.process_directory(input_dir)

        # Generate Q&A pairs
        print("Generating Q&A pairs...")
        processed_documents = []
        for doc in documents:
            try:
                processed_doc = self.llm_processor.process_document(doc)
                processed_documents.append(processed_doc)
            except Exception as e:
                print(f"Failed to process document {doc.file_name}: {str(e)}")
                continue

        # Save to CSV
        print(f"Saving to {output_file}...")
        self._save_to_csv(processed_documents, output_file)
        print(f"Successfully processed {len(processed_documents)} documents")

    def _save_to_csv(self, documents: List[Document], output_file: Path) -> None:
        """Save processed documents to CSV"""
        import pandas as pd

        qa_data = []
        for doc in documents:
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

        df = pd.DataFrame(qa_data)
        df.to_csv(output_file, index=False)

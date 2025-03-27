from pathlib import Path
from typing import List
from ...parser.pdf_processor import PDFProcessor
from ...models.base import Document
from ...llm.processor import LLMProcessor


class DownstreamTaskGenerator:
    def __init__(self, llama_api_key: str, openai_api_key: str):
        """Initialize the downstream task generator"""
        self.pdf_processor = PDFProcessor(llama_api_key)
        self.llm_processor = LLMProcessor(openai_api_key)

    def generate_training_data(
        self, input_dir: Path, output_file: Path, task_type: str = "qa"
    ) -> None:
        """Generate training data for downstream tasks"""
        # Process PDFs
        print("Processing PDF files...")
        documents = self.pdf_processor.process_directory(input_dir)

        # Generate task-specific data
        print(f"Generating {task_type} data...")
        processed_documents = []
        for doc in documents:
            try:
                processed_doc = self.llm_processor.process_document(
                    doc, task_type=task_type
                )
                processed_documents.append(processed_doc)
            except Exception as e:
                print(f"Failed to process document {doc.file_name}: {str(e)}")
                continue

        # Save to CSV
        print(f"Saving to {output_file}...")
        self._save_to_csv(processed_documents, output_file, task_type)
        print(f"Successfully processed {len(processed_documents)} documents")

    def _save_to_csv(
        self, documents: List[Document], output_file: Path, task_type: str
    ) -> None:
        """Save processed documents to CSV"""
        import pandas as pd

        data = []
        for doc in documents:
            if task_type == "qa":
                for qa in doc.qa_pairs:
                    data.append(
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
            elif task_type == "summarization":
                # Add summarization-specific data structure
                pass
            elif task_type == "classification":
                # Add classification-specific data structure
                pass

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

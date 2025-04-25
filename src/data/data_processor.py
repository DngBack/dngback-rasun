from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
        padding: str = 'max_length',
        truncation: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.padding = padding
        self.truncation = truncation

    def prepare_data(
        self,
        input_path: str,
        output_path: str,
        text_column: str = 'text',
        label_column: Optional[str] = None,
    ) -> None:
        """Prepare data for fine-tuning.

        Args:
            input_path: Path to input data file (CSV or JSON)
            output_path: Path to save processed data
            text_column: Name of the text column
            label_column: Name of the label column (if any)
        """
        logger.info(f'Processing data from {input_path}')

        # Read data
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.json'):
            df = pd.read_json(input_path)
        else:
            raise ValueError('Unsupported file format. Use CSV or JSON.')

        # Create dataset
        dataset = Dataset.from_pandas(df)

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_seq_length,
                return_tensors='pt',
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        # Save dataset
        tokenized_dataset.save_to_disk(output_path)
        logger.info(f'Saved processed dataset to {output_path}')

    def load_dataset(self, data_path: str) -> Dataset:
        """Load processed dataset.

        Args:
            data_path: Path to processed dataset

        Returns:
            Loaded dataset
        """
        return Dataset.load_from_disk(data_path)

    def process_csv_for_training(
        self,
        input_path: str,
        output_path: str,
        prompt_template: str = 'Question: {question}\nAnswer: {answer}',
        question_column: str = 'question',
        answer_column: str = 'answer',
    ) -> None:
        """Process CSV data for training with custom prompt template.

        Args:
            input_path: Path to input CSV file
            output_path: Path to save processed data
            prompt_template: Template for formatting questions and answers
            question_column: Name of the question column
            answer_column: Name of the answer column
        """
        logger.info(f'Processing CSV data from {input_path}')

        # Read CSV
        df = pd.read_csv(input_path)

        # Format data with prompt template
        formatted_data = []
        for _, row in df.iterrows():
            prompt = prompt_template.format(
                question=row[question_column], answer=row[answer_column],
            )
            formatted_data.append({'text': prompt})

        # Create dataset
        dataset = Dataset.from_list(formatted_data)

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_seq_length,
                return_tensors='pt',
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        # Save dataset
        tokenized_dataset.save_to_disk(output_path)
        logger.info(f'Saved processed dataset to {output_path}')

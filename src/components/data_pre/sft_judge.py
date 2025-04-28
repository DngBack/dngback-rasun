from __future__ import annotations

from typing import Optional

import pandas as pd
from datasets import Dataset
from src.base.base_data import BaseDataModule
from src.components.prompts.judge import PROMPT_WITH_CONTEXT
from src.components.prompts.judge import PROMPT_WITHOUT_CONTEXT
from transformers import AutoTokenizer


class SFTJudge(BaseDataModule):
    def __init__(
        self,
        prompt_without_context: str = PROMPT_WITHOUT_CONTEXT,
        prompt_with_context: str = PROMPT_WITH_CONTEXT,
    ):
        self.prompt_without_context = prompt_without_context
        self.prompt_with_context = prompt_with_context

    def process(
        self,
        file_path: Optional[str] = None,
        mode: str = 'with_context',
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        if not file_path:
            raise ValueError('file_path is required for processing')
        dataset = self.get_data(file_path)
        dataset = dataset.map(
            self.formatting_prompts_func,
            batched=True,
            fn_kwargs={'mode': mode, 'tokenizer': tokenizer},
        )
        return dataset

    def get_data(self, file_path: Optional[str] = None) -> Dataset:
        """Get data from raw csv file

        Args:
            file_path (str, optional): Path to the data file. If None, uses the instance's file_path.

        Returns:
            Dataset: The processed dataset
        """
        input_path = file_path
        if not input_path:
            raise ValueError(
                'No file path provided. Either pass file_path to get_data() or set it in __init__',
            )

        df = pd.read_csv(input_path)

        df = df.rename(
            columns={
                'query': 'instruction',
                'chunk': 'input',
                'response': 'output',
                'check_response': 'label',
            },
        )

        dataset = Dataset.from_pandas(df)

        return dataset

    def formatting_prompts_func(
        self,
        examples: dict,
        mode: str = 'with_context',
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> dict:
        instructions = examples['instruction']
        inputs = examples['input']
        outputs = examples['output']
        labels = examples['label']
        texts = []
        if mode == 'with_context':
            PROMPT = self.prompt_with_context
            for instruction, input_text, output, label in zip(
                instructions,
                inputs,
                outputs,
                labels,
            ):
                text = (
                    PROMPT.format(
                        instruction=instruction,
                        input=input_text,
                        output=output,
                        label=label,
                    )
                    + tokenizer.eos_token
                )
                texts.append(text)
        else:
            PROMPT = self.prompt_without_context
            for instruction, input_text, output, label in zip(
                instructions,
                inputs,
                outputs,
                labels,
            ):
                text = (
                    PROMPT.format(
                        instruction=instruction,
                        # input=input_text,
                        output=output,
                        label=label,
                    )
                    + tokenizer.eos_token
                )
                texts.append(text)
        return {
            'text': texts,
        }

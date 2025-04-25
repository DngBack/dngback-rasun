from __future__ import annotations

import logging
from typing import Any
from typing import List
from typing import Optional

import torch
import wandb
from datasets import Dataset
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from transformers import Trainer
from transformers import TrainingArguments

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        output_dir: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        learning_rate: float = 2e-5,
        max_seq_length: int = 512,
        use_wandb: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.use_wandb = use_wandb

    def train(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
    ) -> None:
        """Train the model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
        """
        if self.use_wandb:
            wandb.init(project='sunbot-finetuning')

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            learning_rate=self.learning_rate,
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=10,
            save_strategy='epoch',
            evaluation_strategy='epoch' if eval_dataset else 'no',
            load_best_model_at_end=True if eval_dataset else False,
            report_to=['wandb'] if self.use_wandb else None,
            remove_unused_columns=True,
            label_names=['labels'],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        logger.info('Starting training')
        trainer.train()
        logger.info('Training completed')

        if self.use_wandb:
            wandb.finish()

    def evaluate(self, test_dataset: Any) -> dict:
        """Evaluate the model.

        Args:
            test_dataset: Test dataset

        Returns:
            Dictionary containing evaluation metrics
        """
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=self.output_dir,
                per_device_eval_batch_size=self.per_device_train_batch_size,
                remove_unused_columns=True,
                label_names=['labels'],
            ),
        )

        logger.info('Starting evaluation')
        metrics = trainer.evaluate(test_dataset)
        logger.info(f'Evaluation results: {metrics}')

        return metrics

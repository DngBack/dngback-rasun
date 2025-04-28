from __future__ import annotations

from typing import Any

from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported


class SFTJudge:
    def __init__(self) -> None:
        self.trainer = None

    def set_trainer(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        max_seq_length: int = 2048,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 5,
        max_steps: int = 60,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        seed: int = 3407,
        output_dir: str = 'outputs',
        report_to: str = 'none',
    ) -> None:
        self.trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field='text',
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.
            args=TrainingArguments(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
                learning_rate=learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim='adamw_torch_fused',  # Use "adamw_8bit" only if using bitsandbytes
                weight_decay=weight_decay,
                lr_scheduler_type='linear',
                seed=seed,
                output_dir=output_dir,
                report_to=report_to,
            ),
        )

    def train(self) -> None:
        """Start the training process."""
        if self.trainer is None:
            raise ValueError('Trainer not set. Call set_trainer() first.')
        self.trainer.train()  # type: ignore

    def save_model(self, path: str) -> None:
        """Save the trained model."""
        if self.trainer is None:
            raise ValueError('Trainer not set. Call set_trainer() first.')
        self.trainer.save_model(path)

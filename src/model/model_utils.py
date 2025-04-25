from __future__ import annotations

import logging
from typing import Optional

import torch
from peft import get_peft_model
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class ModelUtils:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer.

        Returns:
            Tuple containing model and tokenizer
        """
        logger.info(f'Loading model from {self.model_name_or_path}')

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
        )

        return self.model, self.tokenizer

    def prepare_model_for_training(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
    ) -> PreTrainedModel:
        """Prepare model for fine-tuning with LoRA.

        Args:
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            target_modules: List of modules to fine-tune

        Returns:
            Prepared model
        """
        if self.model is None:
            raise ValueError(
                'Model not loaded. Call load_model_and_tokenizer() first.',
            )

        if target_modules is None:
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias='none',
            task_type='CAUSAL_LM',
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        return self.model

    def save_model(self, output_dir: str) -> None:
        """Save fine-tuned model.

        Args:
            output_dir: Directory to save model
        """
        if self.model is None:
            raise ValueError(
                'Model not loaded. Call load_model_and_tokenizer() first.',
            )

        logger.info(f'Saving model to {output_dir}')
        self.model.save_pretrained(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

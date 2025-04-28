from __future__ import annotations

from typing import List
from typing import Optional

from unsloth import FastLanguageModel


class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(
        self,
        model_name: str,
        max_length: int = 2048,
        dtype: Optional[str] = None,
        load_in_4bit: bool = False,
    ) -> None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_length=max_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        self.model = model
        self.tokenizer = tokenizer

    def get_peft_model(
        self,
        r: int = 16,
        lora_alpha: int = 16,
        bias: str = 'none',
        target_modules: Optional[List[str]] = [
            'q_proj',
            'k_proj',
            'v_proj',
            'o_proj',
            'gate_proj',
            'up_proj',
            'down_proj',
        ],
        use_gradient_checkpointing: bool = True,
        random_start: int = 3407,
        max_seq_length: int = 2048,
        use_rslora: bool = False,
        modules_to_save: Optional[List[str]] = None,
        init_lora_weights: bool = True,
        loftq_config: dict = {},
    ) -> None:
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            lora_alpha=lora_alpha,
            bias=bias,
            target_modules=target_modules,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_start=random_start,
            max_seq_length=max_seq_length,
            use_rslora=use_rslora,
            # modules_to_save=modules_to_save,
            # init_lora_weights=init_lora_weights,
            loftq_config=loftq_config,
        )

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

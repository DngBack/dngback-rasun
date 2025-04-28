from __future__ import annotations

from unsloth import FastLanguageModel


class Inference:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def get_model(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_infer(self, prompt):
        # Enable native 2x faster inference
        FastLanguageModel.for_inference(self.model)

    def get_inputs(self, prompt_base: str, input_dict: dict):
        if self.tokenizer is None:
            raise ValueError(
                'Tokenizer not initialized. Call get_model() first.',
            )
        prompt = prompt_base.format(**input_dict)
        inputs = self.tokenizer(
            [prompt],
            return_tensors='pt',
        ).to('cuda')
        return inputs

    def infer(self, inputs):
        if self.model is None or self.tokenizer is None:
            raise ValueError(
                'Model or tokenizer not initialized. Call get_model() first.',
            )
        outputs = self.model.generate(
            **inputs, max_new_tokens=2048, use_cache=True,
        )
        return self.tokenizer.batch_decode(outputs)[0]

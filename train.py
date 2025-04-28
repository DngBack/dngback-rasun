from __future__ import annotations

from src.pipeline.judge_ft import PipelineJudgeFT

pipeline = PipelineJudgeFT()

data_process = pipeline.create_data_module()
model_manager = pipeline.create_model_manager()
trainer_module = pipeline.create_trainer_module()

# Get model and tokenizer
model_manager.load_model(model_name='unsloth/Qwen2.5-7B')
model_manager.get_peft_model()
model = model_manager.get_model()
tokenizer = model_manager.get_tokenizer()

# Get data
dataset = data_process.process(
    file_path='dataset/sample/train_data.csv',
    mode='with_context',
    tokenizer=tokenizer,
)

# Get trainer
trainer_module.set_trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=2048,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
)

# Train
trainer_module.train()

# Save Model
trainer_module.save_model(path='rasun_v1/')

# push to hub
# trainer_module.push_to_hub(repo_id="rasun-v1", commit_message="End of training")

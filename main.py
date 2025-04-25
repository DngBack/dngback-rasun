from __future__ import annotations

from pathlib import Path

import typer
from src.data.data_processor import DataProcessor
from src.model.model_utils import ModelUtils
from src.training.trainer import ModelTrainer
from src.utils.logging import setup_logging

app = typer.Typer()


@app.command()
def prepare_data(
    input_path: str = typer.Option(..., help='Path to input data'),
    output_path: str = typer.Option(..., help='Path to save processed data'),
    model_name_or_path: str = typer.Option(..., help='Path to base model'),
    max_seq_length: int = typer.Option(512, help='Maximum sequence length'),
):
    """Prepare data for fine-tuning."""
    setup_logging()

    # Load model and tokenizer
    model_utils = ModelUtils(model_name_or_path)
    model, tokenizer = model_utils.load_model_and_tokenizer()

    # Process data
    data_processor = DataProcessor(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    data_processor.prepare_data(input_path, output_path)


@app.command()
def train(
    model_name_or_path: str = typer.Option(..., help='Path to base model'),
    data_path: str = typer.Option(..., help='Path to processed data'),
    output_dir: str = typer.Option(..., help='Directory to save model'),
    num_train_epochs: int = typer.Option(3, help='Number of training epochs'),
    per_device_train_batch_size: int = typer.Option(
        4, help='Batch size per device',
    ),
    learning_rate: float = typer.Option(2e-5, help='Learning rate'),
    max_seq_length: int = typer.Option(512, help='Maximum sequence length'),
    use_wandb: bool = typer.Option(True, help='Use Weights & Biases'),
):
    """Fine-tune the model."""
    setup_logging()

    # Load model and tokenizer
    model_utils = ModelUtils(model_name_or_path)
    model, tokenizer = model_utils.load_model_and_tokenizer()

    # Prepare model for training
    model = model_utils.prepare_model_for_training()

    # Load dataset
    data_processor = DataProcessor(tokenizer=tokenizer)
    train_dataset = data_processor.load_dataset(data_path)

    # Train
    trainer = ModelTrainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        use_wandb=use_wandb,
    )
    trainer.train(train_dataset)

    # Save model
    model_utils.save_model(output_dir)


@app.command()
def evaluate(
    model_path: str = typer.Option(..., help='Path to fine-tuned model'),
    test_data_path: str = typer.Option(..., help='Path to test data'),
    output_dir: str = typer.Option(
        'evaluation', help='Directory to save evaluation results',
    ),
):
    """Evaluate the fine-tuned model."""
    setup_logging()

    # Load model and tokenizer
    model_utils = ModelUtils(model_path)
    model, tokenizer = model_utils.load_model_and_tokenizer()

    # Load dataset
    data_processor = DataProcessor(tokenizer=tokenizer)
    test_dataset = data_processor.load_dataset(test_data_path)

    # Evaluate
    trainer = ModelTrainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
    )
    metrics = trainer.evaluate(test_dataset)

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{output_dir}/metrics.txt', 'w') as f:
        for key, value in metrics.items():
            f.write(f'{key}: {value}\n')


if __name__ == '__main__':
    app()

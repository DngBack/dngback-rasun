# Training Pipeline

This repository contains a training pipeline for fine-tuning language models using the Unsloth framework.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Install Guild AI:

```bash
pip install guildai
```

## Training with Guild

Guild AI is used for experiment tracking and hyperparameter optimization. Here's how to use it:

1. Initialize Guild in your project:

```bash
guild init
```

2. Run the training pipeline with default parameters:

```bash
guild run src/pipeline/train.py
```

3. Run with custom hyperparameters:

```bash
guild run src/pipeline/train.py \
    num_epochs=5 \
    batch_size=8 \
    learning_rate=1e-4
```

4. View training runs:

```bash
guild runs
```

5. Compare different runs:

```bash
guild compare
```

## Hyperparameter Optimization

To perform hyperparameter optimization:

1. Create a hyperparameter search configuration:

```bash
guild run src/pipeline/train.py \
    --optimizer random \
    --max-trials 10 \
    --objective minimize loss \
    num_epochs=[3,5,7] \
    batch_size=[4,8,16] \
    learning_rate=[1e-4,2e-4,5e-4]
```

2. Monitor the optimization progress:

```bash
guild runs
```

## Data

The training data is located at:

- `dataset/sample/evaluated_Data250421_Full.csv`

## Model Output

Trained models are saved to:

- `outputs/trained_model/`

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Unsloth
- Guild AI
- Pandas
- Datasets

## Notes

- The training script uses the Qwen1.5-7B-Chat model by default
- Training is performed using 4-bit quantization for memory efficiency
- The maximum sequence length is set to 2048 tokens
- Default training parameters can be modified in the script or through Guild commands

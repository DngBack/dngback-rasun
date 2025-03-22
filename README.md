# dngback-rasun

Test finetuning LLMs with private dataset

## Description

dngback-rasun is an experimental project focused on fine-tuning large language models (LLMs) using private datasets. The goal is to enhance domain-specific knowledge and improve performance on specialized tasks.

This project utilizes models from the Qwen or LLaMA families—powerful LLMs capable of adapting to various tasks through fine-tuning.

Key objectives of the project include:

- Domain Knowledge Fine-tuning: Training the model with domain-specific data to improve expertise in a particular field.

- Downstream Task Learning: Optimizing the model for specific tasks such as question answering, text generation, or classification.

## To Do List

- [ ] Domain Knowledge Finetuning
- [ ] Downstream Task Learning

## Install and run

### Set up

#### Set up env

Using python 3.10.16

```bash
conda create -n rasun python==3.10.16 -y

conda activate rasun
```

#### Set up pre-commit to format code

    - Install:
    ```bash
    pip install pre-commit
    ```

    - Add pre-commit to git hook:
    ```bash
    pre-commit install
    ```

    - Run pre-commit for formating code (only staged files in git):
    ```bash
    pre-commit run
    ```

    - Run pre-commit for formating code with all files:
    ```bash
    pre-commit run --all-files

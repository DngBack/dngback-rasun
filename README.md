# dngback-rasun

Test finetuning LLMs with private dataset

## Description

## To Do List

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

# SunBot Fine-tuning Pipeline

Pipeline để fine-tuning mô hình ngôn ngữ lớn (LLM) dựa trên SunBot.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc thư mục

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_processor.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── model_utils.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/
│       ├── __init__.py
│       └── logging.py
└── main.py
```

## Sử dụng

### 1. Chuẩn bị dữ liệu

```bash
python main.py prepare-data --input_path path/to/raw/data --output_path path/to/processed/data
```

### 2. Fine-tuning

```bash
python main.py train \
    --model_name_or_path path/to/base/model \
    --data_path path/to/processed/data \
    --output_dir path/to/save/model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --max_seq_length 512
```

### 3. Đánh giá

```bash
python main.py evaluate \
    --model_path path/to/finetuned/model \
    --test_data_path path/to/test/data
```

## Các tham số chính

- `--model_name_or_path`: Đường dẫn đến mô hình cơ sở
- `--data_path`: Đường dẫn đến dữ liệu đã xử lý
- `--output_dir`: Thư mục lưu mô hình đã fine-tune
- `--num_train_epochs`: Số epoch huấn luyện
- `--per_device_train_batch_size`: Batch size cho mỗi device
- `--learning_rate`: Learning rate
- `--max_seq_length`: Độ dài tối đa của sequence

## Lưu ý

- Đảm bảo có đủ GPU memory cho việc fine-tuning
- Sử dụng wandb để theo dõi quá trình huấn luyện
- Có thể điều chỉnh các hyperparameter trong file config

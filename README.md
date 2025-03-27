# PDF to Q&A Pipeline

This project provides a pipeline to convert PDF documents into question-answer pairs using LlamaParser for PDF processing and OpenAI's GPT models for Q&A generation.

## Features

- PDF document processing using LlamaParser
- Question-answer pair generation using OpenAI's GPT models
- Structured data output in CSV format
- Pydantic models for data validation
- Progress tracking with tqdm
- Error handling and logging

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd pdf-to-qa-pipeline
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root with your API keys:

```
LLAMA_API_KEY=your_llama_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Place your PDF files in the `input_pdfs` directory.

2. Run the pipeline:

```bash
python -m src.main
```

The pipeline will:

- Process all PDF files in the input directory
- Generate question-answer pairs using GPT
- Save the results to `output/qa_pairs.csv`

## Output Format

The generated CSV file contains the following columns:

- `file_name`: Name of the source PDF file
- `question`: Generated question
- `answer`: Generated answer
- `context`: Optional context from the source document
- `source_page`: Page number where the content was found
- `processed_at`: Timestamp of processing
- Additional metadata fields

## Project Structure

```
pdf-to-qa-pipeline/
├── src/
│   ├── __init__.py
│   ├── models.py          # Pydantic models
│   ├── pdf_processor.py   # PDF processing with LlamaParser
│   ├── llm_processor.py   # Q&A generation with OpenAI
│   ├── pipeline.py        # Main pipeline orchestration
│   └── main.py           # Entry point
├── input_pdfs/           # Place PDF files here
├── output/              # Generated CSV files
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Error Handling

The pipeline includes robust error handling:

- Failed PDF processing is logged but doesn't stop the pipeline
- Failed Q&A generation for individual documents is logged but continues processing
- Missing API keys are caught early with clear error messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

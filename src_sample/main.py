import os
from pathlib import Path
from dotenv import load_dotenv
from datagen.domain_knowledge.generator import DomainKnowledgeGenerator
from datagen.downstream.generator import DownstreamTaskGenerator


def main():
    # Load environment variables
    load_dotenv()

    # Get API keys
    llama_api_key = os.getenv("LLAMA_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not llama_api_key or not openai_api_key:
        raise ValueError(
            "Please set LLAMA_API_KEY and OPENAI_API_KEY in your .env file"
        )

    # Set up paths
    input_dir = Path("input_pdfs")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate domain knowledge data
    print("\nGenerating domain knowledge data...")
    domain_gen = DomainKnowledgeGenerator(llama_api_key, openai_api_key)
    domain_gen.generate_qa_pairs(
        input_dir=input_dir, output_file=output_dir / "domain_knowledge_qa.csv"
    )

    # Generate downstream task data
    print("\nGenerating downstream task data...")
    downstream_gen = DownstreamTaskGenerator(llama_api_key, openai_api_key)
    downstream_gen.generate_training_data(
        input_dir=input_dir,
        output_file=output_dir / "downstream_qa.csv",
        task_type="qa",
    )


if __name__ == "__main__":
    main()

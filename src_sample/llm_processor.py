from typing import List, Dict, Any
from openai import OpenAI
from .models import Document, QAPair


class LLMProcessor:
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        """Initialize the LLM processor with OpenAI API key"""
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_qa_pairs(self, text: str, num_pairs: int = 5) -> List[QAPair]:
        """Generate question-answer pairs from text content"""
        prompt = f"""Given the following text, generate {num_pairs} high-quality question-answer pairs.
        The questions should be clear, specific, and test understanding of the content.
        The answers should be concise but complete.
        
        Text:
        {text}
        
        Format each pair as:
        Q: [question]
        A: [answer]
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that creates high-quality question-answer pairs from text content.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )

            # Parse the response into QAPair objects
            qa_pairs = []
            content = response.choices[0].message.content

            # Split the content into Q&A pairs
            pairs = content.split("\n\n")
            for pair in pairs:
                if "Q:" in pair and "A:" in pair:
                    question = pair.split("Q:")[1].split("A:")[0].strip()
                    answer = pair.split("A:")[1].strip()
                    qa_pairs.append(QAPair(question=question, answer=answer))

            return qa_pairs

        except Exception as e:
            raise Exception(f"Error generating Q&A pairs: {str(e)}")

    def process_document(self, document: Document) -> Document:
        """Process a document to generate Q&A pairs"""
        # Extract text content from the document
        text_content = document.processing_metadata.get("text_content", "")

        # Generate Q&A pairs
        qa_pairs = self.generate_qa_pairs(text_content)

        # Update document with Q&A pairs
        document.qa_pairs = qa_pairs
        return document

from __future__ import annotations

from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding


class LlamaIndexChunker:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _embedding_client(self) -> OpenAIEmbedding:
        return OpenAIEmbedding(api_key=self.api_key)

    def _splitter(self) -> SemanticSplitterNodeParser:
        return SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self._embedding_client(),
        )

    def chunk_text(self, text: str) -> List[str]:
        nodes = self._splitter().get_nodes_from_documents(
            documents=[Document(text=text)],
        )
        return [node.get_content() for node in nodes]

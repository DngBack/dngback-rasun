from __future__ import annotations

import os

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

os.environ['OPENAI_API_KEY'] = 'sk-...'

embed_model = OpenAIEmbedding()
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model,
)

# also baseline splitter
base_splitter = SentenceSplitter(chunk_size=512)

embed_model = OpenAIEmbedding()
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model,
)

# also baseline splitter
base_splitter = SentenceSplitter(chunk_size=512)

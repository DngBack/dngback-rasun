from __future__ import annotations

from typing import List
from typing import Mapping
from typing import Optional

import nest_asyncio
from llama_cloud_services.parse import LlamaParse
from llama_cloud_services.parse.utils import ResultType
from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from pydantic import BaseModel
from pydantic import Field
from src.base import BaseParse
from src.base import ParseInput
from src.base import ParseOutput
# bring in deps from llama_cloud_services
# bring in deps from llama_index

nest_asyncio.apply()


class ChunkConfig(BaseModel):
    """Configuration for text chunking"""

    chunk_size: int = Field(
        default=1000,
        ge=100,
        description='Size of each text chunk in characters',
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description='Number of characters to overlap between chunks',
    )
    separator: str = Field(
        default='\n',
        description='Text separator for chunk boundaries',
    )

    class Config:
        json_schema_extra = {
            'example': {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'separator': '\n',
            },
        }


class LlamaParsePdf(BaseParse):
    """Parser for PDF files using LlamaParse with chunking support"""

    def __init__(
        self,
        api_key: str,
        result_type: ResultType = ResultType.MD,
        chunk_config: Optional[ChunkConfig] = None,
    ):
        self.api_key = api_key
        self.result_type = result_type
        self.chunk_config = chunk_config or ChunkConfig()

    def _parser_client(self) -> LlamaParse:
        """Get the LlamaParse client"""
        return LlamaParse(api_key=self.api_key, result_type=self.result_type)

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on configuration"""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_config.chunk_size
            if end > text_length:
                end = text_length

            # Find the last separator before the chunk size limit
            if end < text_length:
                last_separator = text.rfind(
                    self.chunk_config.separator,
                    start,
                    end,
                )
                if last_separator != -1:
                    end = last_separator

            chunks.append(text[start:end])
            start = end - self.chunk_config.chunk_overlap

        return chunks

    def parse(self, input: ParseInput) -> ParseOutput:
        """Parse the input file and return chunked content"""
        # use SimpleDirectoryReader to parse our file
        file_extractor: Mapping[str, BaseReader] = {
            '.pdf': self._parser_client(),
        }
        documents = SimpleDirectoryReader(
            input_files=[input.file_path],
            file_extractor=file_extractor,  # type: ignore
        ).load_data()

        # Extract text from the first document
        text = documents[0].text

        # Chunk the text if chunking is enabled
        if self.chunk_config:
            chunks = self._chunk_text(text)
            # Join chunks with a special separator for later processing
            processed_text = '\n---CHUNK---\n'.join(chunks)
        else:
            processed_text = text

        return ParseOutput(parsed_data=processed_text)

from __future__ import annotations

from typing import Mapping

import nest_asyncio
from llama_cloud_services.parse import LlamaParse
from llama_cloud_services.parse.utils import ResultType
from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from src.base import BaseParse
from src.base import ParseInput
from src.base import ParseOutput
# bring in deps from llama_cloud_services
# bring in deps from llama_index

nest_asyncio.apply()


class LlamaParsePdf(BaseParse):
    """Parser for PDF files using LlamaParse"""

    def __init__(self, api_key: str, result_type: ResultType = ResultType.MD):
        self.api_key = api_key
        self.result_type = result_type

    def _parser_client(self) -> LlamaParse:
        """Get the LlamaParse client"""
        return LlamaParse(api_key=self.api_key, result_type=self.result_type)

    def parse(self, input: ParseInput) -> ParseOutput:
        """Parse the single input file"""

        # use SimpleDirectoryReader to parse our file
        file_extractor: Mapping[str, BaseReader] = {
            '.pdf': self._parser_client(),
        }
        documents = SimpleDirectoryReader(
            input_files=[input.file_path],
            file_extractor=file_extractor,  # type: ignore
        ).load_data()
        return ParseOutput(parsed_data=documents[0].text)

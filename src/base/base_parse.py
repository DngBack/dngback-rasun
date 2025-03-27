from __future__ import annotations

from abc import abstractmethod
from typing import List
from typing import Union

from ..models import FileType
from .base_model import CustomBaseModel as BaseModel


class ParseInput(BaseModel):
    """Input for parsing"""

    file_path: Union[str, List[str]]
    file_type: FileType


class ParseOutput(BaseModel):
    """Output for parsing"""

    parsed_data: str


class BaseParse:
    """Base class for parsing"""

    @abstractmethod
    def parse(self, input: ParseInput) -> ParseOutput:
        """Parse the input"""
        pass

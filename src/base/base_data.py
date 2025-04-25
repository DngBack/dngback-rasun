from __future__ import annotations

from abc import abstractmethod


class BaseDataModule:
    """Base class for data modules"""

    @abstractmethod
    def process(self):
        """Process data"""
        raise NotImplementedError

    @abstractmethod
    def get_data(self):
        """Get data from raw file"""
        raise NotImplementedError

    @abstractmethod
    def formatted_data(self):
        """Format data for training"""
        raise NotImplementedError

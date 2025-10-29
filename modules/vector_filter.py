"""
Vector-based Log Filter (Mock)

Interface and mock implementation for vector DB log filtering.
Input: issue description, uploaded log file path
Output: mocked filtered logs string
"""

from abc import ABC, abstractmethod
from modules.vector_db import VectorDb


class VectorLogFilter(ABC):
    """Interface for vector DB based log filtering."""

    @abstractmethod
    def filter(self, issue_description: str, log_file_path: str) -> str:
        """Return filtered logs based on vector similarity."""
        raise NotImplementedError


class VectorLogFilterImpl(VectorLogFilter):

    def filter(self, issue_description: str, log_file_path: str) -> str:
        db = VectorDb(input_document_path=log_file_path,
                      output_directory="temp_vector_db")
        results = db.search(issue_description)
        return "\n".join(results)

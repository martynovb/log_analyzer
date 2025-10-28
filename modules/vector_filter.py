"""
Vector-based Log Filter (Mock)

Interface and mock implementation for vector DB log filtering.
Input: issue description, uploaded log file path
Output: mocked filtered logs string
"""

from abc import ABC, abstractmethod
from typing import Optional


class VectorLogFilter(ABC):
    """Interface for vector DB based log filtering."""

    @abstractmethod
    def filter(self, issue_description: str, log_file_path: str) -> str:
        """Return filtered logs based on vector similarity."""
        raise NotImplementedError


class MockVectorLogFilter(VectorLogFilter):
    """Mock implementation that returns a placeholder string."""

    def filter(self, issue_description: str, log_file_path: str) -> str:
        # This is mocked; replace with real vector retrieval later
        header = "=== VECTOR FILTER (MOCK) ===\n"
        details = (
            f"Issue: {issue_description[:120]}\n"
            f"Source Log: {log_file_path}\n"
            "Matches:\n"
            "[Line 1024] >>> Mocked relevant log line matching semantic meaning\n"
            "[Line 2048]     Surrounding context...\n"
        )
        return header + details



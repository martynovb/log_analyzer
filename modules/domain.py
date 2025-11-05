"""
Domain Models

Data classes for the log analysis system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime

from modules.utils import format_time


class FilterMode(str, Enum):
    vector = "vector"
    llm = "llm"

@dataclass
class AnalysisRequest:
    """Data class for analysis request parameters."""
    log_file_path: str
    issue_description: str
    keywords: Optional[List[str]] = None
    filter_mode: FilterMode = FilterMode.llm  # "llm" or "vector"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    max_tokens: int = 3500
    context_lines: int = 2
    deduplicate: bool = True
    prioritize_by_severity: bool = False


@dataclass
class AnalysisResult:
    """Data class for analysis results."""
    request: AnalysisRequest
    extracted_keywords: List[str]
    filtered_logs: str
    context_info: Dict[str, Any]
    generated_prompt: str
    llm_analysis: Optional[str] = None
    llm_model: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: int = 0

    def processing_time_formatted(self):
        processing_seconds = self.processing_time_ms / 1000.0
        processing_time_formatted = format_time(processing_seconds)
        return processing_time_formatted


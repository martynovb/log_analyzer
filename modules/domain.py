"""
Domain Models

Data classes for the log analysis system.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class AnalysisRequest:
    """Data class for analysis request parameters."""
    log_file_path: str
    issue_description: str
    keywords: Optional[List[str]] = None
    filter_mode: str = "llm"  # "llm" or "vector"
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
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: int = 0


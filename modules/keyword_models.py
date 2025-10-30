from dataclasses import dataclass
from enum import Enum


class KeywordType(Enum):
    """Types of keywords that can be extracted."""
    ERROR = "error"
    TECHNICAL = "technical"
    COMPONENT = "component"
    ACTION = "action"
    FUNCTIONAL = "functional"


@dataclass
class ExtractedKeyword:
    """Represents an extracted keyword with metadata."""
    keyword: str
    keyword_type: KeywordType
    confidence: float
    context: str
    extraction_method: str
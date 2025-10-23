# modules/keyword_extractor.py
import json
import logging
import re
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


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


class LLMInterface(ABC):
    """Abstract interface for LLM integration."""
    
    @abstractmethod
    def extract_keywords(self, issue_description: str) -> List[ExtractedKeyword]:
        """Extract keywords from issue description."""
        pass
    
    @abstractmethod
    def analyze_logs(self, prompt: str) -> str:
        """Analyze logs using the provided prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM is available."""
        pass


class LocalLLMInterface(LLMInterface):
    """Interface for local LLM integration with hardcoded LM Studio settings."""

    def __init__(self,
                 base_url: str = "http://127.0.0.1:1234",
                 model: str = "qwen2.5-coder-32b-instruct",
                 timeout: int = 120,
                 max_tokens: int = 1000):
        """
        Initialize local LLM interface with hardcoded settings.
        
        Args:
            base_url: Base URL of the local LLM server (hardcoded for LM Studio)
            model: Model name to use (hardcoded for your setup)
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens for response
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)

        # Test connection on initialization
        self._test_connection()

    def _test_connection(self) -> bool:
        """Test connection to local LLM."""
        try:
            # Try OpenAI-compatible endpoint first (for LM Studio)
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                self.logger.info(f"Connected to local LLM (OpenAI-compatible) at {self.base_url}")
                return True
        except Exception as e:
            self.logger.debug(f"OpenAI-compatible endpoint test failed: {e}")

        try:
            # Fallback to Ollama-style endpoint
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.logger.info(f"Connected to local LLM (Ollama-compatible) at {self.base_url}")
                return True
        except Exception as e:
            self.logger.debug(f"Ollama-compatible endpoint test failed: {e}")

        self.logger.warning(f"Could not connect to local LLM at {self.base_url}")
        return False

    def is_available(self) -> bool:
        """Check if LLM is available."""
        try:
            # Try OpenAI-compatible endpoint first (for LM Studio)
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass

        try:
            # Fallback to Ollama-style endpoint
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def analyze_logs(self, prompt: str) -> str:
        """Analyze logs using the provided prompt."""
        try:
            response = self._make_llm_request(prompt)
            return response
        except Exception as e:
            self.logger.error(f"LLM log analysis failed: {e}")
            raise Exception(f"Log analysis failed: {e}")

    def extract_keywords(self, issue_description: str) -> List[ExtractedKeyword]:
        """Extract keywords using LLM."""
        prompt = self._create_keyword_extraction_prompt(issue_description)
        
        try:
            response = self._make_llm_request(prompt)
            return self._parse_llm_response(response, issue_description)
        except Exception as e:
            self.logger.error(f"LLM keyword extraction failed: {e}")
            raise Exception(f"LLM extraction failed: {e}")

    def _create_keyword_extraction_prompt(self, issue_description: str) -> str:
        """Create a prompt for keyword extraction."""
        return f"""Analyze the following issue description and extract relevant keywords for log analysis.

Issue Description: "{issue_description}"

Please extract keywords that would be useful for filtering log files to find relevant entries. Focus on:
- Error types and exception names
- Technical components and services
- Actions and operations
- Performance-related terms
- System components

Return the result as a JSON object with this exact format:
{{
    "keywords": [
        {{
            "keyword": "keyword_text",
            "type": "error|technical|component|action|functional",
            "confidence": 0.8
        }}
    ]
}}

Extract 5-10 most relevant keywords. Be specific and technical."""

    def _make_llm_request(self, prompt: str) -> str:
        """Make request to local LLM."""
        # Try OpenAI-compatible endpoint first (for LM Studio)
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            # Fallback to Ollama-style endpoint
            self.logger.warning(f"OpenAI-compatible endpoint failed: {e}, trying Ollama endpoint")

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": self.max_tokens
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                raise Exception(f"LLM request failed with status {response.status_code}")

            return response.json()["response"]

    def _parse_llm_response(self, response: str, original_text: str) -> List[ExtractedKeyword]:
        """Parse LLM response into ExtractedKeyword objects."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            extracted_keywords = []
            for item in data.get("keywords", []):
                keyword_type = self._map_keyword_type(item.get("type", "technical"))
                confidence = float(item.get("confidence", 0.7))

                extracted_keywords.append(ExtractedKeyword(
                    keyword=item["keyword"],
                    keyword_type=keyword_type,
                    confidence=confidence,
                    context=self._extract_context(original_text, item["keyword"]),
                    extraction_method="llm"
                ))

            return extracted_keywords

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            # Fallback: extract keywords from response text
            return self._fallback_keyword_extraction(response, original_text)

    def _map_keyword_type(self, llm_type: str) -> KeywordType:
        """Map LLM keyword type to our KeywordType enum."""
        type_mapping = {
            "technical": KeywordType.TECHNICAL,
            "error": KeywordType.ERROR,
            "component": KeywordType.COMPONENT,
            "action": KeywordType.ACTION,
            "functional": KeywordType.FUNCTIONAL
        }
        return type_mapping.get(llm_type.lower(), KeywordType.TECHNICAL)

    def _extract_context(self, text: str, keyword: str, context_window: int = 50) -> str:
        """Extract context around a keyword."""
        keyword_lower = keyword.lower()
        text_lower = text.lower()

        index = text_lower.find(keyword_lower)
        if index == -1:
            return ""

        start = max(0, index - context_window)
        end = min(len(text), index + len(keyword) + context_window)

        return text[start:end].strip()

    def _fallback_keyword_extraction(self, response: str, original_text: str) -> List[ExtractedKeyword]:
        """Fallback keyword extraction from response text."""
        # Simple fallback: extract words that appear in both response and original text
        response_words = set(re.findall(r'\b\w+\b', response.lower()))
        original_words = set(re.findall(r'\b\w+\b', original_text.lower()))

        common_words = response_words.intersection(original_words)
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        keywords = [word for word in common_words if word not in stop_words and len(word) > 3]

        extracted_keywords = []
        for keyword in keywords[:5]:  # Limit to 5 keywords
            extracted_keywords.append(ExtractedKeyword(
                keyword=keyword,
                keyword_type=KeywordType.TECHNICAL,
                confidence=0.5,
                context=self._extract_context(original_text, keyword),
                extraction_method="llm_fallback"
            ))

        return extracted_keywords


class MockLLMInterface(LLMInterface):
    """Mock LLM interface for testing purposes."""

    def extract_keywords(self, issue_description: str) -> List[ExtractedKeyword]:
        """Mock keyword extraction."""
        return [
            ExtractedKeyword(
                keyword="error",
                keyword_type=KeywordType.ERROR,
                confidence=0.8,
                context=issue_description[:50],
                extraction_method="mock"
            )
        ]

    def analyze_logs(self, prompt: str) -> str:
        """Mock log analysis."""
        return """# Mock Analysis Results

## Root Cause Analysis
Based on the log analysis, the issue appears to be related to snapshot uploading failures. The logs show multiple upload attempts with varying success rates.

## Error Pattern Identification
- Recurring upload failures during peak usage times
- Network timeout errors
- Server response delays

## Timeline Analysis
The issue started around 11:04:00 and persisted throughout the session with intermittent success.

## Impact Assessment
- Snapshot upload functionality is affected
- User experience degraded
- Data synchronization issues

## Recommended Actions
1. Check network connectivity
2. Verify server capacity
3. Implement retry mechanisms
4. Add better error handling

## Prevention Measures
- Implement circuit breaker pattern
- Add monitoring and alerting
- Optimize upload batch sizes
- Add fallback mechanisms"""

    def is_available(self) -> bool:
        """Mock availability check."""
        return True


class KeywordExtractor:
    """Extracts relevant keywords from issue descriptions using LLM only."""
    
    def __init__(self, llm_interface: Optional[LLMInterface] = None):
        """
        Initialize the keyword extractor.
        
        Args:
            llm_interface: LLM interface to use (if None, will create LocalLLMInterface with hardcoded settings)
        """
        self.logger = logging.getLogger(__name__)
        
        # Setup LLM interface - mandatory
        if llm_interface is None:
            try:
                self.llm_interface = LocalLLMInterface()
            except Exception as e:
                self.logger.error(f"Failed to initialize LocalLLMInterface: {e}")
                raise Exception("LLM is required but not available. Please ensure your LM Studio is running at http://127.0.0.1:1234")
        else:
            self.llm_interface = llm_interface
    
    def extract_keywords(self, issue_description: str) -> List[ExtractedKeyword]:
        """
        Extract keywords from issue description using LLM only.
        
        Args:
            issue_description: The issue description text
            
        Returns:
            List of extracted keywords with metadata
        """
        if not issue_description or not issue_description.strip():
            return []
        
        if not self.llm_interface or not self.llm_interface.is_available():
            raise Exception("LLM is not available. Please ensure your LM Studio is running at http://127.0.0.1:1234")
        
        try:
            self.logger.info("Using LLM for keyword extraction")
            llm_keywords = self.llm_interface.extract_keywords(issue_description)
            self.logger.info(f"LLM extracted {len(llm_keywords)} keywords")
            return llm_keywords
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}")
            raise Exception(f"Keyword extraction failed: {e}")
    
    def is_llm_available(self) -> bool:
        """Check if LLM is available."""
        return self.llm_interface and self.llm_interface.is_available()
    
    def get_extraction_methods_used(self, issue_description: str) -> List[str]:
        """Get list of extraction methods used."""
        return ["llm"] if self.is_llm_available() else []
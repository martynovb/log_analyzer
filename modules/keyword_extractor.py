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
                 model: str = "deepseek/deepseek-r1-0528-qwen3-8b",
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
        """Test connection to local LLM and update model name if needed."""
        try:
            # Try OpenAI-compatible endpoint first (for LM Studio)
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                if "data" in models_data and len(models_data["data"]) > 0:
                    # Use the first available model
                    actual_model = models_data["data"][0]["id"]
                    if actual_model != self.model:
                        self.logger.info(f"Detected model: {actual_model} (default was: {self.model})")
                        self.model = actual_model
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
                # Update model name if we got a valid response
                models_data = response.json()
                if "data" in models_data and len(models_data["data"]) > 0:
                    actual_model = models_data["data"][0]["id"]
                    if actual_model != self.model:
                        self.logger.debug(f"Detected model: {actual_model}, updating from {self.model}")
                        self.model = actual_model
                return True
        except Exception as e:
            self.logger.debug(f"OpenAI-compatible endpoint check failed: {e}")

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
            if not response or not isinstance(response, str):
                raise Exception(f"Invalid response from LLM: {type(response)} - {response}")
            return self._clean_llm_response(response)
        except Exception as e:
            self.logger.error(f"LLM log analysis failed: {e}")
            raise Exception(f"Log analysis failed: {str(e)}")

    def extract_keywords(self, issue_description: str) -> List[ExtractedKeyword]:
        """Extract keywords using LLM."""
        prompt = self._create_keyword_extraction_prompt(issue_description)
        
        try:
            response = self._make_llm_request(prompt)
            keywords = self._parse_llm_response(response, issue_description)
            
            # If no keywords were extracted, raise exception
            if not keywords:
                raise Exception("No keywords extracted by LLM")
            
            return keywords
        except Exception as e:
            self.logger.error(f"LLM keyword extraction failed: {e}")
            raise Exception(f"LLM keyword extraction failed: {e}")

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
            self.logger.debug(f"Making LLM request to {url} with model: {self.model}")
            self.logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(url, json=payload, timeout=self.timeout)
            self.logger.debug(f"Response status: {response.status_code}")
            
            response.raise_for_status()
            result = response.json()
            self.logger.debug(f"OpenAI response structure: {list(result.keys())}")
            
            # Handle OpenAI-compatible response structure
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
                elif "text" in choice:
                    return choice["text"]
                else:
                    self.logger.warning(f"Unexpected choice structure: {choice}")
                    return str(choice)
            else:
                self.logger.warning(f"Unexpected OpenAI response structure: {result}")
                return str(result)
        except requests.exceptions.HTTPError as e:
            # Parse error details from response
            try:
                error_detail = response.json()
                self.logger.error(f"HTTP Error {response.status_code}: {error_detail}")
                raise Exception(f"LLM request failed with status {response.status_code}: {error_detail.get('error', {}).get('message', str(error_detail))}")
            except:
                self.logger.error(f"HTTP Error {response.status_code}: {response.text}")
                raise Exception(f"LLM request failed with status {response.status_code}: {response.text}")
        except Exception as e:
            # If OpenAI endpoint fails, raise the error instead of trying Ollama endpoint
            # This prevents the "Unexpected endpoint or method" error
            self.logger.error(f"OpenAI-compatible endpoint failed: {e}")
            raise Exception(f"LLM request failed: {e}")

    def _clean_llm_response(self, response: str) -> str:
        """Clean LLM response by removing thinking tags and reasoning content."""
        import re
        
        # Remove <think>...</think> tags and their content
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove <think>...</think> tags and their content
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
        
        # Remove any remaining thinking-related tags
        cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<reasoning>.*?</reasoning>', '', cleaned, flags=re.DOTALL)
        
        # Clean up extra whitespace and newlines
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Replace multiple newlines with double newlines
        cleaned = cleaned.strip()
        
        return cleaned

    def _parse_llm_response(self, response: str, original_text: str) -> List[ExtractedKeyword]:
        """Parse LLM response into ExtractedKeyword objects."""
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
            raise ValueError("Issue description cannot be empty")
        
        if not self.llm_interface or not self.llm_interface.is_available():
            raise Exception("LLM is not available. Please ensure your LM Studio is running at http://127.0.0.1:1234")
        
        try:
            self.logger.info("Using LLM for keyword extraction")
            llm_keywords = self.llm_interface.extract_keywords(issue_description)
            self.logger.info(f"LLM extracted {len(llm_keywords)} keywords")
            return llm_keywords
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}")
            raise Exception(f"LLM keyword extraction failed: {e}")
    
    def is_llm_available(self) -> bool:
        """Check if LLM is available."""
        return self.llm_interface and self.llm_interface.is_available()
    
    def get_extraction_methods_used(self, issue_description: str) -> List[str]:
        """Get list of extraction methods used."""
        return ["llm"] if self.is_llm_available() else []
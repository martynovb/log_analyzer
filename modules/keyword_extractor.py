# modules/keyword_extractor.py
import logging
from typing import List, Optional

from modules import LLMInterface, LocalLLMInterface
from modules.keyword_models import ExtractedKeyword


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

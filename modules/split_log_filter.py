"""
Split Log Filter Module

This module implements a chunk-based log filtering approach that splits logs
by max_tokens, processes each chunk separately through LLM, and then synthesizes
the results.
"""

from typing import List, Optional
from dataclasses import dataclass

from modules.log_filter import LogFilterConfig, LogFilter
from modules.llm_interface import LLMInterface
from modules.prompt_generator import PromptGenerator


@dataclass
class SplitLogFilterConfig(LogFilterConfig):
    """Configuration for split log filter."""
    llm_interface: LLMInterface
    issue_description: str
    max_tokens: int
    keywords: List[str] = None
    context_description: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if not self.issue_description:
            raise ValueError("issue_description cannot be empty")
        if self.max_tokens is None or self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not isinstance(self.llm_interface, LLMInterface):
            raise ValueError("llm_interface must be an instance of LLMInterface")
        if self.keywords is None:
            self.keywords = []

    @property
    def max_chars(self) -> int:
        """Calculate maximum characters based on token limit."""
        return self.max_tokens * 4  # Rough estimation: 1 token ≈ 4 characters


class SplitLogFilter(LogFilter):
    """Log filter that splits logs into chunks and processes each chunk separately."""
    
    def __init__(self, config: SplitLogFilterConfig):
        """
        Initialize split log filter with configuration.
        
        Args:
            config: Configuration object containing all parameters
        """
        super().__init__(config)
        self.config = config
        self.chunk_responses: List[str] = []
        self.prompt_generator = PromptGenerator()
    
    def filter(self) -> str:
        """
        Filter logs by splitting into chunks and processing each chunk.
        
        Returns:
            Concatenated responses string (chunk_responses stored as instance variable)
        """
        # Step 1: Filter by date range
        lines = self.filter_by_date()
        
        if not lines:
            self.chunk_responses = []
            return "No log entries found for the given criteria."
        
        print(f"Total log lines after date filtering: {len(lines)}")
        
        # Step 2: Split logs into chunks based on max_tokens
        chunks = self._split_logs_by_tokens(lines, self.config.max_tokens)
        
        print(f"Split logs into {len(chunks)} chunks")
        
        if len(chunks) == 1:
            # Single chunk - process normally
            print("Single chunk detected, processing normally...")
            chunk_logs = '\n'.join(chunks[0])
            prompt = self.prompt_generator.generate_chunk_prompt(
                self.config.issue_description,
                chunk_logs,
                self.config.context_description
            )
            response = self.config.llm_interface.analyze_logs(prompt)
            self.chunk_responses = [response]
            return response
        
        # Step 3: Process each chunk through LLM
        print(f"Processing {len(chunks)} chunks through LLM...")
        chunk_responses = []
        
        for i, chunk_lines in enumerate(chunks, 1):
            print(f"  Processing chunk {i}/{len(chunks)} ({len(chunk_lines)} lines)...")
            chunk_logs = '\n'.join(chunk_lines)
            
            # Create prompt for this chunk
            prompt = self.prompt_generator.generate_chunk_prompt(
                self.config.issue_description,
                chunk_logs,
                self.config.context_description
            )
            
            # Send to LLM
            try:
                response = self.config.llm_interface.analyze_logs(prompt)
                chunk_responses.append(response)
                print(f"  ✓ Chunk {i} processed successfully")
            except Exception as e:
                error_msg = f"Error processing chunk {i}: {str(e)}"
                print(f"  ✗ {error_msg}")
                chunk_responses.append(error_msg)
        
        self.chunk_responses = chunk_responses
        
        # Step 4: Return concatenated responses
        concatenated = "\n\n".join([
            f"=== Chunk {i+1} Analysis ===\n{response}"
            for i, response in enumerate(chunk_responses)
        ])
        
        return concatenated
    
    def get_chunk_responses(self) -> List[str]:
        """
        Get the list of chunk responses.
        
        Returns:
            List of chunk analysis responses
        """
        return self.chunk_responses
    
    def _split_logs_by_tokens(self, lines: List[str], max_tokens: int) -> List[List[str]]:
        """
        Split log lines into chunks based on token limit.
        
        Args:
            lines: List of log lines to split
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of chunks, each chunk is a list of log lines
        """
        max_chars = max_tokens * 4  # Rough estimation: 1 token ≈ 4 characters
        
        chunks = []
        current_chunk = []
        current_chars = 0
        
        for line in lines:
            line_chars = len(line) + 1  # +1 for newline
            
            # If adding this line would exceed the limit, start a new chunk
            if current_chars + line_chars > max_chars and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [line]
                current_chars = line_chars
            else:
                current_chunk.append(line)
                current_chars += line_chars
        
        # Add the last chunk if it has any lines
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


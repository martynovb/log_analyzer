"""
Log Analyzer - Modular Architecture
==================================

This module provides a comprehensive log analysis system with:
- Simple UI for log upload and issue description
- Keyword extraction from issue descriptions
- Log analysis with filtering
- Context retrieval (mocked)
- Prompt generation for AI analysis

Architecture:
- UI Layer: Simple web interface
- Business Logic: Core analysis components
- Data Layer: File handling and storage
- Integration Layer: External service interfaces
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import os
import re
from datetime import datetime

# Import our existing log analyzer components
from modules.log_analyzer import LogAnalyzer, LogFilterConfig
from modules.keyword_extractor import KeywordExtractor, KeywordType, ExtractedKeyword
from modules.context_retriever import ContextRetriever


@dataclass
class AnalysisRequest:
    """Data class for analysis request parameters."""
    log_file_path: str
    issue_description: str
    keywords: Optional[List[str]] = None
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
    llm_analysis: Optional[str] = None  # New field for LLM analysis response
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: int = 0


# KeywordExtractor is now imported from modules.keyword_extractor


class ContextRetrieverInterface(ABC):
    """Abstract interface for context retrieval."""
    
    @abstractmethod
    def retrieve_codebase_context(self, keywords: List[str]) -> Dict[str, Any]:
        """Retrieve relevant codebase context."""
        pass
    
    @abstractmethod
    def retrieve_documentation_context(self, keywords: List[str]) -> Dict[str, Any]:
        """Retrieve relevant documentation context."""
        pass


class MockContextRetriever(ContextRetrieverInterface):
    """Mock implementation of context retriever for demonstration."""
    
    def __init__(self):
        self.mock_codebase = {
            'error': ['ErrorHandler.java', 'ExceptionManager.py', 'ErrorLogger.js'],
            'exception': ['ExceptionHandler.java', 'CustomException.py', 'ErrorBoundary.jsx'],
            'timeout': ['TimeoutManager.java', 'ConnectionTimeout.py', 'RequestTimeout.js'],
            'database': ['DatabaseManager.java', 'DBConnection.py', 'DatabaseService.js'],
            'network': ['NetworkManager.java', 'HttpClient.py', 'NetworkService.js'],
            'memory': ['MemoryManager.java', 'MemoryMonitor.py', 'MemoryLeakDetector.js'],
            'crash': ['CrashHandler.java', 'CrashReporter.py', 'CrashAnalytics.js']
        }
        
        self.mock_documentation = {
            'error': [
                'Error Handling Best Practices',
                'Common Error Patterns',
                'Error Recovery Strategies'
            ],
            'exception': [
                'Exception Handling Guide',
                'Custom Exception Design',
                'Exception Propagation Patterns'
            ],
            'timeout': [
                'Timeout Configuration Guide',
                'Handling Timeout Scenarios',
                'Timeout Best Practices'
            ],
            'database': [
                'Database Connection Management',
                'Database Error Handling',
                'Database Performance Tuning'
            ],
            'network': [
                'Network Error Handling',
                'HTTP Error Codes',
                'Network Troubleshooting Guide'
            ],
            'memory': [
                'Memory Management Best Practices',
                'Memory Leak Detection',
                'Memory Optimization Techniques'
            ],
            'crash': [
                'Crash Analysis Guide',
                'Crash Prevention Strategies',
                'Crash Recovery Procedures'
            ]
        }
    
    def retrieve_codebase_context(self, keywords: List[str]) -> Dict[str, Any]:
        """Retrieve relevant codebase context based on keywords."""
        relevant_files = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self.mock_codebase:
                relevant_files.extend(self.mock_codebase[keyword_lower])
        
        return {
            'relevant_files': list(set(relevant_files)),  # Remove duplicates
            'total_files': len(set(relevant_files)),
            'search_keywords': keywords,
            'retrieval_method': 'mock_codebase_search'
        }
    
    def retrieve_documentation_context(self, keywords: List[str]) -> Dict[str, Any]:
        """Retrieve relevant documentation context based on keywords."""
        relevant_docs = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self.mock_documentation:
                relevant_docs.extend(self.mock_documentation[keyword_lower])
        
        return {
            'relevant_documentation': list(set(relevant_docs)),  # Remove duplicates
            'total_docs': len(set(relevant_docs)),
            'search_keywords': keywords,
            'retrieval_method': 'mock_documentation_search'
        }


class PromptCreator:
    """Creates comprehensive prompts for AI analysis."""
    
    def __init__(self):
        self.prompt_template = """
# Log Analysis Request

## Issue Description
{issue_description}

## Extracted Keywords
{keywords}

## Filtered Log Entries
{filtered_logs}

## Codebase Context
{codebase_context}

## Documentation Context
{documentation_context}

## Analysis Instructions
Please analyze the provided log entries in the context of the issue description and provide:

1. **Root Cause Analysis**: What is likely causing the issue?
2. **Error Pattern Identification**: Are there recurring patterns in the logs?
3. **Timeline Analysis**: When did the issue start and how did it progress?
4. **Impact Assessment**: What systems or components are affected?
5. **Recommended Actions**: What steps should be taken to resolve the issue?
6. **Prevention Measures**: How can similar issues be prevented in the future?

Please provide a detailed analysis with specific references to log entries and codebase context.
"""
    
    def create_prompt(self, 
                     issue_description: str,
                     keywords: List[str],
                     filtered_logs: str,
                     codebase_context: Dict[str, Any],
                     documentation_context: Dict[str, Any]) -> str:
        """
        Create a comprehensive prompt for AI analysis.
        
        Args:
            issue_description: The original issue description
            keywords: Extracted keywords
            filtered_logs: Filtered log entries
            codebase_context: Relevant codebase information
            documentation_context: Relevant documentation information
            
        Returns:
            Formatted prompt string
        """
        # Format codebase context
        codebase_text = self._format_context(codebase_context, "Codebase")
        
        # Format documentation context
        docs_text = self._format_context(documentation_context, "Documentation")
        
        # Create the prompt
        prompt = self.prompt_template.format(
            issue_description=issue_description,
            keywords=', '.join(keywords),
            filtered_logs=filtered_logs,
            codebase_context=codebase_text,
            documentation_context=docs_text
        )
        
        return prompt.strip()
    
    def _format_context(self, context: Dict[str, Any], context_type: str) -> str:
        """Format context information for the prompt."""
        if not context:
            return f"No {context_type.lower()} context available."
        
        formatted = f"### {context_type} Information\n"
        
        # Handle codebase/files context
        if 'relevant_files' in context:
            formatted += f"**Relevant Files ({context.get('total_files', 0)}):**\n"
            for file in context['relevant_files']:
                formatted += f"- {file}\n"
        
        # Handle documentation context
        if 'relevant_documentation' in context:
            formatted += f"**Relevant Documentation ({context.get('total_docs', 0)}):**\n"
            for doc in context['relevant_documentation']:
                formatted += f"- {doc}\n"
        
        # Handle detailed items if available (for JSON-based context)
        if 'relevant_items' in context:
            items = context['relevant_items']
            if context_type == "Codebase":
                formatted += f"**Components ({len(items)}):**\n"
            for item in items:
                if isinstance(item, dict):
                    item_title = item.get('title', item.get('id', 'Unknown'))
                    item_content = item.get('content', '')
                    if item_content:
                        # Truncate long content
                        if len(item_content) > 500:
                            item_content = item_content[:500] + "..."
                        formatted += f"\n**{item_title}:**\n{item_content}\n"
                    else:
                        formatted += f"- {item_title}\n"
        
        formatted += f"\n**Search Method:** {context.get('retrieval_method', 'Unknown')}\n"
        if 'search_keywords' in context:
            formatted += f"**Search Keywords:** {', '.join(context.get('search_keywords', []))}\n"
        
        return formatted


class LogAnalysisOrchestrator:
    """Main orchestrator that coordinates all analysis components."""
    
    def __init__(self, context_retriever: Optional[ContextRetrieverInterface] = None):
        """
        Initialize the orchestrator.

        Args:
            context_retriever: Optional context retriever implementation
        """
        self.keyword_extractor = KeywordExtractor()
        # Use JSON-based ContextRetriever
        self.context_retriever = context_retriever or ContextRetriever()
        self.prompt_creator = PromptCreator()
    
    def analyze_issue(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform complete issue analysis.
        
        Args:
            request: Analysis request parameters
            
        Returns:
            Complete analysis result
        """
        start_time = datetime.now()
        
        # Step 1: Extract keywords from issue description
        print("Extracting keywords from issue description...")
        extracted_keywords_objects = self.keyword_extractor.extract_keywords(request.issue_description)
        
        # Convert to simple keyword list for backward compatibility
        extracted_keywords = [kw.keyword for kw in extracted_keywords_objects]
        
        # Add any manually provided keywords
        if request.keywords:
            extracted_keywords.extend(request.keywords)
        
        # Remove duplicates
        all_keywords = list(set(extracted_keywords))
        
        print(f"Extracted keywords: {all_keywords}")
        print(f"LLM available: {self.keyword_extractor.is_llm_available()}")
        
        # Show extraction methods used
        methods_used = self.keyword_extractor.get_extraction_methods_used(request.issue_description)
        print(f"Extraction methods: {methods_used}")
        
        # Step 2: Analyze logs
        print("Analyzing logs...")
        config = LogFilterConfig(
            log_file_path=request.log_file_path,
            max_tokens=request.max_tokens,
            context_lines=request.context_lines,
            deduplicate=request.deduplicate,
            prioritize_by_severity=request.prioritize_by_severity,
            prioritize_matches=True,
            max_results=None
        )
        
        analyzer = LogAnalyzer(config)
        filtered_logs = analyzer.analyze(
            keywords=all_keywords,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Step 3: Retrieve context
        print("Retrieving codebase and documentation context...")
        codebase_context = self.context_retriever.retrieve_codebase_context(all_keywords)
        documentation_context = self.context_retriever.retrieve_documentation_context(all_keywords)
        
        # Step 4: Create comprehensive prompt
        print("Creating analysis prompt...")
        generated_prompt = self.prompt_creator.create_prompt(
            issue_description=request.issue_description,
            keywords=all_keywords,
            filtered_logs=filtered_logs,
            codebase_context=codebase_context,
            documentation_context=documentation_context
        )
        
        # Step 5: Get LLM analysis
        print("Getting LLM analysis...")
        llm_analysis = None
        try:
            llm_analysis = self.keyword_extractor.llm_interface.analyze_logs(generated_prompt)
            print("LLM analysis completed successfully")
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            llm_analysis = f"LLM analysis failed: {str(e)}"
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Create result
        result = AnalysisResult(
            request=request,
            extracted_keywords=all_keywords,
            filtered_logs=filtered_logs,
            context_info={
                'codebase': codebase_context,
                'documentation': documentation_context
            },
            generated_prompt=generated_prompt,
            llm_analysis=llm_analysis,
            timestamp=end_time,
            processing_time_ms=processing_time_ms
        )
        
        print(f"Analysis completed in {processing_time_ms}ms")
        return result
    
    def _parse_filtered_logs(self, filtered_logs: str) -> dict:
        """
        Parse filtered logs string into structured format with summary and entries array.
        
        Args:
            filtered_logs: String containing formatted logs
            
        Returns:
            Dictionary with 'summary' and 'entries' array
        """
        # Extract summary section (everything before the separator line)
        summary_match = re.search(r'(=== LOG ANALYSIS SUMMARY ===.*?Legend:.*?)', filtered_logs, re.DOTALL)
        summary_text = summary_match.group(1) if summary_match else ""
        
        # Extract entries (skip summary and separator lines)
        # Look for lines that start with [Line or are context lines
        log_lines = filtered_logs.split('\n')
        entries = []
        
        # Find where actual log entries start (after separator or ===)
        start_processing = False
        
        for line in log_lines:
            line = line.strip()
            if not line or line.startswith('===') or 'truncated' in line.lower():
                if 'truncated' in line.lower():
                    entries.append({
                        'type': 'truncation_notice',
                        'content': line
                    })
                continue
            
            # Start processing after we see a [Line marker
            if '[Line' in line or start_processing:
                start_processing = True
                
                # Parse log entry
                if '[Line' in line:
                    # Extract line number
                    line_num_match = re.search(r'\[Line (\d+)\]', line)
                    line_number = int(line_num_match.group(1)) if line_num_match else None
                    
                    # Extract if it's a direct match (>>>) or context
                    is_direct_match = '>>>' in line
                    
                    # Extract the actual log content
                    if is_direct_match:
                        log_content = re.sub(r'\[Line \d+\]\s*>>>\s*', '', line).strip()
                    else:
                        log_content = re.sub(r'\[Line \d+\]\s+', '', line).strip()
                    
                    entries.append({
                        'line_number': line_number,
                        'is_direct_match': is_direct_match,
                        'content': log_content
                    })
                elif line:  # Context lines without [Line] marker
                    entries.append({
                        'type': 'continuation',
                        'content': line
                    })
        
        return {
            'summary': summary_text.strip(),
            'entries': entries,
            'total_entries': len(entries)
        }
    
    def save_result(self, result: AnalysisResult, output_dir: str = "analysis_results", custom_timestamp: str = None) -> str:
        """
        Save analysis result to file.
        
        Args:
            result: Analysis result to save
            output_dir: Directory to save results
            custom_timestamp: Optional custom timestamp string to use for filename
            
        Returns:
            Path to saved file
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Use custom timestamp if provided, otherwise use result timestamp
        if custom_timestamp:
            timestamp_str = custom_timestamp
        else:
            timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
        
        filename = f"analysis_result_{timestamp_str}.json"
        filepath = Path(output_dir) / filename
        
        # Parse filtered logs into structured format
        parsed_logs = self._parse_filtered_logs(result.filtered_logs)
        
        # Convert result to dictionary for JSON serialization
        result_dict = {
            'timestamp': result.timestamp.isoformat(),
            'processing_time_ms': result.processing_time_ms,
            'request': {
                'log_file_path': result.request.log_file_path,
                'issue_description': result.request.issue_description,
                'keywords': result.request.keywords,
                'start_date': result.request.start_date,
                'end_date': result.request.end_date,
                'max_tokens': result.request.max_tokens,
                'context_lines': result.request.context_lines,
                'deduplicate': result.request.deduplicate,
                'prioritize_by_severity': result.request.prioritize_by_severity
            },
            'extracted_keywords': result.extracted_keywords,
            'filtered_logs': parsed_logs,
            'context_info': result.context_info,
            'generated_prompt': result.generated_prompt,
            'llm_analysis': result.llm_analysis
        }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis result saved to: {filepath}")
        return str(filepath)


def main():
    """Example usage of the modular log analysis system."""
    
    # Create analysis request
    request = AnalysisRequest(
        log_file_path="app.log",
        issue_description="The application is experiencing periodic crashes with memory errors. Users report the app freezes and then crashes with segmentation faults. This happens especially during high load periods.",
        start_date="2025-10-15",
        end_date="2025-10-16",
        max_tokens=3000,
        context_lines=2,
        deduplicate=True,
        prioritize_by_severity=True
    )
    
    # Create orchestrator
    orchestrator = LogAnalysisOrchestrator()
    
    # Perform analysis
    result = orchestrator.analyze_issue(request)
    
    # Save result
    output_file = orchestrator.save_result(result)
    
    # Display summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Issue Description: {request.issue_description[:100]}...")
    print(f"Extracted Keywords: {', '.join(result.extracted_keywords)}")
    print(f"Processing Time: {result.processing_time_ms}ms")
    print(f"Codebase Files Found: {result.context_info['codebase']['total_files']}")
    print(f"Documentation Found: {result.context_info['documentation']['total_docs']}")
    print(f"Result saved to: {output_file}")
    
    # Show prompt preview
    print("\n" + "="*80)
    print("GENERATED PROMPT PREVIEW")
    print("="*80)
    print(result.generated_prompt[:1000] + "..." if len(result.generated_prompt) > 1000 else result.generated_prompt)


if __name__ == "__main__":
    main()

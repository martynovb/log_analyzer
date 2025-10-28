"""
Log Analyzer - Modular Architecture
==================================

This module provides a comprehensive log analysis system with:
- Orchestration layer for web UI
- Keyword extraction from issue descriptions
- Log analysis with filtering
- Context retrieval from JSON database
- Prompt generation for AI analysis

Architecture:
- UI Layer: Simple web interface
- Business Logic: Core analysis components
- Data Layer: File handling and storage
- Integration Layer: External service interfaces
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import os
import re
from datetime import datetime

# Import from modules - clean, modular approach
from modules import (
    LogAnalyzer, LogFilterConfig,
    KeywordExtractor, KeywordType, ExtractedKeyword,
    ContextRetriever,
    PromptGenerator, AnalysisData
)


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


# Using PromptGenerator from modules/prompt_generator.py instead of duplicate PromptCreator


class LogAnalysisOrchestrator:
    """Main orchestrator that coordinates all analysis components."""
    
    def __init__(self):
        """
        Initialize the orchestrator.
        """
        self.keyword_extractor = KeywordExtractor()
        # Use JSON-based ContextRetriever
        self.context_retriever = ContextRetriever()
        self.prompt_generator = PromptGenerator()
    
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
        
        # Step 4: Format context and create comprehensive prompt
        print("Creating analysis prompt...")
        
        # Format context descriptions from retrieved data
        codebase_text = self._format_context(codebase_context, "Codebase")
        docs_text = self._format_context(documentation_context, "Documentation")
        combined_context = f"{codebase_text}\n\n{docs_text}" if docs_text else codebase_text
        
        # Create AnalysisData for prompt generation
        analysis_data = AnalysisData(
            issue_description=request.issue_description,
            extracted_keywords=all_keywords,
            filtered_logs=filtered_logs,
            context_description=combined_context,
            log_file_path=request.log_file_path,
            analysis_date_range=f"{request.start_date} to {request.end_date}" if request.start_date and request.end_date else request.start_date or request.end_date
        )
        
        # Generate prompt using PromptGenerator
        generated_prompt = self.prompt_generator.generate_prompt(analysis_data)
        
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



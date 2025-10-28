"""
Prompt Generation Module

This module handles creating comprehensive prompts for LLM analysis
by combining issue descriptions, filtered logs, and relevant context.
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class AnalysisData:
    """Container for all analysis data."""
    issue_description: str
    extracted_keywords: List[str]
    filtered_logs: str
    context_description: str
    log_file_path: str
    analysis_date_range: Optional[str] = None


class PromptGenerator:
    """Generates simple, focused prompts for LLM log analysis."""
    
    def format_context(self, context: dict, context_type: str) -> str:
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
    
    def generate_prompt(self, analysis_data: AnalysisData) -> str:
        """
        Generate a simple, focused prompt for log analysis.
        
        Args:
            analysis_data: Container with all analysis data
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""# Log Analysis Request

## Issue Description
{analysis_data.issue_description}

## Extracted Keywords
{', '.join(analysis_data.extracted_keywords)}

## Context Information
{analysis_data.context_description}

## Log File
- Path: {analysis_data.log_file_path}
- Date Range: {analysis_data.analysis_date_range or 'Not specified'}

## Filtered Log Entries
{analysis_data.filtered_logs}

## Analysis Task

Analyze the logs above and provide:

1. **Root Cause**: What is causing the issue? Be specific and reference log entries.
2. **How to Reproduce**: What sequence of events or actions leads to this issue?
3. **Key Moments**: What log entries or patterns should we pay attention to? Quote the exact lines.
4. **Recommendations**: What should be done to fix or prevent this issue?

Provide a clear, structured analysis with specific references to log entries."""
        
        return prompt

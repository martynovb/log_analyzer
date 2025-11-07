"""
Prompt Generation Module

This module handles creating comprehensive prompts for LLM analysis
by combining issue descriptions, filtered logs, and relevant context.
"""

from typing import List, Optional
from dataclasses import dataclass

from modules.domain import FilterMode


@dataclass
class AnalysisData:
    """Container for all analysis data."""
    issue_description: str
    extracted_keywords: List[str]
    filter_mode: FilterMode
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

{self.get_logs_prompt(analysis_data)}

## Analysis Task

Analyze the logs above and provide:

**Root Cause**: What is causing the issue? Be specific and reference log entries.

Provide a clear, structured analysis with specific references to log entries."""

        return prompt

    def get_logs_prompt(self, analysis_data: AnalysisData) -> str:
        common_part = f"""## Log File
- Path: {analysis_data.log_file_path}
- Date Range: {analysis_data.analysis_date_range or 'Not specified'}

## Filtered Log Entries
{analysis_data.filtered_logs}"""

        match analysis_data.filter_mode:
            case FilterMode.llm:
                return common_part
            case FilterMode.vector:
                vector_specific_part = """Logs are presented in a json format as a list of dictionaries:
"[{\"score\":0.3708814,\"logs\":\"timestamp1 <log_level1> log_tag1: message1\ntimestamp2 <log_level2> log_tag2: message2\"},
{\"score\":0.786563,\"logs\":\"timestamp3 <log_level3> log_tag3: message3\ntimestamp4 <log_level4> log_tag4: message4\"}]"
where score represents how much the logs are similar to the requested string.
Lower score represents more similarity.
The logs in a json are sorted by score. Pay attention to the score and consider it in your answer."""
                return f"{vector_specific_part}\n{common_part}"

    def get_logs_prompt(self, analysis_data: AnalysisData) -> str:
        common_part = f"""## Log File
- Path: {analysis_data.log_file_path}
- Date Range: {analysis_data.analysis_date_range or 'Not specified'}

## Filtered Log Entries
{analysis_data.filtered_logs}"""

        match analysis_data.filter_mode:
            case FilterMode.llm:
                return common_part
            case FilterMode.vector:
                vector_specific_part = """Logs are presented in a json format as a list of dictionaries:
"[{\"score\":0.3708814,\"logs\":\"timestamp1 <log_level1> log_tag1: message1\ntimestamp2 <log_level2> log_tag2: message2\"},
{\"score\":0.786563,\"logs\":\"timestamp3 <log_level3> log_tag3: message3\ntimestamp4 <log_level4> log_tag4: message4\"}]"
where score represents how much the logs are similar to the requested string.
Lower score represents more similarity.
The logs in a json are sorted by score. Pay attention to the score and consider it in your answer."""
                return f"{vector_specific_part}\n{common_part}"

    def generate_synthesis_prompt(
        self,
        issue_description: str,
        context_description: str,
        chunk_responses: List[str],
    ) -> str:
        """
        Generate a synthesis prompt that combines analysis from multiple chunks.

        Args:
            issue_description: The issue description
            extracted_keywords: List of extracted keywords (empty for split mode)
            context_description: Formatted context information (empty for split mode)
            chunk_responses: List of LLM responses from each chunk
            log_file_path: Path to the log file
            analysis_date_range: Optional date range string

        Returns:
            Formatted synthesis prompt string
        """
        # Format chunk responses
        chunk_analyses = "\n\n".join(
            [
                f"### Chunk {i+1} Analysis\n{response}"
                for i, response in enumerate(chunk_responses)
            ]
        )

        context_section = ""
        if context_description:
            context_section = f"\n## Context Information\n{context_description}\n"

        prompt = f"""# Log Analysis Synthesis Request

## Issue Description
{issue_description}
{context_section}
## Log File

## Chunk Analyses
The logs were split into {len(chunk_responses)} chunks and each chunk was analyzed separately. Below are the analyses from each chunk:

{chunk_analyses}

## Synthesis Task

Synthesize the analyses from all chunks above and provide a comprehensive final analysis:

**Root Cause**: What is causing the issue? Be specific and reference findings from the chunk analyses.

**Pattern Identification**: Are there any patterns or trends across the chunks?

**Timeline**: If time-based patterns are evident, describe the timeline of events.

**Impact Assessment**: What is the overall impact of the issue?

**Recommended Actions**: What actions should be taken based on the complete analysis?

Provide a clear, structured analysis that synthesizes insights from all chunks into a cohesive understanding of the issue."""

        return prompt

    def generate_chunk_prompt(
        self,
        issue_description: str,
        chunk_logs: str,
        context_description: str = "",
    ) -> str:
        """
        Create a prompt for analyzing a single chunk of logs.

        Args:
            issue_description: The issue description
            chunk_logs: The log content for this chunk
            context_description: Optional context information

        Returns:
            Formatted prompt string for LLM analysis
        """
        context_section = ""
        if context_description:
            context_section = f"\n## Context Information\n{context_description}\n"

        prompt = f"""# Log Analysis Request

## Issue Description
{issue_description}
{context_section}## Log Entries (Chunk)

This is one chunk of the log file. Analyze these log entries:

{chunk_logs}

## Analysis Task

Provide a brief analysis:

1. **Does the issue exist in these logs?** (Yes/No with brief explanation)

2. **If yes, what is the root cause?** (Be specific and reference log entries)

Keep your response concise and focused."""

        return prompt
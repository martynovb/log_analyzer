"""
Result Handler Module

Handles parsing and saving analysis results.
"""

import re
import json
from pathlib import Path

from .domain import AnalysisResult


class ResultHandler:
    """Handles result parsing and saving."""
    
    def parse_filtered_logs(self, filtered_logs: str) -> dict:
        """
        Parse filtered logs string into structured format with summary and entries array.
        
        Handles two formats:
        1. Formatted format (keyword-based) with [Line X] markers and summary
        2. Plain format (vector DB) with just log lines
        
        Args:
            filtered_logs: String containing formatted logs
            
        Returns:
            Dictionary with 'summary' and 'entries' array
        """
        if not filtered_logs or not filtered_logs.strip():
            return {
                'summary': '',
                'entries': [],
                'total_entries': 0
            }
        
        # Extract summary section (everything before the separator line) - only for formatted logs
        summary_match = re.search(r'(=== LOG ANALYSIS SUMMARY ===.*?Legend:.*?)', filtered_logs, re.DOTALL)
        summary_text = summary_match.group(1) if summary_match else ""
        
        # Check if this is formatted output (keyword-based) or plain output (vector DB)
        has_line_markers = '[Line' in filtered_logs
        has_summary = bool(summary_match)
        
        log_lines = filtered_logs.split('\n')
        entries = []
        
        if has_line_markers or has_summary:
            # Format 1: Formatted output with [Line X] markers (keyword-based)
            start_processing = False
            
            for line in log_lines:
                line = line.strip()
                if not line or (line.startswith('===') and 'LOG ANALYSIS' in line) or 'truncated' in line.lower():
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
        else:
            # Format 2: Plain output (vector DB) - just log lines
            for line_num, line in enumerate(log_lines, start=1):
                line = line.strip()
                if line:  # Only process non-empty lines
                    entries.append({
                        'line_number': line_num,
                        'is_direct_match': True,  # All vector DB results are considered matches
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
        parsed_logs = self.parse_filtered_logs(result.filtered_logs)
        
        # Convert result to dictionary for JSON serialization
        result_dict = {
            'timestamp': result.timestamp.isoformat(),
            'processing_time_ms': result.processing_time_ms,
            'llm_model': getattr(result, 'llm_model', None),
            'request': {
                'log_file_path': result.request.log_file_path,
                'issue_description': result.request.issue_description,
                'keywords': result.request.keywords,
                'filter_mode': getattr(result.request, 'filter_mode', 'llm'),
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
            'llm_analysis': result.llm_analysis,
            'chunk_responses': result.chunk_responses if result.chunk_responses else None
        }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis result saved to: {filepath}")
        return str(filepath)


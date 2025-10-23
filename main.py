import re
import sys
from datetime import datetime
from typing import List, Optional, Tuple, Dict
from collections import defaultdict


class LogFilter:
    def __init__(self, log_file_path: str, max_tokens: int = 3500):
        """
        Initialize log filter.
        
        Args:
            log_file_path: Path to the log file
            max_tokens: Maximum tokens for LLM (leaving buffer from 4000)
        """
        if not log_file_path:
            raise ValueError("log_file_path cannot be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        self.log_file_path = log_file_path
        self.max_tokens = max_tokens
        # Rough estimation: 1 token ≈ 4 characters
        self.max_chars = max_tokens * 4
        
    def parse_log_timestamp(self, line: str) -> Optional[datetime]:
        """
        Parse timestamp from log line. Adjust patterns based on your log format.
        Common formats:
        - 2025-10-21 14:30:45.123
        - 2025-10-21T14:30:45Z
        - [2025-10-21 14:30:45]
        """
        # Timestamp patterns
        patterns = [
            r'(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})',  # ISO format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                timestamp_str = match.group(1)
                # Try parsing with different formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%d/%m/%Y %H:%M:%S']:
                    try:
                        return datetime.strptime(timestamp_str.split('.')[0], fmt)
                    except ValueError:
                        continue
        return None
    
    def filter_by_date_range(self, lines: List[str], start_date: Optional[str], 
                            end_date: Optional[str]) -> List[str]:
        """
        Filter lines by date range.
        
        Args:
            lines: List of log lines
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
        """
        if not start_date and not end_date:
            return lines
        
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59) if end_date else None
        except ValueError as e:
            print(f"Error parsing date: {e}")
            return lines
        
        filtered = []
        last_valid_timestamp = None
        
        for line in lines:
            log_dt = self.parse_log_timestamp(line)
            
            # If we found a timestamp, remember it and check if it's in range
            if log_dt:
                last_valid_timestamp = log_dt
                if start_dt and log_dt < start_dt:
                    continue
                if end_dt and log_dt > end_dt:
                    continue
                filtered.append(line)
            else:
                # No timestamp - include line if last valid timestamp was in range
                if last_valid_timestamp:
                    if start_dt and last_valid_timestamp < start_dt:
                        continue
                    if end_dt and last_valid_timestamp > end_dt:
                        continue
                filtered.append(line)
        
        return filtered
    
    def filter_by_keywords(self, lines: List[str], keywords: List[str],
                          context_lines: int = 2, word_boundary: bool = True,
                          max_results: Optional[int] = None) -> Tuple[List[Tuple[int, str, bool]], Dict[str, int]]:
        """
        Filter lines by keywords with context - OPTIMIZED VERSION.
        
        Args:
            lines: List of log lines
            keywords: List of keywords to search for
            context_lines: Number of lines before/after to include
            word_boundary: If True, match whole words only (reduces false positives)
            max_results: Maximum number of matching lines (not including context)
            
        Returns:
            Tuple of:
            - List of tuples (line_number, line_content, is_direct_match)
              is_direct_match=True for actual keyword matches, False for context lines
            - Dict of keyword match counts
        """
        if not keywords:
            return [], {}
        
        # Compile patterns once
        if word_boundary:
            patterns = [r'\b' + re.escape(kw) + r'\b' for kw in keywords]
        else:
            patterns = [re.escape(kw) for kw in keywords]
        
        keyword_pattern = re.compile('|'.join(patterns), re.IGNORECASE)
        
        # Track individual keyword statistics
        keyword_stats = defaultdict(int)
        individual_patterns = {
            kw: re.compile(
                (r'\b' + re.escape(kw) + r'\b') if word_boundary else re.escape(kw),
                re.IGNORECASE
            ) for kw in keywords
        }
        
        # Two-pass approach for better performance
        matches_with_context = {}  # line_num -> (line, is_direct_match)
        matched_line_numbers = []
        
        # PASS 1: Find all matching lines
        for i, line in enumerate(lines):
            if keyword_pattern.search(line):
                matched_line_numbers.append(i)
                matches_with_context[i] = (line, True)
                
                # Count individual keyword matches
                for kw, pattern in individual_patterns.items():
                    if pattern.search(line):
                        keyword_stats[kw] += 1
                
                # Early termination if max_results reached
                if max_results and len(matched_line_numbers) >= max_results:
                    print(f"  Reached max_results limit ({max_results}), stopping search")
                    break
        
        # PASS 2: Add context lines
        context_to_add = set()
        for match_idx in matched_line_numbers:
            start = max(0, match_idx - context_lines)
            end = min(len(lines), match_idx + context_lines + 1)
            context_to_add.update(range(start, end))
        
        # Add context lines that aren't already direct matches
        for ctx_idx in context_to_add:
            if ctx_idx not in matches_with_context:
                matches_with_context[ctx_idx] = (lines[ctx_idx], False)
        
        # Sort by line number and format output
        result = [
            (line_num, line, is_match)
            for line_num, (line, is_match) in sorted(matches_with_context.items())
        ]
        
        print(f"  Found {len(matched_line_numbers)} lines with direct keyword matches")
        print(f"  Total with context: {len(result)} lines")
        if keyword_stats:
            print(f"  Keyword breakdown: {dict(keyword_stats)}")
        
        return result, dict(keyword_stats)
    
    def deduplicate_similar_lines(self, filtered_lines: List[Tuple[int, str, bool]], 
                                 max_duplicates: int = 3) -> List[Tuple[int, str, bool]]:
        """
        Remove very similar repetitive log entries using fast hash-based approach.
        
        Args:
            filtered_lines: Lines to deduplicate (line_num, line, is_match)
            max_duplicates: Keep only this many copies of similar messages
        """
        if not filtered_lines:
            return filtered_lines
        
        print(f"  Deduplicating {len(filtered_lines)} entries...")
        
        # Fast hash-based deduplication
        pattern_counts = {}
        unique_lines = []
        
        for line_num, line, is_match in filtered_lines:
            # Extract the core message (remove timestamp, line numbers, etc.)
            core_message = re.sub(r'\d{4}-\d{2}-\d{2}', '', line)
            core_message = re.sub(r'\d{2}:\d{2}:\d{2}[.,]\d+', '', core_message)
            core_message = re.sub(r'\[.*?\]', '', core_message)
            core_message = re.sub(r'\d+', '#', core_message)  # Replace all numbers with #
            core_message = core_message.strip()
            
            # Use hash for fast comparison
            pattern_hash = hash(core_message)
            
            if pattern_hash not in pattern_counts:
                pattern_counts[pattern_hash] = 0
            
            if pattern_counts[pattern_hash] < max_duplicates:
                unique_lines.append((line_num, line, is_match))
                pattern_counts[pattern_hash] += 1
        
        print(f"  Kept {len(unique_lines)} unique entries")
        return unique_lines
    
    def prioritize_lines(self, filtered_lines: List[Tuple[int, str, bool]]) -> List[Tuple[int, str, bool]]:
        """
        Prioritize lines by severity/importance.
        Direct matches get highest priority, then by log level.
        
        Args:
            filtered_lines: Lines to prioritize (line_num, line, is_match)
            
        Returns:
            Sorted list with most important lines first
        """
        def get_priority(item):
            line_num, line, is_match = item
            line_upper = line.upper()
            
            score = 0
            if is_match:
                score += 1000
            
            if 'LogLevel.e' in line_upper or 'exception' in line_upper:
                score += 200
            elif 'LogLevel.d' in line_upper or 'LogLevel.i' in line_upper:
                score += 100
            elif 'LogLevel.analytics' in line_upper:
                score += 50
            
            # Keep original order for same priority (use negative line_num)
            return (-score, line_num)
        
        print(f"  Prioritizing {len(filtered_lines)} lines by importance...")
        sorted_lines = sorted(filtered_lines, key=get_priority)
        
        return sorted_lines
    
    def truncate_to_token_limit(self, filtered_lines: List[Tuple[int, str, bool]], 
                               add_summary: bool = True,
                               prioritize_matches: bool = True) -> str:
        """
        Truncate filtered lines to fit within token limit.
        PRIORITY: Direct matches first, then context lines.
        
        Args:
            filtered_lines: Lines with (line_num, line, is_match)
            add_summary: Whether to add summary header
            prioritize_matches: If True, include all direct matches first, then fill with context
        """
        # Reserve space for summary
        summary_chars = 500 if add_summary else 0
        available_chars = self.max_chars - summary_chars
        
        if prioritize_matches:
            # Separate direct matches from context
            direct_matches = [(ln, l, m) for ln, l, m in filtered_lines if m]
            context_lines = [(ln, l, m) for ln, l, m in filtered_lines if not m]
            
            print(f"  Prioritizing: {len(direct_matches)} direct matches, {len(context_lines)} context lines")
            
            output_lines = []
            total_chars = 0
            included_matches = 0
            included_context = 0
            
            # PHASE 1: Add all direct matches (priority)
            for line_num, line, is_match in direct_matches:
                line_with_num = f"[Line {line_num}] >>> {line}"  # >>> marks direct match
                if total_chars + len(line_with_num) <= available_chars:
                    output_lines.append((line_num, line_with_num))
                    total_chars += len(line_with_num)
                    included_matches += 1
                else:
                    break
            
            # PHASE 2: Fill remaining space with context lines
            if total_chars < available_chars:
                for line_num, line, is_match in context_lines:
                    line_with_num = f"[Line {line_num}]     {line}"  # Indent for context
                    if total_chars + len(line_with_num) <= available_chars:
                        output_lines.append((line_num, line_with_num))
                        total_chars += len(line_with_num)
                        included_context += 1
                    else:
                        break
            
            # Sort by line number for readability
            output_lines.sort(key=lambda x: x[0])
            result_lines = [line for _, line in output_lines]
            
            truncation_info = ""
            if included_matches < len(direct_matches):
                truncation_info += f"\n... (truncated, {len(direct_matches) - included_matches} more direct matches) ...\n"
            if included_context < len(context_lines):
                truncation_info += f"... (truncated, {len(context_lines) - included_context} more context lines) ...\n"
            
            if truncation_info:
                result_lines.append(truncation_info)
            
            print(f"  Included: {included_matches}/{len(direct_matches)} matches, {included_context}/{len(context_lines)} context")
        
        else:
            # Original behavior: keep order, truncate when full
            output_lines = []
            total_chars = 0
            
            for line_num, line, is_match in filtered_lines:
                marker = ">>>" if is_match else "   "
                line_with_num = f"[Line {line_num}] {marker} {line}"
                if total_chars + len(line_with_num) > available_chars:
                    output_lines.append(f"\n... (truncated, {len(filtered_lines) - len(output_lines)} more entries) ...\n")
                    break
                output_lines.append(line_with_num)
                total_chars += len(line_with_num)
            
            result_lines = output_lines
        
        result = '\n'.join(result_lines)
        
        if add_summary:
            summary = self._generate_summary(filtered_lines, len(result_lines))
            result = summary + "\n" + "="*80 + "\n\n" + result
        
        return result
    
    def _generate_summary(self, all_lines: List[Tuple[int, str, bool]], included_count: int) -> str:
        """Generate a summary header."""
        direct_matches = sum(1 for _, _, is_match in all_lines if is_match)
        context_lines_count = len(all_lines) - direct_matches
        error_count = sum(1 for _, line, _ in all_lines if 'ERROR' in line.upper() or 'EXCEPTION' in line.upper())
        warn_count = sum(1 for _, line, _ in all_lines if 'WARN' in line.upper())
        
        summary = f"""=== LOG ANALYSIS SUMMARY ===
Total filtered entries: {len(all_lines)}
  - Direct keyword matches: {direct_matches}
  - Context lines: {context_lines_count}
Included in output: {included_count}
Errors/Exceptions: {error_count}
Warnings: {warn_count}
Estimated tokens: ~{included_count * 20} (approximate)
Legend: >>> = direct match, (indented) = context line
"""
        return summary
    
    def process(self, keywords: List[str], start_date: Optional[str] = None, 
                end_date: Optional[str] = None, context_lines: int = 2,
                deduplicate: bool = True, prioritize_by_severity: bool = False,
                prioritize_matches: bool = True, max_results: Optional[int] = None) -> str:
        """
        Main processing function with optimizations.
        
        Args:
            keywords: List of keywords to search for
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            context_lines: Number of context lines around matches
            deduplicate: Whether to remove similar repetitive entries
            prioritize_by_severity: Whether to prioritize ERROR/FATAL lines (optional)
            prioritize_matches: Prioritize direct matches over context when truncating
            max_results: Maximum number of direct matches to find (for performance)
            
        Returns:
            Filtered and formatted log content ready for LLM
        """
        if not keywords:
            return "Error: No keywords provided for filtering."
        
        if context_lines < 0:
            return "Error: context_lines must be non-negative."
        
        if max_results is not None and max_results <= 0:
            return "Error: max_results must be positive if specified."
        
        print("Reading log file...")
        try:
            with open(self.log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return f"Error: Log file '{self.log_file_path}' not found."
        except PermissionError:
            return f"Error: Permission denied accessing '{self.log_file_path}'."
        except Exception as e:
            return f"Error reading log file: {e}"
        
        print(f"Total lines: {len(lines)}")
        
        # Step 1: Filter by date range
        if start_date or end_date:
            print("Filtering by date range...")
            lines = self.filter_by_date_range(lines, start_date, end_date)
            print(f"After date filtering: {len(lines)} lines")
        
        # Step 2: Filter by keywords (OPTIMIZED with statistics)
        print(f"Filtering by keywords: {keywords}")
        filtered_lines, keyword_stats = self.filter_by_keywords(
            lines, keywords, context_lines, 
            word_boundary=True, max_results=max_results
        )
        print(f"After keyword filtering: {len(filtered_lines)} entries")
        
        if not filtered_lines:
            return "No matching log entries found for the given criteria."
        
        # Step 3: Optional prioritization by severity
        if prioritize_by_severity:
            filtered_lines = self.prioritize_lines(filtered_lines)
        
        # Step 4: Deduplicate similar entries
        if deduplicate:
            print("Deduplicating similar entries...")
            filtered_lines = self.deduplicate_similar_lines(filtered_lines)
            print(f"After deduplication: {len(filtered_lines)} entries")
        
        # Step 5: Truncate to token limit (with match prioritization)
        print("Truncating to token limit...")
        result = self.truncate_to_token_limit(
            filtered_lines, 
            add_summary=True,
            prioritize_matches=prioritize_matches
        )
        
        print(f"Final output size: {len(result)} characters (~{len(result)//4} tokens)")
        
        return result


def main():
    """Example usage"""
    # Configuration
    log_file = "app.log"  # Log file path
    keywords = ["snapshot", "periodic", "measurement"]  # Keywords
    start_date = "2025-10-15"  # Optional
    end_date = "2025-10-16"    # Optional
    
    # Create filter and process
    log_filter = LogFilter(log_file, max_tokens=3000)
    result = log_filter.process(
        keywords=keywords,
        start_date=start_date,
        end_date=end_date,
        context_lines=2,
        deduplicate=False,
        prioritize_by_severity=False,
        prioritize_matches=False,
        max_results=None
    )
    
    # Save output
    output_file = "filtered_logs.txt"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
    except Exception as e:
        print(f"Error saving output file: {e}")
        return
    
    print(f"\nFiltered logs saved to: {output_file}")
    print("\nPreview:")
    print(result[:1000] + "..." if len(result) > 1000 else result)


if __name__ == "__main__":
    main()
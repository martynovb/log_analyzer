import re
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from collections import defaultdict

from modules.log_filter import LogFilterConfig, LogFilter


@dataclass
class LLMLogFilterConfig(LogFilterConfig):
    keywords: list[str]
    context_lines: int = 2
    deduplicate: bool = True
    prioritize_by_severity: bool = False
    prioritize_matches: bool = True
    max_results: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.keywords:
            raise ValueError("Error: No keywords provided for filtering.")
        if self.context_lines < 0:
            raise ValueError("context_lines must be non-negative")
        if self.max_results is not None and self.max_results <= 0:
            raise ValueError("max_results must be positive if specified")


class LLMLogFilter(LogFilter):
    """Log filter class that picks only necessary logs."""

    def __init__(self, config: LLMLogFilterConfig):
        """
        Initialize log analyzer with configuration.

        Args:
            config: Configuration object containing all parameters
        """
        super().__init__(config)
        self.config = config

        # Initialize components
        self.keyword_filter = LogKeywordFilter()
        self.deduplicator = LogDeduplicator()
        self.prioritizer = LogPrioritizer()
        self.formatter = TokenLimitedFormatter()

    def filter(self) -> str:
        """
        Returns:
            Filtered and formatted log content ready for LLM
        """

        # Step 1: Read log file
        # Step 2: Filter by date range
        lines = self.filter_by_date()

        # Step 3: Filter by keywords
        print(f"Filtering by keywords: {self.config.keywords}")
        filtered_lines, keyword_stats = self.keyword_filter.filter_by_keywords(
            lines, self.config.keywords, self.config.context_lines,
            word_boundary=True, max_results=self.config.max_results
        )
        print(f"After keyword filtering: {len(filtered_lines)} entries")

        if not filtered_lines:
            return "No matching log entries found for the given criteria."

        # Step 4: Optional prioritization by severity
        if self.config.prioritize_by_severity:
            filtered_lines = self.prioritizer.prioritize_lines(filtered_lines)

        # Step 5: Deduplicate similar entries
        if self.config.deduplicate:
            print("Deduplicating similar entries...")
            filtered_lines = self.deduplicator.deduplicate_similar_lines(
                filtered_lines)
            print(f"After deduplication: {len(filtered_lines)} entries")

        # Step 6: Format output
        print("Formatting output...")
        result = self.formatter.format(filtered_lines, self.config)

        print(
            f"Final output size: {len(result)} characters (~{len(result) // 4} tokens)")

        return result


class LogKeywordFilter:
    """Handles filtering log lines by keywords with context."""

    def filter_by_keywords(self, lines: List[str], keywords: List[str],
                           context_lines: int = 2, word_boundary: bool = True,
                           max_results: Optional[int] = None) -> Tuple[
        List[Tuple[int, str, bool]], Dict[str, int]]:
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
                (r'\b' + re.escape(kw) + r'\b') if word_boundary else re.escape(
                    kw),
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
                    print(
                        f"  Reached max_results limit ({max_results}), stopping search")
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
            for line_num, (line, is_match) in
            sorted(matches_with_context.items())
        ]

        print(
            f"  Found {len(matched_line_numbers)} lines with direct keyword matches")
        print(f"  Total with context: {len(result)} lines")
        if keyword_stats:
            print(f"  Keyword breakdown: {dict(keyword_stats)}")

        return result, dict(keyword_stats)


class LogDeduplicator:
    """Handles deduplication of similar log entries."""

    def deduplicate_similar_lines(self,
                                  filtered_lines: List[Tuple[int, str, bool]],
                                  max_duplicates: int = 3) -> List[
        Tuple[int, str, bool]]:
        """
        Remove very similar repetitive log entries using fast hash-based approach.

        Args:
            filtered_lines: Lines to deduplicate (line_num, line, is_match)
            max_duplicates: Keep only this many copies of similar messages

        Returns:
            Deduplicated list of log lines
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
            core_message = re.sub(r'\d+', '#',
                                  core_message)  # Replace all numbers with #
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


class LogPrioritizer:
    """Handles prioritization of log lines by severity/importance."""

    def prioritize_lines(self, filtered_lines: List[Tuple[int, str, bool]]) -> \
            List[Tuple[int, str, bool]]:
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


class LogFormatter(ABC):
    """Abstract base class for log formatters."""

    @abstractmethod
    def format(self, filtered_lines: List[Tuple[int, str, bool]],
               config: LogFilterConfig) -> str:
        """Format filtered log lines."""
        pass


class TokenLimitedFormatter(LogFormatter):
    """Formats log lines with token limit constraints."""

    def format(self, filtered_lines: List[Tuple[int, str, bool]],
               config: LLMLogFilterConfig) -> str:
        """
        Format filtered lines to fit within token limit.
        PRIORITY: Direct matches first, then context lines.

        Args:
            filtered_lines: Lines with (line_num, line, is_match)
            config: Configuration object

        Returns:
            Formatted log content ready for LLM
        """
        # Reserve space for summary
        summary_chars = 500
        available_chars = config.max_chars - summary_chars

        if config.prioritize_matches:
            # Separate direct matches from context
            direct_matches = [(ln, l, m) for ln, l, m in filtered_lines if m]
            context_lines = [(ln, l, m) for ln, l, m in filtered_lines if not m]

            print(
                f"  Prioritizing: {len(direct_matches)} direct matches, {len(context_lines)} context lines")

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

            print(
                f"  Included: {included_matches}/{len(direct_matches)} matches, {included_context}/{len(context_lines)} context")

        else:
            # Original behavior: keep order, truncate when full
            output_lines = []
            total_chars = 0

            for line_num, line, is_match in filtered_lines:
                marker = ">>>" if is_match else "   "
                line_with_num = f"[Line {line_num}] {marker} {line}"
                if total_chars + len(line_with_num) > available_chars:
                    output_lines.append(
                        f"\n... (truncated, {len(filtered_lines) - len(output_lines)} more entries) ...\n")
                    break
                output_lines.append(line_with_num)
                total_chars += len(line_with_num)

            result_lines = output_lines

        result = '\n'.join(result_lines)

        # Add summary
        summary = self._generate_summary(filtered_lines, len(result_lines))
        result = summary + "\n" + "=" * 80 + "\n\n" + result

        return result

    def _generate_summary(self, all_lines: List[Tuple[int, str, bool]],
                          included_count: int) -> str:
        """Generate a summary header."""
        direct_matches = sum(1 for _, _, is_match in all_lines if is_match)
        context_lines_count = len(all_lines) - direct_matches
        error_count = sum(1 for _, line, _ in all_lines if
                          'ERROR' in line.upper() or 'EXCEPTION' in line.upper())
        warn_count = sum(
            1 for _, line, _ in all_lines if 'WARN' in line.upper())

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

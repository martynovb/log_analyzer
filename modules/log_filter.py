"""
Core Log Analyzer Components
============================

Contains the main LogAnalyzer class and LogFilterConfig for log analysis.
"""

import re
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass(kw_only=True)
class LogFilterConfig:
    """Configuration class for log filtering parameters."""
    log_file_path: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if not self.log_file_path:
            raise ValueError("log_file_path cannot be empty")


class LogFilter(ABC):
    """Abstract class for log filtering."""

    def __init__(self, config: LogFilterConfig):
        self.config = config
        timestamp_parser = LogTimestampParser()
        self.date_filter = LogDateFilter(timestamp_parser)
        self.file_reader = LogFileReader()

    def filter_by_date(self) -> list[str]:
        print("Reading log file...")
        try:
            lines = self.file_reader.read_log_file(self.config.log_file_path)
        except (FileNotFoundError, PermissionError, Exception) as e:
            raise ValueError(f"Error: {e}")

        print(f"Total lines: {len(lines)}")

        # Step 2: Filter by date range
        if self.config.start_date or self.config.end_date:
            print("Filtering by date range...")
            lines = self.date_filter.filter_by_date_range(lines,
                                                          self.config.start_date,
                                                          self.config.end_date)
            print(f"After date filtering: {len(lines)} lines")
        return  lines

    @abstractmethod
    def filter(self) -> str:
        """Return filtered logs."""
        raise NotImplementedError


class LogTimestampParser:
    """Handles parsing of timestamps from log lines."""

    def __init__(self):
        self.patterns = [
            r'(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})',  # ISO format
        ]
        self.formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
                        '%d/%m/%Y %H:%M:%S']

    def parse_timestamp(self, line: str) -> Optional[datetime]:
        """
        Parse timestamp from log line.
        
        Args:
            line: Log line to parse
            
        Returns:
            Parsed datetime object or None if no timestamp found
        """
        for pattern in self.patterns:
            match = re.search(pattern, line)
            if match:
                timestamp_str = match.group(1)
                # Try parsing with different formats
                for fmt in self.formats:
                    try:
                        return datetime.strptime(timestamp_str.split('.')[0],
                                                 fmt)
                    except ValueError:
                        continue
        return None


class LogDateFilter:
    """Handles filtering log lines by date range."""

    def __init__(self, parser: LogTimestampParser):
        self.parser = parser

    def filter_by_date_range(self, lines: List[str], start_date: Optional[str],
                             end_date: Optional[str]) -> List[str]:
        """
        Filter lines by date range.
        
        Args:
            lines: List of log lines
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            Filtered list of log lines
        """
        if not start_date and not end_date:
            return lines

        try:
            start_dt = datetime.strptime(start_date,
                                         '%Y-%m-%d') if start_date else None
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23,
                                                                     minute=59,
                                                                     second=59) if end_date else None
        except ValueError as e:
            print(f"Error parsing date: {e}")
            return lines

        filtered = []
        last_valid_timestamp = None

        for line in lines:
            log_dt = last_valid_timestamp
            next_log_dt = self.parser.parse_timestamp(line)
            if next_log_dt:
                log_dt = next_log_dt
                last_valid_timestamp = next_log_dt

            if log_dt:
                if start_dt and log_dt < start_dt:
                    continue
                if end_dt and log_dt > end_dt:
                    continue
            filtered.append(line)

        return filtered


class LogFileReader:
    """Handles reading log files with error handling."""

    def read_log_file(self, file_path: str) -> List[str]:
        """
        Read log file with proper error handling.
        
        Args:
            file_path: Path to the log file
            
        Returns:
            List of log lines
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If permission denied
            Exception: For other I/O errors
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Log file '{file_path}' not found.")
        except PermissionError:
            raise PermissionError(f"Permission denied accessing '{file_path}'.")
        except Exception as e:
            raise Exception(f"Error reading log file: {e}")

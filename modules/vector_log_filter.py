"""
Vector-based Log Filter (Mock)

Interface and mock implementation for vector DB log filtering.
Input: issue description, uploaded log file path
Output: mocked filtered logs string
"""
import os
import shutil
from dataclasses import dataclass
from typing import Optional

from modules.log_filter import LogFilterConfig, LogFilter
from modules.vector_db import VectorDb


@dataclass
class VectorLogFilterConfig(LogFilterConfig):
    issue_description: str

    def __post_init__(self):
        super().__post_init__()
        if not self.issue_description:
            raise ValueError(
                "Error: No issue description provided for filtering.")


@dataclass
class DbSignature:
    log_file_path: str
    start_date: Optional[str]
    end_date: Optional[str]


@dataclass
class DbCache:
    db: VectorDb
    db_signature: DbSignature


class VectorLogFilter(LogFilter):
    WORKING_DIRECTORY = "temp_vector_db"
    _cached_db_cache: DbCache | None = None

    def __init__(self, config: VectorLogFilterConfig):
        super().__init__(config)
        self.config = config

    def _get_db_signature(self) -> DbSignature:
        """Create a signature for the current DB parameters."""
        return DbSignature(
            log_file_path=self.config.log_file_path,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )

    def _can_reuse_db(self, directory: str) -> bool:
        """Check if existing DB can be reused based on cached signature."""
        if VectorLogFilter._cached_db_cache is None:
            print("  DEBUG: _cached_db_cache is None - cannot reuse DB")
            return False
        
        if VectorLogFilter._cached_db_cache.db is None:
            print("  DEBUG: _cached_db_cache.db is None (will attempt to reload if signature matches)")

        current_signature = self._get_db_signature()
        cached_signature = VectorLogFilter._cached_db_cache.db_signature

        # Compare each field and print which one differs
        if cached_signature.log_file_path != current_signature.log_file_path:
            print("  DEBUG: Field 'log_file_path' changed:")
            print(f"    Cached: {repr(cached_signature.log_file_path)}")
            print(f"    Current: {repr(current_signature.log_file_path)}")
            return False
        
        if cached_signature.start_date != current_signature.start_date:
            print("  DEBUG: Field 'start_date' changed:")
            print(f"    Cached: {repr(cached_signature.start_date)}")
            print(f"    Current: {repr(current_signature.start_date)}")
            return False
        
        if cached_signature.end_date != current_signature.end_date:
            print("  DEBUG: Field 'end_date' changed:")
            print(f"    Cached: {repr(cached_signature.end_date)}")
            print(f"    Current: {repr(current_signature.end_date)}")
            return False

        # Check if directory exists
        if not os.path.exists(directory):
            print(f"  DEBUG: Directory does not exist: {directory}")
            return False
        
        if not os.path.isdir(directory):
            print(f"  DEBUG: Path exists but is not a directory: {directory}")
            return False

        # All checks passed
        print("  DEBUG: All signature fields match and directory exists - can reuse DB")
        return True

    def filter(self) -> str:
        directory = VectorLogFilter.WORKING_DIRECTORY

        # Check if we can reuse existing DB
        can_reuse = self._can_reuse_db(directory)

        if can_reuse:
            print(
                "  Reusing existing vector DB (log_file_path, start_date, end_date unchanged)"
            )
            # Load existing DB (keep it open for reuse)
            db = VectorDb(persist_directory=directory, load_existing=True)
            # Update the cached instance reference
            VectorLogFilter._cached_db_cache.db = db
        else:
            print("  Creating new vector DB (parameters changed or DB not found)")
            # Close previous DB instance if it exists to release file handles
            # This is important on Windows before deleting the directory
            if VectorLogFilter._cached_db_cache is not None and VectorLogFilter._cached_db_cache.db is not None:
                VectorLogFilter._cached_db_cache.db.close()
                VectorLogFilter._cached_db_cache = None
            
            # Clear out the working directory first.
            # After closing, we should be able to delete without ignore_errors
            # but keep it as a safety measure
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=True)
            os.makedirs(directory, exist_ok=True)

            lines = self.filter_by_date()
            # Storing logs filtered by date
            filtered_logs_by_date_file_path = f"{directory}/filtered_logs_by_date.txt"
            with open(filtered_logs_by_date_file_path, "w") as f:
                f.writelines(lines)

            # Create new DB
            db = VectorDb(
                input_document_path=filtered_logs_by_date_file_path,
                persist_directory=directory
            )

            # Cache signature and instance in memory for future reuse
            db_signature = self._get_db_signature()
            VectorLogFilter._cached_db_cache = DbCache(db=db, db_signature=db_signature)

        # Perform search with current issue_description
        # (issue_description can change without needing to recreate DB)
        results = db.search(self.config.issue_description)

        # Storing logs filtered by date and context
        filtered_logs_file_path = f"{directory}/filtered_logs.txt"
        with open(filtered_logs_file_path, "w") as f:
            f.writelines([f"\n{score=}, {logs=}" for logs, score in results])

        return "\n".join([logs for logs, _ in results])

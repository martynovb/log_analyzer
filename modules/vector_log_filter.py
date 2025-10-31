"""
Vector-based Log Filter (Mock)

Interface and mock implementation for vector DB log filtering.
Input: issue description, uploaded log file path
Output: mocked filtered logs string
"""
import os
import shutil
from dataclasses import dataclass

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


class VectorLogFilter(LogFilter):
    WORKING_DIRECTORY = "temp_vector_db"
    _cached_db_signature: dict | None = None
    _cached_db_instance: VectorDb | None = None  # Keep reference to last DB for cleanup

    def __init__(self, config: VectorLogFilterConfig):
        super().__init__(config)
        self.config = config

    def _get_db_signature(self) -> dict:
        """Create a signature for the current DB parameters."""
        return {
            "log_file_path": self.config.log_file_path,
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
        }

    def _can_reuse_db(self, directory: str) -> bool:
        """Check if existing DB can be reused based on cached signature."""
        if VectorLogFilter._cached_db_signature is None:
            return False

        current_signature = self._get_db_signature()

        # Check if all key parameters match and directory exists
        return (
            VectorLogFilter._cached_db_signature.get("log_file_path")
            == current_signature["log_file_path"]
            and VectorLogFilter._cached_db_signature.get("start_date")
            == current_signature["start_date"]
            and VectorLogFilter._cached_db_signature.get("end_date")
            == current_signature["end_date"]
            and os.path.exists(directory)
            and os.path.isdir(directory)
        )

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
            VectorLogFilter._cached_db_instance = db
        else:
            print("  Creating new vector DB (parameters changed or DB not found)")
            # Close previous DB instance if it exists to release file handles
            # This is important on Windows before deleting the directory
            if VectorLogFilter._cached_db_instance is not None:
                VectorLogFilter._cached_db_instance.close()
                VectorLogFilter._cached_db_instance = None
            
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
            VectorLogFilter._cached_db_signature = self._get_db_signature()
            VectorLogFilter._cached_db_instance = db

        # Perform search with current issue_description
        # (issue_description can change without needing to recreate DB)
        results = db.search(self.config.issue_description)

        # Storing logs filtered by date and context
        filtered_logs_file_path = f"{directory}/filtered_logs.txt"
        with open(filtered_logs_file_path, "w") as f:
            f.writelines([f"\n{score=}, {logs=}" for logs, score in results])

        return "\n".join([logs for logs, _ in results])

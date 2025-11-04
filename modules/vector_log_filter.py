"""
Vector-based Log Filter

Interface and implementation for vector DB log filtering.
Input: issue description, uploaded log file path
Output: filtered logs string
"""
import json
import os
import shutil

from typing import Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from modules.log_filter import LogFilterConfig, LogFilter
from modules.vector_db import VectorDb


@dataclass
class VectorLogFilterConfig(LogFilterConfig):
    issue_description: str
    chunk_size: int = 800
    chunk_overlap: int = 200
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

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

    def __eq__(self, other) -> bool:
        if not isinstance(other, DbSignature):
            return False
        return (
                self.log_file_path == other.log_file_path
                and self.start_date == other.start_date
                and self.end_date == other.end_date
        )


@dataclass
class DbCache:
    db: VectorDb
    db_signature: DbSignature


@dataclass
class VectorLogFilterResponse:
    logs: str
    # Score represents how much the chunk is similar to the requested string.
    # Lower score represents more similarity.
    score: float


class VectorLogFilter(LogFilter):
    WORKING_DIRECTORY = "temp_vector_db"
    _db_cache: DbCache | None = None

    def __init__(self, config: VectorLogFilterConfig):
        super().__init__(config)
        self.config = config

    def _get_current_db_signature(self) -> DbSignature:
        """Create a signature for the current DB parameters."""
        return DbSignature(
            log_file_path=self.config.log_file_path,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )

    def _get_cached_db(self, directory: str) -> VectorDb | None:
        """Check if existing DB can be reused based on cached signature."""
        if not VectorLogFilter._db_cache:
            return None

        if not os.path.exists(directory) or not os.path.isdir(directory):
            return None

        current_signature = self._get_current_db_signature()
        cached_signature = VectorLogFilter._db_cache.db_signature

        if cached_signature != current_signature:
            return None

        return VectorLogFilter._db_cache.db

    def filter(self) -> str:
        directory = VectorLogFilter.WORKING_DIRECTORY

        # Check if we can reuse existing DB
        if cached_db := self._get_cached_db(directory):
            print(
                "  Reusing existing vector DB (log_file_path, start_date, end_date unchanged)"
            )
            db = cached_db
        else:
            print(
                "  Creating new vector DB (parameters changed or DB not found)")
            # Close previous DB instance if it exists to release file handles
            # This is important on Windows before deleting the directory
            if VectorLogFilter._db_cache:
                VectorLogFilter._db_cache.db.close()
                VectorLogFilter._db_cache = None

            # Clear out the working directory first.
            # After closing, we should be able to delete without ignore_errors
            # but keep it as a safety measure
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=True)
            os.makedirs(directory, exist_ok=True)

            lines = self.filter_by_date()
            # Storing logs filtered by date
            filtered_logs_by_date_file_path = Path(
                f"{directory}/filtered_logs_by_date.txt")
            with open(filtered_logs_by_date_file_path, "w") as f:
                f.writelines(lines)

            # Create new DB
            db = VectorDb(
                persist_directory=directory,
                input_document_path=filtered_logs_by_date_file_path,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                embedding_model_name=self.config.embedding_model_name,
            )
            filtered_logs_by_date_file_path.unlink()

            # Cache signature and instance in memory for future reuse
            VectorLogFilter._db_cache = DbCache(db=db,
                                                db_signature=self._get_current_db_signature())

        # Perform search with current issue_description
        # (issue_description can change without needing to recreate DB)
        db_entries = db.search(self.config.issue_description,
                               k=self.get_number_of_filtered_entries())

        # Convert response to json
        filter_response = [
            asdict(
                VectorLogFilterResponse(
                    logs=entry.chunk,
                    score=entry.score)
            )
            for entry in db_entries
        ]
        json_str = json.dumps(filter_response, indent=2)

        # Storing logs filtered by date and context
        filtered_logs_file_path = f"{directory}/filtered_logs.txt"
        with open(filtered_logs_file_path, "w") as f:
            f.writelines(json_str)

        return json_str

    def get_number_of_filtered_entries(self) -> int:
        k = self.config.max_chars // self.config.chunk_size
        return k

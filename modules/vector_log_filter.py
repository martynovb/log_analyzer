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

    def __init__(self, config: VectorLogFilterConfig):
        super().__init__(config)
        self.config = config

    def filter(self) -> str:
        directory = VectorLogFilter.WORKING_DIRECTORY

        # Clear out the working directory first.
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

        lines = self.filter_by_date()
        # Storing logs filtered by date
        filtered_logs_by_date_file_path = f"{directory}/filtered_logs_by_date.txt"
        with open(filtered_logs_by_date_file_path, "w") as f:
            f.writelines(lines)

        db = VectorDb(input_document_path=filtered_logs_by_date_file_path,
                      output_directory=directory)
        results = db.search(self.config.issue_description)

        # Storing logs filtered by date and context
        filtered_logs_file_path = f"{directory}/filtered_logs.txt"
        with open(filtered_logs_file_path, "w") as f:
            f.writelines(results)

        return "\n".join(results)

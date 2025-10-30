import hashlib
import os
from pathlib import Path


class FilteredLogsCache:
    """Disk-backed cache for filtered logs keyed by log content and params."""

    def __init__(self, base_dir: Path | None = None):
        project_root = Path(__file__).resolve().parents[1]
        default_dir = project_root / 'assets' / 'results' / 'cache'
        self.base_dir = base_dir or default_dir
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _compute_log_identity(self, file_path: str) -> str:
        if not file_path:
            return ""
        sha1 = hashlib.sha1()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    sha1.update(chunk)
            return sha1.hexdigest()
        except Exception:
            try:
                mtime = str(os.path.getmtime(file_path))
            except Exception:
                mtime = "0"
            return hashlib.sha1((file_path + '|' + mtime).encode('utf-8')).hexdigest()

    def _cache_file_path(self,
                         log_file_path: str,
                         issue_description: str | None,
                         filter_mode: str | None,
                         start_date: str | None,
                         end_date: str | None) -> Path:
        log_id = self._compute_log_identity(log_file_path or "")
        key_str = "|".join([
            log_id,
            issue_description or "",
            filter_mode or "",
            start_date or "",
            end_date or "",
        ])
        key_hash = hashlib.sha1(key_str.encode('utf-8')).hexdigest()
        return self.base_dir / f"filtered_{key_hash}.txt"

    def get(self,
            log_file_path: str,
            issue_description: str | None,
            filter_mode: str | None,
            start_date: str | None,
            end_date: str | None) -> str | None:
        cache_file = self._cache_file_path(log_file_path, issue_description,
                                           filter_mode, start_date, end_date)
        try:
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = f.read()
                if data:
                    print(f"Using cached filtered logs from previous session: {cache_file}")
                    return data
        except Exception:
            return None
        return None

    def save(self,
             log_file_path: str,
             issue_description: str | None,
             filter_mode: str | None,
             start_date: str | None,
             end_date: str | None,
             logs: str) -> None:
        cache_file = self._cache_file_path(log_file_path, issue_description,
                                           filter_mode, start_date, end_date)
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(logs or "")
        except Exception:
            pass



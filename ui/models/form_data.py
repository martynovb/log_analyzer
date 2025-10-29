from dataclasses import dataclass
from flask import request


@dataclass
class FormData:
    issue_description: str
    start_date: str
    end_date: str
    max_tokens: int
    context_lines: int
    filter_mode: str
    deduplicate: bool
    prioritize_severity: bool
    llm_url: str
    llm_model: str
    llm_timeout: int
    llm_max_tokens: int

    @classmethod
    def from_request(cls) -> "FormData":
        """Create and validate FormData instance from Flask request.form."""
        form = request.form

        instance = cls(
            issue_description=form.get('issue_description', '').strip(),
            start_date=form.get('start_date', ''),
            end_date=form.get('end_date', ''),
            max_tokens=int(form.get('max_tokens', 3500)),
            context_lines=int(form.get('context_lines', 2)),
            filter_mode=form.get('filter_mode', 'llm'),
            deduplicate=form.get('deduplicate') == 'on',
            prioritize_severity=form.get('prioritize_severity') == 'on',
            llm_url=form.get('llm_url', '').strip(),
            llm_model=form.get('llm_model', '').strip(),
            llm_timeout=int(form.get('llm_timeout', 120)),
            llm_max_tokens=int(form.get('llm_max_tokens', 1000))
        )

        instance.validate()
        return instance

    def validate(self) -> None:
        """Raise ValueError if required fields are missing or invalid."""
        if not self.issue_description:
            raise ValueError("Issue description is required.")
        if not self.llm_url or not self.llm_model:
            raise ValueError("LLM URL and Model name are required.")

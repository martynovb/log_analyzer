# Log Analyzer

An intelligent log analysis tool with LLM-powered keyword extraction, context retrieval, and automated issue analysis.

## 🚀 Installation

### Prerequisites
- Python 3.10+ (3.11, 3.12, or 3.13 recommended)
- pip (Python package installer)
- Optional: Local LLM server (LM Studio or Ollama) for keyword extraction

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd log_analyzer

# Create virtual environment (recommended)
python -m venv log_analyzer_env

# Activate virtual environment
# Windows:
log_analyzer_env\Scripts\activate
# macOS/Linux:
source log_analyzer_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🏃 Running the Application

### Web UI (Recommended)

```bash
# Start the web server
python ui/app.py

# Open browser to: http://localhost:5000
```

**Usage:**
1. Upload a log file
2. Describe the issue
3. **Select filter method:**
   - **LLM keywords**: Extracts keywords and filters log (requires LLM setup)
   - **Vector DB**: Semantic search (mock implementation)
4. Configure analysis options
5. Click "Analyze Logs"

### CLI Tool

```bash
# Basic usage
python main.py --log-file app.log --keywords "error,timeout"

# Advanced usage
python main.py --log-file app.log --keywords "snapshot,periodic" \
    --start-date 2025-10-15 --end-date 2025-10-16 \
    --max-tokens 3500 --deduplicate --prioritize-severity
```

## 📁 Project Structure

```
log_analyzer/
├── main.py                    # CLI entry point
├── log_analyzer_system.py     # Orchestration layer (for web UI)
│
├── modules/                   # Core modules (REPLACEABLE)
│   ├── __init__.py           # Exports all modules
│   ├── domain.py             # Data models (Request, Result)
│   ├── result_handler.py     # Result parsing & saving
│   ├── log_analyzer.py       # Log filtering & analysis
│   ├── keyword_extractor.py  # LLM-based keyword extraction
│   ├── context_retriever.py  # Context from JSON files
│   └── prompt_generator.py   # Prompt generation for LLM
│
├── ui/
│   ├── app.py                # Flask web application
│   └── templates/
│       └── index.html        # Web UI
│
├── assets/
│   ├── context/              # JSON context files
│   │   ├── components/       # Codebase components
│   │   ├── docs/            # Documentation
│   │   └── errors/           # Error patterns
│   ├── uploads/              # Uploaded log files
│   └── results/              # Analysis results
│
├── requirements.txt
└── README.md
```

## 📝 What Each File Does

### Entry Points
- **`main.py`** - CLI tool for command-line log analysis
- **`ui/app.py`** - Web UI server (Flask application)

### Orchestration
- **`log_analyzer_system.py`** - Orchestrates analysis workflow, coordinates all modules

### Core Modules (All in `modules/`)
- **`domain.py`** - Data models (AnalysisRequest, AnalysisResult)
- **`log_analyzer.py`** - Filters and analyzes log files using keyword matching
- **`keyword_extractor.py`** - Extracts keywords from issue descriptions using LLM
- **`vector_filter.py`** - Filters logs using vector DB semantic search (mock)
- **`context_retriever.py`** - Retrieves relevant context from JSON files
- **`prompt_generator.py`** - Generates prompts for LLM analysis
- **`result_handler.py`** - Parses and saves analysis results

### Context Data
- **`assets/context/components.json`** - Codebase components
- **`assets/context/documentation.json`** - Documentation
- **`assets/context/errors.json`** - Error patterns and solutions

## 🔧 How to Replace Modules

All modules in `modules/` can be independently replaced.

### Example 1: Replacing Keyword Extractor

1. Create new file: `modules/my_keyword_extractor.py`
2. Implement the same interface:
```python
class MyKeywordExtractor:
    def extract_keywords(self, issue_description: str) -> List[ExtractedKeyword]:
        # Your implementation
        pass
    
    def is_llm_available(self) -> bool:
        # Your implementation
        pass
```

3. Update `modules/__init__.py`:
```python
from .my_keyword_extractor import MyKeywordExtractor
__all__ = [..., 'MyKeywordExtractor']
```

4. Update imports in `log_analyzer_system.py`:
```python
from modules import MyKeywordExtractor
self.keyword_extractor = MyKeywordExtractor()
```

### Example 2: Implementing Real Vector DB Filter

Replace `MockVectorLogFilter` in `modules/vector_filter.py`:

```python
class MyVectorLogFilter(VectorLogFilter):
    def filter(self, issue_description: str, log_file_path: str) -> str:
        # Connect to your vector DB
        # Embed issue description
        # Find similar log entries
        # Return filtered logs
        pass
```

### Module Interface Requirements

| Module | Key Methods |
|--------|------------|
| **KeywordExtractor** | `extract_keywords(description) -> List[ExtractedKeyword]`, `is_llm_available() -> bool` |
| **VectorLogFilter** | `filter(issue_description, log_file_path) -> str` |
| **ContextRetriever** | `retrieve_codebase_context(keywords) -> dict`, `retrieve_documentation_context(keywords) -> dict`, `retrieve_error_context(keywords) -> dict` |
| **PromptGenerator** | `format_context(context, type) -> str`, `generate_prompt(analysis_data) -> str` |
| **LogAnalyzer** | `analyze(keywords, start_date, end_date) -> str` |
| **ResultHandler** | `parse_filtered_logs(logs) -> dict`, `save_result(result, output_dir, timestamp) -> str` |

## 🎯 Quick Start

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run web UI**: `python ui/app.py`
3. **Open browser**: `http://localhost:5000`
4. **Upload log file** and describe issue
5. **Get AI-powered analysis**

## 🤖 LLM Setup (Optional)

The system uses LLM for keyword extraction. Configure it in the web UI:

1. Start LM Studio or Ollama
2. Enter LLM URL (default: `http://127.0.0.1:1234`)
3. Enter model name
4. Click "Analyze Logs"

Without LLM, the system will prompt you to enter keywords manually.

## 📊 Output

- **CLI**: Saves filtered logs to `filtered_logs.txt` (or custom path)
- **Web UI**: Saves full analysis results to `assets/results/analysis_result_<timestamp>.json`

## 🔍 Troubleshooting

**Import errors**: Make sure you're in the project root and have activated the virtual environment.

**Flask not found**: Run `pip install -r requirements.txt`

**Port 5000 in use**: Change port in `ui/app.py` or kill the process using that port.

**LLM connection failed**: Ensure your LLM server (LM Studio/Ollama) is running and the URL/model are correct.

# Log Analyzer - Modular Architecture

A comprehensive log analysis system with intelligent issue analysis, keyword extraction, and context-aware prompt generation.

## 🚀 Complete Setup Guide (From Scratch)

### Prerequisites
- **Python 3.10+** (3.11, 3.12, 3.13 recommended)
- **pip** (Python package installer)
- **Git** (for cloning the repository)
- **Web browser** (for the UI)
- **Optional**: Local LLM server (Ollama) for enhanced keyword extraction

### Step 1: Environment Setup

#### Option A: Using Virtual Environment (Recommended)
```bash
# Create a new virtual environment
python -m venv log_analyzer_env

# Activate the virtual environment
# On Windows:
log_analyzer_env\Scripts\activate
# On macOS/Linux:
source log_analyzer_env/bin/activate

# Verify Python version
python --version
```

#### Option B: Using Conda
```bash
# Create a new conda environment
conda create -n log_analyzer python=3.11
conda activate log_analyzer
```

### Step 2: Clone and Install Dependencies

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd log_analyzer

# Install all required dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

**Expected dependencies:**
- Flask==2.3.3
- Werkzeug==2.3.7
- Jinja2==3.1.2
- MarkupSafe==2.1.3
- itsdangerous==2.1.2
- click==8.1.7
- blinker==1.6.2
- requests==2.31.0

### Step 3: Test Installation

```bash
# Test CLI functionality
python main.py --help

# Test with sample keywords (if you have app.log)
python main.py --keywords "test,error" --output test_output.txt
```

### Step 4: Run the Web UI

```bash
# Start the Flask web server
python ui/app.py
```

**Expected output:**
```
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://[your-ip]:5000
 * Debug mode: on
```

**Access the UI:**
- Open your web browser
- Navigate to: `http://localhost:5000` or `http://127.0.0.1:5000`

### Step 5: Optional - LLM Enhancement Setup

For enhanced keyword extraction using AI:

#### Install Ollama (Local LLM Server)
1. **Download Ollama**: Visit https://ollama.ai and download for your OS
2. **Install and start Ollama**:
   ```bash
   # After installation, start the server
   ollama serve
   ```
3. **Pull a model**:
   ```bash
   # Pull a recommended model
   ollama pull mistral
   # Or use the default configured model
   ollama pull mistralai/mistral-7b-instruct-v0.3
   ```

#### Configure LLM Settings (if needed)
The system is pre-configured for:
- **Base URL**: `http://127.0.0.1:1234` (default Ollama port)
- **Model**: `mistralai/mistral-7b-instruct-v0.3`

To change these settings, edit `modules/keyword_extractor.py`:
```python
# In LocalLLMInterface.__init__()
base_url: str = "http://127.0.0.1:1234",  # Change port if needed
model: str = "mistralai/mistral-7b-instruct-v0.3",  # Change model if needed
```

## 🎯 Quick Start Guide

### CLI Usage
```bash
# Basic analysis
python main.py --log-file app.log --keywords "error,timeout,exception"

# Advanced analysis with date range
python main.py --log-file app.log --keywords "snapshot,periodic" --start-date 2025-10-15 --end-date 2025-10-16 --deduplicate --prioritize-severity

# Help and all options
python main.py --help
```

### Web UI Usage
1. **Start the UI**: `python ui/app.py`
2. **Open browser**: `http://localhost:5000`
3. **Upload log file**: Click "Choose File" and select your `.log` file
4. **Describe the issue**: Enter a detailed description of the problem
5. **Preview keywords**: Click "Preview Keywords" to see extracted keywords
6. **Configure analysis**: Set date range, context lines, etc.
7. **Run analysis**: Click "Analyze Logs"
8. **Download results**: Click "Download Full Results"

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'log_analyzer_system'
# Solution: Ensure you're in the project root directory
cd /path/to/log_analyzer
python ui/app.py
```

#### 2. Flask Not Found
```bash
# Error: ModuleNotFoundError: No module named 'flask'
# Solution: Install dependencies
pip install -r requirements.txt
```

#### 3. Virtual Environment Issues
```bash
# If virtual environment is not activated
# Windows:
log_analyzer_env\Scripts\activate
# macOS/Linux:
source log_analyzer_env/bin/activate
```

#### 4. Port Already in Use
```bash
# Error: Address already in use
# Solution: Kill existing process or use different port
# Kill process on port 5000:
# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F
# macOS/Linux:
lsof -ti:5000 | xargs kill -9
```

#### 5. LLM Connection Issues
```bash
# If LLM is not available, the system automatically falls back to heuristic keyword extraction
# Check Ollama status:
curl http://127.0.0.1:1234/api/tags
# If not running, start Ollama:
ollama serve
```

### File Structure Verification
Ensure your project structure looks like this:
```
log_analyzer/
├── main.py
├── log_analyzer_system.py
├── modules/
│   ├── keyword_extractor.py
│   ├── context_retriever.py
│   └── ...
├── ui/
│   ├── app.py
│   └── templates/
│       └── index.html
├── requirements.txt
└── README.md
```

## 📊 Output Locations

- **CLI Output**: `filtered_logs.txt` (or custom `--output` path)
- **Web UI Results**: `analysis_results/analysis_result_<timestamp>.json`
- **Web UI Preview**: Displayed in browser after analysis

## 🧪 Testing the Installation

### Test CLI
```bash
# Create a test log file
echo "2025-10-23 10:00:00 ERROR: Test error message" > test.log
echo "2025-10-23 10:01:00 INFO: Test info message" >> test.log

# Test analysis
python main.py --log-file test.log --keywords "error,test" --output test_result.txt

# Check results
cat test_result.txt
```

### Test Web UI
1. Start UI: `python ui/app.py`
2. Open: `http://localhost:5000`
3. Upload the test.log file
4. Enter issue description: "Application has test errors"
5. Click "Analyze Logs"
6. Verify results appear

## 🎉 Success Indicators

You'll know everything is working when:
- ✅ CLI runs without errors: `python main.py --help`
- ✅ Web UI loads: `http://localhost:5000` shows the upload form
- ✅ Analysis completes: Results are generated and saved
- ✅ LLM integration (optional): Keywords show "llm" extraction method

---

## 🧩 Architecture Overview

The system follows a modular, object-oriented architecture with clear separation of concerns:

```
log_analyzer/
├── main.py                          # CLI entrypoint (minimal)
├── log_analyzer_system.py           # Modular orchestration (keyword extraction, context, prompts)
├── modules/
│   ├── keyword_extractor.py         # LLM + heuristic keyword extraction
│   ├── context_retriever.py         # Context retriever interfaces/impls
│   ├── prompt_generator.py          # Prompt creation (if used separately)
│   └── components.py                # Other shared components
├── ui/
│   ├── app.py                       # Flask web application
│   └── templates/
│       └── index.html               # Web UI template
├── analysis_results/                # Saved analysis outputs (UI)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🧪 Testing

Run the system-level tests (if present):
```bash
python test_system.py
```

## 🔌 Extensibility
- Swap in real context retrievers by implementing `ContextRetrieverInterface`.
- Add custom formatters by extending the `LogFormatter` strategy.
- Tweak keyword extraction by configuring `LocalLLMInterface` or changing rule patterns.

## ❓ Troubleshooting
- If no output: verify `--keywords` are provided or the issue description contains extractable terms.
- If LLM not used: ensure your LLM server is running and reachable; otherwise the system falls back to rules.
- Large logs: consider narrowing with `--start-date`/`--end-date` and using `--deduplicate`.

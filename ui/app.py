"""
Simple Web UI for Log Analyzer
==============================

A Flask-based web interface for uploading logs and describing issues.
"""

import sys
from pathlib import Path

# Add project root to Python path BEFORE importing project modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ui.models.form_data import FormData

from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime

from log_analyzer_system import LogAnalysisOrchestrator
from modules import LocalLLMInterface, AnalysisRequest

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Store for uploaded files - use assets folder structure
project_root = Path(__file__).parent.parent
UPLOAD_FOLDER = str(project_root / 'assets' / 'uploads')
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)


@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_log():
    """Handle log file upload and analysis request."""
    try:
        form_data = FormData.from_request()
        orchestrator = LogAnalysisOrchestrator()
        configure_llm(orchestrator, form_data)
        filepath, timestamp = handle_file_upload(UPLOAD_FOLDER)
        analysis_request = create_analysis_request(filepath, form_data)
        result, output_file = run_analysis(orchestrator, analysis_request,
                                           timestamp)

        return jsonify({
            'success': True,
            'analysis_id': timestamp,
            'extracted_keywords': result.extracted_keywords,
            'processing_time_ms': result.processing_time_ms,
            'processing_time_formatted': result.processing_time_formatted(),
            'filter_mode': form_data.filter_mode,
            'codebase_files': result.context_info['codebase']['total_files'],
            'documentation_items': result.context_info['documentation'][
                'total_docs'],
            'result_file': output_file,
            'llm_analysis': result.llm_analysis
        })

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except RuntimeError as re:
        return jsonify({'error': str(re)}), 400
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


def configure_llm(orchestrator: LogAnalysisOrchestrator, form_data: FormData):
    """Create and attach a custom LLM interface based on form data."""
    try:
        custom_llm = LocalLLMInterface(
            base_url=form_data.llm_url,
            model=form_data.llm_model,
            timeout=form_data.llm_timeout,
            max_tokens=form_data.llm_max_tokens
        )
        orchestrator.llm_interface = custom_llm
        orchestrator.keyword_extractor.llm_interface = custom_llm
    except Exception as e:
        raise RuntimeError(f'Failed to configure LLM: {str(e)}')


def handle_file_upload(upload_folder: str) -> tuple[Path, str]:
    """Validate and save uploaded log file, returning (filepath, timestamp)."""
    if 'log_file' not in request.files:
        raise ValueError('No log file uploaded')

    file = request.files['log_file']
    if not file.filename:
        raise ValueError('No file selected')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    filepath = Path(upload_folder) / filename
    file.save(filepath)

    return filepath, timestamp


def create_analysis_request(filepath: Path,
                            form_data: FormData) -> AnalysisRequest:
    """Build an AnalysisRequest object from the parsed form data."""
    return AnalysisRequest(
        log_file_path=str(filepath),
        issue_description=form_data.issue_description,
        filter_mode=form_data.filter_mode,
        start_date=form_data.start_date or None,
        end_date=form_data.end_date or None,
        max_tokens=form_data.max_tokens,
        context_lines=form_data.context_lines,
        deduplicate=form_data.deduplicate,
        prioritize_by_severity=form_data.prioritize_severity
    )


def run_analysis(orchestrator: LogAnalysisOrchestrator,
                 analysis_request: AnalysisRequest,
                 timestamp: str):
    """Execute the orchestrator analysis and save results."""
    result = orchestrator.analyze_issue(analysis_request)
    # Save result - use assets folder structure with consistent timestamp
    project_root = Path(__file__).parent.parent
    output_file = orchestrator.save_result(
        result, str(project_root / 'assets' / 'results'), timestamp
    )
    return result, output_file


@app.route('/download/<analysis_id>')
def download_result(analysis_id):
    """Download analysis result file."""
    try:
        # Find the result file - use assets folder structure
        project_root = Path(__file__).parent.parent
        results_dir = project_root / 'assets' / 'results'

        # Debug: print the search pattern and directory
        print(
            f"Searching for files matching: analysis_result_{analysis_id}*.json")
        print(f"Search directory: {results_dir}")

        result_files = list(
            results_dir.glob(f'analysis_result_{analysis_id}*.json'))

        print(f"Found files: {result_files}")

        if not result_files:
            return jsonify({'error': 'Analysis result not found'}), 404

        # Get the most recent file if multiple matches
        result_file = max(result_files, key=lambda f: f.stat().st_mtime)

        return send_file(result_file, as_attachment=True,
                         download_name=f'analysis_result_{analysis_id}.json')

    except Exception as e:
        print(f"Download error: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """API endpoint to fetch available LLM models from LM Studio."""
    try:
        llm_url = request.args.get('url', 'http://127.0.0.1:1234').strip()
        if not llm_url:
            return jsonify({'error': 'URL parameter is required'}), 400

        # Ensure URL doesn't end with a slash
        llm_url = llm_url.rstrip('/')

        # Fetch models from LM Studio
        import requests
        models_url = f"{llm_url}/v1/models"

        try:
            response = requests.get(models_url, timeout=5)
            response.raise_for_status()
            data = response.json()

            # Extract model IDs from the response
            models = []
            if 'data' in data and isinstance(data['data'], list):
                for model in data['data']:
                    if 'id' in model:
                        models.append({
                            'id': model['id'],
                            'name': model.get('id', '').split('/')[-1],
                            # Extract model name
                            'object': model.get('object', 'model'),
                            'owned_by': model.get('owned_by', '')
                        })

            return jsonify({
                'success': True,
                'models': models,
                'url': llm_url
            })
        except requests.exceptions.RequestException as e:
            return jsonify({
                'success': False,
                'error': f'Failed to connect to LLM server: {str(e)}',
                'models': []
            }), 200  # Return 200 but with error message

    except Exception as e:
        return jsonify({'error': f'Failed to fetch models: {str(e)}'}), 500


if __name__ == '__main__':
    print("Starting Log Analyzer Web UI...")
    print("Open your browser and go to: http://localhost:5000")

    app.run(debug=True, host='0.0.0.0', port=5000)

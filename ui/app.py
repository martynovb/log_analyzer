"""
Simple Web UI for Log Analyzer
==============================

A Flask-based web interface for uploading logs and describing issues.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, request, jsonify, send_file
import tempfile
from datetime import datetime
import json

from log_analyzer_system import LogAnalysisOrchestrator
from modules import LocalLLMInterface, MockLLMInterface, AnalysisRequest, AnalysisResult

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global orchestrator instance
orchestrator = LogAnalysisOrchestrator()

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
        # Get form data
        issue_description = request.form.get('issue_description', '')
        start_date = request.form.get('start_date', '')
        end_date = request.form.get('end_date', '')
        max_tokens = int(request.form.get('max_tokens', 3500))
        context_lines = int(request.form.get('context_lines', 2))
        filter_mode = request.form.get('filter_mode', 'llm')
        deduplicate = request.form.get('deduplicate') == 'on'
        prioritize_severity = request.form.get('prioritize_severity') == 'on'

        # Get LLM configuration from form
        llm_url = request.form.get('llm_url', '').strip()
        llm_model = request.form.get('llm_model', '').strip()
        llm_timeout = int(request.form.get('llm_timeout', 120))
        llm_max_tokens = int(request.form.get('llm_max_tokens', 1000))

        # Configure LLM (required for both filter modes for final analysis)
        if not llm_url or not llm_model:
            return jsonify({'error': 'LLM URL and Model name are required'}), 400

        try:
            custom_llm = LocalLLMInterface(
                base_url=llm_url,
                model=llm_model,
                timeout=llm_timeout,
                max_tokens=llm_max_tokens
            )
            orchestrator.keyword_extractor.llm_interface = custom_llm
        except Exception as e:
            return jsonify({'error': f'Failed to configure LLM: {str(e)}'}), 400

        # Handle file upload
        if 'log_file' not in request.files:
            return jsonify({'error': 'No log file uploaded'}), 400
        
        file = request.files['log_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not issue_description.strip():
            return jsonify({'error': 'Issue description is required'}), 400
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = Path(UPLOAD_FOLDER) / filename
        
        file.save(filepath)
        
        # Create analysis request
        analysis_request = AnalysisRequest(
            log_file_path=str(filepath),
            issue_description=issue_description,
            filter_mode=filter_mode,
            start_date=start_date if start_date else None,
            end_date=end_date if end_date else None,
            max_tokens=max_tokens,
            context_lines=context_lines,
            deduplicate=deduplicate,
            prioritize_by_severity=prioritize_severity
        )
        
        # Perform analysis
        result = orchestrator.analyze_issue(analysis_request)
        
        # Save result - use assets folder structure with consistent timestamp
        project_root = Path(__file__).parent.parent
        output_file = orchestrator.save_result(result, str(project_root / 'assets' / 'results'), timestamp)
        
        # Format processing time for readable display
        processing_time_ms = result.processing_time_ms
        processing_seconds = processing_time_ms / 1000.0
        
        def format_time(seconds):
            """Format time in a readable format."""
            if seconds < 1:
                return f"{int(seconds * 1000)}ms"
            elif seconds < 60:
                return f"{seconds:.1f} seconds"
            else:
                minutes = int(seconds // 60)
                secs = seconds % 60
                if secs < 1:
                    return f"{minutes} minute{'s' if minutes != 1 else ''}"
                else:
                    return f"{minutes} minute{'s' if minutes != 1 else ''} {secs:.1f} seconds"
        
        processing_time_formatted = format_time(processing_seconds)
        
        # Return analysis result
        return jsonify({
            'success': True,
            'analysis_id': timestamp,
            'extracted_keywords': result.extracted_keywords,
            'processing_time_ms': processing_time_ms,
            'processing_time_formatted': processing_time_formatted,
            'filter_mode': filter_mode,
            'codebase_files': result.context_info['codebase']['total_files'],
            'documentation_items': result.context_info['documentation']['total_docs'],
            'result_file': output_file,
            'llm_analysis': result.llm_analysis
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/download/<analysis_id>')
def download_result(analysis_id):
    """Download analysis result file."""
    try:
        # Find the result file - use assets folder structure
        project_root = Path(__file__).parent.parent
        results_dir = project_root / 'assets' / 'results'
        
        # Debug: print the search pattern and directory
        print(f"Searching for files matching: analysis_result_{analysis_id}*.json")
        print(f"Search directory: {results_dir}")
        
        result_files = list(results_dir.glob(f'analysis_result_{analysis_id}*.json'))
        
        print(f"Found files: {result_files}")
        
        if not result_files:
            return jsonify({'error': 'Analysis result not found'}), 404
        
        # Get the most recent file if multiple matches
        result_file = max(result_files, key=lambda f: f.stat().st_mtime)
        
        return send_file(result_file, as_attachment=True, download_name=f'analysis_result_{analysis_id}.json')
        
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
                            'name': model.get('id', '').split('/')[-1],  # Extract model name
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


@app.route('/api/keywords', methods=['POST'])
def extract_keywords():
    """API endpoint to extract keywords from issue description."""
    try:
        data = request.get_json()
        issue_description = data.get('issue_description', '')
        llm_config = data.get('llm_config', {})

        if not issue_description.strip():
            return jsonify({'error': 'Issue description is required'}), 400

        # Configure LLM if provided
        if llm_config and llm_config.get('url') and llm_config.get('model'):
            try:
                custom_llm = LocalLLMInterface(
                    base_url=llm_config.get('url', ''),
                    model=llm_config.get('model', ''),
                    timeout=llm_config.get('timeout', 120),
                    max_tokens=llm_config.get('max_tokens', 1000)
                )
                orchestrator.keyword_extractor.llm_interface = custom_llm
            except Exception as e:
                return jsonify({'error': f'Failed to configure LLM: {str(e)}'}), 400

        # Extract keywords using the LLM-only method
        keywords = orchestrator.keyword_extractor.extract_keywords(issue_description)

        # Convert to serializable format
        keywords_data = []
        for kw in keywords:
            keywords_data.append({
                'keyword': kw.keyword,
                'type': kw.keyword_type.value,
                'confidence': kw.confidence,
                'context': kw.context,
                'extraction_method': kw.extraction_method
            })

        return jsonify({
            'keywords': keywords_data,
            'llm_used': orchestrator.keyword_extractor.is_llm_available(),
            'extraction_methods': orchestrator.keyword_extractor.get_extraction_methods_used(issue_description)
        })

    except Exception as e:
        return jsonify({'error': f'Keyword extraction failed: {str(e)}'}), 500


if __name__ == '__main__':
    print("Starting Log Analyzer Web UI...")
    print("Open your browser and go to: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

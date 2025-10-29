"""
Log Analyzer Orchestration Layer

This module provides orchestration logic that coordinates all analysis components.
All business logic is in the modules directory.
"""

from typing import List
from datetime import datetime

# Import from modules - clean, modular approach
from modules import (
    LogAnalyzer, LogFilterConfig,
    KeywordExtractor,
    ContextRetriever,
    PromptGenerator, AnalysisData,
    AnalysisRequest, AnalysisResult,
    ResultHandler,
    VectorLogFilterImpl
)


class LogAnalysisOrchestrator:
    """Main orchestrator that coordinates all analysis components."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.keyword_extractor = KeywordExtractor()
        self.context_retriever = ContextRetriever()
        self.prompt_generator = PromptGenerator()
        self.result_handler = ResultHandler()
    
    def analyze_issue(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform complete issue analysis by orchestrating all modules.
        
        Args:
            request: Analysis request parameters
            
        Returns:
            Complete analysis result
        """
        start_time = datetime.now()
        
        def format_time(seconds: float) -> str:
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
        
        def print_step_time(step_name: str, step_start: datetime, step_end: datetime):
            """Print time taken for a step."""
            elapsed = (step_end - step_start).total_seconds()
            print(f"  ✓ {step_name} completed in {format_time(elapsed)}")
        
        # Step 1: Extract keywords from issue description (skip LLM if vector mode)
        filter_mode = getattr(request, 'filter_mode', 'llm')
        
        step1_start = datetime.now()
        if filter_mode == 'vector':
            # For vector DB mode, skip LLM keyword extraction
            print("Step 1: Skipping keyword extraction (vector DB mode - keywords not needed for filtering)...")
            extracted_keywords = []
        else:
            # For LLM/keyword mode, use LLM-based keyword extraction
            print("Step 1: Extracting keywords from issue description using LLM...")
            extracted_keywords_objects = self.keyword_extractor.extract_keywords(request.issue_description)
            extracted_keywords = [kw.keyword for kw in extracted_keywords_objects]
        
        # Add any manually provided keywords
        if request.keywords:
            extracted_keywords.extend(request.keywords)
        
        # Remove duplicates
        all_keywords = list(set(extracted_keywords))
        step1_end = datetime.now()
        if filter_mode != 'vector':
            print(f"  Extracted keywords: {all_keywords}")
        print_step_time("Step 1: Keyword extraction", step1_start, step1_end)
        
        # Step 2: Analyze logs (branch by filter mode)
        step2_start = datetime.now()
        print(f"Step 2: Analyzing logs using {'Vector DB' if filter_mode == 'vector' else 'keyword-based'} approach...")
        if filter_mode == 'vector':
            vector_filter = VectorLogFilterImpl()
            filtered_logs = vector_filter.filter(
                issue_description=request.issue_description,
                log_file_path=request.log_file_path,
            )
        else:
            config = LogFilterConfig(
                log_file_path=request.log_file_path,
                max_tokens=request.max_tokens,
                context_lines=request.context_lines,
                deduplicate=request.deduplicate,
                prioritize_by_severity=request.prioritize_by_severity,
                prioritize_matches=True,
                max_results=None
            )
            analyzer = LogAnalyzer(config)
            filtered_logs = analyzer.analyze(
                keywords=all_keywords,
                start_date=request.start_date,
                end_date=request.end_date
            )
        step2_end = datetime.now()
        print_step_time("Step 2: Log filtering", step2_start, step2_end)
        
        # Step 3: Retrieve context
        step3_start = datetime.now()
        print("Step 3: Retrieving codebase, documentation and error contexts...")
        codebase_context = self.context_retriever.retrieve_codebase_context(all_keywords)
        documentation_context = self.context_retriever.retrieve_documentation_context(all_keywords)
        error_context = self.context_retriever.retrieve_error_context(all_keywords)
        step3_end = datetime.now()
        print_step_time("Step 3: Context retrieval", step3_start, step3_end)
        
        # Step 4: Format context and create comprehensive prompt
        step4_start = datetime.now()
        print("Step 4: Creating analysis prompt...")
        
        # Format context using PromptGenerator
        codebase_text = self.prompt_generator.format_context(codebase_context, "Codebase")
        docs_text = self.prompt_generator.format_context(documentation_context, "Documentation")
        errors_text = self.prompt_generator.format_context(error_context, "Errors")
        parts = [p for p in [codebase_text, docs_text, errors_text] if p]
        combined_context = "\n\n".join(parts)
        
        # Create AnalysisData for prompt generation
        analysis_data = AnalysisData(
            issue_description=request.issue_description,
            extracted_keywords=all_keywords,
            filtered_logs=filtered_logs,
            context_description=combined_context,
            log_file_path=request.log_file_path,
            analysis_date_range=f"{request.start_date} to {request.end_date}" if request.start_date and request.end_date else request.start_date or request.end_date
        )
        
        # Generate prompt using PromptGenerator
        generated_prompt = self.prompt_generator.generate_prompt(analysis_data)
        step4_end = datetime.now()
        print_step_time("Step 4: Prompt generation", step4_start, step4_end)
        
        # Step 5: Get LLM analysis
        step5_start = datetime.now()
        print("Step 5: Getting LLM analysis...")
        llm_analysis = None
        try:
            llm_analysis = self.keyword_extractor.llm_interface.analyze_logs(generated_prompt)
            step5_end = datetime.now()
            print_step_time("Step 5: LLM analysis", step5_start, step5_end)
        except Exception as e:
            step5_end = datetime.now()
            print(f"  ✗ LLM analysis failed: {e}")
            print_step_time("Step 5: LLM analysis (failed)", step5_start, step5_end)
            llm_analysis = f"LLM analysis failed: {str(e)}"
        
        # Calculate total processing time
        end_time = datetime.now()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        total_seconds = (end_time - start_time).total_seconds()
        print(f"\n{'='*60}")
        print(f"Total processing time: {format_time(total_seconds)}")
        print(f"{'='*60}\n")
        
        # Create result
        result = AnalysisResult(
            request=request,
            extracted_keywords=all_keywords,
            filtered_logs=filtered_logs,
            context_info={
                'codebase': codebase_context,
                'documentation': documentation_context,
                'errors': error_context
            },
            generated_prompt=generated_prompt,
            llm_analysis=llm_analysis,
            timestamp=end_time,
            processing_time_ms=processing_time_ms
        )
        
        return result
    
    def save_result(self, result: AnalysisResult, output_dir: str = "analysis_results", custom_timestamp: str = None) -> str:
        """
        Save analysis result to file.
        
        Args:
            result: Analysis result to save
            output_dir: Directory to save results
            custom_timestamp: Optional custom timestamp string to use for filename
            
        Returns:
            Path to saved file
        """
        return self.result_handler.save_result(result, output_dir, custom_timestamp)

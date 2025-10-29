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
        
        # Step 1: Extract keywords from issue description
        print("Extracting keywords from issue description...")
        extracted_keywords_objects = self.keyword_extractor.extract_keywords(request.issue_description)
        extracted_keywords = [kw.keyword for kw in extracted_keywords_objects]
        
        # Add any manually provided keywords
        if request.keywords:
            extracted_keywords.extend(request.keywords)
        
        # Remove duplicates
        all_keywords = list(set(extracted_keywords))
        print(f"Extracted keywords: {all_keywords}")
        
        # Step 2: Analyze logs (branch by filter mode)
        print("Analyzing logs...")
        if getattr(request, 'filter_mode', 'llm') == 'vector':
            print("Using vector DB approach")
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
        
        # Step 3: Retrieve context
        print("Retrieving codebase, documentation and error contexts...")
        codebase_context = self.context_retriever.retrieve_codebase_context(all_keywords)
        documentation_context = self.context_retriever.retrieve_documentation_context(all_keywords)
        error_context = self.context_retriever.retrieve_error_context(all_keywords)
        
        # Step 4: Format context and create comprehensive prompt
        print("Creating analysis prompt...")
        
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
        
        # Step 5: Get LLM analysis
        print("Getting LLM analysis...")
        llm_analysis = None
        try:
            llm_analysis = self.keyword_extractor.llm_interface.analyze_logs(generated_prompt)
            print("LLM analysis completed successfully")
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            llm_analysis = f"LLM analysis failed: {str(e)}"
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
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
        
        print(f"Analysis completed in {processing_time_ms}ms")
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

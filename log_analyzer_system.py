"""
Log Analyzer Orchestration Layer

This module provides orchestration logic that coordinates all analysis components.
All business logic is in the modules directory.
"""

from datetime import datetime

# Import from modules - clean, modular approach
from modules import (
    LLMLogFilterConfig, LLMLogFilter,
    VectorLogFilterConfig, VectorLogFilter,
    KeywordExtractor,
    ContextRetriever,
    PromptGenerator, AnalysisData,
    AnalysisRequest, AnalysisResult,
    ResultHandler,
    LocalLLMInterface
)
from modules.utils import format_time


class LogAnalysisOrchestrator:
    """Main orchestrator that coordinates all analysis components."""

    def __init__(self):
        """Initialize the orchestrator."""
        self.llm_interface = LocalLLMInterface()
        self.keyword_extractor = KeywordExtractor(
            llm_interface=self.llm_interface)
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
        step1_start = datetime.now()
        all_keywords = self.extract_keywords(request)
        step1_end = datetime.now()
        self.print_step_time("Step 1: Keyword extraction", step1_start,
                             step1_end)

        # Step 2: Filter logs (branch by filter mode)
        step2_start = datetime.now()
        print(
            f"Step 2: Analyzing logs using {'Vector DB' if request.filter_mode == 'vector' else 'keyword-based'} approach...")
        filtered_logs = self.filter_logs(request, keywords=all_keywords)
        step2_end = datetime.now()
        self.print_step_time("Step 2: Log filtering", step2_start, step2_end)

        # Step 3: Retrieve context
        step3_start = datetime.now()
        print(
            "Step 3: Retrieving codebase, documentation and error contexts...")
        codebase_context = self.context_retriever.retrieve_codebase_context(
            all_keywords)
        documentation_context = self.context_retriever.retrieve_documentation_context(
            all_keywords)
        error_context = self.context_retriever.retrieve_error_context(
            all_keywords)
        step3_end = datetime.now()
        self.print_step_time("Step 3: Context retrieval", step3_start,
                             step3_end)

        # Step 4: Format context and create comprehensive prompt
        step4_start = datetime.now()
        print("Step 4: Creating analysis prompt...")

        # Format context using PromptGenerator
        codebase_text = self.prompt_generator.format_context(codebase_context,
                                                             "Codebase")
        docs_text = self.prompt_generator.format_context(documentation_context,
                                                         "Documentation")
        errors_text = self.prompt_generator.format_context(error_context,
                                                           "Errors")
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
        self.print_step_time("Step 4: Prompt generation", step4_start,
                             step4_end)

        # Step 5: Get LLM analysis
        step5_start = datetime.now()
        print("Step 5: Getting LLM analysis...")
        try:
            llm_analysis = self.llm_interface.analyze_logs(generated_prompt)
            step5_end = datetime.now()
            self.print_step_time("Step 5: LLM analysis", step5_start, step5_end)
        except Exception as e:
            step5_end = datetime.now()
            print(f"  ✗ LLM analysis failed: {e}")
            self.print_step_time("Step 5: LLM analysis (failed)", step5_start,
                                 step5_end)
            llm_analysis = f"LLM analysis failed: {str(e)}"

        # Calculate total processing time
        end_time = datetime.now()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

        total_seconds = (end_time - start_time).total_seconds()
        print(f"\n{'=' * 60}")
        print(f"Total processing time: {format_time(total_seconds)}")
        print(f"{'=' * 60}\n")

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
            llm_model=getattr(self.llm_interface, 'model', None),
            timestamp=end_time,
            processing_time_ms=processing_time_ms
        )

        print(f"Analysis completed in {processing_time_ms}ms")
        return result

    def save_result(self, result: AnalysisResult,
                    output_dir: str = "analysis_results",
                    custom_timestamp: str = None) -> str:
        """
        Save analysis result to file.
        
        Args:
            result: Analysis result to save
            output_dir: Directory to save results
            custom_timestamp: Optional custom timestamp string to use for filename
            
        Returns:
            Path to saved file
        """
        return self.result_handler.save_result(result, output_dir,
                                               custom_timestamp)

    def filter_logs(self, request: AnalysisRequest, keywords: list[str]) -> str:
        if request.filter_mode == 'vector':
            print("Using vector DB approach to filter logs")
            config = VectorLogFilterConfig(
                issue_description=request.issue_description,
                log_file_path=request.log_file_path,
                start_date=request.start_date,
                end_date=request.end_date
            )
            log_filter = VectorLogFilter(config=config)
        else:
            print("Using LLM approach to filter logs")
            config = LLMLogFilterConfig(
                keywords=keywords,
                log_file_path=request.log_file_path,
                max_tokens=request.max_tokens,
                context_lines=request.context_lines,
                deduplicate=request.deduplicate,
                prioritize_by_severity=request.prioritize_by_severity,
                prioritize_matches=True,
                max_results=None,
                start_date=request.start_date,
                end_date=request.end_date
            )
            log_filter = LLMLogFilter(config=config)
        filtered_logs = log_filter.filter()
        return filtered_logs

    def extract_keywords(self, request: AnalysisRequest) -> list[str]:
        if request.filter_mode == 'vector':
            # For vector DB mode, skip LLM keyword extraction
            print(
                "Step 1: Skipping keyword extraction (vector DB mode - keywords not needed for filtering)...")
            return []
        else:
            # For LLM/keyword mode, use LLM-based keyword extraction
            print(
                "Step 1: Extracting keywords from issue description using LLM...")
            extracted_keywords_objects = self.keyword_extractor.extract_keywords(
                request.issue_description)
            extracted_keywords = [kw.keyword for kw in
                                  extracted_keywords_objects]

            # Add any manually provided keywords
            if request.keywords:
                extracted_keywords.extend(request.keywords)

            # Remove duplicates
            all_keywords = list(set(extracted_keywords))
            print(f"  Extracted keywords: {all_keywords}")
            return all_keywords

    def print_step_time(self, step_name: str, step_start: datetime,
                        step_end: datetime):
        """Print time taken for a step."""
        elapsed = (step_end - step_start).total_seconds()
        print(f"  ✓ {step_name} completed in {format_time(elapsed)}")

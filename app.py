"""
Main Application Orchestrator

This module coordinates all components of the log analyzer application,
providing a high-level interface for the complete analysis workflow.
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime

from modules.log_analyzer import LogAnalyzer, LogFilterConfig
from modules.keyword_extractor import KeywordExtractor
from modules.context_retriever import ContextRetriever
from modules.prompt_generator import PromptGenerator, AnalysisData, PromptType
from ui.log_analyzer_ui import LogAnalyzerUI


class LogAnalyzerApp:
    """Main application orchestrator for the log analyzer."""
    
    def __init__(self):
        """Initialize the application with all required components."""
        self.keyword_extractor = KeywordExtractor()
        self.context_retriever = ContextRetriever()
        self.prompt_generator = PromptGenerator()
        
        # Initialize UI with callback
        self.ui = LogAnalyzerUI(on_analyze_callback=self._perform_analysis)
        
        # Analysis results storage
        self.last_analysis_data: Optional[AnalysisData] = None
        self.last_prompt: Optional[str] = None
    
    def _perform_analysis(self, form_data: Dict[str, Any]) -> None:
        """
        Perform complete log analysis workflow.
        
        Args:
            form_data: Data from the UI form
        """
        try:
            # Step 1: Extract keywords from issue description
            self.ui.show_info("Extracting keywords from issue description...")
            keywords = self.keyword_extractor.get_top_keywords(
                form_data["issue_description"], 
                limit=15
            )
            
            if not keywords:
                self.ui.show_error("No relevant keywords found in the issue description.")
                return
            
            # Step 2: Create log analyzer configuration
            config = LogFilterConfig(
                log_file_path=form_data["log_file_path"],
                max_tokens=form_data["max_tokens"],
                context_lines=2,
                deduplicate=True,
                prioritize_by_severity=True,
                prioritize_matches=True,
                max_results=None
            )
            
            # Step 3: Analyze logs
            self.ui.show_info("Analyzing log files...")
            log_analyzer = LogAnalyzer(config)
            filtered_logs = log_analyzer.analyze(
                keywords=keywords,
                start_date=form_data["start_date"],
                end_date=form_data["end_date"]
            )
            
            if filtered_logs.startswith("Error:"):
                self.ui.show_error(filtered_logs)
                return
            
            # Step 4: Retrieve context
            self.ui.show_info("Retrieving relevant context...")
            context_results = self.context_retriever.retrieve_context(keywords)
            context_description = self.context_retriever.get_combined_context(keywords)
            
            # Step 5: Create analysis data
            date_range = None
            if form_data["start_date"] and form_data["end_date"]:
                date_range = f"{form_data['start_date']} to {form_data['end_date']}"
            elif form_data["start_date"]:
                date_range = f"From {form_data['start_date']}"
            elif form_data["end_date"]:
                date_range = f"Until {form_data['end_date']}"
            
            analysis_data = AnalysisData(
                issue_description=form_data["issue_description"],
                extracted_keywords=keywords,
                filtered_logs=filtered_logs,
                context_description=context_description,
                log_file_path=form_data["log_file_path"],
                analysis_date_range=date_range
            )
            
            # Step 6: Generate prompt
            self.ui.show_info("Generating analysis prompt...")
            prompt_type = PromptType(form_data["analysis_type"])
            prompt = self.prompt_generator.generate_prompt(analysis_data, prompt_type)
            
            # Store results
            self.last_analysis_data = analysis_data
            self.last_prompt = prompt
            
            # Step 7: Display results
            self._display_results(analysis_data, prompt, keywords, context_results)
            
        except Exception as e:
            self.ui.show_error(f"Analysis failed: {str(e)}")
    
    def _display_results(self, analysis_data: AnalysisData, prompt: str, 
                         keywords: list, context_results: dict) -> None:
        """
        Display analysis results in the UI.
        
        Args:
            analysis_data: Complete analysis data
            prompt: Generated prompt
            keywords: Extracted keywords
            context_results: Context retrieval results
        """
        # Create results summary
        results_summary = f"""
=== LOG ANALYSIS COMPLETE ===

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Log File: {os.path.basename(analysis_data.log_file_path)}
Date Range: {analysis_data.analysis_date_range or 'All available data'}
Analysis Type: {prompt.split('#')[1].strip() if '#' in prompt else 'General Analysis'}

=== EXTRACTED KEYWORDS ===
{', '.join(keywords)}

=== CONTEXT SOURCES QUERIED ===
"""
        
        for source_name, context_items in context_results.items():
            results_summary += f"\n{source_name}: {len(context_items)} items found"
        
        results_summary += f"""

=== GENERATED PROMPT ===
{prompt}

=== ANALYSIS STATISTICS ===
"""
        
        # Add prompt statistics
        stats = self.prompt_generator.get_prompt_statistics(analysis_data)
        for key, value in stats.items():
            results_summary += f"{key.replace('_', ' ').title()}: {value}\n"
        
        results_summary += f"""

=== NEXT STEPS ===
1. Copy the generated prompt above
2. Paste it into your preferred LLM (ChatGPT, Claude, etc.)
3. Review the analysis results
4. Use the insights to debug and resolve the issue

=== KEYWORD BREAKDOWN ===
"""
        
        # Add keyword breakdown
        keyword_summary = self.keyword_extractor.get_keyword_summary(analysis_data.issue_description)
        for kw_type, kw_list in keyword_summary.items():
            if kw_list:
                results_summary += f"\n{kw_type.title()}: {', '.join(kw_list)}"
        
        self.ui.show_results(results_summary)
        self.ui.show_info("Analysis complete! Check the results below.")
    
    def save_results(self, output_file: str) -> bool:
        """
        Save the last analysis results to a file.
        
        Args:
            output_file: Path to output file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.last_prompt:
            return False
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(self.last_prompt)
            return True
        except Exception:
            return False
    
    def get_analysis_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get summary of the last analysis.
        
        Returns:
            Dictionary with analysis summary or None if no analysis performed
        """
        if not self.last_analysis_data:
            return None
        
        return {
            "issue_description": self.last_analysis_data.issue_description,
            "keywords": self.last_analysis_data.extracted_keywords,
            "log_file": self.last_analysis_data.log_file_path,
            "date_range": self.last_analysis_data.analysis_date_range,
            "prompt_length": len(self.last_prompt) if self.last_prompt else 0,
            "context_sources": len(self.context_retriever.get_available_sources())
        }
    
    def run(self):
        """Start the application."""
        self.ui.run()
    
    def add_context_source(self, name: str, source):
        """
        Add a new context source to the retriever.
        
        Args:
            name: Name of the source
            source: ContextSource implementation
        """
        self.context_retriever.add_context_source(name, source)
    
    def get_available_context_sources(self) -> list:
        """Get list of available context sources."""
        return self.context_retriever.get_available_sources()


def main():
    """Main entry point for the application."""
    app = LogAnalyzerApp()
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        app.ui.destroy()


if __name__ == "__main__":
    main()

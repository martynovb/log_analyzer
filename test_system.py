#!/usr/bin/env python3
"""
Comprehensive Test Suite for Modular Log Analyzer
=================================================

Tests all components of the modular log analysis system.
"""

import unittest
import tempfile
import os
from pathlib import Path
from datetime import datetime

from log_analyzer_system import (
    AnalysisRequest, AnalysisResult, KeywordExtractor, 
    MockContextRetriever, PromptCreator, LogAnalysisOrchestrator
)


class TestKeywordExtractor(unittest.TestCase):
    """Test keyword extraction functionality."""
    
    def setUp(self):
        self.extractor = KeywordExtractor()
    
    def test_extract_keywords_basic(self):
        """Test basic keyword extraction."""
        description = "The application crashes with memory errors and segmentation faults"
        keywords = self.extractor.extract_keywords(description)
        
        self.assertIn("application", keywords)
        self.assertIn("crashes", keywords)
        self.assertIn("memory", keywords)
        self.assertIn("errors", keywords)
        self.assertIn("segmentation", keywords)
        self.assertIn("faults", keywords)
    
    def test_extract_keywords_max_limit(self):
        """Test keyword extraction with max limit."""
        description = "application crashes memory errors segmentation faults database timeout network connection"
        keywords = self.extractor.extract_keywords(description, max_keywords=5)
        
        self.assertEqual(len(keywords), 5)
    
    def test_extract_technical_keywords(self):
        """Test technical keyword extraction."""
        description = "The system throws a NullPointerException and has timeout errors"
        keywords = self.extractor.extract_technical_keywords(description)
        
        self.assertIn("errors", keywords)
        self.assertIn("timeout", keywords)
        self.assertIn("nullpointerexception", keywords)
    
    def test_empty_description(self):
        """Test handling of empty description."""
        keywords = self.extractor.extract_keywords("")
        self.assertEqual(keywords, [])
        
        technical_keywords = self.extractor.extract_technical_keywords("")
        self.assertEqual(technical_keywords, [])


class TestMockContextRetriever(unittest.TestCase):
    """Test mock context retriever functionality."""
    
    def setUp(self):
        self.retriever = MockContextRetriever()
    
    def test_retrieve_codebase_context(self):
        """Test codebase context retrieval."""
        keywords = ["error", "exception", "timeout"]
        context = self.retriever.retrieve_codebase_context(keywords)
        
        self.assertIn("relevant_files", context)
        self.assertIn("total_files", context)
        self.assertIn("search_keywords", context)
        self.assertIn("retrieval_method", context)
        self.assertGreater(context["total_files"], 0)
    
    def test_retrieve_documentation_context(self):
        """Test documentation context retrieval."""
        keywords = ["database", "network", "memory"]
        context = self.retriever.retrieve_documentation_context(keywords)
        
        self.assertIn("relevant_documentation", context)
        self.assertIn("total_docs", context)
        self.assertIn("search_keywords", context)
        self.assertIn("retrieval_method", context)
        self.assertGreater(context["total_docs"], 0)
    
    def test_empty_keywords(self):
        """Test handling of empty keywords."""
        codebase_context = self.retriever.retrieve_codebase_context([])
        docs_context = self.retriever.retrieve_documentation_context([])
        
        self.assertEqual(codebase_context["total_files"], 0)
        self.assertEqual(docs_context["total_docs"], 0)


class TestPromptCreator(unittest.TestCase):
    """Test prompt creation functionality."""
    
    def setUp(self):
        self.creator = PromptCreator()
    
    def test_create_prompt(self):
        """Test prompt creation."""
        issue_description = "Application crashes with memory errors"
        keywords = ["crash", "memory", "error"]
        filtered_logs = "Sample log entries..."
        codebase_context = {
            "relevant_files": ["ErrorHandler.java"],
            "total_files": 1,
            "search_keywords": ["error"],
            "retrieval_method": "mock_search"
        }
        documentation_context = {
            "relevant_documentation": ["Error Handling Guide"],
            "total_docs": 1,
            "search_keywords": ["error"],
            "retrieval_method": "mock_search"
        }
        
        prompt = self.creator.create_prompt(
            issue_description, keywords, filtered_logs,
            codebase_context, documentation_context
        )
        
        self.assertIn(issue_description, prompt)
        self.assertIn("crash, memory, error", prompt)
        self.assertIn("Sample log entries...", prompt)
        self.assertIn("ErrorHandler.java", prompt)
        self.assertIn("Error Handling Guide", prompt)
    
    def test_format_context(self):
        """Test context formatting."""
        context = {
            "relevant_files": ["file1.java", "file2.py"],
            "total_files": 2,
            "search_keywords": ["error"],
            "retrieval_method": "mock_search"
        }
        
        formatted = self.creator._format_context(context, "Codebase")
        
        self.assertIn("Codebase Information", formatted)
        self.assertIn("file1.java", formatted)
        self.assertIn("file2.py", formatted)
        self.assertIn("mock_search", formatted)


class TestAnalysisRequest(unittest.TestCase):
    """Test analysis request data class."""
    
    def test_analysis_request_creation(self):
        """Test analysis request creation."""
        request = AnalysisRequest(
            log_file_path="test.log",
            issue_description="Test issue",
            keywords=["test"],
            max_tokens=2000
        )
        
        self.assertEqual(request.log_file_path, "test.log")
        self.assertEqual(request.issue_description, "Test issue")
        self.assertEqual(request.keywords, ["test"])
        self.assertEqual(request.max_tokens, 2000)
        self.assertEqual(request.context_lines, 2)  # Default value


class TestLogAnalysisOrchestrator(unittest.TestCase):
    """Test main orchestrator functionality."""
    
    def setUp(self):
        self.orchestrator = LogAnalysisOrchestrator()
        
        # Create a temporary log file for testing
        self.temp_log = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
        self.temp_log.write("""2025-10-15T10:00:00 LogLevel.i Application started
2025-10-15T10:01:00 LogLevel.e Error: Memory allocation failed
2025-10-15T10:02:00 LogLevel.d Debug: Processing request
2025-10-15T10:03:00 LogLevel.e Exception: NullPointerException
2025-10-15T10:04:00 LogLevel.i Application shutdown
""")
        self.temp_log.close()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_log.name):
            os.unlink(self.temp_log.name)
    
    def test_analyze_issue(self):
        """Test complete issue analysis."""
        request = AnalysisRequest(
            log_file_path=self.temp_log.name,
            issue_description="The application crashes with memory errors and exceptions",
            max_tokens=1000
        )
        
        result = self.orchestrator.analyze_issue(request)
        
        # Verify result structure
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.request, request)
        self.assertIsInstance(result.extracted_keywords, list)
        self.assertIsInstance(result.filtered_logs, str)
        self.assertIsInstance(result.context_info, dict)
        self.assertIsInstance(result.generated_prompt, str)
        self.assertIsInstance(result.timestamp, datetime)
        self.assertIsInstance(result.processing_time_ms, int)
        
        # Verify keywords were extracted
        self.assertGreater(len(result.extracted_keywords), 0)
        
        # Verify context was retrieved
        self.assertIn("codebase", result.context_info)
        self.assertIn("documentation", result.context_info)
    
    def test_save_result(self):
        """Test result saving functionality."""
        request = AnalysisRequest(
            log_file_path=self.temp_log.name,
            issue_description="Test issue",
            max_tokens=1000
        )
        
        result = self.orchestrator.analyze_issue(request)
        output_file = self.orchestrator.save_result(result)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Verify file content
        with open(output_file, 'r') as f:
            content = f.read()
            self.assertIn("timestamp", content)
            self.assertIn("Test issue", content)
        
        # Clean up
        os.unlink(output_file)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        self.orchestrator = LogAnalysisOrchestrator()
        
        # Create a more comprehensive test log file
        self.temp_log = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False)
        self.temp_log.write("""2025-10-15T10:00:00 LogLevel.i Application started successfully
2025-10-15T10:01:00 LogLevel.d Loading configuration from database
2025-10-15T10:02:00 LogLevel.e ERROR: Database connection timeout
2025-10-15T10:03:00 LogLevel.e EXCEPTION: java.sql.SQLTimeoutException
2025-10-15T10:04:00 LogLevel.w WARNING: Memory usage high (85%)
2025-10-15T10:05:00 LogLevel.e FATAL: OutOfMemoryError - Application crash
2025-10-15T10:06:00 LogLevel.i Application terminated
""")
        self.temp_log.close()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_log.name):
            os.unlink(self.temp_log.name)
    
    def test_complete_analysis_workflow(self):
        """Test the complete analysis workflow."""
        request = AnalysisRequest(
            log_file_path=self.temp_log.name,
            issue_description="The application crashes with database timeout errors and memory issues. Users experience OutOfMemoryError exceptions.",
            start_date="2025-10-15",
            end_date="2025-10-15",
            max_tokens=2000,
            context_lines=1,
            deduplicate=True,
            prioritize_by_severity=True
        )
        
        # Perform analysis
        result = self.orchestrator.analyze_issue(request)
        
        # Verify comprehensive results
        self.assertGreater(len(result.extracted_keywords), 0)
        self.assertIn("database", result.extracted_keywords)
        self.assertIn("timeout", result.extracted_keywords)
        self.assertIn("memory", result.extracted_keywords)
        
        # Verify filtered logs contain relevant entries
        self.assertIn("Database connection timeout", result.filtered_logs)
        self.assertIn("OutOfMemoryError", result.filtered_logs)
        
        # Verify context retrieval
        self.assertGreater(result.context_info["codebase"]["total_files"], 0)
        self.assertGreater(result.context_info["documentation"]["total_docs"], 0)
        
        # Verify prompt generation
        self.assertIn("Database connection timeout", result.generated_prompt)
        self.assertIn("OutOfMemoryError", result.generated_prompt)
        self.assertIn("Root Cause Analysis", result.generated_prompt)
        
        # Save and verify result
        output_file = self.orchestrator.save_result(result)
        self.assertTrue(os.path.exists(output_file))
        
        # Clean up
        os.unlink(output_file)


def run_tests():
    """Run all tests."""
    print("Running Log Analyzer Test Suite...")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestKeywordExtractor,
        TestMockContextRetriever,
        TestPromptCreator,
        TestAnalysisRequest,
        TestLogAnalysisOrchestrator,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nTest Suite {'PASSED' if success else 'FAILED'}")
    
    return success


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

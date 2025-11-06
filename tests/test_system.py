#!/usr/bin/env python3
"""
Comprehensive Test Suite for Modular Log Analyzer
=================================================

Tests all components of the modular log analysis system.
"""

import unittest
import tempfile
import os
from datetime import datetime

from log_analyzer_system import LogAnalysisOrchestrator
from modules import (
    AnalysisRequest, AnalysisResult, KeywordExtractor,
    PromptGenerator, AnalysisData
)
from modules.domain import FilterMode


class TestKeywordExtractor(unittest.TestCase):
    """Test keyword extraction functionality."""
    
    def setUp(self):
        self.extractor = KeywordExtractor()
    
    def test_extract_keywords_basic(self):
        """Test basic keyword extraction."""
        description = "The application crashes with memory errors and segmentation faults"
        keywords = self.extractor.extract_keywords(description)

        self.assertTrue(
            any("application" in keyword.keyword.lower() for keyword in keywords))
        self.assertTrue(
            any("crashes" in keyword.keyword.lower() for keyword in keywords))
        self.assertTrue(
            any("memory" in keyword.keyword.lower() for keyword in keywords))
        self.assertTrue(
            any("errors" in keyword.keyword.lower() for keyword in keywords))
        self.assertTrue(
            any("segmentation" in keyword.keyword.lower() for keyword in keywords))
        self.assertTrue(
            any("faults" in keyword.keyword.lower() for keyword in keywords))


    def test_empty_description(self):
        """Test handling of empty description."""
        with self.assertRaises(ValueError) as cm:
            keywords = self.extractor.extract_keywords("")
        self.assertEqual(str(cm.exception), "Issue description cannot be empty")


# TestMockContextRetriever removed - now using real ContextRetriever from modules

class TestPromptGenerator(unittest.TestCase):
    """Test prompt generation functionality."""
    
    def setUp(self):
        self.generator = PromptGenerator()
    
    def test_generate_prompt(self):
        """Test prompt generation."""
        analysis_data = AnalysisData(
            issue_description="Application crashes with memory errors",
            extracted_keywords=["crash", "memory", "error"],
            filtered_logs="Sample log entries...",
            context_description="Error Handler module",
            log_file_path="app.log",
            filter_mode=FilterMode.llm
        )
        
        prompt = self.generator.generate_prompt(analysis_data)
        
        self.assertIn("Application crashes with memory errors", prompt)
        self.assertIn("crash, memory, error", prompt)
        self.assertIn("Sample log entries...", prompt)
        self.assertIn("Error Handler module", prompt)
        self.assertIn("app.log", prompt)


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
            prioritize_by_severity=True,
            filter_mode=FilterMode.llm
        )
        
        # Perform analysis
        result = self.orchestrator.analyze_issue(request)
        
        # Verify comprehensive results
        self.assertGreater(len(result.extracted_keywords), 0)
        self.assertTrue(any("database" in keyword.lower() for keyword in result.extracted_keywords))
        self.assertTrue(any("timeout" in keyword.lower() for keyword in result.extracted_keywords))
        self.assertTrue(any("memory" in keyword.lower() for keyword in result.extracted_keywords))
        
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
        TestPromptGenerator,
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

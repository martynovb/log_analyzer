#!/usr/bin/env python3
"""
Unit tests for VectorDb module
=============================

Tests the vector database functionality including initialization,
document loading, and search capabilities.
"""

import os
import sys
import tempfile
import shutil
import unittest

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.vector_db import VectorDb
from modules.vector_log_filter import VectorLogFilter, VectorLogFilterConfig


class TestVectorDb(unittest.TestCase):
    """Test cases for VectorDb class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a temporary test document file
        self.test_file = os.path.join(self.test_dir, "test_document.txt")
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write("""This is a test document for vector database testing.
It contains multiple sentences to test chunking functionality.
We need enough content to create multiple chunks for testing.
The vector database should be able to search through this content effectively.
Error handling and exception scenarios are important to test as well.
""")
        
        # Create a test directory with multiple files
        self.test_input_dir = os.path.join(self.test_dir, "input_dir")
        os.makedirs(self.test_input_dir, exist_ok=True)
        
        with open(os.path.join(self.test_input_dir, "file1.txt"), 'w', encoding='utf-8') as f:
            f.write("First document content for testing.\n")
        with open(os.path.join(self.test_input_dir, "file2.txt"), 'w', encoding='utf-8') as f:
            f.write("Second document with different content.\n")
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directories
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_init_with_file_path(self):
        """Test VectorDb initialization with input_document_path."""
        persist_dir = os.path.join(self.test_dir, "db1")
        os.makedirs(persist_dir, exist_ok=True)
        
        db = VectorDb(
            input_document_path=self.test_file,
            persist_directory=persist_dir
        )
        
        self.assertIsNotNone(db.db)
        self.assertGreater(db.chunk_number, 0)
        self.assertTrue(os.path.exists(persist_dir))

    def test_init_with_directory(self):
        """Test VectorDb initialization with input_directory."""
        persist_dir = os.path.join(self.test_dir, "db2")
        os.makedirs(persist_dir, exist_ok=True)
        
        db = VectorDb(
            input_directory=self.test_input_dir,
            persist_directory=persist_dir
        )
        
        self.assertIsNotNone(db.db)
        self.assertGreater(db.chunk_number, 0)

    def test_init_without_input_error(self):
        """Test that initialization raises error when no input is provided."""
        with self.assertRaises(RuntimeError) as context:
            VectorDb()
        
        self.assertIn("You must specify either a directory or a file", str(context.exception))

    def test_init_custom_chunk_size(self):
        """Test VectorDb with custom chunk size."""
        persist_dir = os.path.join(self.test_dir, "db3")
        os.makedirs(persist_dir, exist_ok=True)
        
        custom_chunk_size = 300
        custom_chunk_overlap = 50  # Must be smaller than chunk_size
        db = VectorDb(
            input_document_path=self.test_file,
            persist_directory=persist_dir,
            chunk_size=custom_chunk_size,
            chunk_overlap=custom_chunk_overlap
        )
        
        self.assertIsNotNone(db.db)
        # With smaller chunk size, we should get more chunks
        self.assertGreater(db.chunk_number, 0)

    def test_init_custom_chunk_overlap(self):
        """Test VectorDb with custom chunk overlap."""
        persist_dir = os.path.join(self.test_dir, "db4")
        os.makedirs(persist_dir, exist_ok=True)
        
        db = VectorDb(
            input_document_path=self.test_file,
            persist_directory=persist_dir,
            chunk_overlap=50
        )
        
        self.assertIsNotNone(db.db)

    def test_search_functionality(self):
        """Test search functionality returns results."""
        persist_dir = os.path.join(self.test_dir, "db5")
        os.makedirs(persist_dir, exist_ok=True)
        
        db = VectorDb(
            input_document_path=self.test_file,
            persist_directory=persist_dir
        )
        
        # Perform a search
        results = db.search("test document")
        
        # Verify results format
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Verify result format: (content, score)
        for result in results:
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            self.assertIsInstance(result[0], str)  # content
            self.assertIsInstance(result[1], (int, float))  # score

    def test_search_empty_query(self):
        """Test search with empty query."""
        persist_dir = os.path.join(self.test_dir, "db6")
        os.makedirs(persist_dir, exist_ok=True)
        
        db = VectorDb(
            input_document_path=self.test_file,
            persist_directory=persist_dir
        )
        
        results = db.search("")
        
        # Should return results even with empty query
        self.assertIsInstance(results, list)

    def test_search_specific_term(self):
        """Test search for specific terms in the document."""
        persist_dir = os.path.join(self.test_dir, "db7")
        os.makedirs(persist_dir, exist_ok=True)
        
        db = VectorDb(
            input_document_path=self.test_file,
            persist_directory=persist_dir
        )
        
        # Search for term that should be in the document
        results = db.search("vector database")
        
        self.assertGreater(len(results), 0)
        # Check that results contain relevant content
        found_relevant = any("vector" in content.lower() or "database" in content.lower() 
                            for content, _ in results)
        self.assertTrue(found_relevant, "Search should return relevant results")

    def test_init_with_persist_directory(self):
        """Test that persist_directory is properly used."""
        persist_dir = os.path.join(self.test_dir, "db8")
        os.makedirs(persist_dir, exist_ok=True)
        
        db = VectorDb(
            input_document_path=self.test_file,
            persist_directory=persist_dir
        )
        
        # Verify persist directory exists and contains DB files
        self.assertTrue(os.path.exists(persist_dir))
        # ChromaDB creates files in the persist directory
        files = os.listdir(persist_dir)
        self.assertGreater(len(files), 0, "Persist directory should contain DB files")

    def test_init_without_persist_directory(self):
        """Test initialization without persist_directory (in-memory DB)."""
        db = VectorDb(
            input_document_path=self.test_file,
            persist_directory=None
        )
        
        self.assertIsNotNone(db.db)
        self.assertGreater(db.chunk_number, 0)

    def test_init_custom_embedding_model(self):
        """Test initialization with custom embedding model name."""
        persist_dir = os.path.join(self.test_dir, "db9")
        os.makedirs(persist_dir, exist_ok=True)
        
        db = VectorDb(
            input_document_path=self.test_file,
            persist_directory=persist_dir,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.assertIsNotNone(db.db)

    def test_chunk_number_attribute(self):
        """Test that chunk_number attribute is set correctly."""
        persist_dir = os.path.join(self.test_dir, "db10")
        os.makedirs(persist_dir, exist_ok=True)
        
        db = VectorDb(
            input_document_path=self.test_file,
            persist_directory=persist_dir
        )
        
        self.assertIsInstance(db.chunk_number, int)
        self.assertGreater(db.chunk_number, 0)

    def test_multiple_searches(self):
        """Test performing multiple searches on the same DB."""
        persist_dir = os.path.join(self.test_dir, "db11")
        os.makedirs(persist_dir, exist_ok=True)
        
        db = VectorDb(
            input_document_path=self.test_file,
            persist_directory=persist_dir
        )
        
        results1 = db.search("test")
        results2 = db.search("document")
        results3 = db.search("content")
        
        self.assertIsInstance(results1, list)
        self.assertIsInstance(results2, list)
        self.assertIsInstance(results3, list)
        self.assertGreater(len(results1), 0)
        self.assertGreater(len(results2), 0)
        self.assertGreater(len(results3), 0)

    def test_load_existing_db(self):
        """Test loading an existing vector DB."""
        persist_dir = os.path.join(self.test_dir, "db12")
        os.makedirs(persist_dir, exist_ok=True)
        
        # Create a new DB
        db1 = VectorDb(
            input_document_path=self.test_file,
            persist_directory=persist_dir
        )
        
        # Verify DB was created
        self.assertIsNotNone(db1.db)
        self.assertGreater(db1.chunk_number, 0)
        
        # Load the existing DB
        db2 = VectorDb(
            persist_directory=persist_dir,
            load_existing=True
        )
        
        # Verify DB was loaded
        self.assertIsNotNone(db2.db)
        # chunk_number is not set when loading existing DBs (early return in __init__)
        self.assertFalse(hasattr(db2, 'chunk_number'))
        
        # Verify both can search
        results1 = db1.search("test")
        results2 = db2.search("test")
        
        self.assertGreater(len(results1), 0)
        self.assertGreater(len(results2), 0)

    def test_vector_log_filter_reuse_same_parameters(self):
        """Test VectorLogFilter reuses DB when parameters are unchanged."""
        # Create test log file
        test_log = os.path.join(self.test_dir, "test_log.log")
        with open(test_log, 'w', encoding='utf-8') as f:
            f.write("""2025-10-15T10:00:00 INFO Application started
2025-10-15T10:01:00 ERROR Memory allocation failed
2025-10-15T10:02:00 DEBUG Processing request
2025-10-16T10:00:00 INFO Session ended
""")
        
        # Clear cached DB cache
        VectorLogFilter._cached_db_cache = None
        
        # Use test directory instead of temp directory
        work_dir = os.path.join(self.test_dir, "test_vector_db_reuse")
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        
        original_work_dir = VectorLogFilter.WORKING_DIRECTORY
        VectorLogFilter.WORKING_DIRECTORY = work_dir
        
        try:
            # First analysis - should create new DB
            config1 = VectorLogFilterConfig(
                log_file_path=test_log,
                start_date="2025-10-15",
                end_date="2025-10-16",
                issue_description="Memory errors"
            )
            filter1 = VectorLogFilter(config1)
            results1 = filter1.filter()
            
            # Verify DB was created and signature cached
            self.assertTrue(os.path.exists(work_dir))
            self.assertIsNotNone(VectorLogFilter._cached_db_cache)
            self.assertIsNotNone(VectorLogFilter._cached_db_cache.db_signature)
            self.assertGreater(len(results1), 0)
            
            # Force garbage collection to release file handles
            import gc
            gc.collect()
            
            # Second analysis with same parameters, different issue - should REUSE
            config2 = VectorLogFilterConfig(
                log_file_path=test_log,
                start_date="2025-10-15",
                end_date="2025-10-16",
                issue_description="Different issue description"  # Changed
            )
            filter2 = VectorLogFilter(config2)
            results2 = filter2.filter()
            
            # Verify results are returned (DB was reused)
            self.assertGreater(len(results2), 0)
            # Verify signature still matches
            self.assertEqual(
                VectorLogFilter._cached_db_cache.db_signature.log_file_path,
                test_log
            )
            
        finally:
            # Restore original working directory
            VectorLogFilter.WORKING_DIRECTORY = original_work_dir

    def test_vector_log_filter_recreate_on_date_change(self):
        """Test VectorLogFilter creates new DB when date range changes."""
        # Create test log file
        test_log = os.path.join(self.test_dir, "test_log2.log")
        with open(test_log, 'w', encoding='utf-8') as f:
            f.write("""2025-10-15T10:00:00 INFO Application started
2025-10-15T10:01:00 ERROR Memory allocation failed
2025-10-17T10:00:00 INFO New session
""")
        
        # Use test directory
        work_dir = os.path.join(self.test_dir, "test_vector_db_recreate")
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        
        original_work_dir = VectorLogFilter.WORKING_DIRECTORY
        VectorLogFilter.WORKING_DIRECTORY = work_dir
        VectorLogFilter._cached_db_signature = None
        
        try:
            # First analysis
            config1 = VectorLogFilterConfig(
                log_file_path=test_log,
                start_date="2025-10-15",
                end_date="2025-10-15",
                issue_description="First issue"
            )
            filter1 = VectorLogFilter(config1)
            results1 = filter1.filter()
            
            original_signature = VectorLogFilter._cached_db_cache.db_signature
            
            # Force garbage collection to release file handles
            import gc
            gc.collect()
            
            # Second analysis with changed date - should CREATE NEW DB
            config2 = VectorLogFilterConfig(
                log_file_path=test_log,
                start_date="2025-10-17",  # Changed date
                end_date="2025-10-17",
                issue_description="Second issue"
            )
            filter2 = VectorLogFilter(config2)
            results2 = filter2.filter()
            
            # Verify new signature was cached (different from original)
            self.assertIsNotNone(VectorLogFilter._cached_db_cache)
            self.assertIsNotNone(VectorLogFilter._cached_db_cache.db_signature)
            self.assertNotEqual(
                VectorLogFilter._cached_db_cache.db_signature.start_date,
                original_signature.start_date
            )
            self.assertGreater(len(results2), 0)
            
        finally:
            VectorLogFilter.WORKING_DIRECTORY = original_work_dir

    def test_vector_log_filter_recreate_on_file_change(self):
        """Test VectorLogFilter creates new DB when log file changes."""
        # Create two different log files
        test_log1 = os.path.join(self.test_dir, "test_log3a.log")
        test_log2 = os.path.join(self.test_dir, "test_log3b.log")
        
        with open(test_log1, 'w', encoding='utf-8') as f:
            f.write("2025-10-15T10:00:00 INFO File 1 content\n")
        with open(test_log2, 'w', encoding='utf-8') as f:
            f.write("2025-10-15T10:00:00 INFO File 2 content\n")
        
        # Use test directory
        work_dir = os.path.join(self.test_dir, "test_vector_db_file_change")
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        
        original_work_dir = VectorLogFilter.WORKING_DIRECTORY
        VectorLogFilter.WORKING_DIRECTORY = work_dir
        VectorLogFilter._cached_db_signature = None
        
        try:
            # First analysis with file 1
            config1 = VectorLogFilterConfig(
                log_file_path=test_log1,
                start_date="2025-10-15",
                end_date="2025-10-15",
                issue_description="File 1 analysis"
            )
            filter1 = VectorLogFilter(config1)
            results1 = filter1.filter()
            
            original_signature = VectorLogFilter._cached_db_cache.db_signature
            
            # Force garbage collection to release file handles
            import gc
            gc.collect()
            
            # Second analysis with file 2 - should CREATE NEW DB
            config2 = VectorLogFilterConfig(
                log_file_path=test_log2,  # Changed file
                start_date="2025-10-15",
                end_date="2025-10-15",
                issue_description="File 2 analysis"
            )
            filter2 = VectorLogFilter(config2)
            results2 = filter2.filter()
            
            # Verify new signature was cached (different file path)
            self.assertIsNotNone(VectorLogFilter._cached_db_cache)
            self.assertIsNotNone(VectorLogFilter._cached_db_cache.db_signature)
            self.assertNotEqual(
                VectorLogFilter._cached_db_cache.db_signature.log_file_path,
                original_signature.log_file_path
            )
            self.assertGreater(len(results2), 0)
            
        finally:
            VectorLogFilter.WORKING_DIRECTORY = original_work_dir


def run_tests():
    """Run all tests."""
    print("Running VectorDb Unit Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestVectorDb)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
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
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)


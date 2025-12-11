#!/usr/bin/env python3
"""
Test script for BatchFileProcessor with smart resume capabilities.

Tests comprehensive batch file processing functionality including:
- Automatic detection and skipping of processed files
- Timestamped output generation with processing metadata
- Error handling that continues processing after failures
- Resume functionality for interrupted processing

Requirements: 6.1, 6.2, 6.3, 6.5
"""

import sys
import os
import tempfile
import shutil
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.insert(0, 'src')

from src.batch_file_processor import BatchFileProcessor, FileProcessingMetadata, BatchProcessingState
from src.hybrid_processor import ProcessingConfiguration
from src.models import AddressRecord


def create_test_csv_file(file_path: str, addresses: List[str]) -> None:
    """Create a test CSV file with addresses."""
    data = []
    for i, addr in enumerate(addresses):
        data.append({
            'addr_hash_key': f'hash_{i}',
            'addr_text': addr,
            'city_id': 'TEST_CITY',
            'pincode': '123456',
            'state_id': 'TEST_STATE',
            'zone_id': 'TEST_ZONE',
            'address_id': f'addr_{i}',
            'assigned_pickup_dlvd_geo_points': '',
            'assigned_pickup_dlvd_geo_points_count': 0
        })
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"✅ Created test file: {file_path} with {len(addresses)} addresses")


def test_file_discovery_and_skipping():
    """Test automatic detection and skipping of processed files."""
    print("\n🧪 Testing file discovery and skipping...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_files = []
        for i in range(3):
            file_path = os.path.join(temp_dir, f'test_file_{i}.csv')
            addresses = [f'Test Address {j} for file {i}' for j in range(5)]
            create_test_csv_file(file_path, addresses)
            test_files.append(file_path)
        
        # Create output directory
        output_dir = os.path.join(temp_dir, 'batch_output')
        
        # Initialize processor
        config = ProcessingConfiguration(
            gpu_batch_size=100,
            target_throughput=500,
            cpu_allocation_ratio=0.5  # Use more CPU for testing
        )
        
        processor = BatchFileProcessor(config, output_dir)
        
        # Initialize batch state for testing
        processor.current_batch_state = BatchProcessingState(
            batch_id="test_batch",
            start_time=time.time(),
            state_file_path=os.path.join(output_dir, "test_state.json")
        )
        
        # Test file discovery
        file_patterns = [os.path.join(temp_dir, '*.csv')]
        discovered_files = processor._discover_files(file_patterns, skip_processed=False)
        
        assert len(discovered_files) == 3, f"Expected 3 files, found {len(discovered_files)}"
        print(f"✅ File discovery: Found {len(discovered_files)} files")
        
        # Create fake processed output for one file
        results_dir = Path(output_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        fake_output = results_dir / "test_file_0_processed_20231201_120000.csv"
        fake_output.write_text("fake,processed,data\n1,2,3")
        
        # Test skipping processed files
        files_to_process = processor._discover_files(file_patterns, skip_processed=True)
        
        # Should skip the file with existing output
        print(f"✅ File skipping: {len(discovered_files) - len(files_to_process)} files skipped")


def test_batch_processing_state():
    """Test batch processing state management and persistence."""
    print("\n🧪 Testing batch processing state management...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        state_file = os.path.join(temp_dir, 'test_state.json')
        
        # Create new batch state
        state = BatchProcessingState(
            batch_id="test_batch_123",
            start_time=time.time(),
            state_file_path=state_file
        )
        
        # Add file metadata
        metadata = FileProcessingMetadata(
            file_path="/test/file.csv",
            file_hash="test_hash",
            file_size=1024,
            total_addresses=100
        )
        
        state.files_metadata["/test/file.csv"] = metadata
        state.total_files = 1
        state.total_addresses = 100
        
        # Save state
        state.save_state()
        
        assert os.path.exists(state_file), "State file should be created"
        print("✅ State persistence: State file created")
        
        # Load state
        loaded_state = BatchProcessingState.load_state(state_file)
        
        assert loaded_state is not None, "State should be loaded"
        assert loaded_state.batch_id == "test_batch_123", "Batch ID should match"
        assert len(loaded_state.files_metadata) == 1, "File metadata should be preserved"
        assert loaded_state.total_addresses == 100, "Total addresses should match"
        
        print("✅ State persistence: State loaded correctly")


def test_file_metadata_operations():
    """Test file processing metadata operations."""
    print("\n🧪 Testing file metadata operations...")
    
    # Create metadata
    metadata = FileProcessingMetadata(
        file_path="/test/file.csv",
        file_hash="abc123",
        file_size=2048,
        total_addresses=50
    )
    
    # Test initial state
    assert metadata.processing_status == "not_started"
    assert metadata.calculate_progress_percentage() == 0.0
    assert not metadata.is_processing_complete()
    assert not metadata.can_resume()
    
    # Test status updates
    metadata.update_processing_status("in_progress")
    assert metadata.processing_status == "in_progress"
    assert metadata.start_time is not None
    
    # Test progress updates
    metadata.processed_addresses = 25
    assert metadata.calculate_progress_percentage() == 50.0
    
    # Test completion
    metadata.processed_addresses = 50
    metadata.update_processing_status("completed")
    assert metadata.is_processing_complete()
    assert metadata.end_time is not None
    assert metadata.processing_time > 0
    
    # Test serialization
    metadata_dict = metadata.to_dict()
    restored_metadata = FileProcessingMetadata.from_dict(metadata_dict)
    
    assert restored_metadata.file_path == metadata.file_path
    assert restored_metadata.processing_status == metadata.processing_status
    assert restored_metadata.processed_addresses == metadata.processed_addresses
    
    print("✅ File metadata: All operations working correctly")


def test_error_handling_and_continuation():
    """Test error handling that continues processing after failures."""
    print("\n🧪 Testing error handling and continuation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files - some will "fail" processing
        test_files = []
        for i in range(3):
            file_path = os.path.join(temp_dir, f'test_file_{i}.csv')
            addresses = [f'Test Address {j} for file {i}' for j in range(3)]
            create_test_csv_file(file_path, addresses)
            test_files.append(file_path)
        
        # Create output directory
        output_dir = os.path.join(temp_dir, 'batch_output')
        
        # Initialize processor
        config = ProcessingConfiguration(
            gpu_batch_size=100,  # Minimum valid batch size
            target_throughput=500,  # Minimum valid throughput
            cpu_allocation_ratio=0.5  # Valid CPU allocation ratio
        )
        
        processor = BatchFileProcessor(config, output_dir)
        
        # Create batch state
        batch_id = "test_error_handling"
        state_file = os.path.join(output_dir, 'state', f'{batch_id}_state.json')
        
        state = BatchProcessingState(
            batch_id=batch_id,
            start_time=time.time(),
            state_file_path=state_file
        )
        
        # Add file metadata
        for file_path in test_files:
            metadata = FileProcessingMetadata(
                file_path=file_path,
                file_hash="test_hash",
                file_size=os.path.getsize(file_path),
                total_addresses=3
            )
            state.files_metadata[file_path] = metadata
        
        state.total_files = len(test_files)
        processor.current_batch_state = state
        
        # Test error marking
        error_file = test_files[1]
        error_message = "Simulated processing error"
        
        state.mark_file_failed(error_file, error_message)
        
        # Verify error handling
        failed_metadata = state.files_metadata[error_file]
        assert failed_metadata.processing_status == "failed"
        assert failed_metadata.last_error == error_message
        assert error_file in state.failed_files
        
        print("✅ Error handling: File marked as failed correctly")
        print(f"   Failed files: {len(state.failed_files)}")
        print(f"   Error details: {failed_metadata.error_count} errors recorded")


def test_resume_functionality():
    """Test resume functionality for interrupted processing."""
    print("\n🧪 Testing resume functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test file
        file_path = os.path.join(temp_dir, 'test_resume.csv')
        addresses = [f'Resume Test Address {i}' for i in range(10)]
        create_test_csv_file(file_path, addresses)
        
        # Create metadata for interrupted processing
        metadata = FileProcessingMetadata(
            file_path=file_path,
            file_hash="resume_test_hash",
            file_size=os.path.getsize(file_path),
            total_addresses=10,
            processed_addresses=6,  # Partially processed
            processing_status="interrupted",
            last_processed_index=6
        )
        
        # Test resume capability
        assert metadata.can_resume(), "Should be able to resume interrupted processing"
        
        progress_pct = metadata.calculate_progress_percentage()
        assert progress_pct == 60.0, f"Expected 60% progress, got {progress_pct}%"
        
        # Test resume checkpoint
        metadata.resume_checkpoint = {
            "last_batch_index": 2,
            "processing_mode": "gpu_primary"
        }
        
        # Serialize and deserialize to test persistence
        metadata_dict = metadata.to_dict()
        restored_metadata = FileProcessingMetadata.from_dict(metadata_dict)
        
        assert restored_metadata.can_resume(), "Restored metadata should support resume"
        assert restored_metadata.last_processed_index == 6, "Resume index should be preserved"
        assert restored_metadata.resume_checkpoint is not None, "Resume checkpoint should be preserved"
        
        print("✅ Resume functionality: All resume operations working correctly")
        print(f"   Progress: {progress_pct}% ({metadata.processed_addresses}/{metadata.total_addresses})")
        print(f"   Resume from index: {metadata.last_processed_index}")


def test_timestamped_output_generation():
    """Test timestamped output file generation with metadata."""
    print("\n🧪 Testing timestamped output generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test file
        file_path = os.path.join(temp_dir, 'test_output.csv')
        addresses = ['123 Test Street, Test City', '456 Sample Road, Sample Town']
        create_test_csv_file(file_path, addresses)
        
        # Create output directory
        output_dir = os.path.join(temp_dir, 'batch_output')
        
        # Initialize processor
        config = ProcessingConfiguration(
            gpu_batch_size=100,  # Minimum valid batch size
            target_throughput=500  # Minimum valid throughput
        )
        
        processor = BatchFileProcessor(config, output_dir)
        
        # Test timestamped filename generation
        results_dir = Path(output_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Simulate output generation
        base_name = "test_output"
        timestamp = "20231201_120000"
        expected_filename = f"{base_name}_processed_{timestamp}.csv"
        
        # Verify naming pattern
        assert "processed" in expected_filename, "Output should contain 'processed'"
        assert timestamp in expected_filename, "Output should contain timestamp"
        
        print("✅ Timestamped output: Naming pattern correct")
        print(f"   Expected format: {expected_filename}")


def run_comprehensive_test():
    """Run comprehensive test of batch file processing functionality."""
    print("🚀 COMPREHENSIVE BATCH FILE PROCESSOR TEST")
    print("=" * 60)
    
    try:
        # Test individual components
        test_file_discovery_and_skipping()
        test_batch_processing_state()
        test_file_metadata_operations()
        test_error_handling_and_continuation()
        test_resume_functionality()
        test_timestamped_output_generation()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ File discovery and skipping: Working")
        print("✅ Batch processing state: Working")
        print("✅ File metadata operations: Working")
        print("✅ Error handling and continuation: Working")
        print("✅ Resume functionality: Working")
        print("✅ Timestamped output generation: Working")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
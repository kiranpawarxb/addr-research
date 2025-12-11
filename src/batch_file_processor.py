"""Batch File Processing with Smart Resume Capabilities.

This module implements comprehensive batch file processing functionality for the
GPU-CPU Hybrid Address Processing System. It provides automatic detection and
skipping of processed files, timestamped output generation with processing metadata,
error handling that continues processing after failures, and resume functionality
for interrupted processing.

Requirements: 6.1, 6.2, 6.3, 6.5
"""

import logging
import os
import time
import json
import glob
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import threading
from concurrent.futures import ThreadPoolExecutor, Future

try:
    from .hybrid_processor import GPUCPUHybridProcessor, ProcessingConfiguration, ProcessingResult, BatchProcessingReport
    from .models import ParsedAddress
    from .csv_reader import CSVReader
    from .output_writer import OutputWriter
    from .comprehensive_output_generator import ComprehensiveOutputGenerator, ComprehensiveOutputConfig
except ImportError:
    # Fallback for direct execution
    from hybrid_processor import GPUCPUHybridProcessor, ProcessingConfiguration, ProcessingResult, BatchProcessingReport
    from models import ParsedAddress
    from csv_reader import CSVReader
    from output_writer import OutputWriter
    from comprehensive_output_generator import ComprehensiveOutputGenerator, ComprehensiveOutputConfig


@dataclass
class FileProcessingMetadata:
    """Metadata for tracking file processing status and resume capabilities.
    
    Stores comprehensive information about file processing including timestamps,
    device information, performance metrics, and processing status for smart resume.
    
    Requirements: 6.2, 6.5
    """
    file_path: str
    file_hash: str  # SHA-256 hash for file integrity verification
    file_size: int  # File size in bytes
    total_addresses: int
    processed_addresses: int = 0
    
    # Processing Status
    processing_status: str = "not_started"  # not_started, in_progress, completed, failed, interrupted
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    processing_time: float = 0.0
    
    # Output Information
    output_file: Optional[str] = None
    output_timestamp: Optional[str] = None
    
    # Performance Metrics
    throughput_rate: float = 0.0
    gpu_utilization: float = 0.0
    success_rate: float = 0.0
    
    # Device Information
    device_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Error Information
    error_count: int = 0
    error_details: List[str] = field(default_factory=list)
    last_error: Optional[str] = None
    
    # Resume Information
    last_processed_index: int = 0
    resume_checkpoint: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileProcessingMetadata':
        """Create metadata from dictionary (JSON deserialization)."""
        return cls(**data)
    
    def update_processing_status(self, status: str, error: Optional[str] = None) -> None:
        """Update processing status with optional error information."""
        self.processing_status = status
        
        if status == "in_progress" and self.start_time is None:
            self.start_time = time.time()
        elif status in ["completed", "failed", "interrupted"]:
            if self.end_time is None:
                self.end_time = time.time()
            if self.start_time:
                self.processing_time = self.end_time - self.start_time
        
        if error:
            self.last_error = error
            if error not in self.error_details:
                self.error_details.append(error)
            self.error_count += 1
    
    def calculate_progress_percentage(self) -> float:
        """Calculate processing progress percentage."""
        if self.total_addresses == 0:
            return 0.0
        return (self.processed_addresses / self.total_addresses) * 100.0
    
    def is_processing_complete(self) -> bool:
        """Check if file processing is complete."""
        return self.processing_status == "completed"
    
    def can_resume(self) -> bool:
        """Check if processing can be resumed from interruption."""
        return (self.processing_status in ["interrupted", "in_progress"] and 
                self.processed_addresses > 0 and 
                self.processed_addresses < self.total_addresses)


@dataclass
class BatchProcessingState:
    """State management for batch processing operations with resume capabilities.
    
    Maintains comprehensive state information for batch processing including
    file metadata, processing progress, and resume checkpoints.
    
    Requirements: 6.1, 6.5
    """
    batch_id: str
    start_time: float
    state_file_path: str
    
    # File Processing State
    files_metadata: Dict[str, FileProcessingMetadata] = field(default_factory=dict)
    processed_files: List[str] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    
    # Batch Progress
    total_files: int = 0
    completed_files: int = 0
    total_addresses: int = 0
    processed_addresses: int = 0
    
    # Configuration
    processing_config: Optional[Dict[str, Any]] = None
    
    def save_state(self) -> None:
        """Save current batch processing state to file for resume capability."""
        try:
            state_data = {
                "batch_id": self.batch_id,
                "start_time": self.start_time,
                "files_metadata": {path: metadata.to_dict() for path, metadata in self.files_metadata.items()},
                "processed_files": self.processed_files,
                "failed_files": self.failed_files,
                "skipped_files": self.skipped_files,
                "total_files": self.total_files,
                "completed_files": self.completed_files,
                "total_addresses": self.total_addresses,
                "processed_addresses": self.processed_addresses,
                "processing_config": self.processing_config
            }
            
            # Ensure state directory exists
            os.makedirs(os.path.dirname(self.state_file_path), exist_ok=True)
            
            # Write state with atomic operation
            temp_file = f"{self.state_file_path}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            # Atomic rename
            os.replace(temp_file, self.state_file_path)
            
        except Exception as e:
            logging.error(f"Failed to save batch processing state: {e}")
    
    @classmethod
    def load_state(cls, state_file_path: str) -> Optional['BatchProcessingState']:
        """Load batch processing state from file for resume capability."""
        try:
            if not os.path.exists(state_file_path):
                return None
            
            with open(state_file_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # Reconstruct metadata objects
            files_metadata = {}
            for path, metadata_dict in state_data.get("files_metadata", {}).items():
                files_metadata[path] = FileProcessingMetadata.from_dict(metadata_dict)
            
            # Create state object
            state = cls(
                batch_id=state_data["batch_id"],
                start_time=state_data["start_time"],
                state_file_path=state_file_path
            )
            
            # Restore state
            state.files_metadata = files_metadata
            state.processed_files = state_data.get("processed_files", [])
            state.failed_files = state_data.get("failed_files", [])
            state.skipped_files = state_data.get("skipped_files", [])
            state.total_files = state_data.get("total_files", 0)
            state.completed_files = state_data.get("completed_files", 0)
            state.total_addresses = state_data.get("total_addresses", 0)
            state.processed_addresses = state_data.get("processed_addresses", 0)
            state.processing_config = state_data.get("processing_config")
            
            return state
            
        except Exception as e:
            logging.error(f"Failed to load batch processing state: {e}")
            return None
    
    def update_file_progress(self, file_path: str, processed_count: int) -> None:
        """Update processing progress for a specific file."""
        if file_path in self.files_metadata:
            metadata = self.files_metadata[file_path]
            metadata.processed_addresses = processed_count
            
            # Update batch totals
            self.processed_addresses = sum(
                meta.processed_addresses for meta in self.files_metadata.values()
            )
            
            # Save state periodically
            self.save_state()
    
    def mark_file_completed(self, file_path: str, result: ProcessingResult) -> None:
        """Mark a file as completed and update metadata."""
        if file_path in self.files_metadata:
            metadata = self.files_metadata[file_path]
            metadata.update_processing_status("completed")
            metadata.processed_addresses = len(result.parsed_addresses)
            metadata.success_rate = result.calculate_success_rate()
            
            if result.performance_metrics:
                metadata.throughput_rate = result.performance_metrics.throughput_rate
                metadata.gpu_utilization = result.performance_metrics.gpu_utilization
            
            metadata.device_statistics = result.device_statistics
            
            if file_path not in self.processed_files:
                self.processed_files.append(file_path)
            
            self.completed_files += 1
            self.save_state()
    
    def mark_file_failed(self, file_path: str, error: str) -> None:
        """Mark a file as failed and record error information."""
        if file_path in self.files_metadata:
            metadata = self.files_metadata[file_path]
            metadata.update_processing_status("failed", error)
            
            if file_path not in self.failed_files:
                self.failed_files.append(file_path)
            
            self.save_state()


class BatchFileProcessor:
    """Comprehensive batch file processing with smart resume capabilities.
    
    Implements automated file processing with detection and skipping of processed files,
    timestamped output generation with processing metadata, error handling that continues
    processing after failures, and resume functionality for interrupted processing.
    
    Requirements: 6.1, 6.2, 6.3, 6.5
    """
    
    def __init__(self, config: ProcessingConfiguration, output_dir: str = "batch_output"):
        """Initialize batch file processor with configuration.
        
        Args:
            config: Processing configuration for hybrid processing
            output_dir: Directory for batch processing outputs and state files
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory structure
        self.output_dir.mkdir(exist_ok=True)
        self.state_dir = self.output_dir / "state"
        self.state_dir.mkdir(exist_ok=True)
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize hybrid processor
        self.hybrid_processor = GPUCPUHybridProcessor(config)
        
        # Processing state
        self.current_batch_state: Optional[BatchProcessingState] = None
        self.processing_lock = threading.Lock()
        
        self.logger.info(f"Initialized BatchFileProcessor with output directory: {self.output_dir}")
    
    def process_files_batch(
        self, 
        file_patterns: List[str], 
        resume_batch_id: Optional[str] = None,
        skip_processed: bool = True,
        continue_on_error: bool = True
    ) -> BatchProcessingReport:
        """Process multiple files with smart resume capabilities and error handling.
        
        Implements comprehensive batch processing with automatic detection and skipping
        of processed files, error handling that continues processing after failures,
        and resume functionality for interrupted processing.
        
        Args:
            file_patterns: List of file patterns (glob patterns) to process
            resume_batch_id: Optional batch ID to resume from previous interruption
            skip_processed: Whether to automatically skip already processed files
            continue_on_error: Whether to continue processing after individual file failures
            
        Returns:
            BatchProcessingReport with comprehensive processing results
            
        Requirements: 6.1, 6.2, 6.3, 6.5
        """
        # Generate or resume batch ID
        if resume_batch_id:
            batch_id = resume_batch_id
            self.logger.info(f"🔄 Resuming batch processing: {batch_id}")
        else:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.logger.info(f"🚀 Starting new batch processing: {batch_id}")
        
        # Initialize or load batch state
        state_file = self.state_dir / f"{batch_id}_state.json"
        
        if resume_batch_id and state_file.exists():
            # Load existing state for resume
            self.current_batch_state = BatchProcessingState.load_state(str(state_file))
            if self.current_batch_state:
                self.logger.info(f"📂 Loaded existing batch state: {len(self.current_batch_state.files_metadata)} files tracked")
            else:
                self.logger.warning("Failed to load batch state, starting fresh")
                self.current_batch_state = self._create_new_batch_state(batch_id, str(state_file))
        else:
            # Create new batch state
            self.current_batch_state = self._create_new_batch_state(batch_id, str(state_file))
        
        try:
            # Initialize hybrid processing
            if not self.hybrid_processor.is_initialized:
                self.logger.info("🔧 Initializing hybrid processing components...")
                self.hybrid_processor.initialize_hybrid_processing()
            
            # Discover and prepare files for processing
            files_to_process = self._discover_files(file_patterns, skip_processed)
            
            if not files_to_process:
                self.logger.warning("No files found to process")
                return self._generate_batch_report()
            
            self.logger.info(f"📊 Batch processing plan: {len(files_to_process)} files")
            
            # Update batch state with discovered files
            self._update_batch_state_with_files(files_to_process)
            
            # Process files with error handling and resume capability
            self._process_files_with_error_handling(files_to_process, continue_on_error)
            
            # Generate comprehensive batch report
            batch_report = self._generate_batch_report()
            
            self.logger.info(f"🎉 Batch processing completed: {batch_report.total_files_processed} files processed")
            
            return batch_report
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            if self.current_batch_state:
                # Save state for potential resume
                self.current_batch_state.save_state()
            raise
        finally:
            # Cleanup
            if self.hybrid_processor.is_initialized:
                self.hybrid_processor.shutdown()
    
    def _create_new_batch_state(self, batch_id: str, state_file_path: str) -> BatchProcessingState:
        """Create new batch processing state."""
        state = BatchProcessingState(
            batch_id=batch_id,
            start_time=time.time(),
            state_file_path=state_file_path,
            processing_config=self._serialize_config()
        )
        return state
    
    def _serialize_config(self) -> Dict[str, Any]:
        """Serialize processing configuration for state persistence."""
        return {
            "gpu_batch_size": self.config.gpu_batch_size,
            "dataset_batch_size": self.config.dataset_batch_size,
            "gpu_memory_fraction": self.config.gpu_memory_fraction,
            "gpu_queue_size": self.config.gpu_queue_size,
            "num_gpu_streams": self.config.num_gpu_streams,
            "cpu_allocation_ratio": self.config.cpu_allocation_ratio,
            "cpu_batch_size": self.config.cpu_batch_size,
            "target_throughput": self.config.target_throughput,
            "gpu_utilization_threshold": self.config.gpu_utilization_threshold
        }
    
    def _discover_files(self, file_patterns: List[str], skip_processed: bool) -> List[str]:
        """Discover files to process with automatic detection and skipping of processed files.
        
        Implements automatic detection and skipping of already processed files based on
        file metadata and processing state information.
        
        Args:
            file_patterns: List of file patterns to search
            skip_processed: Whether to skip already processed files
            
        Returns:
            List of file paths to process
            
        Requirements: 6.1
        """
        discovered_files = []
        
        # Expand file patterns
        for pattern in file_patterns:
            matching_files = glob.glob(pattern, recursive=True)
            discovered_files.extend(matching_files)
        
        # Remove duplicates and sort
        discovered_files = sorted(list(set(discovered_files)))
        
        self.logger.info(f"📂 Discovered {len(discovered_files)} files from patterns")
        
        if not skip_processed:
            return discovered_files
        
        # Filter out already processed files
        files_to_process = []
        
        for file_path in discovered_files:
            if self._should_skip_file(file_path):
                if self.current_batch_state:
                    self.current_batch_state.skipped_files.append(file_path)
                self.logger.info(f"⏭️ Skipping already processed file: {os.path.basename(file_path)}")
            else:
                files_to_process.append(file_path)
        
        skipped_count = len(self.current_batch_state.skipped_files) if self.current_batch_state else 0
        self.logger.info(f"📋 Files to process: {len(files_to_process)} (skipped: {skipped_count})")
        
        return files_to_process
    
    def _should_skip_file(self, file_path: str) -> bool:
        """Determine if a file should be skipped based on processing history.
        
        Checks file metadata and processing state to determine if file has already
        been successfully processed and should be skipped.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file should be skipped, False otherwise
        """
        # Check if file is in current batch state as completed
        if (self.current_batch_state and 
            file_path in self.current_batch_state.files_metadata):
            metadata = self.current_batch_state.files_metadata[file_path]
            if metadata.is_processing_complete():
                # Verify output file still exists
                if metadata.output_file and os.path.exists(metadata.output_file):
                    return True
        
        # Check for existing output files with timestamped naming
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        
        # Look for existing output files in results directory
        output_pattern = str(self.results_dir / f"{base_name}_processed_*.csv")
        existing_outputs = glob.glob(output_pattern)
        
        if existing_outputs:
            # Check if any existing output is newer than source file
            source_mtime = os.path.getmtime(file_path)
            for output_file in existing_outputs:
                output_mtime = os.path.getmtime(output_file)
                if output_mtime > source_mtime:
                    self.logger.debug(f"Found newer output file for {file_name}: {os.path.basename(output_file)}")
                    return True
        
        return False
    
    def _update_batch_state_with_files(self, files_to_process: List[str]) -> None:
        """Update batch state with file metadata for processing tracking."""
        self.current_batch_state.total_files = len(files_to_process)
        
        total_addresses = 0
        
        for file_path in files_to_process:
            if file_path not in self.current_batch_state.files_metadata:
                # Create metadata for new file
                file_hash = self._calculate_file_hash(file_path)
                file_size = os.path.getsize(file_path)
                
                # Quick count of addresses in file
                address_count = self._count_addresses_in_file(file_path)
                total_addresses += address_count
                
                metadata = FileProcessingMetadata(
                    file_path=file_path,
                    file_hash=file_hash,
                    file_size=file_size,
                    total_addresses=address_count
                )
                
                self.current_batch_state.files_metadata[file_path] = metadata
        
        self.current_batch_state.total_addresses = total_addresses
        self.current_batch_state.save_state()
        
        self.logger.info(f"📊 Batch state updated: {len(files_to_process)} files, {total_addresses:,} total addresses")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file for integrity verification."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to calculate hash for {file_path}: {e}")
            return "unknown"
    
    def _count_addresses_in_file(self, file_path: str) -> int:
        """Quick count of addresses in CSV file."""
        try:
            csv_reader = CSVReader(file_path)
            records = csv_reader.read()
            return len(records)
        except Exception as e:
            self.logger.warning(f"Failed to count addresses in {file_path}: {e}")
            return 0
    
    def _process_files_with_error_handling(self, files_to_process: List[str], continue_on_error: bool) -> None:
        """Process files with comprehensive error handling and continuation logic.
        
        Implements error handling that continues processing after individual file failures
        and maintains processing state for resume capability.
        
        Args:
            files_to_process: List of file paths to process
            continue_on_error: Whether to continue processing after failures
            
        Requirements: 6.3
        """
        for i, file_path in enumerate(files_to_process, 1):
            file_name = os.path.basename(file_path)
            
            self.logger.info(f"📄 Processing file {i}/{len(files_to_process)}: {file_name}")
            
            # Check if file can be resumed
            metadata = self.current_batch_state.files_metadata[file_path]
            
            if metadata.can_resume():
                self.logger.info(f"🔄 Resuming interrupted processing from {metadata.processed_addresses} addresses")
            
            try:
                # Update processing status
                metadata.update_processing_status("in_progress")
                self.current_batch_state.save_state()
                
                # Process single file with resume capability
                result = self._process_single_file_with_resume(file_path)
                
                if result:
                    # Mark file as completed
                    self.current_batch_state.mark_file_completed(file_path, result)
                    
                    self.logger.info(f"✅ {file_name} completed: {len(result.parsed_addresses)} addresses processed")
                    self.logger.info(f"   Success rate: {result.calculate_success_rate():.1f}%")
                    if result.performance_metrics:
                        self.logger.info(f"   Throughput: {result.performance_metrics.throughput_rate:.1f} addr/sec")
                else:
                    raise RuntimeError("Processing returned no result")
                
            except Exception as e:
                error_msg = f"Processing failed for {file_name}: {str(e)}"
                self.logger.error(error_msg)
                
                # Mark file as failed
                self.current_batch_state.mark_file_failed(file_path, error_msg)
                
                if not continue_on_error:
                    self.logger.error("Stopping batch processing due to error (continue_on_error=False)")
                    raise
                else:
                    self.logger.info(f"⚠️ Continuing with next file (continue_on_error=True)")
    
    def _process_single_file_with_resume(self, file_path: str) -> Optional[ProcessingResult]:
        """Process a single file with resume capability for interrupted processing.
        
        Implements resume functionality for interrupted processing by checking processing
        state and continuing from the last processed position.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            ProcessingResult with parsed addresses and metadata
            
        Requirements: 6.5
        """
        metadata = self.current_batch_state.files_metadata[file_path]
        
        try:
            # Load addresses from file
            csv_reader = CSVReader(file_path)
            address_records = csv_reader.read()
            
            if not address_records:
                raise ValueError("No addresses found in file")
            
            # Extract address texts for processing
            addresses = [record.addr_text for record in address_records]
            
            # Check for resume capability
            start_index = 0
            if metadata.can_resume() and metadata.last_processed_index > 0:
                start_index = metadata.last_processed_index
                addresses = addresses[start_index:]
                self.logger.info(f"🔄 Resuming from address {start_index + 1}/{metadata.total_addresses}")
            
            # Process addresses using hybrid processor
            result = self.hybrid_processor.process_addresses_hybrid(addresses)
            
            # Generate timestamped output file with processing metadata
            output_file = self._generate_timestamped_output(file_path, address_records, result, start_index)
            
            # Update metadata with output information
            metadata.output_file = output_file
            metadata.output_timestamp = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process file {file_path}: {e}")
            # Save current progress for potential resume
            metadata.update_processing_status("interrupted", str(e))
            self.current_batch_state.save_state()
            raise
    
    def _generate_timestamped_output(
        self, 
        file_path: str, 
        address_records: List, 
        result: ProcessingResult, 
        start_index: int = 0
    ) -> str:
        """Generate timestamped output file with comprehensive processing metadata.
        
        Creates comprehensive output files with timestamps, all parsed fields,
        processing metadata, device information, and performance metrics.
        
        Args:
            file_path: Original input file path
            address_records: Original address records from CSV
            result: Processing result with parsed addresses
            start_index: Starting index for resume processing
            
        Returns:
            Path to generated output file
            
        Requirements: 6.2, 9.1, 9.2
        """
        # Generate timestamped filename
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_processed_{timestamp}"
        output_path = str(self.results_dir / output_filename)
        
        try:
            # Use comprehensive output generator for enhanced output
            comprehensive_config = ComprehensiveOutputConfig(
                include_metadata_header=True,
                include_performance_summary=True,
                include_device_information=True,
                include_error_details=True,
                include_optimization_suggestions=True,
                generate_json_metadata=True,
                generate_performance_report=True,
                generate_error_report=result.error_count > 0,
                include_processing_timestamps=True,
                include_device_columns=True,
                include_performance_columns=True
            )
            
            # Create comprehensive output generator
            output_generator = ComprehensiveOutputGenerator(comprehensive_config)
            
            # Get relevant address records (handle resume case)
            relevant_records = address_records[start_index:start_index + len(result.parsed_addresses)]
            
            # Generate comprehensive output
            output_files = output_generator.generate_comprehensive_output(
                result.parsed_addresses,
                relevant_records,
                result,
                output_path
            )
            
            # Return the main CSV file path
            main_csv_path = output_files.get('csv', output_path + '.csv')
            
            self.logger.info(f"📁 Generated comprehensive output: {os.path.basename(main_csv_path)}")
            self.logger.info(f"   Additional files: {len(output_files) - 1} (metadata, performance, etc.)")
            
            return main_csv_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive output, falling back to basic output: {e}")
            
            # Fallback to basic output generation
            return self._generate_basic_timestamped_output(file_path, address_records, result, start_index, timestamp)
    
    def _generate_basic_timestamped_output(
        self, 
        file_path: str, 
        address_records: List, 
        result: ProcessingResult, 
        start_index: int,
        timestamp: str
    ) -> str:
        """Generate basic timestamped output as fallback."""
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        output_filename = f"{base_name}_processed_{timestamp}.csv"
        output_path = str(self.results_dir / output_filename)
        
        # Create consolidated groups for output writer
        from .models import ConsolidatedGroup
        
        # Combine original records with parsed results
        combined_records = []
        for i, (record, parsed) in enumerate(zip(address_records[start_index:], result.parsed_addresses)):
            combined_records.append((record, parsed))
        
        # Create a single group for all records (batch processing doesn't consolidate)
        consolidated_group = ConsolidatedGroup(
            group_id=f"batch_{timestamp}",
            society_name="BATCH_PROCESSING",
            pin_code="000000",
            records=combined_records
        )
        
        # Write output with metadata
        output_writer = OutputWriter(output_path)
        records_written = output_writer.write([consolidated_group])
        
        # Add processing metadata to output file
        self._add_processing_metadata_to_output(output_path, result, file_path, timestamp)
        
        self.logger.info(f"📁 Generated basic output: {output_filename} ({records_written} records)")
        
        return output_path
    
    def _add_processing_metadata_to_output(
        self, 
        output_path: str, 
        result: ProcessingResult, 
        source_file: str, 
        timestamp: str
    ) -> None:
        """Add comprehensive processing metadata to output file.
        
        Appends processing metadata including timestamps, device information,
        and performance metrics to the output file for comprehensive analysis.
        
        Args:
            output_path: Path to output CSV file
            result: Processing result with metadata
            source_file: Original source file path
            timestamp: Processing timestamp
        """
        try:
            metadata_lines = [
                "",
                "# PROCESSING METADATA",
                f"# Source File: {source_file}",
                f"# Processing Timestamp: {timestamp}",
                f"# Processing Time: {result.processing_time:.2f} seconds",
                f"# GPU Processing Time: {result.gpu_processing_time:.2f} seconds",
                f"# CPU Processing Time: {result.cpu_processing_time:.2f} seconds",
                f"# Total Addresses: {len(result.parsed_addresses)}",
                f"# Success Rate: {result.calculate_success_rate():.1f}%",
                f"# Error Count: {result.error_count}",
            ]
            
            # Add performance metrics
            if result.performance_metrics:
                metadata_lines.extend([
                    f"# Throughput Rate: {result.performance_metrics.throughput_rate:.1f} addresses/second",
                    f"# GPU Utilization: {result.performance_metrics.gpu_utilization:.1f}%",
                    f"# Processing Efficiency: {result.performance_metrics.processing_efficiency:.1f}%",
                ])
            
            # Add device statistics
            if result.device_statistics:
                metadata_lines.append("# Device Statistics:")
                for key, value in result.device_statistics.items():
                    metadata_lines.append(f"#   {key}: {value}")
            
            # Add optimization suggestions
            if result.optimization_suggestions:
                metadata_lines.append("# Optimization Suggestions:")
                for suggestion in result.optimization_suggestions:
                    metadata_lines.append(f"#   - {suggestion}")
            
            # Append metadata to file
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(metadata_lines))
            
        except Exception as e:
            self.logger.warning(f"Failed to add metadata to output file: {e}")
    
    def _generate_batch_report(self) -> BatchProcessingReport:
        """Generate comprehensive batch processing report with performance analysis."""
        if not self.current_batch_state:
            return BatchProcessingReport()
        
        report = BatchProcessingReport()
        
        # Collect file results
        for file_path, metadata in self.current_batch_state.files_metadata.items():
            if metadata.is_processing_complete():
                # Create processing result from metadata
                file_result = ProcessingResult(
                    processing_time=metadata.processing_time,
                    device_statistics=metadata.device_statistics,
                    error_count=metadata.error_count
                )
                
                # Add performance metrics if available
                if metadata.throughput_rate > 0:
                    from .hybrid_processor import PerformanceMetrics
                    file_result.performance_metrics = PerformanceMetrics(
                        throughput_rate=metadata.throughput_rate,
                        gpu_utilization=metadata.gpu_utilization
                    )
                
                # Add to report
                report.add_file_result(file_result, os.path.basename(file_path))
        
        # Generate summary
        report.generate_summary()
        
        # Generate comprehensive batch processing reports
        try:
            batch_reports_dir = self.output_dir / "batch_reports"
            batch_reports_dir.mkdir(exist_ok=True)
            
            # Create comprehensive output generator
            comprehensive_config = ComprehensiveOutputConfig(
                generate_batch_summary=True,
                generate_comparative_analysis=True,
                include_per_file_metrics=True
            )
            
            output_generator = ComprehensiveOutputGenerator(comprehensive_config)
            
            # Generate comprehensive batch reports
            report_files = output_generator.generate_batch_processing_report(
                report,
                str(batch_reports_dir),
                self.current_batch_state.batch_id
            )
            
            self.logger.info(f"📊 Generated comprehensive batch reports: {len(report_files)} files")
            for report_type, file_path in report_files.items():
                self.logger.info(f"   {report_type}: {os.path.basename(file_path)}")
                
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive batch reports: {e}")
        
        return report
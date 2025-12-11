"""AsynchronousQueueManager - Manages GPU queues and asynchronous processing.

This module implements the AsynchronousQueueManager class for sustained GPU utilization
through continuous batch feeding, multiple concurrent GPU workers, and asynchronous
result collection.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from queue import Queue, Empty, Full
from typing import List, Optional, Callable, Any, Dict
from dataclasses import dataclass

try:
    from .models import ParsedAddress
    from .hybrid_processor import ProcessingConfiguration
    from .error_handling import ErrorRecoveryManager
except ImportError:
    # Fallback for direct execution
    from models import ParsedAddress
    from hybrid_processor import ProcessingConfiguration
    from error_handling import ErrorRecoveryManager


@dataclass
class QueueStatus:
    """Status information for GPU queues."""
    input_queue_size: int
    output_queue_size: int
    max_queue_size: int
    active_workers: int
    total_processed: int
    processing_rate: float
    queue_utilization: float


class AsynchronousQueueManager:
    """Manages GPU queues and asynchronous processing for sustained GPU utilization.
    
    Implements asynchronous batch processing with GPU input queues, pre-loaded batches,
    multiple concurrent GPU workers, dedicated data feeder threads, and output queues
    for result collection to maintain continuous GPU feeding without idle time.
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
    """
    
    def __init__(self, config: ProcessingConfiguration):
        """Initialize AsynchronousQueueManager with configuration.
        
        Args:
            config: Processing configuration with queue settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # GPU Queues
        self.input_queue: Optional[Queue] = None
        self.output_queue: Optional[Queue] = None
        self.queue_size = config.gpu_queue_size
        
        # Worker Management
        self.gpu_workers: List[Future] = []
        self.data_feeder_thread: Optional[threading.Thread] = None
        self.result_collector_thread: Optional[threading.Thread] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        
        # Processing State
        self.is_initialized = False
        self.is_running = False
        self.shutdown_requested = False
        
        # Processing Function (will be set by caller)
        self.gpu_processing_function: Optional[Callable] = None
        
        # Error handling
        self.error_recovery_manager = None
        
        # Statistics and Monitoring
        self.processing_stats = {
            "total_batches_queued": 0,
            "total_batches_processed": 0,
            "total_addresses_processed": 0,
            "queue_wait_times": [],
            "processing_start_time": 0,
            "last_activity_time": time.time()
        }
        
        # Thread synchronization
        self.stats_lock = threading.Lock()
        self.worker_lock = threading.Lock()
        
        self.logger.info(f"Initialized AsynchronousQueueManager with queue size: {self.queue_size}")
    
    def initialize_gpu_queues(self, queue_size: Optional[int] = None) -> bool:
        """Initialize GPU input and output queues with configurable sizes.
        
        Sets up input and output queues for asynchronous GPU processing with
        configurable queue sizes to maintain continuous GPU feeding.
        
        Args:
            queue_size: Optional override for queue size (uses config if None)
            
        Returns:
            True if initialization successful, False otherwise
            
        Requirements: 3.1, 3.2
        """
        if self.is_initialized:
            self.logger.warning("GPU queues already initialized")
            return True
        
        try:
            # Use provided queue size or fall back to configuration
            if queue_size is not None:
                self.queue_size = queue_size
            
            self.logger.info(f"🔧 Initializing GPU queues with size: {self.queue_size}")
            
            # Create input queue for batches ready for GPU processing
            # Queue size should accommodate 10+ pre-loaded batches as per requirements
            self.input_queue = Queue(maxsize=self.queue_size)
            
            # Create output queue for processed results
            # Output queue can be larger to handle burst processing
            self.output_queue = Queue(maxsize=self.queue_size * 2)
            
            # Initialize thread executor for GPU workers
            # Number of workers based on GPU streams configuration
            max_workers = max(2, self.config.num_gpu_streams)
            self.executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="GPUWorker"
            )
            
            # Reset statistics
            with self.stats_lock:
                self.processing_stats = {
                    "total_batches_queued": 0,
                    "total_batches_processed": 0,
                    "total_addresses_processed": 0,
                    "queue_wait_times": [],
                    "processing_start_time": time.time(),
                    "last_activity_time": time.time()
                }
            
            self.is_initialized = True
            self.logger.info("✅ GPU queues initialized successfully")
            self.logger.info(f"  Input queue capacity: {self.input_queue.maxsize}")
            self.logger.info(f"  Output queue capacity: {self.output_queue.maxsize}")
            self.logger.info(f"  GPU worker threads: {max_workers}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GPU queues: {e}")
            return False
    
    def start_data_feeder(self, addresses: List[str], batch_size: Optional[int] = None) -> bool:
        """Start continuous batch preparation and feeding to GPU input queue.
        
        Launches dedicated data feeder thread that continuously prepares batches
        and feeds them to the GPU input queue to eliminate GPU starvation.
        
        Args:
            addresses: List of addresses to process
            batch_size: Optional batch size override (uses config if None)
            
        Returns:
            True if data feeder started successfully, False otherwise
            
        Requirements: 3.2, 3.4
        """
        if not self.is_initialized:
            self.logger.error("GPU queues not initialized. Call initialize_gpu_queues() first.")
            return False
        
        if self.data_feeder_thread and self.data_feeder_thread.is_alive():
            self.logger.warning("Data feeder already running")
            return True
        
        if not addresses:
            self.logger.warning("No addresses provided for data feeding")
            return False
        
        try:
            # Use provided batch size or fall back to configuration
            effective_batch_size = batch_size or self.config.gpu_batch_size
            
            self.logger.info(f"🚀 Starting data feeder for {len(addresses)} addresses "
                           f"with batch size: {effective_batch_size}")
            
            # Create and start data feeder thread
            self.data_feeder_thread = threading.Thread(
                target=self._data_feeder_worker,
                args=(addresses, effective_batch_size),
                name="DataFeeder",
                daemon=True
            )
            
            self.data_feeder_thread.start()
            
            # Wait a moment to ensure thread started successfully
            time.sleep(0.1)
            
            # Check if thread was created successfully
            # For small datasets, thread may complete quickly, which is normal
            if self.data_feeder_thread.ident is not None:
                self.logger.info("✅ Data feeder thread started successfully")
                return True
            else:
                self.logger.error("Data feeder thread failed to start")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start data feeder: {e}")
            return False
    
    def start_gpu_workers(self, num_workers: Optional[int] = None, 
                         processing_function: Optional[Callable] = None) -> bool:
        """Start multiple concurrent GPU workers for batch processing.
        
        Launches multiple GPU processing workers that consume batches from the
        input queue and process them concurrently using multiple GPU streams.
        
        Args:
            num_workers: Optional number of workers (uses config if None)
            processing_function: Function to process batches on GPU
            
        Returns:
            True if GPU workers started successfully, False otherwise
            
        Requirements: 3.3, 3.4
        """
        if not self.is_initialized:
            self.logger.error("GPU queues not initialized. Call initialize_gpu_queues() first.")
            return False
        
        if self.gpu_workers:
            self.logger.warning("GPU workers already running")
            return True
        
        try:
            # Use provided parameters or fall back to configuration/stored values
            effective_num_workers = num_workers or self.config.num_gpu_streams
            if processing_function:
                self.gpu_processing_function = processing_function
            
            if not self.gpu_processing_function:
                self.logger.error("No GPU processing function provided")
                return False
            
            self.logger.info(f"🚀 Starting {effective_num_workers} GPU workers")
            
            # Start GPU worker threads
            with self.worker_lock:
                self.gpu_workers = []
                for worker_id in range(effective_num_workers):
                    future = self.executor.submit(
                        self._gpu_worker,
                        worker_id,
                        self.gpu_processing_function
                    )
                    self.gpu_workers.append(future)
            
            # Start result collector thread
            if not self.result_collector_thread or not self.result_collector_thread.is_alive():
                self.result_collector_thread = threading.Thread(
                    target=self._result_collector_worker,
                    name="ResultCollector",
                    daemon=True
                )
                self.result_collector_thread.start()
            
            self.is_running = True
            
            self.logger.info(f"✅ Started {len(self.gpu_workers)} GPU workers successfully")
            self.logger.info("✅ Result collector thread started")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start GPU workers: {e}")
            return False
    
    def collect_results(self, timeout: Optional[float] = None) -> List[ParsedAddress]:
        """Collect processed results from GPU workers asynchronously.
        
        Gathers processed results from the output queue without blocking
        the GPU processing pipeline.
        
        Args:
            timeout: Optional timeout for result collection (seconds)
            
        Returns:
            List of processed ParsedAddress objects
            
        Requirements: 3.5
        """
        if not self.is_initialized:
            self.logger.error("GPU queues not initialized")
            return []
        
        results = []
        start_time = time.time()
        
        try:
            self.logger.debug("🔍 Collecting results from output queue...")
            
            while True:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    self.logger.debug(f"Result collection timeout after {timeout}s")
                    break
                
                try:
                    # Try to get result from output queue (non-blocking)
                    result_batch = self.output_queue.get(timeout=0.1)
                    
                    if result_batch is None:
                        # Sentinel value indicating end of processing
                        self.logger.debug("Received end-of-processing sentinel")
                        break
                    
                    # Add batch results to collection
                    if isinstance(result_batch, list):
                        results.extend(result_batch)
                    else:
                        results.append(result_batch)
                    
                    # Mark task as done
                    self.output_queue.task_done()
                    
                    # Update statistics
                    with self.stats_lock:
                        self.processing_stats["last_activity_time"] = time.time()
                    
                except Empty:
                    # No results available, check if processing is still active
                    if not self.is_running and self.output_queue.empty():
                        self.logger.debug("No more results available and processing stopped")
                        break
                    
                    # Continue waiting for results
                    continue
                
                except Exception as e:
                    self.logger.error(f"Error collecting result: {e}")
                    continue
            
            collection_time = time.time() - start_time
            self.logger.info(f"✅ Collected {len(results)} results in {collection_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to collect results: {e}")
            return results
    
    def get_queue_status(self) -> QueueStatus:
        """Get current status of GPU queues and processing pipeline.
        
        Returns:
            QueueStatus with current queue sizes and processing metrics
        """
        if not self.is_initialized:
            return QueueStatus(0, 0, 0, 0, 0, 0.0, 0.0)
        
        try:
            with self.stats_lock:
                # Calculate processing rate
                elapsed_time = time.time() - self.processing_stats["processing_start_time"]
                processing_rate = (
                    self.processing_stats["total_addresses_processed"] / elapsed_time
                    if elapsed_time > 0 else 0.0
                )
                
                # Calculate queue utilization
                input_utilization = (
                    self.input_queue.qsize() / self.input_queue.maxsize
                    if self.input_queue.maxsize > 0 else 0.0
                )
                
                # Count active workers
                active_workers = sum(1 for worker in self.gpu_workers if not worker.done())
            
            return QueueStatus(
                input_queue_size=self.input_queue.qsize() if self.input_queue else 0,
                output_queue_size=self.output_queue.qsize() if self.output_queue else 0,
                max_queue_size=self.queue_size,
                active_workers=active_workers,
                total_processed=self.processing_stats["total_addresses_processed"],
                processing_rate=processing_rate,
                queue_utilization=input_utilization
            )
            
        except Exception as e:
            self.logger.error(f"Error getting queue status: {e}")
            return QueueStatus(0, 0, 0, 0, 0, 0.0, 0.0)
    
    def shutdown(self) -> None:
        """Gracefully shutdown the queue manager and cleanup resources."""
        self.logger.info("🔄 Shutting down AsynchronousQueueManager...")
        
        # Signal shutdown
        self.shutdown_requested = True
        self.is_running = False
        
        try:
            # Stop data feeder thread
            if self.data_feeder_thread and self.data_feeder_thread.is_alive():
                self.logger.info("Stopping data feeder thread...")
                self.data_feeder_thread.join(timeout=5.0)
                if self.data_feeder_thread.is_alive():
                    self.logger.warning("Data feeder thread did not stop gracefully")
            
            # Stop GPU workers
            if self.gpu_workers:
                self.logger.info(f"Stopping {len(self.gpu_workers)} GPU workers...")
                
                # Cancel pending futures
                for worker in self.gpu_workers:
                    worker.cancel()
                
                # Wait for workers to complete
                for worker in as_completed(self.gpu_workers, timeout=10.0):
                    try:
                        worker.result(timeout=1.0)
                    except Exception as e:
                        self.logger.debug(f"Worker completed with exception: {e}")
                
                self.gpu_workers.clear()
            
            # Stop result collector thread
            if self.result_collector_thread and self.result_collector_thread.is_alive():
                self.logger.info("Stopping result collector thread...")
                # Add sentinel to signal end
                if self.output_queue:
                    try:
                        self.output_queue.put(None, timeout=1.0)
                    except Full:
                        pass
                
                self.result_collector_thread.join(timeout=5.0)
                if self.result_collector_thread.is_alive():
                    self.logger.warning("Result collector thread did not stop gracefully")
            
            # Shutdown thread executor
            if self.executor:
                self.logger.info("Shutting down thread executor...")
                self.executor.shutdown(wait=True)
                self.executor = None
            
            # Clear queues
            if self.input_queue:
                while not self.input_queue.empty():
                    try:
                        self.input_queue.get_nowait()
                    except Empty:
                        break
            
            if self.output_queue:
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except Empty:
                        break
            
            # Reset state
            self.input_queue = None
            self.output_queue = None
            self.data_feeder_thread = None
            self.result_collector_thread = None
            self.is_initialized = False
            
            self.logger.info("✅ AsynchronousQueueManager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    # Private worker methods
    
    def _data_feeder_worker(self, addresses: List[str], batch_size: int) -> None:
        """Worker thread that continuously feeds batches to the GPU input queue.
        
        This dedicated thread prepares batches and feeds them to the GPU input queue
        to maintain continuous GPU feeding and eliminate starvation.
        
        Args:
            addresses: List of addresses to process
            batch_size: Size of each batch
        """
        self.logger.info(f"📥 Data feeder started: {len(addresses)} addresses, batch size: {batch_size}")
        
        try:
            # Split addresses into batches
            batches = []
            for i in range(0, len(addresses), batch_size):
                batch = addresses[i:i + batch_size]
                batches.append(batch)
            
            self.logger.info(f"Created {len(batches)} batches for processing")
            
            # Feed batches to input queue
            for batch_idx, batch in enumerate(batches):
                if self.shutdown_requested:
                    self.logger.info("Data feeder shutdown requested")
                    break
                
                try:
                    # Add batch to input queue (blocking if queue is full)
                    queue_start_time = time.time()
                    self.input_queue.put(batch, timeout=30.0)
                    queue_wait_time = time.time() - queue_start_time
                    
                    # Update statistics
                    with self.stats_lock:
                        self.processing_stats["total_batches_queued"] += 1
                        self.processing_stats["queue_wait_times"].append(queue_wait_time)
                        self.processing_stats["last_activity_time"] = time.time()
                    
                    self.logger.debug(f"Queued batch {batch_idx + 1}/{len(batches)} "
                                    f"({len(batch)} addresses) in {queue_wait_time:.3f}s")
                    
                    # Brief pause to prevent overwhelming the queue
                    if queue_wait_time < 0.001:  # If queue accepted immediately
                        time.sleep(0.001)  # Small delay to prevent CPU spinning
                
                except Full:
                    self.logger.warning(f"Input queue full, skipping batch {batch_idx + 1}")
                    continue
                
                except Exception as e:
                    self.logger.error(f"Error queuing batch {batch_idx + 1}: {e}")
                    continue
            
            # Signal end of data by adding sentinel
            try:
                self.input_queue.put(None, timeout=5.0)
                self.logger.info("📥 Data feeder completed - all batches queued")
            except Full:
                self.logger.warning("Could not add end-of-data sentinel to input queue")
            
        except Exception as e:
            self.logger.error(f"Data feeder worker failed: {e}")
        
        finally:
            self.logger.info("📥 Data feeder thread exiting")
    
    def _gpu_worker(self, worker_id: int, processing_function: Callable) -> None:
        """GPU worker thread that processes batches from the input queue.
        
        Each worker continuously consumes batches from the input queue,
        processes them using the provided GPU processing function, and
        puts results in the output queue.
        
        Args:
            worker_id: Unique identifier for this worker
            processing_function: Function to process batches on GPU
        """
        self.logger.info(f"🔥 GPU Worker {worker_id} started")
        
        processed_batches = 0
        processed_addresses = 0
        
        try:
            while not self.shutdown_requested:
                try:
                    # Get batch from input queue
                    batch = self.input_queue.get(timeout=1.0)
                    
                    if batch is None:
                        # Sentinel value indicating end of processing
                        self.logger.info(f"🔥 GPU Worker {worker_id} received end signal")
                        break
                    
                    # Process batch using GPU
                    batch_start_time = time.time()
                    try:
                        results = processing_function(batch)
                        batch_processing_time = time.time() - batch_start_time
                        
                        # Put results in output queue
                        self.output_queue.put(results, timeout=5.0)
                        
                        # Update statistics
                        processed_batches += 1
                        processed_addresses += len(batch)
                        
                        with self.stats_lock:
                            self.processing_stats["total_batches_processed"] += 1
                            self.processing_stats["total_addresses_processed"] += len(batch)
                            self.processing_stats["last_activity_time"] = time.time()
                        
                        self.logger.debug(f"🔥 Worker {worker_id} processed batch "
                                        f"({len(batch)} addresses) in {batch_processing_time:.3f}s")
                    
                    except Exception as e:
                        self.logger.error(f"🔥 Worker {worker_id} processing error: {e}")
                        # Create failed results for this batch
                        failed_results = [
                            ParsedAddress(
                                parse_success=False,
                                parse_error=f"GPU worker {worker_id} processing error: {str(e)}"
                            ) for _ in batch
                        ]
                        self.output_queue.put(failed_results, timeout=5.0)
                    
                    finally:
                        # Mark input task as done
                        self.input_queue.task_done()
                
                except Empty:
                    # No batch available, continue waiting
                    continue
                
                except Exception as e:
                    self.logger.error(f"🔥 Worker {worker_id} error: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"🔥 GPU Worker {worker_id} failed: {e}")
        
        finally:
            self.logger.info(f"🔥 GPU Worker {worker_id} exiting "
                           f"(processed {processed_batches} batches, {processed_addresses} addresses)")
    
    def _result_collector_worker(self) -> None:
        """Worker thread that manages result collection and queue cleanup.
        
        This thread monitors the output queue and ensures proper cleanup
        of processed results.
        """
        self.logger.info("📤 Result collector started")
        
        try:
            while not self.shutdown_requested:
                try:
                    # Monitor output queue size
                    queue_size = self.output_queue.qsize()
                    
                    if queue_size > self.queue_size:
                        self.logger.warning(f"Output queue size high: {queue_size}")
                    
                    # Brief pause
                    time.sleep(1.0)
                
                except Exception as e:
                    self.logger.error(f"Result collector error: {e}")
                    time.sleep(1.0)
        
        except Exception as e:
            self.logger.error(f"Result collector worker failed: {e}")
        
        finally:
            self.logger.info("📤 Result collector thread exiting")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics.
        
        Returns:
            Dictionary with processing statistics and performance metrics
        """
        with self.stats_lock:
            elapsed_time = time.time() - self.processing_stats["processing_start_time"]
            
            avg_queue_wait = (
                sum(self.processing_stats["queue_wait_times"]) / 
                len(self.processing_stats["queue_wait_times"])
                if self.processing_stats["queue_wait_times"] else 0.0
            )
            
            processing_rate = (
                self.processing_stats["total_addresses_processed"] / elapsed_time
                if elapsed_time > 0 else 0.0
            )
            
            return {
                "is_initialized": self.is_initialized,
                "is_running": self.is_running,
                "queue_size": self.queue_size,
                "total_batches_queued": self.processing_stats["total_batches_queued"],
                "total_batches_processed": self.processing_stats["total_batches_processed"],
                "total_addresses_processed": self.processing_stats["total_addresses_processed"],
                "processing_rate": processing_rate,
                "average_queue_wait_time": avg_queue_wait,
                "elapsed_time": elapsed_time,
                "active_workers": len([w for w in self.gpu_workers if not w.done()]),
                "input_queue_size": self.input_queue.qsize() if self.input_queue else 0,
                "output_queue_size": self.output_queue.qsize() if self.output_queue else 0,
                "data_feeder_active": (
                    self.data_feeder_thread.is_alive() 
                    if self.data_feeder_thread else False
                ),
                "result_collector_active": (
                    self.result_collector_thread.is_alive() 
                    if self.result_collector_thread else False
                )
            }
    
    def set_error_recovery_manager(self, error_manager: 'ErrorRecoveryManager') -> None:
        """Set error recovery manager for comprehensive error handling."""
        self.error_recovery_manager = error_manager
"""Comprehensive Error Handling and Recovery Mechanisms.

This module implements robust error handling and graceful degradation for the
GPU-CPU hybrid processing system, including automatic fallback strategies,
memory allocation recovery, model loading alternatives, timeout handling,
and critical error management with partial result saving.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple, Union
from enum import Enum
import traceback
import pickle
import os
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from .models import ParsedAddress
except ImportError:
    # Fallback for direct execution
    from models import ParsedAddress


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorType(Enum):
    """Types of errors that can occur during processing."""
    GPU_PROCESSING = "gpu_processing"
    MEMORY_ALLOCATION = "memory_allocation"
    MODEL_LOADING = "model_loading"
    TIMEOUT = "timeout"
    CRITICAL_SYSTEM = "critical_system"
    NETWORK = "network"
    DATA_CORRUPTION = "data_corruption"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class ErrorContext:
    """Context information for error handling and recovery."""
    error_type: ErrorType
    severity: ErrorSeverity
    error_message: str
    exception: Optional[Exception] = None
    timestamp: float = field(default_factory=time.time)
    component: str = ""
    operation: str = ""
    retry_count: int = 0
    max_retries: int = 3
    recovery_attempted: bool = False
    recovery_successful: bool = False
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary for logging/serialization."""
        return {
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "error_message": self.error_message,
            "exception_type": type(self.exception).__name__ if self.exception else None,
            "exception_str": str(self.exception) if self.exception else None,
            "timestamp": self.timestamp,
            "component": self.component,
            "operation": self.operation,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "additional_data": self.additional_data
        }


@dataclass
class RecoveryStrategy:
    """Recovery strategy configuration for different error types."""
    strategy_name: str
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    fallback_enabled: bool = True
    partial_results_save: bool = False
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt with exponential backoff."""
        if not self.exponential_backoff:
            return self.base_delay
        
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)


class ErrorRecoveryManager:
    """Manages error handling and recovery strategies for hybrid processing.
    
    Implements comprehensive error handling including GPU processing errors,
    memory allocation failures, model loading fallbacks, timeout handling,
    and critical error management with partial result saving.
    
    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
    """
    
    def __init__(self, config: Any):
        """Initialize error recovery manager with configuration.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.recovery_statistics = {
            "total_errors": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "fallback_activations": 0,
            "partial_saves": 0
        }
        
        # Recovery strategies
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # State management
        self.error_lock = threading.Lock()
        self.partial_results_dir = "partial_results"
        self.ensure_partial_results_dir()
        
        # Component references (will be set by hybrid processor)
        self.gpu_processor = None
        self.cpu_processor = None
        self.performance_monitor = None
        
        self.logger.info("Initialized ErrorRecoveryManager with comprehensive error handling")
    
    def _initialize_recovery_strategies(self) -> Dict[ErrorType, RecoveryStrategy]:
        """Initialize recovery strategies for different error types.
        
        Returns:
            Dictionary mapping error types to recovery strategies
        """
        strategies = {
            ErrorType.GPU_PROCESSING: RecoveryStrategy(
                strategy_name="GPU Processing Recovery",
                max_attempts=2,
                base_delay=0.5,
                max_delay=5.0,
                exponential_backoff=True,
                fallback_enabled=True,
                partial_results_save=True
            ),
            ErrorType.MEMORY_ALLOCATION: RecoveryStrategy(
                strategy_name="Memory Allocation Recovery",
                max_attempts=3,
                base_delay=1.0,
                max_delay=10.0,
                exponential_backoff=True,
                fallback_enabled=True,
                partial_results_save=True
            ),
            ErrorType.MODEL_LOADING: RecoveryStrategy(
                strategy_name="Model Loading Recovery",
                max_attempts=3,
                base_delay=2.0,
                max_delay=30.0,
                exponential_backoff=True,
                fallback_enabled=True,
                partial_results_save=False
            ),
            ErrorType.TIMEOUT: RecoveryStrategy(
                strategy_name="Timeout Recovery",
                max_attempts=3,
                base_delay=5.0,
                max_delay=60.0,
                exponential_backoff=True,
                fallback_enabled=True,
                partial_results_save=True
            ),
            ErrorType.CRITICAL_SYSTEM: RecoveryStrategy(
                strategy_name="Critical System Recovery",
                max_attempts=1,
                base_delay=10.0,
                max_delay=10.0,
                exponential_backoff=False,
                fallback_enabled=True,
                partial_results_save=True
            )
        }
        
        return strategies
    
    def ensure_partial_results_dir(self) -> None:
        """Ensure partial results directory exists."""
        try:
            os.makedirs(self.partial_results_dir, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Failed to create partial results directory: {e}")
    
    def handle_gpu_processing_error(self, 
                                  addresses: List[str], 
                                  error: Exception, 
                                  component: str = "GPU") -> List[ParsedAddress]:
        """Handle GPU processing errors with automatic CPU fallback.
        
        Implements automatic fallback to CPU processing when GPU processing fails,
        with retry mechanisms and error recovery strategies.
        
        Args:
            addresses: List of addresses that failed GPU processing
            error: The exception that occurred during GPU processing
            component: Component name for logging
            
        Returns:
            List of parsed addresses from fallback processing
            
        Requirements: 7.1
        """
        error_context = ErrorContext(
            error_type=ErrorType.GPU_PROCESSING,
            severity=ErrorSeverity.HIGH,
            error_message=f"GPU processing failed: {str(error)}",
            exception=error,
            component=component,
            operation="gpu_processing"
        )
        
        self._log_error(error_context)
        
        try:
            # Attempt GPU recovery first
            recovery_result = self._attempt_gpu_recovery(addresses, error_context)
            if recovery_result is not None:
                self.logger.info(f"✅ GPU recovery successful for {len(recovery_result)} addresses")
                error_context.recovery_successful = True
                self._update_recovery_statistics(True)
                return recovery_result
            
            # GPU recovery failed, fallback to CPU processing
            self.logger.warning(f"GPU recovery failed, falling back to CPU for {len(addresses)} addresses")
            
            if self.cpu_processor and hasattr(self.cpu_processor, 'handle_gpu_fallback'):
                fallback_results = self.cpu_processor.handle_gpu_fallback(addresses)
                
                # Save partial results if enabled
                if self.recovery_strategies[ErrorType.GPU_PROCESSING].partial_results_save:
                    self._save_partial_results(fallback_results, "gpu_fallback", error_context)
                
                self._update_recovery_statistics(True, fallback=True)
                self.logger.info(f"✅ CPU fallback completed for {len(fallback_results)} addresses")
                return fallback_results
            else:
                self.logger.error("CPU processor not available for fallback")
                self._update_recovery_statistics(False)
                return self._create_failed_results(addresses, "CPU fallback not available")
        
        except Exception as fallback_error:
            self.logger.error(f"Fallback processing also failed: {fallback_error}")
            self._update_recovery_statistics(False)
            return self._create_failed_results(addresses, f"Both GPU and CPU processing failed: {str(fallback_error)}")
    
    def handle_memory_allocation_error(self, 
                                     addresses: List[str], 
                                     error: Exception, 
                                     current_batch_size: int,
                                     component: str = "GPU") -> Tuple[List[ParsedAddress], int]:
        """Handle memory allocation failures with batch size reduction and retry.
        
        Implements automatic batch size reduction and retry mechanisms when
        memory allocation fails, with exponential backoff and fallback strategies.
        
        Args:
            addresses: List of addresses to process
            error: Memory allocation error
            current_batch_size: Current batch size that caused the error
            component: Component name for logging
            
        Returns:
            Tuple of (processed results, new batch size)
            
        Requirements: 7.2
        """
        error_context = ErrorContext(
            error_type=ErrorType.MEMORY_ALLOCATION,
            severity=ErrorSeverity.HIGH,
            error_message=f"Memory allocation failed: {str(error)}",
            exception=error,
            component=component,
            operation="memory_allocation",
            additional_data={"original_batch_size": current_batch_size}
        )
        
        self._log_error(error_context)
        
        strategy = self.recovery_strategies[ErrorType.MEMORY_ALLOCATION]
        results = []
        new_batch_size = current_batch_size
        
        for attempt in range(strategy.max_attempts):
            try:
                # Reduce batch size progressively
                reduction_factor = 2 ** (attempt + 1)
                new_batch_size = max(1, current_batch_size // reduction_factor)
                
                self.logger.info(f"Memory recovery attempt {attempt + 1}: "
                                f"reducing batch size from {current_batch_size} to {new_batch_size}")
                
                # Clear GPU memory if available
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    self.logger.info("Cleared GPU memory cache")
                
                # Wait with exponential backoff
                delay = strategy.calculate_delay(attempt)
                if delay > 0:
                    time.sleep(delay)
                
                # Attempt processing with reduced batch size
                if self.gpu_processor and hasattr(self.gpu_processor, 'process_with_dataset_batching'):
                    # Process in smaller batches
                    batch_results = []
                    for i in range(0, len(addresses), new_batch_size):
                        batch = addresses[i:i + new_batch_size]
                        try:
                            batch_result = self.gpu_processor.process_with_dataset_batching(batch)
                            batch_results.extend(batch_result)
                        except Exception as batch_error:
                            self.logger.warning(f"Batch processing failed even with reduced size: {batch_error}")
                            # Continue with remaining batches
                            batch_results.extend(self._create_failed_results(batch, str(batch_error)))
                    
                    results = batch_results
                    error_context.recovery_successful = True
                    self._update_recovery_statistics(True)
                    
                    self.logger.info(f"✅ Memory allocation recovery successful with batch size {new_batch_size}")
                    return results, new_batch_size
                
            except Exception as retry_error:
                self.logger.warning(f"Memory recovery attempt {attempt + 1} failed: {retry_error}")
                error_context.retry_count = attempt + 1
                
                # If this is the last attempt, try CPU fallback
                if attempt == strategy.max_attempts - 1:
                    self.logger.warning("All memory recovery attempts failed, trying CPU fallback")
                    
                    if self.cpu_processor and hasattr(self.cpu_processor, 'process_cpu_overflow'):
                        try:
                            fallback_results = self.cpu_processor.process_cpu_overflow(addresses)
                            self._update_recovery_statistics(True, fallback=True)
                            return fallback_results, new_batch_size
                        except Exception as cpu_error:
                            self.logger.error(f"CPU fallback also failed: {cpu_error}")
        
        # All recovery attempts failed
        self.logger.error(f"Memory allocation recovery failed after {strategy.max_attempts} attempts")
        self._update_recovery_statistics(False)
        
        # Save partial results if any were processed
        if results:
            self._save_partial_results(results, "memory_recovery", error_context)
        
        return self._create_failed_results(addresses, "Memory allocation recovery failed"), new_batch_size
    
    def handle_model_loading_error(self, 
                                 error: Exception, 
                                 model_name: str,
                                 component: str = "GPU") -> bool:
        """Handle model loading failures with fallback strategies and different precision levels.
        
        Implements progressive fallback through different precision levels and
        alternative model loading strategies when model loading fails.
        
        Args:
            error: Model loading error
            model_name: Name of the model that failed to load
            component: Component name for logging
            
        Returns:
            True if recovery successful, False otherwise
            
        Requirements: 7.3
        """
        error_context = ErrorContext(
            error_type=ErrorType.MODEL_LOADING,
            severity=ErrorSeverity.CRITICAL,
            error_message=f"Model loading failed: {str(error)}",
            exception=error,
            component=component,
            operation="model_loading",
            additional_data={"model_name": model_name}
        )
        
        self._log_error(error_context)
        
        strategy = self.recovery_strategies[ErrorType.MODEL_LOADING]
        
        # Progressive fallback strategies
        fallback_strategies = [
            {"precision": "float16", "device_map": "auto", "trust_remote_code": True},
            {"precision": "float32", "device_map": "auto", "trust_remote_code": True},
            {"precision": "float32", "device_map": None, "trust_remote_code": True},
            {"precision": "float32", "device_map": None, "trust_remote_code": False}
        ]
        
        for attempt, fallback_config in enumerate(fallback_strategies):
            try:
                self.logger.info(f"Model loading recovery attempt {attempt + 1}: "
                                f"trying {fallback_config}")
                
                # Wait with exponential backoff
                delay = strategy.calculate_delay(attempt)
                if delay > 0:
                    time.sleep(delay)
                
                # Attempt model loading with fallback configuration
                success = self._attempt_model_loading_with_config(
                    model_name, fallback_config, component
                )
                
                if success:
                    error_context.recovery_successful = True
                    self._update_recovery_statistics(True)
                    self.logger.info(f"✅ Model loading recovery successful with config: {fallback_config}")
                    return True
                
            except Exception as retry_error:
                self.logger.warning(f"Model loading attempt {attempt + 1} failed: {retry_error}")
                error_context.retry_count = attempt + 1
        
        # All model loading attempts failed
        self.logger.error(f"Model loading recovery failed after {len(fallback_strategies)} attempts")
        self._update_recovery_statistics(False)
        return False
    
    def handle_timeout_error(self, 
                           operation: Callable, 
                           args: Tuple, 
                           kwargs: Dict[str, Any],
                           timeout_seconds: float,
                           error: Exception,
                           component: str = "Processing") -> Any:
        """Handle timeout errors with exponential backoff retry mechanisms.
        
        Implements timeout handling with exponential backoff and retry mechanisms
        for operations that exceed time limits.
        
        Args:
            operation: The operation that timed out
            args: Arguments for the operation
            kwargs: Keyword arguments for the operation
            timeout_seconds: Original timeout that was exceeded
            error: Timeout error
            component: Component name for logging
            
        Returns:
            Result of successful operation or None if all retries failed
            
        Requirements: 7.4
        """
        error_context = ErrorContext(
            error_type=ErrorType.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            error_message=f"Operation timeout: {str(error)}",
            exception=error,
            component=component,
            operation=operation.__name__ if hasattr(operation, '__name__') else str(operation),
            additional_data={"timeout_seconds": timeout_seconds}
        )
        
        self._log_error(error_context)
        
        strategy = self.recovery_strategies[ErrorType.TIMEOUT]
        
        for attempt in range(strategy.max_attempts):
            try:
                # Increase timeout with each attempt
                new_timeout = timeout_seconds * (2 ** attempt)
                new_timeout = min(new_timeout, 300)  # Max 5 minutes
                
                self.logger.info(f"Timeout recovery attempt {attempt + 1}: "
                                f"increasing timeout to {new_timeout}s")
                
                # Wait with exponential backoff
                delay = strategy.calculate_delay(attempt)
                if delay > 0:
                    time.sleep(delay)
                
                # Attempt operation with increased timeout
                result = self._execute_with_timeout(operation, args, kwargs, new_timeout)
                
                if result is not None:
                    error_context.recovery_successful = True
                    self._update_recovery_statistics(True)
                    self.logger.info(f"✅ Timeout recovery successful with {new_timeout}s timeout")
                    return result
                
            except Exception as retry_error:
                self.logger.warning(f"Timeout recovery attempt {attempt + 1} failed: {retry_error}")
                error_context.retry_count = attempt + 1
        
        # All timeout recovery attempts failed
        self.logger.error(f"Timeout recovery failed after {strategy.max_attempts} attempts")
        self._update_recovery_statistics(False)
        return None
    
    def handle_critical_error(self, 
                            addresses: List[str],
                            partial_results: List[ParsedAddress],
                            error: Exception,
                            component: str = "System") -> List[ParsedAddress]:
        """Handle critical errors with partial result saving and detailed diagnostics.
        
        Implements critical error handling that saves partial results and provides
        detailed error diagnostics for system failures.
        
        Args:
            addresses: Original addresses being processed
            partial_results: Any results processed before the error
            error: Critical error that occurred
            component: Component name for logging
            
        Returns:
            Combined results including partial results and failed entries
            
        Requirements: 7.5
        """
        error_context = ErrorContext(
            error_type=ErrorType.CRITICAL_SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            error_message=f"Critical system error: {str(error)}",
            exception=error,
            component=component,
            operation="critical_error_handling",
            additional_data={
                "total_addresses": len(addresses),
                "partial_results_count": len(partial_results),
                "traceback": traceback.format_exc()
            }
        )
        
        self._log_error(error_context)
        
        try:
            # Save partial results immediately
            saved_file = self._save_partial_results(partial_results, "critical_error", error_context)
            
            # Generate detailed diagnostics
            diagnostics = self._generate_error_diagnostics(error_context)
            
            # Create comprehensive error report
            error_report = {
                "timestamp": datetime.now().isoformat(),
                "error_context": error_context.to_dict(),
                "diagnostics": diagnostics,
                "partial_results_file": saved_file,
                "system_state": self._capture_system_state()
            }
            
            # Save error report
            report_file = self._save_error_report(error_report, "critical_error")
            
            # Create results for remaining addresses
            processed_count = len(partial_results)
            remaining_addresses = addresses[processed_count:]
            
            failed_results = self._create_failed_results(
                remaining_addresses, 
                f"Critical error occurred. Partial results saved to {saved_file}. "
                f"Error report: {report_file}"
            )
            
            # Combine partial results with failed results
            all_results = partial_results + failed_results
            
            self._update_recovery_statistics(False, partial_save=True)
            
            self.logger.critical(f"Critical error handled: {len(partial_results)} results saved, "
                               f"{len(failed_results)} failed. Report: {report_file}")
            
            return all_results
            
        except Exception as handling_error:
            self.logger.critical(f"Failed to handle critical error: {handling_error}")
            # Emergency fallback - return what we can
            return partial_results + self._create_failed_results(
                addresses[len(partial_results):], 
                f"Critical error handling failed: {str(handling_error)}"
            )
    
    # Private helper methods
    
    def _attempt_gpu_recovery(self, addresses: List[str], error_context: ErrorContext) -> Optional[List[ParsedAddress]]:
        """Attempt GPU recovery strategies before falling back to CPU."""
        try:
            # Clear GPU memory and reset state
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Wait briefly for GPU to stabilize
            time.sleep(1.0)
            
            # Attempt processing with smaller batch size
            if self.gpu_processor and hasattr(self.gpu_processor, 'process_with_dataset_batching'):
                # Try with half the normal batch size
                reduced_batch_size = max(1, self.config.gpu_batch_size // 2)
                
                results = []
                for i in range(0, len(addresses), reduced_batch_size):
                    batch = addresses[i:i + reduced_batch_size]
                    try:
                        batch_results = self.gpu_processor.process_with_dataset_batching(batch)
                        results.extend(batch_results)
                    except Exception:
                        # If even small batches fail, give up on GPU recovery
                        return None
                
                return results
            
        except Exception as recovery_error:
            self.logger.debug(f"GPU recovery attempt failed: {recovery_error}")
        
        return None
    
    def _attempt_model_loading_with_config(self, model_name: str, config: Dict[str, Any], component: str) -> bool:
        """Attempt model loading with specific configuration."""
        try:
            if not TORCH_AVAILABLE:
                return False
            
            # This is a placeholder - actual implementation would depend on the specific component
            # For now, we'll simulate the attempt
            self.logger.info(f"Attempting model loading for {component} with config: {config}")
            
            # Simulate model loading attempt
            time.sleep(0.5)  # Simulate loading time
            
            # In a real implementation, this would actually attempt to load the model
            # with the specified configuration and return True/False based on success
            
            return True  # Placeholder success
            
        except Exception as e:
            self.logger.debug(f"Model loading attempt failed: {e}")
            return False
    
    def _execute_with_timeout(self, operation: Callable, args: Tuple, kwargs: Dict[str, Any], timeout: float) -> Any:
        """Execute operation with specified timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {timeout} seconds")
        
        try:
            # Set up timeout signal (Unix-like systems only)
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout))
            
            # Execute operation
            result = operation(*args, **kwargs)
            
            # Clear timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
            
        except Exception as e:
            # Clear timeout on exception
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            raise e
    
    def _save_partial_results(self, results: List[ParsedAddress], operation: str, error_context: ErrorContext) -> str:
        """Save partial results to disk for recovery."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"partial_results_{operation}_{timestamp}.pkl"
            filepath = os.path.join(self.partial_results_dir, filename)
            
            # Save results using pickle
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'results': results,
                    'error_context': error_context.to_dict(),
                    'timestamp': timestamp,
                    'operation': operation
                }, f)
            
            self.logger.info(f"Saved {len(results)} partial results to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save partial results: {e}")
            return ""
    
    def _save_error_report(self, report: Dict[str, Any], operation: str) -> str:
        """Save detailed error report to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"error_report_{operation}_{timestamp}.pkl"
            filepath = os.path.join(self.partial_results_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(report, f)
            
            self.logger.info(f"Saved error report to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save error report: {e}")
            return ""
    
    def _generate_error_diagnostics(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Generate detailed error diagnostics."""
        diagnostics = {
            "error_analysis": {
                "error_type": error_context.error_type.value,
                "severity": error_context.severity.value,
                "component": error_context.component,
                "operation": error_context.operation
            },
            "system_info": self._capture_system_state(),
            "recovery_history": [ctx.to_dict() for ctx in self.error_history[-10:]],  # Last 10 errors
            "recommendations": self._generate_recovery_recommendations(error_context)
        }
        
        return diagnostics
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for diagnostics."""
        state = {
            "timestamp": time.time(),
            "gpu_available": TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
            "error_statistics": self.recovery_statistics.copy()
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                state["gpu_memory_allocated"] = torch.cuda.memory_allocated()
                state["gpu_memory_reserved"] = torch.cuda.memory_reserved()
                state["gpu_device_count"] = torch.cuda.device_count()
            except Exception:
                pass
        
        return state
    
    def _generate_recovery_recommendations(self, error_context: ErrorContext) -> List[str]:
        """Generate recovery recommendations based on error context."""
        recommendations = []
        
        if error_context.error_type == ErrorType.GPU_PROCESSING:
            recommendations.extend([
                "Consider reducing GPU batch size",
                "Check GPU memory availability",
                "Verify CUDA drivers are up to date",
                "Enable CPU fallback processing"
            ])
        elif error_context.error_type == ErrorType.MEMORY_ALLOCATION:
            recommendations.extend([
                "Reduce batch size significantly",
                "Clear GPU memory cache",
                "Close other GPU-intensive applications",
                "Consider using CPU processing for large datasets"
            ])
        elif error_context.error_type == ErrorType.MODEL_LOADING:
            recommendations.extend([
                "Check model file integrity",
                "Verify sufficient disk space",
                "Try loading with different precision (float32 instead of float16)",
                "Check network connectivity for model downloads"
            ])
        elif error_context.error_type == ErrorType.TIMEOUT:
            recommendations.extend([
                "Increase timeout values",
                "Reduce batch sizes to speed up processing",
                "Check system load and available resources",
                "Consider processing in smaller chunks"
            ])
        
        return recommendations
    
    def _create_failed_results(self, addresses: List[str], error_message: str) -> List[ParsedAddress]:
        """Create failed ParsedAddress results for addresses."""
        return [
            ParsedAddress(
                parse_success=False,
                parse_error=error_message
            ) for _ in addresses
        ]
    
    def _log_error(self, error_context: ErrorContext) -> None:
        """Log error with appropriate severity level."""
        with self.error_lock:
            self.error_history.append(error_context)
            self.recovery_statistics["total_errors"] += 1
        
        log_message = (f"Error in {error_context.component}.{error_context.operation}: "
                      f"{error_context.error_message}")
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _update_recovery_statistics(self, success: bool, fallback: bool = False, partial_save: bool = False) -> None:
        """Update recovery statistics."""
        with self.error_lock:
            if success:
                self.recovery_statistics["successful_recoveries"] += 1
            else:
                self.recovery_statistics["failed_recoveries"] += 1
            
            if fallback:
                self.recovery_statistics["fallback_activations"] += 1
            
            if partial_save:
                self.recovery_statistics["partial_saves"] += 1
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error and recovery statistics."""
        with self.error_lock:
            return {
                "recovery_statistics": self.recovery_statistics.copy(),
                "recent_errors": [ctx.to_dict() for ctx in self.error_history[-5:]],
                "error_types_count": self._count_error_types(),
                "recovery_success_rate": (
                    self.recovery_statistics["successful_recoveries"] / 
                    max(1, self.recovery_statistics["total_errors"])
                ) * 100
            }
    
    def _count_error_types(self) -> Dict[str, int]:
        """Count occurrences of each error type."""
        counts = {}
        for error_ctx in self.error_history:
            error_type = error_ctx.error_type.value
            counts[error_type] = counts.get(error_type, 0) + 1
        return counts
    
    def set_component_references(self, gpu_processor=None, cpu_processor=None, performance_monitor=None):
        """Set references to other components for error handling."""
        self.gpu_processor = gpu_processor
        self.cpu_processor = cpu_processor
        self.performance_monitor = performance_monitor
        
        self.logger.info("Component references set for error recovery manager")
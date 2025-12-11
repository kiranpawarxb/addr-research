"""Configuration management for GPU-CPU Hybrid Processing System.

Extends the existing configuration system with hybrid processing specific settings
including GPU optimization parameters, CPU allocation ratios, and performance thresholds.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
try:
    from .config_loader import Config, ConfigLoader, ConfigurationError
    from .hybrid_processor import ProcessingConfiguration
except ImportError:
    # Fallback for direct execution
    from config_loader import Config, ConfigLoader, ConfigurationError
    from hybrid_processor import ProcessingConfiguration


@dataclass
class HybridProcessingConfig:
    """Extended configuration for hybrid processing system.
    
    Combines existing system configuration with hybrid processing specific settings.
    """
    # Base system configuration
    base_config: Config
    
    # Hybrid processing configuration
    processing_config: ProcessingConfiguration
    
    # Logging configuration for hybrid processing
    hybrid_log_level: str = "INFO"
    hybrid_log_file: Optional[str] = None
    enable_performance_logging: bool = True
    
    # Advanced GPU settings
    cuda_device_id: int = 0  # CUDA device to use
    enable_gpu_monitoring: bool = True  # Enable nvidia-smi monitoring
    gpu_monitoring_interval: int = 5  # GPU monitoring interval in seconds
    
    # System resource limits
    max_memory_usage_gb: float = 16.0  # Maximum system memory usage
    cpu_core_limit: Optional[int] = None  # Limit CPU cores (None = auto-detect)
    
    def __post_init__(self):
        """Validate hybrid configuration parameters."""
        # Validate CUDA device ID
        if self.cuda_device_id < 0:
            raise ValueError(f"cuda_device_id must be non-negative, got {self.cuda_device_id}")
        
        # Validate memory limits
        if self.max_memory_usage_gb <= 0:
            raise ValueError(f"max_memory_usage_gb must be positive, got {self.max_memory_usage_gb}")
        
        # Validate monitoring interval
        if self.gpu_monitoring_interval <= 0:
            raise ValueError(f"gpu_monitoring_interval must be positive, got {self.gpu_monitoring_interval}")


class HybridConfigLoader:
    """Configuration loader for hybrid processing system.
    
    Loads both base system configuration and hybrid processing specific settings
    from YAML files with validation and environment variable substitution.
    """
    
    def __init__(self, config_path: str, hybrid_config_path: Optional[str] = None):
        """Initialize hybrid configuration loader.
        
        Args:
            config_path: Path to base system configuration file
            hybrid_config_path: Optional path to hybrid-specific configuration
        """
        self.config_path = config_path
        self.hybrid_config_path = hybrid_config_path or self._get_default_hybrid_config_path()
        self.base_loader = ConfigLoader(config_path)
    
    def load(self) -> HybridProcessingConfig:
        """Load complete hybrid processing configuration.
        
        Returns:
            HybridProcessingConfig with validated settings
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Load base configuration
            base_config = self.base_loader.load()
            
            # Load hybrid-specific configuration
            hybrid_settings = self._load_hybrid_settings()
            
            # Create processing configuration
            processing_config = self._create_processing_config(hybrid_settings)
            
            # Create hybrid configuration
            return HybridProcessingConfig(
                base_config=base_config,
                processing_config=processing_config,
                hybrid_log_level=hybrid_settings.get('logging', {}).get('level', 'INFO'),
                hybrid_log_file=hybrid_settings.get('logging', {}).get('file_path'),
                enable_performance_logging=hybrid_settings.get('logging', {}).get('enable_performance', True),
                cuda_device_id=hybrid_settings.get('gpu', {}).get('device_id', 0),
                enable_gpu_monitoring=hybrid_settings.get('gpu', {}).get('enable_monitoring', True),
                gpu_monitoring_interval=hybrid_settings.get('gpu', {}).get('monitoring_interval', 5),
                max_memory_usage_gb=hybrid_settings.get('system', {}).get('max_memory_gb', 16.0),
                cpu_core_limit=hybrid_settings.get('system', {}).get('cpu_core_limit')
            )
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load hybrid configuration: {e}")
    
    def _get_default_hybrid_config_path(self) -> str:
        """Get default path for hybrid configuration file."""
        base_dir = os.path.dirname(self.config_path)
        return os.path.join(base_dir, "hybrid_config.yaml")
    
    def _load_hybrid_settings(self) -> Dict[str, Any]:
        """Load hybrid-specific settings from YAML file."""
        if not os.path.exists(self.hybrid_config_path):
            # Return default settings if hybrid config file doesn't exist
            return self._get_default_hybrid_settings()
        
        try:
            with open(self.hybrid_config_path, 'r') as f:
                settings = yaml.safe_load(f) or {}
                return self._substitute_env_vars(settings)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in hybrid config file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading hybrid config file: {e}")
    
    def _get_default_hybrid_settings(self) -> Dict[str, Any]:
        """Get default hybrid processing settings."""
        return {
            'gpu': {
                'batch_size': 400,
                'dataset_batch_size': 1000,
                'memory_fraction': 0.95,
                'queue_size': 10,
                'num_streams': 2,
                'enable_compilation': True,
                'use_half_precision': True,
                'enable_cudnn_benchmark': True,
                'enable_tensor_float32': True,
                'device_id': 0,
                'enable_monitoring': True,
                'monitoring_interval': 5
            },
            'cpu': {
                'allocation_ratio': 0.02,
                'batch_size': 50,
                'worker_count': 2
            },
            'performance': {
                'log_interval': 10,
                'target_throughput': 2000,
                'gpu_utilization_threshold': 0.90
            },
            'logging': {
                'level': 'INFO',
                'file_path': None,
                'enable_performance': True
            },
            'system': {
                'max_memory_gb': 16.0,
                'cpu_core_limit': None
            }
        }
    
    def _substitute_env_vars(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in settings (reuse from base loader)."""
        return self.base_loader._substitute_in_value(settings)
    
    def _create_processing_config(self, hybrid_settings: Dict[str, Any]) -> ProcessingConfiguration:
        """Create ProcessingConfiguration from hybrid settings."""
        gpu_settings = hybrid_settings.get('gpu', {})
        cpu_settings = hybrid_settings.get('cpu', {})
        perf_settings = hybrid_settings.get('performance', {})
        
        return ProcessingConfiguration(
            # GPU Configuration
            gpu_batch_size=gpu_settings.get('batch_size', 400),
            dataset_batch_size=gpu_settings.get('dataset_batch_size', 1000),
            gpu_memory_fraction=gpu_settings.get('memory_fraction', 0.95),
            gpu_queue_size=gpu_settings.get('queue_size', 10),
            num_gpu_streams=gpu_settings.get('num_streams', 2),
            
            # CPU Configuration
            cpu_allocation_ratio=cpu_settings.get('allocation_ratio', 0.02),
            cpu_batch_size=cpu_settings.get('batch_size', 50),
            cpu_worker_count=cpu_settings.get('worker_count', 2),
            
            # Performance Configuration
            performance_log_interval=perf_settings.get('log_interval', 10),
            target_throughput=perf_settings.get('target_throughput', 2000),
            gpu_utilization_threshold=perf_settings.get('gpu_utilization_threshold', 0.90),
            
            # Advanced GPU Optimizations
            enable_model_compilation=gpu_settings.get('enable_compilation', True),
            use_half_precision=gpu_settings.get('use_half_precision', True),
            enable_cudnn_benchmark=gpu_settings.get('enable_cudnn_benchmark', True),
            enable_tensor_float32=gpu_settings.get('enable_tensor_float32', True)
        )


def create_default_hybrid_config(output_path: str) -> None:
    """Create a default hybrid configuration file.
    
    Args:
        output_path: Path where to create the configuration file
    """
    default_config = {
        'gpu': {
            'batch_size': 400,
            'dataset_batch_size': 1000,
            'memory_fraction': 0.95,
            'queue_size': 10,
            'num_streams': 2,
            'enable_compilation': True,
            'use_half_precision': True,
            'enable_cudnn_benchmark': True,
            'enable_tensor_float32': True,
            'device_id': 0,
            'enable_monitoring': True,
            'monitoring_interval': 5
        },
        'cpu': {
            'allocation_ratio': 0.02,
            'batch_size': 50,
            'worker_count': 2
        },
        'performance': {
            'log_interval': 10,
            'target_throughput': 2000,
            'gpu_utilization_threshold': 0.90
        },
        'logging': {
            'level': 'INFO',
            'file_path': None,
            'enable_performance': True
        },
        'system': {
            'max_memory_gb': 16.0,
            'cpu_core_limit': None
        }
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)


def load_hybrid_config(
    config_path: str = "config/config.yaml",
    hybrid_config_path: Optional[str] = None
) -> HybridProcessingConfig:
    """Convenience function to load hybrid processing configuration.
    
    Args:
        config_path: Path to base system configuration file
        hybrid_config_path: Optional path to hybrid-specific configuration
        
    Returns:
        HybridProcessingConfig with validated settings
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    loader = HybridConfigLoader(config_path, hybrid_config_path)
    return loader.load()
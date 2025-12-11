"""Address Consolidation System - Main package."""

__version__ = "0.1.0"

# Import core components
from .models import AddressRecord, ParsedAddress, ConsolidatedGroup, ConsolidationStats

# Import hybrid processing components
from .hybrid_processor import (
    GPUCPUHybridProcessor,
    ProcessingConfiguration,
    PerformanceMetrics,
    ProcessingResult,
    BatchProcessingReport
)

# Import configuration management
from .config_loader import Config, ConfigLoader, load_config
from .hybrid_config import HybridProcessingConfig, HybridConfigLoader, load_hybrid_config

# Import logging utilities
from .hybrid_logging import setup_hybrid_logging, get_hybrid_logger

__all__ = [
    # Core models
    'AddressRecord',
    'ParsedAddress', 
    'ConsolidatedGroup',
    'ConsolidationStats',
    
    # Hybrid processing
    'GPUCPUHybridProcessor',
    'ProcessingConfiguration',
    'PerformanceMetrics',
    'ProcessingResult',
    'BatchProcessingReport',
    
    # Configuration
    'Config',
    'ConfigLoader',
    'load_config',
    'HybridProcessingConfig',
    'HybridConfigLoader',
    'load_hybrid_config',
    
    # Logging
    'setup_hybrid_logging',
    'get_hybrid_logger'
]

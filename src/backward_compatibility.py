"""Backward Compatibility Layer for GPU-CPU Hybrid Processing System.

This module provides backward compatibility with existing processing configurations,
API interfaces, and data formats. It ensures that existing code can work with
the new hybrid processing system with minimal changes.

Requirements: 9.1, 9.2
"""

import logging
import warnings
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

try:
    from .models import ParsedAddress, AddressRecord
    from .hybrid_processor import ProcessingConfiguration, ProcessingResult
    from .hybrid_integration_adapter import HybridIntegrationAdapter
    from .migration_utilities import ConfigurationMigrator, CompatibilityValidator
except ImportError:
    # Fallback for direct execution
    from models import ParsedAddress, AddressRecord
    from hybrid_processor import ProcessingConfiguration, ProcessingResult
    from hybrid_integration_adapter import HybridIntegrationAdapter
    from migration_utilities import ConfigurationMigrator, CompatibilityValidator


logger = logging.getLogger(__name__)


class LegacyProcessorAdapter:
    """Backward compatibility adapter for legacy processing interfaces.
    
    Provides a compatibility layer that mimics the interface of older processing
    systems while using the new GPU-CPU hybrid processing system underneath.
    This allows existing code to work without modification.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize legacy processor adapter.
        
        Args:
            config: Legacy configuration dictionary (will be migrated automatically)
        """
        self.logger = logging.getLogger(__name__)
        
        # Migrate legacy configuration
        if config is None:
            config = {}
        
        self.migrated_config = self._migrate_legacy_config(config)
        self.processing_config = ProcessingConfiguration(**self.migrated_config)
        
        # Initialize hybrid adapter
        self.hybrid_adapter = HybridIntegrationAdapter(self.processing_config)
        
        # Legacy state tracking
        self.is_setup = False
        self.processed_count = 0
        
        self.logger.info("Initialized LegacyProcessorAdapter with backward compatibility")
    
    def setup_parser(self, **kwargs) -> None:
        """Legacy setup method for backward compatibility.
        
        Mimics the interface of older parser setup methods while initializing
        the hybrid processing system underneath.
        
        Args:
            **kwargs: Legacy configuration parameters (ignored, uses migrated config)
        """
        if kwargs:
            self.logger.info(f"Legacy setup parameters detected: {list(kwargs.keys())}")
            warnings.warn(
                "Legacy setup parameters are deprecated. Use ProcessingConfiguration instead.",
                DeprecationWarning,
                stacklevel=2
            )
        
        try:
            self.hybrid_adapter.initialize()
            self.is_setup = True
            self.logger.info("✅ Legacy parser setup completed (using hybrid processing)")
        except Exception as e:
            self.logger.error(f"Legacy parser setup failed: {e}")
            raise
    
    def parse_addresses(self, addresses: List[str]) -> List[ParsedAddress]:
        """Legacy address parsing method for backward compatibility.
        
        Provides the same interface as older parsing methods while using
        the hybrid GPU-CPU processing system for actual processing.
        
        Args:
            addresses: List of address strings to parse
            
        Returns:
            List of ParsedAddress objects (same format as legacy system)
        """
        if not self.is_setup:
            self.logger.warning("Parser not setup, initializing automatically")
            self.setup_parser()
        
        if not addresses:
            return []
        
        try:
            # Process using hybrid system
            result = self.hybrid_adapter.process_address_list(addresses)
            
            # Update legacy counters
            self.processed_count += len(addresses)
            
            # Log legacy-style summary
            success_count = sum(1 for addr in result.parsed_addresses if addr.parse_success)
            self.logger.info(f"Legacy parsing completed: {success_count}/{len(addresses)} addresses parsed")
            
            return result.parsed_addresses
            
        except Exception as e:
            self.logger.error(f"Legacy address parsing failed: {e}")
            # Return error results for backward compatibility
            return [ParsedAddress(parse_success=False, parse_error=f"Processing error: {str(e)}") 
                   for _ in addresses]
    
    def process_csv(
        self,
        input_file: str,
        output_file: str,
        **kwargs
    ) -> int:
        """Legacy CSV processing method for backward compatibility.
        
        Provides the same interface as older CSV processing methods while using
        the hybrid processing system with comprehensive output generation.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            **kwargs: Legacy parameters (batch_size, etc.)
            
        Returns:
            Number of records processed (for backward compatibility)
        """
        if not self.is_setup:
            self.logger.warning("Parser not setup, initializing automatically")
            self.setup_parser()
        
        # Handle legacy parameters
        consolidate = kwargs.get('consolidate', True)
        comprehensive = kwargs.get('comprehensive_output', True)
        
        if kwargs:
            legacy_params = [k for k in kwargs.keys() if k not in ['consolidate', 'comprehensive_output']]
            if legacy_params:
                self.logger.info(f"Legacy CSV parameters detected: {legacy_params}")
                warnings.warn(
                    f"Legacy CSV parameters are deprecated: {legacy_params}",
                    DeprecationWarning,
                    stacklevel=2
                )
        
        try:
            # Process using hybrid integration
            result, consolidation_stats = self.hybrid_adapter.process_csv_file(
                input_file=input_file,
                output_file=output_file,
                consolidate_results=consolidate,
                comprehensive_output=comprehensive
            )
            
            # Update legacy counters
            records_processed = len(result.parsed_addresses)
            self.processed_count += records_processed
            
            # Log legacy-style summary
            success_count = sum(1 for addr in result.parsed_addresses if addr.parse_success)
            self.logger.info(f"Legacy CSV processing completed: {success_count}/{records_processed} records processed")
            
            if consolidation_stats:
                self.logger.info(f"Consolidation: {consolidation_stats.total_groups} groups from {consolidation_stats.total_records} records")
            
            return records_processed
            
        except Exception as e:
            self.logger.error(f"Legacy CSV processing failed: {e}")
            raise
    
    def cleanup_parser(self) -> None:
        """Legacy cleanup method for backward compatibility.
        
        Mimics the interface of older parser cleanup methods while properly
        shutting down the hybrid processing system.
        """
        try:
            if self.hybrid_adapter:
                self.hybrid_adapter.shutdown()
            
            self.is_setup = False
            self.logger.info(f"✅ Legacy parser cleanup completed (processed {self.processed_count} total addresses)")
            
        except Exception as e:
            self.logger.error(f"Legacy parser cleanup failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Legacy statistics method for backward compatibility.
        
        Returns:
            Dictionary with processing statistics in legacy format
        """
        return {
            'total_processed': self.processed_count,
            'is_setup': self.is_setup,
            'processor_type': 'hybrid_gpu_cpu',
            'config': self.migrated_config
        }
    
    def _migrate_legacy_config(self, legacy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy configuration to hybrid processing format.
        
        Args:
            legacy_config: Legacy configuration dictionary
            
        Returns:
            Migrated configuration dictionary
        """
        migrator = ConfigurationMigrator()
        
        # Start with defaults
        migrated = migrator.default_values.copy()
        
        # Apply legacy configuration mappings
        for old_key, value in legacy_config.items():
            new_key = migrator.config_mapping.get(old_key, old_key)
            if new_key in migrated:
                try:
                    # Convert to appropriate type
                    if new_key in ['gpu_batch_size', 'dataset_batch_size', 'gpu_queue_size', 
                                  'num_gpu_streams', 'cpu_batch_size', 'cpu_worker_count', 
                                  'target_throughput', 'performance_log_interval']:
                        migrated[new_key] = int(value)
                    elif new_key in ['gpu_memory_fraction', 'cpu_allocation_ratio', 'gpu_utilization_threshold']:
                        migrated[new_key] = float(value)
                    elif new_key in ['enable_model_compilation', 'use_half_precision', 
                                   'enable_cudnn_benchmark', 'enable_tensor_float32']:
                        if isinstance(value, str):
                            migrated[new_key] = value.lower() in ('true', '1', 'yes', 'on')
                        else:
                            migrated[new_key] = bool(value)
                    else:
                        migrated[new_key] = value
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Failed to migrate legacy config {old_key}={value}: {e}")
        
        return migrated


class LegacyConfigurationLoader:
    """Loads and converts legacy configuration formats.
    
    Provides automatic detection and conversion of various legacy configuration
    formats to the new ProcessingConfiguration format.
    """
    
    def __init__(self):
        """Initialize legacy configuration loader."""
        self.logger = logging.getLogger(__name__)
        self.migrator = ConfigurationMigrator()
    
    def load_legacy_config(
        self,
        config_source: Union[str, Dict[str, Any], None]
    ) -> ProcessingConfiguration:
        """Load configuration from various legacy sources.
        
        Args:
            config_source: Configuration file path, dictionary, or None for defaults
            
        Returns:
            ProcessingConfiguration object
        """
        if config_source is None:
            # Use defaults
            config_dict = self.migrator.default_values.copy()
        elif isinstance(config_source, str):
            # Load from file
            if Path(config_source).exists():
                legacy_config = self.migrator._read_config_file(config_source)
                config_dict = self.migrator._migrate_config_dict(legacy_config)
            else:
                self.logger.warning(f"Configuration file not found: {config_source}, using defaults")
                config_dict = self.migrator.default_values.copy()
        elif isinstance(config_source, dict):
            # Migrate dictionary
            config_dict = self.migrator._migrate_config_dict(config_source)
        else:
            raise ValueError(f"Unsupported configuration source type: {type(config_source)}")
        
        # Create and validate configuration
        try:
            config = ProcessingConfiguration(**config_dict)
            self.logger.info("✅ Legacy configuration loaded and validated successfully")
            return config
        except Exception as e:
            self.logger.error(f"Failed to create configuration from legacy source: {e}")
            raise


class BackwardCompatibilityManager:
    """Manages backward compatibility for the entire hybrid processing system.
    
    Provides centralized management of backward compatibility features,
    deprecation warnings, and migration assistance.
    """
    
    def __init__(self):
        """Initialize backward compatibility manager."""
        self.logger = logging.getLogger(__name__)
        self.compatibility_warnings = []
        self.migration_suggestions = []
    
    def check_compatibility(
        self,
        legacy_code_path: Optional[str] = None,
        legacy_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Check compatibility of legacy code and configuration.
        
        Args:
            legacy_code_path: Path to legacy Python script (optional)
            legacy_config: Legacy configuration dictionary (optional)
            
        Returns:
            Dictionary with compatibility analysis results
        """
        self.logger.info("Performing backward compatibility analysis")
        
        compatibility_report = {
            'overall_compatibility': True,
            'warnings': [],
            'migration_required': [],
            'suggestions': [],
            'validation_results': {}
        }
        
        # Check configuration compatibility
        if legacy_config:
            try:
                loader = LegacyConfigurationLoader()
                migrated_config = loader.load_legacy_config(legacy_config)
                
                validator = CompatibilityValidator()
                is_valid, messages = validator.validate_configuration(migrated_config)
                
                compatibility_report['validation_results']['config'] = {
                    'valid': is_valid,
                    'messages': messages
                }
                
                if not is_valid:
                    compatibility_report['overall_compatibility'] = False
                    compatibility_report['migration_required'].append('Configuration validation failed')
                
            except Exception as e:
                compatibility_report['overall_compatibility'] = False
                compatibility_report['warnings'].append(f"Configuration compatibility check failed: {e}")
        
        # Check code compatibility
        if legacy_code_path and Path(legacy_code_path).exists():
            try:
                with open(legacy_code_path, 'r') as f:
                    code_content = f.read()
                
                # Check for deprecated patterns
                deprecated_patterns = [
                    ('LLMParser', 'Use HybridIntegrationAdapter instead'),
                    ('parse_addresses', 'Use process_address_list method'),
                    ('setup_parser', 'Use initialize method'),
                    ('cleanup_parser', 'Use shutdown method'),
                    ('batch_size=', 'Use gpu_batch_size in ProcessingConfiguration'),
                ]
                
                for pattern, suggestion in deprecated_patterns:
                    if pattern in code_content:
                        compatibility_report['warnings'].append(f"Deprecated pattern found: {pattern}")
                        compatibility_report['suggestions'].append(suggestion)
                
                compatibility_report['validation_results']['code'] = {
                    'deprecated_patterns_found': len([p for p, _ in deprecated_patterns if p in code_content]),
                    'migration_recommended': any(p in code_content for p, _ in deprecated_patterns)
                }
                
            except Exception as e:
                compatibility_report['warnings'].append(f"Code compatibility check failed: {e}")
        
        # Generate migration suggestions
        if compatibility_report['warnings'] or compatibility_report['migration_required']:
            compatibility_report['suggestions'].extend([
                "Consider using migration_utilities.migrate_existing_script() for automatic code migration",
                "Use migration_utilities.migrate_existing_configuration() for configuration migration",
                "Review the hybrid processing documentation for new features and optimizations"
            ])
        
        return compatibility_report
    
    def create_legacy_adapter(
        self,
        legacy_config: Optional[Dict[str, Any]] = None
    ) -> LegacyProcessorAdapter:
        """Create a legacy processor adapter for backward compatibility.
        
        Args:
            legacy_config: Legacy configuration dictionary
            
        Returns:
            LegacyProcessorAdapter instance
        """
        self.logger.info("Creating legacy processor adapter for backward compatibility")
        
        adapter = LegacyProcessorAdapter(legacy_config)
        
        # Issue deprecation warning
        warnings.warn(
            "LegacyProcessorAdapter is deprecated. "
            "Consider migrating to HybridIntegrationAdapter for full feature support.",
            DeprecationWarning,
            stacklevel=2
        )
        
        return adapter
    
    def get_migration_guide(self) -> str:
        """Get a comprehensive migration guide for upgrading to hybrid processing.
        
        Returns:
            String with detailed migration instructions
        """
        return """
GPU-CPU Hybrid Processing Migration Guide
========================================

1. Configuration Migration:
   - Use migration_utilities.migrate_existing_configuration() to convert old config files
   - Update environment variables to new naming conventions
   - Review ProcessingConfiguration parameters for new optimization options

2. Code Migration:
   - Replace LLMParser with HybridIntegrationAdapter
   - Update method calls: parse_addresses() -> process_address_list()
   - Add proper initialization: adapter.initialize()
   - Add proper cleanup: adapter.shutdown()

3. Performance Optimization:
   - Configure GPU batch sizes (400-800 recommended)
   - Set target throughput (1500-2500 addresses/second)
   - Enable GPU optimizations (model compilation, half-precision)
   - Monitor GPU utilization (target 90%+)

4. Output Format:
   - Enable comprehensive_output for detailed metadata
   - Use consolidation for backward compatibility
   - Review new performance metrics and optimization suggestions

5. Testing:
   - Use CompatibilityValidator to verify migrated configuration
   - Test with small datasets before full migration
   - Monitor performance improvements and GPU utilization

For detailed examples, see the examples/ directory and documentation.
        """.strip()


# Convenience functions for backward compatibility

def create_legacy_processor(config: Optional[Dict[str, Any]] = None) -> LegacyProcessorAdapter:
    """Create a legacy processor adapter for backward compatibility.
    
    Convenience function for creating a legacy processor that works with
    existing code while using the new hybrid processing system underneath.
    
    Args:
        config: Legacy configuration dictionary
        
    Returns:
        LegacyProcessorAdapter instance
    """
    return LegacyProcessorAdapter(config)


def load_legacy_configuration(
    config_source: Union[str, Dict[str, Any], None]
) -> ProcessingConfiguration:
    """Load configuration from legacy sources.
    
    Convenience function for loading and migrating legacy configuration
    to the new ProcessingConfiguration format.
    
    Args:
        config_source: Configuration file path, dictionary, or None for defaults
        
    Returns:
        ProcessingConfiguration object
    """
    loader = LegacyConfigurationLoader()
    return loader.load_legacy_config(config_source)


def check_backward_compatibility(
    legacy_code_path: Optional[str] = None,
    legacy_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Check backward compatibility of legacy code and configuration.
    
    Convenience function for performing comprehensive backward compatibility
    analysis and generating migration recommendations.
    
    Args:
        legacy_code_path: Path to legacy Python script (optional)
        legacy_config: Legacy configuration dictionary (optional)
        
    Returns:
        Dictionary with compatibility analysis results
    """
    manager = BackwardCompatibilityManager()
    return manager.check_compatibility(legacy_code_path, legacy_config)
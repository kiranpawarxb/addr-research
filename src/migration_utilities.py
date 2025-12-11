"""Migration Utilities for GPU-CPU Hybrid Processing System.

This module provides utilities for migrating existing processing scripts to use
the new GPU-CPU hybrid processing system. It includes configuration migration,
script conversion helpers, and compatibility validation tools.

Requirements: 9.1, 9.2
"""

import logging
import os
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import re
import ast

try:
    from .hybrid_processor import ProcessingConfiguration
    from .hybrid_integration_adapter import HybridIntegrationAdapter
except ImportError:
    # Fallback for direct execution
    from hybrid_processor import ProcessingConfiguration
    from hybrid_integration_adapter import HybridIntegrationAdapter


logger = logging.getLogger(__name__)


class ConfigurationMigrator:
    """Migrates existing processing configurations to hybrid processing format.
    
    Provides automatic migration of configuration files, environment variables,
    and script parameters to the new GPU-CPU hybrid processing configuration.
    """
    
    def __init__(self):
        """Initialize the configuration migrator."""
        self.logger = logging.getLogger(__name__)
        
        # Mapping of old configuration keys to new ProcessingConfiguration fields
        self.config_mapping = {
            # GPU Configuration
            'batch_size': 'gpu_batch_size',
            'gpu_batch_size': 'gpu_batch_size',
            'dataset_batch_size': 'dataset_batch_size',
            'gpu_memory': 'gpu_memory_fraction',
            'gpu_memory_fraction': 'gpu_memory_fraction',
            'queue_size': 'gpu_queue_size',
            'gpu_queue_size': 'gpu_queue_size',
            'num_streams': 'num_gpu_streams',
            'gpu_streams': 'num_gpu_streams',
            
            # CPU Configuration
            'cpu_ratio': 'cpu_allocation_ratio',
            'cpu_allocation': 'cpu_allocation_ratio',
            'cpu_batch_size': 'cpu_batch_size',
            'cpu_workers': 'cpu_worker_count',
            
            # Performance Configuration
            'target_rate': 'target_throughput',
            'throughput': 'target_throughput',
            'target_throughput': 'target_throughput',
            'gpu_threshold': 'gpu_utilization_threshold',
            'utilization_threshold': 'gpu_utilization_threshold',
            'log_interval': 'performance_log_interval',
            
            # Advanced Options
            'compile_model': 'enable_model_compilation',
            'half_precision': 'use_half_precision',
            'cudnn_benchmark': 'enable_cudnn_benchmark',
            'tensor_float32': 'enable_tensor_float32'
        }
        
        # Default values for missing configurations
        self.default_values = {
            'gpu_batch_size': 400,
            'dataset_batch_size': 1000,
            'gpu_memory_fraction': 0.95,
            'gpu_queue_size': 10,
            'num_gpu_streams': 2,
            'cpu_allocation_ratio': 0.02,
            'cpu_batch_size': 50,
            'cpu_worker_count': 2,
            'target_throughput': 2000,
            'gpu_utilization_threshold': 0.90,
            'performance_log_interval': 10,
            'enable_model_compilation': True,
            'use_half_precision': True,
            'enable_cudnn_benchmark': True,
            'enable_tensor_float32': True
        }
    
    def migrate_config_file(self, old_config_path: str, new_config_path: str) -> ProcessingConfiguration:
        """Migrate an existing configuration file to hybrid processing format.
        
        Args:
            old_config_path: Path to existing configuration file
            new_config_path: Path for new hybrid configuration file
            
        Returns:
            ProcessingConfiguration object with migrated settings
        """
        self.logger.info(f"Migrating configuration from {old_config_path} to {new_config_path}")
        
        # Read existing configuration
        old_config = self._read_config_file(old_config_path)
        
        # Migrate configuration
        new_config_dict = self._migrate_config_dict(old_config)
        
        # Create ProcessingConfiguration
        config = ProcessingConfiguration(**new_config_dict)
        
        # Write new configuration file
        self._write_config_file(new_config_dict, new_config_path)
        
        self.logger.info(f"✅ Configuration migrated successfully to {new_config_path}")
        return config
    
    def migrate_environment_variables(self) -> Dict[str, Any]:
        """Migrate environment variables to hybrid processing configuration.
        
        Returns:
            Dictionary with migrated configuration values
        """
        self.logger.info("Migrating environment variables to hybrid configuration")
        
        migrated_config = {}
        
        # Check for common environment variable patterns
        env_patterns = {
            'BATCH_SIZE': 'gpu_batch_size',
            'GPU_BATCH_SIZE': 'gpu_batch_size',
            'GPU_MEMORY': 'gpu_memory_fraction',
            'CPU_RATIO': 'cpu_allocation_ratio',
            'TARGET_THROUGHPUT': 'target_throughput',
            'QUEUE_SIZE': 'gpu_queue_size'
        }
        
        for env_var, config_key in env_patterns.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Convert to appropriate type
                    if config_key in ['gpu_batch_size', 'dataset_batch_size', 'gpu_queue_size', 
                                    'num_gpu_streams', 'cpu_batch_size', 'cpu_worker_count', 
                                    'target_throughput', 'performance_log_interval']:
                        migrated_config[config_key] = int(value)
                    elif config_key in ['gpu_memory_fraction', 'cpu_allocation_ratio', 'gpu_utilization_threshold']:
                        migrated_config[config_key] = float(value)
                    elif config_key in ['enable_model_compilation', 'use_half_precision', 
                                      'enable_cudnn_benchmark', 'enable_tensor_float32']:
                        migrated_config[config_key] = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        migrated_config[config_key] = value
                    
                    self.logger.info(f"Migrated {env_var}={value} -> {config_key}={migrated_config[config_key]}")
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Failed to convert environment variable {env_var}={value}: {e}")
        
        # Fill in defaults for missing values
        for key, default_value in self.default_values.items():
            if key not in migrated_config:
                migrated_config[key] = default_value
        
        return migrated_config
    
    def _read_config_file(self, config_path: str) -> Dict[str, Any]:
        """Read configuration from various file formats."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                return json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML configuration files")
        else:
            # Try to parse as key=value format
            config = {}
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            config[key.strip()] = value.strip()
            return config
    
    def _migrate_config_dict(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate configuration dictionary to new format."""
        new_config = {}
        
        # Map old keys to new keys
        for old_key, value in old_config.items():
            new_key = self.config_mapping.get(old_key, old_key)
            
            # Convert value to appropriate type
            try:
                if new_key in ['gpu_batch_size', 'dataset_batch_size', 'gpu_queue_size', 
                              'num_gpu_streams', 'cpu_batch_size', 'cpu_worker_count', 
                              'target_throughput', 'performance_log_interval']:
                    new_config[new_key] = int(value)
                elif new_key in ['gpu_memory_fraction', 'cpu_allocation_ratio', 'gpu_utilization_threshold']:
                    new_config[new_key] = float(value)
                elif new_key in ['enable_model_compilation', 'use_half_precision', 
                               'enable_cudnn_benchmark', 'enable_tensor_float32']:
                    if isinstance(value, str):
                        new_config[new_key] = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        new_config[new_key] = bool(value)
                else:
                    new_config[new_key] = value
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Failed to convert config value {old_key}={value}: {e}")
                continue
        
        # Fill in defaults for missing values
        for key, default_value in self.default_values.items():
            if key not in new_config:
                new_config[key] = default_value
        
        return new_config
    
    def _write_config_file(self, config_dict: Dict[str, Any], config_path: str) -> None:
        """Write configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML required for YAML configuration files")
        else:
            # Write as key=value format
            with open(config_path, 'w') as f:
                for key, value in config_dict.items():
                    f.write(f"{key}={value}\n")


class ScriptMigrator:
    """Migrates existing processing scripts to use hybrid processing system.
    
    Provides automated migration of Python scripts that use the old processing
    pipeline to the new GPU-CPU hybrid processing system.
    """
    
    def __init__(self):
        """Initialize the script migrator."""
        self.logger = logging.getLogger(__name__)
        
        # Import patterns to replace
        self.import_replacements = {
            'from src.llm_parser import': 'from src.hybrid_integration_adapter import',
            'from llm_parser import': 'from hybrid_integration_adapter import',
            'import llm_parser': 'import hybrid_integration_adapter',
            'LLMParser': 'HybridIntegrationAdapter',
            'llm_parser': 'hybrid_integration_adapter'
        }
        
        # Method call replacements
        self.method_replacements = {
            'parse_addresses': 'process_address_list',
            'process_csv': 'process_csv_file',
            'setup_parser': 'initialize',
            'cleanup_parser': 'shutdown'
        }
    
    def migrate_script(self, script_path: str, output_path: str, backup: bool = True) -> None:
        """Migrate a Python script to use hybrid processing.
        
        Args:
            script_path: Path to existing script
            output_path: Path for migrated script
            backup: Whether to create backup of original script
        """
        self.logger.info(f"Migrating script from {script_path} to {output_path}")
        
        # Create backup if requested
        if backup and script_path != output_path:
            backup_path = f"{script_path}.backup"
            shutil.copy2(script_path, backup_path)
            self.logger.info(f"Created backup: {backup_path}")
        
        # Read original script
        with open(script_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Migrate content
        migrated_content = self._migrate_script_content(original_content)
        
        # Write migrated script
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(migrated_content)
        
        self.logger.info(f"✅ Script migrated successfully to {output_path}")
    
    def _migrate_script_content(self, content: str) -> str:
        """Migrate script content to use hybrid processing."""
        migrated_content = content
        
        # Replace imports
        for old_import, new_import in self.import_replacements.items():
            migrated_content = migrated_content.replace(old_import, new_import)
        
        # Replace method calls
        for old_method, new_method in self.method_replacements.items():
            # Use regex to replace method calls more precisely
            pattern = rf'\b{re.escape(old_method)}\b'
            migrated_content = re.sub(pattern, new_method, migrated_content)
        
        # Add hybrid processing initialization template if needed
        if 'HybridIntegrationAdapter' in migrated_content and 'initialize()' not in migrated_content:
            init_template = """
# Initialize hybrid processing adapter
adapter = HybridIntegrationAdapter(config)
adapter.initialize()

try:
    # Your processing code here
    pass
finally:
    # Cleanup
    adapter.shutdown()
"""
            # Insert template after imports
            lines = migrated_content.split('\n')
            import_end = 0
            for i, line in enumerate(lines):
                if line.strip() and not (line.startswith('import ') or line.startswith('from ') or line.startswith('#')):
                    import_end = i
                    break
            
            lines.insert(import_end, init_template)
            migrated_content = '\n'.join(lines)
        
        return migrated_content


class CompatibilityValidator:
    """Validates compatibility between existing and hybrid processing systems.
    
    Provides validation tools to ensure that migrated scripts and configurations
    work correctly with the new hybrid processing system.
    """
    
    def __init__(self):
        """Initialize the compatibility validator."""
        self.logger = logging.getLogger(__name__)
    
    def validate_configuration(self, config: ProcessingConfiguration) -> Tuple[bool, List[str]]:
        """Validate hybrid processing configuration for compatibility.
        
        Args:
            config: ProcessingConfiguration to validate
            
        Returns:
            Tuple of (is_valid, validation_messages)
        """
        self.logger.info("Validating hybrid processing configuration")
        
        is_valid = True
        messages = []
        
        try:
            # Validate configuration by creating instance
            # This will trigger __post_init__ validation
            test_config = ProcessingConfiguration(
                gpu_batch_size=config.gpu_batch_size,
                dataset_batch_size=config.dataset_batch_size,
                gpu_memory_fraction=config.gpu_memory_fraction,
                gpu_queue_size=config.gpu_queue_size,
                num_gpu_streams=config.num_gpu_streams,
                cpu_allocation_ratio=config.cpu_allocation_ratio,
                cpu_batch_size=config.cpu_batch_size,
                cpu_worker_count=config.cpu_worker_count,
                target_throughput=config.target_throughput,
                gpu_utilization_threshold=config.gpu_utilization_threshold
            )
            
            messages.append("✅ Configuration validation passed")
            
        except ValueError as e:
            is_valid = False
            messages.append(f"❌ Configuration validation failed: {e}")
        
        # Additional compatibility checks
        if config.gpu_batch_size < 100:
            messages.append("⚠️ GPU batch size is very small, may impact performance")
        
        if config.cpu_allocation_ratio > 0.1:
            messages.append("⚠️ CPU allocation ratio is high, may reduce GPU efficiency")
        
        if config.target_throughput > 3000:
            messages.append("⚠️ Target throughput is very high, may not be achievable")
        
        return is_valid, messages
    
    def validate_integration(self, adapter: HybridIntegrationAdapter, test_addresses: List[str]) -> Tuple[bool, List[str]]:
        """Validate hybrid integration adapter with test data.
        
        Args:
            adapter: HybridIntegrationAdapter to test
            test_addresses: Test addresses for validation
            
        Returns:
            Tuple of (is_valid, validation_messages)
        """
        self.logger.info(f"Validating hybrid integration with {len(test_addresses)} test addresses")
        
        is_valid = True
        messages = []
        
        try:
            # Test initialization
            if not adapter.is_initialized:
                adapter.initialize()
            messages.append("✅ Adapter initialization successful")
            
            # Test processing
            if test_addresses:
                result = adapter.process_address_list(test_addresses[:5])  # Test with first 5 addresses
                
                if result.parsed_addresses:
                    success_count = sum(1 for addr in result.parsed_addresses if addr.parse_success)
                    messages.append(f"✅ Processing test successful: {success_count}/{len(result.parsed_addresses)} addresses parsed")
                else:
                    is_valid = False
                    messages.append("❌ Processing test failed: No results returned")
            else:
                messages.append("⚠️ No test addresses provided for processing validation")
            
        except Exception as e:
            is_valid = False
            messages.append(f"❌ Integration validation failed: {e}")
        
        return is_valid, messages


def migrate_existing_configuration(
    old_config_path: str,
    new_config_path: str = None
) -> ProcessingConfiguration:
    """Migrate existing configuration to hybrid processing format.
    
    Convenience function for migrating configuration files with automatic
    path generation and validation.
    
    Args:
        old_config_path: Path to existing configuration file
        new_config_path: Path for new configuration (auto-generated if None)
        
    Returns:
        ProcessingConfiguration object with migrated settings
    """
    if new_config_path is None:
        old_path = Path(old_config_path)
        new_config_path = old_path.parent / f"hybrid_{old_path.name}"
    
    migrator = ConfigurationMigrator()
    config = migrator.migrate_config_file(old_config_path, new_config_path)
    
    # Validate migrated configuration
    validator = CompatibilityValidator()
    is_valid, messages = validator.validate_configuration(config)
    
    for message in messages:
        logger.info(message)
    
    if not is_valid:
        raise ValueError("Configuration migration validation failed")
    
    return config


def migrate_existing_script(
    script_path: str,
    output_path: str = None,
    backup: bool = True
) -> None:
    """Migrate existing processing script to use hybrid processing.
    
    Convenience function for migrating Python scripts with automatic
    path generation and validation.
    
    Args:
        script_path: Path to existing script
        output_path: Path for migrated script (auto-generated if None)
        backup: Whether to create backup of original script
    """
    if output_path is None:
        script_path_obj = Path(script_path)
        output_path = script_path_obj.parent / f"hybrid_{script_path_obj.name}"
    
    migrator = ScriptMigrator()
    migrator.migrate_script(script_path, output_path, backup)
    
    logger.info(f"✅ Script migration completed: {output_path}")
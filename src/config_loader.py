"""Configuration loader for the Address Consolidation System.

This module provides functionality to load and validate configuration from YAML files,
with support for environment variable substitution and default values.
"""

import os
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import yaml


@dataclass
class InputConfig:
    """Configuration for input CSV file."""
    file_path: str
    required_columns: List[str] = field(default_factory=lambda: ["addr_text", "pincode", "city_id"])


@dataclass
class LLMConfig:
    """Configuration for LLM parser."""
    api_endpoint: str = "https://api.openai.com/v1/chat/completions"
    api_key: str = ""
    model: str = "gpt-4"
    max_retries: int = 3
    timeout_seconds: int = 30
    batch_size: int = 10


@dataclass
class ConsolidationConfig:
    """Configuration for address consolidation."""
    fuzzy_matching: bool = True
    similarity_threshold: float = 0.85
    normalize_society_names: bool = True


@dataclass
class OutputConfig:
    """Configuration for output files."""
    file_path: str = "consolidated_addresses.csv"
    include_statistics: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    file_path: str = "consolidation.log"


@dataclass
class Config:
    """Main configuration object."""
    input: InputConfig
    llm: LLMConfig
    consolidation: ConsolidationConfig
    output: OutputConfig
    logging: LoggingConfig


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class ConfigLoader:
    """Loads and validates configuration from YAML files."""
    
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')
    
    def __init__(self, config_path: str):
        """Initialize the configuration loader.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self._raw_config: Optional[Dict[str, Any]] = None
    
    def load(self) -> Config:
        """Load and validate configuration from file.
        
        Returns:
            Config object with validated configuration
            
        Raises:
            ConfigurationError: If configuration is invalid or file cannot be read
        """
        try:
            self._raw_config = self._load_yaml()
            self._substitute_env_vars()
            return self._build_config()
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML file and return parsed content."""
        with open(self.config_path, 'r') as f:
            content = yaml.safe_load(f)
            if content is None:
                raise ConfigurationError("Configuration file is empty")
            if not isinstance(content, dict):
                raise ConfigurationError("Configuration must be a YAML dictionary")
            return content
    
    def _substitute_env_vars(self) -> None:
        """Recursively substitute environment variables in configuration."""
        self._raw_config = self._substitute_in_value(self._raw_config)
    
    def _substitute_in_value(self, value: Any) -> Any:
        """Recursively substitute environment variables in a value.
        
        Args:
            value: The value to process (can be dict, list, str, or other)
            
        Returns:
            Value with environment variables substituted
        """
        if isinstance(value, dict):
            return {k: self._substitute_in_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._substitute_in_value(item) for item in value]
        elif isinstance(value, str):
            return self._substitute_env_var(value)
        else:
            return value
    
    def _substitute_env_var(self, value: str) -> str:
        """Substitute environment variables in a string.
        
        Supports ${VAR_NAME} syntax. If the environment variable is not set,
        the original placeholder is kept.
        
        Args:
            value: String that may contain environment variable placeholders
            
        Returns:
            String with environment variables substituted
        """
        def replace_match(match):
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                # Keep the original placeholder if env var not set
                return match.group(0)
            return env_value
        
        return self.ENV_VAR_PATTERN.sub(replace_match, value)
    
    def _build_config(self) -> Config:
        """Build Config object from raw configuration with validation."""
        errors = []
        
        # Build input config
        try:
            input_config = self._build_input_config()
        except Exception as e:
            errors.append(f"Input configuration error: {e}")
            input_config = InputConfig(file_path="")
        
        # Build LLM config
        try:
            llm_config = self._build_llm_config()
        except Exception as e:
            errors.append(f"LLM configuration error: {e}")
            llm_config = LLMConfig()
        
        # Build consolidation config
        try:
            consolidation_config = self._build_consolidation_config()
        except Exception as e:
            errors.append(f"Consolidation configuration error: {e}")
            consolidation_config = ConsolidationConfig()
        
        # Build output config
        try:
            output_config = self._build_output_config()
        except Exception as e:
            errors.append(f"Output configuration error: {e}")
            output_config = OutputConfig()
        
        # Build logging config
        try:
            logging_config = self._build_logging_config()
        except Exception as e:
            errors.append(f"Logging configuration error: {e}")
            logging_config = LoggingConfig()
        
        # Report errors if any critical fields are missing
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ConfigurationError(error_msg)
        
        return Config(
            input=input_config,
            llm=llm_config,
            consolidation=consolidation_config,
            output=output_config,
            logging=logging_config
        )
    
    def _build_input_config(self) -> InputConfig:
        """Build and validate input configuration."""
        input_dict = self._raw_config.get('input', {})
        
        file_path = input_dict.get('file_path')
        if not file_path:
            raise ValueError("input.file_path is required")
        
        required_columns = input_dict.get('required_columns', ["addr_text", "pincode", "city_id"])
        if not isinstance(required_columns, list):
            raise ValueError("input.required_columns must be a list")
        
        return InputConfig(
            file_path=str(file_path),
            required_columns=[str(col) for col in required_columns]
        )
    
    def _build_llm_config(self) -> LLMConfig:
        """Build and validate LLM configuration."""
        llm_dict = self._raw_config.get('llm', {})
        
        api_endpoint = llm_dict.get('api_endpoint', "https://api.openai.com/v1/chat/completions")
        api_key = llm_dict.get('api_key', "")
        model = llm_dict.get('model', "gpt-4")
        max_retries = llm_dict.get('max_retries', 3)
        timeout_seconds = llm_dict.get('timeout_seconds', 30)
        batch_size = llm_dict.get('batch_size', 10)
        
        # Validate types and ranges
        try:
            max_retries = int(max_retries)
            if max_retries < 0:
                raise ValueError("llm.max_retries must be non-negative")
        except (TypeError, ValueError) as e:
            raise ValueError(f"llm.max_retries must be a non-negative integer: {e}")
        
        try:
            timeout_seconds = int(timeout_seconds)
            if timeout_seconds <= 0:
                raise ValueError("llm.timeout_seconds must be positive")
        except (TypeError, ValueError) as e:
            raise ValueError(f"llm.timeout_seconds must be a positive integer: {e}")
        
        try:
            batch_size = int(batch_size)
            if batch_size <= 0:
                raise ValueError("llm.batch_size must be positive")
        except (TypeError, ValueError) as e:
            raise ValueError(f"llm.batch_size must be a positive integer: {e}")
        
        return LLMConfig(
            api_endpoint=str(api_endpoint),
            api_key=str(api_key),
            model=str(model),
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size
        )
    
    def _build_consolidation_config(self) -> ConsolidationConfig:
        """Build and validate consolidation configuration."""
        cons_dict = self._raw_config.get('consolidation', {})
        
        fuzzy_matching = cons_dict.get('fuzzy_matching', True)
        similarity_threshold = cons_dict.get('similarity_threshold', 0.85)
        normalize_society_names = cons_dict.get('normalize_society_names', True)
        
        # Validate types
        if not isinstance(fuzzy_matching, bool):
            raise ValueError("consolidation.fuzzy_matching must be a boolean")
        
        if not isinstance(normalize_society_names, bool):
            raise ValueError("consolidation.normalize_society_names must be a boolean")
        
        try:
            similarity_threshold = float(similarity_threshold)
            if not (0.0 <= similarity_threshold <= 1.0):
                raise ValueError("consolidation.similarity_threshold must be between 0.0 and 1.0")
        except (TypeError, ValueError) as e:
            raise ValueError(f"consolidation.similarity_threshold must be a float between 0.0 and 1.0: {e}")
        
        return ConsolidationConfig(
            fuzzy_matching=fuzzy_matching,
            similarity_threshold=similarity_threshold,
            normalize_society_names=normalize_society_names
        )
    
    def _build_output_config(self) -> OutputConfig:
        """Build and validate output configuration."""
        output_dict = self._raw_config.get('output', {})
        
        file_path = output_dict.get('file_path', "consolidated_addresses.csv")
        include_statistics = output_dict.get('include_statistics', True)
        
        if not isinstance(include_statistics, bool):
            raise ValueError("output.include_statistics must be a boolean")
        
        return OutputConfig(
            file_path=str(file_path),
            include_statistics=include_statistics
        )
    
    def _build_logging_config(self) -> LoggingConfig:
        """Build and validate logging configuration."""
        logging_dict = self._raw_config.get('logging', {})
        
        level = logging_dict.get('level', "INFO")
        file_path = logging_dict.get('file_path', "consolidation.log")
        
        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        level_upper = str(level).upper()
        if level_upper not in valid_levels:
            raise ValueError(f"logging.level must be one of {valid_levels}, got: {level}")
        
        return LoggingConfig(
            level=level_upper,
            file_path=str(file_path)
        )


def load_config(config_path: str) -> Config:
    """Convenience function to load configuration.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Config object with validated configuration
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    loader = ConfigLoader(config_path)
    return loader.load()

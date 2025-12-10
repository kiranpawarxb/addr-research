"""Tests for the configuration loader."""

import os
import tempfile
import pytest
from pathlib import Path

from src.config_loader import (
    ConfigLoader,
    Config,
    InputConfig,
    LLMConfig,
    ConsolidationConfig,
    OutputConfig,
    LoggingConfig,
    ConfigurationError,
    load_config
)


class TestConfigLoader:
    """Test suite for ConfigLoader class."""
    
    def test_load_valid_config(self, tmp_path):
        """Test loading a valid configuration file."""
        config_content = """
input:
  file_path: "test_input.csv"
  required_columns:
    - addr_text
    - pincode
    - city_id

llm:
  api_endpoint: "https://api.openai.com/v1/chat/completions"
  api_key: "test_key_123"
  model: "gpt-4"
  max_retries: 3
  timeout_seconds: 30
  batch_size: 10

consolidation:
  fuzzy_matching: true
  similarity_threshold: 0.85
  normalize_society_names: true

output:
  file_path: "test_output.csv"
  include_statistics: true

logging:
  level: "INFO"
  file_path: "test.log"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        config = loader.load()
        
        assert isinstance(config, Config)
        assert config.input.file_path == "test_input.csv"
        assert config.input.required_columns == ["addr_text", "pincode", "city_id"]
        assert config.llm.api_key == "test_key_123"
        assert config.llm.max_retries == 3
        assert config.consolidation.fuzzy_matching is True
        assert config.consolidation.similarity_threshold == 0.85
        assert config.output.file_path == "test_output.csv"
        assert config.logging.level == "INFO"
    
    def test_env_var_substitution(self, tmp_path):
        """Test environment variable substitution."""
        config_content = """
llm:
  api_key: "${TEST_API_KEY}"
  model: "gpt-4"

input:
  file_path: "test.csv"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        # Set environment variable
        os.environ['TEST_API_KEY'] = "secret_key_from_env"
        
        try:
            loader = ConfigLoader(str(config_file))
            config = loader.load()
            
            assert config.llm.api_key == "secret_key_from_env"
        finally:
            # Clean up
            del os.environ['TEST_API_KEY']
    
    def test_env_var_not_set_keeps_placeholder(self, tmp_path):
        """Test that missing environment variables keep the placeholder."""
        config_content = """
llm:
  api_key: "${NONEXISTENT_VAR}"
  model: "gpt-4"

input:
  file_path: "test.csv"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        config = loader.load()
        
        # Should keep the placeholder if env var not set
        assert config.llm.api_key == "${NONEXISTENT_VAR}"
    
    def test_nested_env_var_substitution(self, tmp_path):
        """Test environment variable substitution in nested structures."""
        config_content = """
input:
  file_path: "${INPUT_FILE}"
  required_columns:
    - addr_text

output:
  file_path: "${OUTPUT_FILE}"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        os.environ['INPUT_FILE'] = "input_from_env.csv"
        os.environ['OUTPUT_FILE'] = "output_from_env.csv"
        
        try:
            loader = ConfigLoader(str(config_file))
            config = loader.load()
            
            assert config.input.file_path == "input_from_env.csv"
            assert config.output.file_path == "output_from_env.csv"
        finally:
            del os.environ['INPUT_FILE']
            del os.environ['OUTPUT_FILE']
    
    def test_missing_config_file(self):
        """Test error handling for missing configuration file."""
        loader = ConfigLoader("nonexistent_config.yaml")
        
        with pytest.raises(ConfigurationError) as exc_info:
            loader.load()
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_invalid_yaml(self, tmp_path):
        """Test error handling for invalid YAML syntax."""
        config_content = """
input:
  file_path: "test.csv"
  invalid yaml syntax here: [unclosed bracket
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ConfigurationError) as exc_info:
            loader.load()
        
        assert "invalid yaml" in str(exc_info.value).lower()
    
    def test_empty_config_file(self, tmp_path):
        """Test error handling for empty configuration file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ConfigurationError) as exc_info:
            loader.load()
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_missing_required_input_file_path(self, tmp_path):
        """Test error handling for missing required input.file_path."""
        config_content = """
input:
  required_columns:
    - addr_text
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ConfigurationError) as exc_info:
            loader.load()
        
        assert "file_path" in str(exc_info.value).lower()
    
    def test_default_values(self, tmp_path):
        """Test that default values are used when optional fields are missing."""
        config_content = """
input:
  file_path: "test.csv"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        config = loader.load()
        
        # Check defaults
        assert config.input.required_columns == ["addr_text", "pincode", "city_id"]
        assert config.llm.api_endpoint == "https://api.openai.com/v1/chat/completions"
        assert config.llm.model == "gpt-4"
        assert config.llm.max_retries == 3
        assert config.consolidation.fuzzy_matching is True
        assert config.consolidation.similarity_threshold == 0.85
        assert config.output.file_path == "consolidated_addresses.csv"
        assert config.logging.level == "INFO"
    
    def test_invalid_max_retries(self, tmp_path):
        """Test validation of max_retries field."""
        config_content = """
input:
  file_path: "test.csv"

llm:
  max_retries: -1
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ConfigurationError) as exc_info:
            loader.load()
        
        assert "max_retries" in str(exc_info.value).lower()
    
    def test_invalid_timeout_seconds(self, tmp_path):
        """Test validation of timeout_seconds field."""
        config_content = """
input:
  file_path: "test.csv"

llm:
  timeout_seconds: 0
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ConfigurationError) as exc_info:
            loader.load()
        
        assert "timeout_seconds" in str(exc_info.value).lower()
    
    def test_invalid_batch_size(self, tmp_path):
        """Test validation of batch_size field."""
        config_content = """
input:
  file_path: "test.csv"

llm:
  batch_size: -5
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ConfigurationError) as exc_info:
            loader.load()
        
        assert "batch_size" in str(exc_info.value).lower()
    
    def test_invalid_similarity_threshold_too_high(self, tmp_path):
        """Test validation of similarity_threshold above 1.0."""
        config_content = """
input:
  file_path: "test.csv"

consolidation:
  similarity_threshold: 1.5
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ConfigurationError) as exc_info:
            loader.load()
        
        assert "similarity_threshold" in str(exc_info.value).lower()
    
    def test_invalid_similarity_threshold_negative(self, tmp_path):
        """Test validation of negative similarity_threshold."""
        config_content = """
input:
  file_path: "test.csv"

consolidation:
  similarity_threshold: -0.5
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ConfigurationError) as exc_info:
            loader.load()
        
        assert "similarity_threshold" in str(exc_info.value).lower()
    
    def test_invalid_log_level(self, tmp_path):
        """Test validation of invalid log level."""
        config_content = """
input:
  file_path: "test.csv"

logging:
  level: "INVALID_LEVEL"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ConfigurationError) as exc_info:
            loader.load()
        
        assert "level" in str(exc_info.value).lower()
    
    def test_log_level_case_insensitive(self, tmp_path):
        """Test that log level is case-insensitive."""
        config_content = """
input:
  file_path: "test.csv"

logging:
  level: "debug"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        config = loader.load()
        
        assert config.logging.level == "DEBUG"
    
    def test_invalid_fuzzy_matching_type(self, tmp_path):
        """Test validation of fuzzy_matching type."""
        config_content = """
input:
  file_path: "test.csv"

consolidation:
  fuzzy_matching: "not_a_boolean"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ConfigurationError) as exc_info:
            loader.load()
        
        assert "fuzzy_matching" in str(exc_info.value).lower()
    
    def test_invalid_required_columns_type(self, tmp_path):
        """Test validation of required_columns type."""
        config_content = """
input:
  file_path: "test.csv"
  required_columns: "not_a_list"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        
        with pytest.raises(ConfigurationError) as exc_info:
            loader.load()
        
        assert "required_columns" in str(exc_info.value).lower()
    
    def test_load_config_convenience_function(self, tmp_path):
        """Test the convenience load_config function."""
        config_content = """
input:
  file_path: "test.csv"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        config = load_config(str(config_file))
        
        assert isinstance(config, Config)
        assert config.input.file_path == "test.csv"
    
    def test_partial_config_sections(self, tmp_path):
        """Test that partial configuration sections use defaults."""
        config_content = """
input:
  file_path: "test.csv"

llm:
  api_key: "my_key"
  # Other LLM fields should use defaults

consolidation:
  fuzzy_matching: false
  # Other consolidation fields should use defaults
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        config = loader.load()
        
        # Check that defaults are used for missing fields
        assert config.llm.api_key == "my_key"
        assert config.llm.model == "gpt-4"  # default
        assert config.llm.max_retries == 3  # default
        assert config.consolidation.fuzzy_matching is False
        assert config.consolidation.similarity_threshold == 0.85  # default
    
    def test_type_coercion(self, tmp_path):
        """Test that values are coerced to correct types."""
        config_content = """
input:
  file_path: "test.csv"

llm:
  max_retries: "5"
  timeout_seconds: "60"
  batch_size: "20"

consolidation:
  similarity_threshold: "0.9"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        loader = ConfigLoader(str(config_file))
        config = loader.load()
        
        assert config.llm.max_retries == 5
        assert config.llm.timeout_seconds == 60
        assert config.llm.batch_size == 20
        assert config.consolidation.similarity_threshold == 0.9
    
    def test_multiple_env_vars_in_single_value(self, tmp_path):
        """Test multiple environment variable substitutions in a single value."""
        config_content = """
input:
  file_path: "${BASE_PATH}/${FILE_NAME}"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        os.environ['BASE_PATH'] = "/data/input"
        os.environ['FILE_NAME'] = "addresses.csv"
        
        try:
            loader = ConfigLoader(str(config_file))
            config = loader.load()
            
            assert config.input.file_path == "/data/input/addresses.csv"
        finally:
            del os.environ['BASE_PATH']
            del os.environ['FILE_NAME']

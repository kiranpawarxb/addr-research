"""Unit tests for the CLI module.

Tests command-line argument parsing, configuration loading, and error handling.
"""

import pytest
import sys
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

from src.cli import (
    parse_arguments,
    setup_logging,
    load_and_override_config,
    validate_paths,
    main
)
from src.config_loader import ConfigurationError, Config, InputConfig, LLMConfig, ConsolidationConfig, OutputConfig, LoggingConfig


class TestArgumentParsing:
    """Test command-line argument parsing."""
    
    def test_parse_arguments_defaults(self):
        """Test parsing with default arguments."""
        with patch('sys.argv', ['prog']):
            args = parse_arguments()
            assert args.config == 'config/config.yaml'
            assert args.input is None
            assert args.output is None
            assert args.verbose is False
            assert args.quiet is False
            assert args.log_file is None
            assert args.no_stats is False
    
    def test_parse_arguments_with_config(self):
        """Test parsing with custom config file."""
        with patch('sys.argv', ['prog', '--config', 'custom.yaml']):
            args = parse_arguments()
            assert args.config == 'custom.yaml'
    
    def test_parse_arguments_with_input_output(self):
        """Test parsing with input and output overrides."""
        with patch('sys.argv', ['prog', '--input', 'in.csv', '--output', 'out.csv']):
            args = parse_arguments()
            assert args.input == 'in.csv'
            assert args.output == 'out.csv'
    
    def test_parse_arguments_verbose(self):
        """Test parsing with verbose flag."""
        with patch('sys.argv', ['prog', '-v']):
            args = parse_arguments()
            assert args.verbose is True
    
    def test_parse_arguments_quiet(self):
        """Test parsing with quiet flag."""
        with patch('sys.argv', ['prog', '-q']):
            args = parse_arguments()
            assert args.quiet is True
    
    def test_parse_arguments_log_file(self):
        """Test parsing with log file."""
        with patch('sys.argv', ['prog', '--log-file', 'test.log']):
            args = parse_arguments()
            assert args.log_file == 'test.log'
    
    def test_parse_arguments_no_stats(self):
        """Test parsing with no-stats flag."""
        with patch('sys.argv', ['prog', '--no-stats']):
            args = parse_arguments()
            assert args.no_stats is True
    
    def test_parse_arguments_short_options(self):
        """Test parsing with short option flags."""
        with patch('sys.argv', ['prog', '-c', 'cfg.yaml', '-i', 'in.csv', '-o', 'out.csv', '-v']):
            args = parse_arguments()
            assert args.config == 'cfg.yaml'
            assert args.input == 'in.csv'
            assert args.output == 'out.csv'
            assert args.verbose is True


class TestLoggingSetup:
    """Test logging configuration."""
    
    def test_setup_logging_info_level(self):
        """Test setting up logging at INFO level."""
        setup_logging('INFO')
        import logging
        logger = logging.getLogger()
        assert logger.level == logging.INFO
    
    def test_setup_logging_debug_level(self):
        """Test setting up logging at DEBUG level."""
        setup_logging('DEBUG')
        import logging
        logger = logging.getLogger()
        assert logger.level == logging.DEBUG
    
    def test_setup_logging_with_file(self, tmp_path):
        """Test setting up logging with file handler."""
        log_file = tmp_path / "test.log"
        setup_logging('INFO', str(log_file))
        
        import logging
        logger = logging.getLogger()
        
        # Check that file handler was added
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0
        
        # Test that logging works
        logger.info("Test message")
        assert log_file.exists()


class TestConfigOverrides:
    """Test configuration loading and overrides."""
    
    def test_load_and_override_config_no_overrides(self, tmp_path):
        """Test loading config without overrides."""
        # Create a minimal config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
input:
  file_path: "test.csv"
  required_columns: ["addr_text", "pincode", "city_id"]
llm:
  api_endpoint: "https://api.openai.com/v1/chat/completions"
  api_key: "test-key"
consolidation:
  fuzzy_matching: true
output:
  file_path: "output.csv"
logging:
  level: "INFO"
""")
        
        args = argparse.Namespace(
            config=str(config_file),
            input=None,
            output=None,
            verbose=False,
            quiet=False,
            log_file=None,
            no_stats=False
        )
        
        config = load_and_override_config(args)
        assert config.input.file_path == "test.csv"
        assert config.output.file_path == "output.csv"
        assert config.logging.level == "INFO"
    
    def test_load_and_override_config_with_input_output(self, tmp_path):
        """Test loading config with input/output overrides."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
input:
  file_path: "test.csv"
llm:
  api_key: "test-key"
output:
  file_path: "output.csv"
""")
        
        args = argparse.Namespace(
            config=str(config_file),
            input="override_input.csv",
            output="override_output.csv",
            verbose=False,
            quiet=False,
            log_file=None,
            no_stats=False
        )
        
        config = load_and_override_config(args)
        assert config.input.file_path == "override_input.csv"
        assert config.output.file_path == "override_output.csv"
    
    def test_load_and_override_config_verbose(self, tmp_path):
        """Test loading config with verbose flag."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
input:
  file_path: "test.csv"
llm:
  api_key: "test-key"
logging:
  level: "INFO"
""")
        
        args = argparse.Namespace(
            config=str(config_file),
            input=None,
            output=None,
            verbose=True,
            quiet=False,
            log_file=None,
            no_stats=False
        )
        
        config = load_and_override_config(args)
        assert config.logging.level == "DEBUG"
    
    def test_load_and_override_config_quiet(self, tmp_path):
        """Test loading config with quiet flag."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
input:
  file_path: "test.csv"
llm:
  api_key: "test-key"
logging:
  level: "INFO"
""")
        
        args = argparse.Namespace(
            config=str(config_file),
            input=None,
            output=None,
            verbose=False,
            quiet=True,
            log_file=None,
            no_stats=False
        )
        
        config = load_and_override_config(args)
        assert config.logging.level == "WARNING"
    
    def test_load_and_override_config_no_stats(self, tmp_path):
        """Test loading config with no-stats flag."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
input:
  file_path: "test.csv"
llm:
  api_key: "test-key"
output:
  include_statistics: true
""")
        
        args = argparse.Namespace(
            config=str(config_file),
            input=None,
            output=None,
            verbose=False,
            quiet=False,
            log_file=None,
            no_stats=True
        )
        
        config = load_and_override_config(args)
        assert config.output.include_statistics is False


class TestPathValidation:
    """Test path validation."""
    
    def test_validate_paths_success(self, tmp_path):
        """Test path validation with valid paths."""
        # Create input file
        input_file = tmp_path / "input.csv"
        input_file.write_text("test")
        
        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        config = Config(
            input=InputConfig(file_path=str(input_file)),
            llm=LLMConfig(),
            consolidation=ConsolidationConfig(),
            output=OutputConfig(file_path=str(output_dir / "output.csv")),
            logging=LoggingConfig()
        )
        
        # Should not raise
        validate_paths(config)
    
    def test_validate_paths_input_not_found(self, tmp_path):
        """Test path validation with missing input file."""
        config = Config(
            input=InputConfig(file_path=str(tmp_path / "nonexistent.csv")),
            llm=LLMConfig(),
            consolidation=ConsolidationConfig(),
            output=OutputConfig(file_path=str(tmp_path / "output.csv")),
            logging=LoggingConfig()
        )
        
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            validate_paths(config)
    
    def test_validate_paths_creates_output_dir(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("test")
        
        output_dir = tmp_path / "new_output_dir"
        
        config = Config(
            input=InputConfig(file_path=str(input_file)),
            llm=LLMConfig(),
            consolidation=ConsolidationConfig(),
            output=OutputConfig(file_path=str(output_dir / "output.csv")),
            logging=LoggingConfig()
        )
        
        validate_paths(config)
        assert output_dir.exists()


class TestMainFunction:
    """Test main entry point."""
    
    def test_main_missing_config(self):
        """Test main with missing config file."""
        with patch('sys.argv', ['prog', '--config', 'nonexistent.yaml']):
            exit_code = main()
            assert exit_code == 1  # ConfigurationError
    
    def test_main_keyboard_interrupt(self, tmp_path):
        """Test main handles keyboard interrupt."""
        # Create input file so validation passes
        input_file = tmp_path / "test.csv"
        input_file.write_text("test")
        
        # Use forward slashes for YAML compatibility
        input_path_yaml = str(input_file).replace('\\', '/')
        
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"""
input:
  file_path: "{input_path_yaml}"
llm:
  api_key: "test-key"
""")
        
        with patch('sys.argv', ['prog', '--config', str(config_file)]):
            with patch('src.cli.AddressConsolidationPipeline') as mock_pipeline:
                mock_pipeline.return_value.run.side_effect = KeyboardInterrupt()
                exit_code = main()
                assert exit_code == 130  # SIGINT

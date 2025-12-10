"""Tests for logging infrastructure."""

import logging
import tempfile
from pathlib import Path
import pytest

from src.cli import setup_logging


class TestLoggingInfrastructure:
    """Test logging setup and configuration."""
    
    def test_logging_levels_configured(self):
        """Test that all logging levels can be configured."""
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in levels:
            setup_logging(level)
            root_logger = logging.getLogger()
            assert root_logger.level == getattr(logging, level)
    
    def test_console_handler_configured(self):
        """Test that console handler is properly configured."""
        setup_logging('INFO')
        root_logger = logging.getLogger()
        
        # Should have at least one handler
        assert len(root_logger.handlers) > 0
        
        # Should have a StreamHandler
        stream_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) > 0
    
    def test_file_handler_configured(self):
        """Test that file handler is properly configured when log file is specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging('INFO', str(log_file))
            
            root_logger = logging.getLogger()
            
            # Should have at least two handlers (console + file)
            assert len(root_logger.handlers) >= 2
            
            # Should have a FileHandler
            file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) > 0
            
            # Log file should be created
            assert log_file.exists()
            
            # Close file handlers to release the file on Windows
            for handler in file_handlers:
                handler.close()
                root_logger.removeHandler(handler)
    
    def test_logging_format_includes_timestamp(self):
        """Test that log format includes timestamp."""
        setup_logging('INFO')
        root_logger = logging.getLogger()
        
        # Check that handlers have formatters
        for handler in root_logger.handlers:
            assert handler.formatter is not None
            format_str = handler.formatter._fmt
            assert '%(asctime)s' in format_str
    
    def test_logging_format_includes_level(self):
        """Test that log format includes log level."""
        setup_logging('INFO')
        root_logger = logging.getLogger()
        
        # Check that handlers have formatters with level
        for handler in root_logger.handlers:
            assert handler.formatter is not None
            format_str = handler.formatter._fmt
            assert '%(levelname)s' in format_str
    
    def test_logging_format_includes_module_name(self):
        """Test that log format includes module name."""
        setup_logging('INFO')
        root_logger = logging.getLogger()
        
        # Check that handlers have formatters with module name
        for handler in root_logger.handlers:
            assert handler.formatter is not None
            format_str = handler.formatter._fmt
            assert '%(name)s' in format_str
    
    def test_structured_logging_in_components(self):
        """Test that components use structured logging."""
        # Import components to check they have loggers
        from src import csv_reader, llm_parser, consolidation_engine, output_writer, pipeline
        
        # Each module should have a logger
        assert hasattr(csv_reader, 'logger')
        assert hasattr(llm_parser, 'logger')
        assert hasattr(consolidation_engine, 'logger')
        assert hasattr(output_writer, 'logger')
        assert hasattr(pipeline, 'logger')
        
        # Loggers should be properly named
        assert csv_reader.logger.name == 'src.csv_reader'
        assert llm_parser.logger.name == 'src.llm_parser'
        assert consolidation_engine.logger.name == 'src.consolidation_engine'
        assert output_writer.logger.name == 'src.output_writer'
        assert pipeline.logger.name == 'src.pipeline'
    
    def test_log_file_invalid_path_handled_gracefully(self):
        """Test that invalid log file path is handled gracefully."""
        # Try to create log file in non-existent directory
        invalid_path = "/nonexistent/directory/test.log"
        
        # Should not raise exception
        setup_logging('INFO', invalid_path)
        
        # Should still have console handler
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0

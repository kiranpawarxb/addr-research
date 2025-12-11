"""Tests for the LLM Parser component."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from src.llm_parser import LLMParser
from src.models import ParsedAddress


class TestLLMParser:
    """Test suite for LLMParser class."""
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance for testing."""
        return LLMParser(
            api_endpoint="https://api.example.com/v1/chat/completions",
            api_key="test-api-key",
            model="gpt-4",
            max_retries=3,
            timeout_seconds=30,
            batch_size=5
        )
    
    @pytest.fixture
    def sample_address(self):
        """Sample Indian address for testing."""
        return "Flat 301, Prestige Shantiniketan, Whitefield Main Road, Whitefield, Bangalore, Karnataka 560066"
    
    @pytest.fixture
    def sample_llm_response(self):
        """Sample LLM API response."""
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "UN": "Flat 301",
                            "SN": "Prestige Shantiniketan",
                            "LN": "",
                            "RD": "Whitefield Main Road",
                            "SL": "Whitefield",
                            "LOC": "Whitefield",
                            "CY": "Bangalore",
                            "DIS": "Bangalore Urban",
                            "ST": "Karnataka",
                            "CN": "India",
                            "PIN": "560066",
                            "Note": ""
                        })
                    }
                }
            ]
        }
    
    def test_parser_initialization(self, parser):
        """Test that parser initializes with correct configuration."""
        assert parser.api_endpoint == "https://api.example.com/v1/chat/completions"
        assert parser.api_key == "test-api-key"
        assert parser.model == "gpt-4"
        assert parser.max_retries == 3
        assert parser.timeout_seconds == 30
        assert parser.batch_size == 5
    
    def test_build_prompt(self, parser, sample_address):
        """Test that prompt is built correctly with examples."""
        prompt = parser._build_prompt(sample_address)
        
        assert sample_address in prompt
        assert "UN (Unit Number)" in prompt
        assert "SN (Society Name)" in prompt
        assert "Prestige Shantiniketan" in prompt  # Example in prompt
        assert "Lodha Splendora" in prompt  # Another example
    
    def test_extract_fields_success(self, parser, sample_llm_response):
        """Test successful field extraction from LLM response."""
        parsed = parser._extract_fields(sample_llm_response)
        
        assert parsed.unit_number == "Flat 301"
        assert parsed.society_name == "Prestige Shantiniketan"
        assert parsed.road == "Whitefield Main Road"
        assert parsed.city == "Bangalore"
        assert parsed.state == "Karnataka"
        assert parsed.pin_code == "560066"
        assert parsed.country == "India"
    
    def test_extract_fields_with_markdown(self, parser):
        """Test field extraction when JSON is wrapped in markdown code blocks."""
        response_with_markdown = {
            "choices": [
                {
                    "message": {
                        "content": "```json\n" + json.dumps({
                            "UN": "A-101",
                            "SN": "Test Society",
                            "LN": "",
                            "RD": "Test Road",
                            "SL": "",
                            "LOC": "Test Locality",
                            "CY": "Mumbai",
                            "DIS": "Mumbai",
                            "ST": "Maharashtra",
                            "CN": "India",
                            "PIN": "400001",
                            "Note": ""
                        }) + "\n```"
                    }
                }
            ]
        }
        
        parsed = parser._extract_fields(response_with_markdown)
        
        assert parsed.unit_number == "A-101"
        assert parsed.society_name == "Test Society"
        assert parsed.city == "Mumbai"
    
    def test_extract_fields_missing_fields(self, parser):
        """Test that missing fields get default empty string values."""
        response_partial = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "UN": "B-202",
                            "SN": "Partial Society",
                            "CY": "Delhi",
                            "PIN": "110001"
                            # Missing other fields
                        })
                    }
                }
            ]
        }
        
        parsed = parser._extract_fields(response_partial)
        
        assert parsed.unit_number == "B-202"
        assert parsed.society_name == "Partial Society"
        assert parsed.city == "Delhi"
        assert parsed.pin_code == "110001"
        # Missing fields should be empty strings
        assert parsed.landmark == ""
        assert parsed.road == ""
        assert parsed.sub_locality == ""
    
    def test_extract_fields_invalid_json(self, parser):
        """Test that invalid JSON raises JSONDecodeError."""
        response_invalid = {
            "choices": [
                {
                    "message": {
                        "content": "This is not valid JSON"
                    }
                }
            ]
        }
        
        with pytest.raises(json.JSONDecodeError):
            parser._extract_fields(response_invalid)
    
    def test_extract_fields_unexpected_structure(self, parser):
        """Test that unexpected response structure raises KeyError."""
        response_bad_structure = {
            "wrong_key": "wrong_value"
        }
        
        with pytest.raises(KeyError):
            parser._extract_fields(response_bad_structure)
    
    @patch('requests.post')
    def test_parse_address_success(self, mock_post, parser, sample_address, sample_llm_response):
        """Test successful address parsing."""
        # Mock the API response
        mock_response = Mock()
        mock_response.json.return_value = sample_llm_response
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        result = parser.parse_address(sample_address)
        
        assert result.parse_success is True
        assert result.parse_error is None
        assert result.society_name == "Prestige Shantiniketan"
        assert result.city == "Bangalore"
        assert parser._total_parsed == 1
        assert parser._total_failed == 0
    
    @patch('requests.post')
    def test_parse_address_empty_input(self, mock_post, parser):
        """Test parsing with empty address text."""
        result = parser.parse_address("")
        
        assert result.parse_success is False
        assert result.parse_error == "Empty address text"
        # Should not make API call
        mock_post.assert_not_called()
    
    @patch('requests.post')
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_parse_address_timeout_with_retry(self, mock_sleep, mock_post, parser, sample_address):
        """Test that timeout triggers retry logic."""
        # First two attempts timeout, third succeeds
        mock_post.side_effect = [
            requests.exceptions.Timeout("Timeout 1"),
            requests.exceptions.Timeout("Timeout 2"),
            Mock(json=lambda: {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "UN": "", "SN": "Test", "LN": "", "RD": "",
                            "SL": "", "LOC": "", "CY": "Mumbai", "DIS": "",
                            "ST": "Maharashtra", "CN": "India", "PIN": "400001", "Note": ""
                        })
                    }
                }]
            }, raise_for_status=Mock())
        ]
        
        result = parser.parse_address(sample_address)
        
        assert result.parse_success is True
        assert mock_post.call_count == 3
        # Check exponential backoff: 2^0=1, 2^1=2
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)
    
    @patch('requests.post')
    @patch('time.sleep')
    def test_parse_address_timeout_all_retries(self, mock_sleep, mock_post, parser, sample_address):
        """Test that all retries exhausted returns failure."""
        # All attempts timeout
        mock_post.side_effect = requests.exceptions.Timeout("Timeout")
        
        result = parser.parse_address(sample_address)
        
        assert result.parse_success is False
        assert "timeout after 3 retries" in result.parse_error.lower()
        assert mock_post.call_count == 3
        assert parser._total_failed == 1
    
    @patch('requests.post')
    @patch('time.sleep')
    def test_parse_address_api_error_with_retry(self, mock_sleep, mock_post, parser, sample_address):
        """Test that API errors trigger retry logic."""
        # First attempt fails, second succeeds
        mock_post.side_effect = [
            requests.exceptions.RequestException("API Error"),
            Mock(json=lambda: {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "UN": "", "SN": "Test", "LN": "", "RD": "",
                            "SL": "", "LOC": "", "CY": "Delhi", "DIS": "",
                            "ST": "Delhi", "CN": "India", "PIN": "110001", "Note": ""
                        })
                    }
                }]
            }, raise_for_status=Mock())
        ]
        
        result = parser.parse_address(sample_address)
        
        assert result.parse_success is True
        assert mock_post.call_count == 2
    
    @patch('requests.post')
    @patch('time.sleep')
    def test_parse_address_invalid_json_with_retry(self, mock_sleep, mock_post, parser, sample_address):
        """Test that invalid JSON triggers retry logic."""
        # First two attempts return invalid JSON, third succeeds
        mock_post.side_effect = [
            Mock(json=lambda: {"choices": [{"message": {"content": "Not JSON"}}]}, raise_for_status=Mock()),
            Mock(json=lambda: {"choices": [{"message": {"content": "Still not JSON"}}]}, raise_for_status=Mock()),
            Mock(json=lambda: {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "UN": "", "SN": "Valid", "LN": "", "RD": "",
                            "SL": "", "LOC": "", "CY": "Pune", "DIS": "",
                            "ST": "Maharashtra", "CN": "India", "PIN": "411001", "Note": ""
                        })
                    }
                }]
            }, raise_for_status=Mock())
        ]
        
        result = parser.parse_address(sample_address)
        
        assert result.parse_success is True
        assert result.society_name == "Valid"
        assert mock_post.call_count == 3
    
    @patch('requests.post')
    def test_parse_batch(self, mock_post, parser):
        """Test batch processing of multiple addresses."""
        addresses = [
            "Address 1",
            "Address 2",
            "Address 3"
        ]
        
        # Mock response that returns consistent data regardless of order
        def mock_response():
            return Mock(
                json=lambda: {
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "UN": "Unit Test", "SN": "Society Test", "LN": "", "RD": "",
                                "SL": "", "LOC": "", "CY": "City", "DIS": "",
                                "ST": "State", "CN": "India", "PIN": "123456", "Note": ""
                            })
                        }
                    }]
                },
                raise_for_status=Mock()
            )
        
        mock_post.return_value = mock_response()
        
        results = parser.parse_batch(addresses)
        
        assert len(results) == 3
        assert all(r.parse_success for r in results)
        # Check that all results have consistent data (since parallel processing order is not guaranteed)
        for result in results:
            assert result.unit_number == "Unit Test"
            assert result.society_name == "Society Test"
            assert result.city == "City"
    
    def test_parse_batch_empty_list(self, parser):
        """Test batch processing with empty list."""
        results = parser.parse_batch([])
        assert results == []
    
    def test_get_statistics(self, parser):
        """Test statistics tracking."""
        stats = parser.get_statistics()
        
        assert "total_parsed" in stats
        assert "total_failed" in stats
        assert "total_retries" in stats
        assert stats["total_parsed"] == 0
        assert stats["total_failed"] == 0
        assert stats["total_retries"] == 0
    
    @patch('requests.post')
    def test_make_api_request(self, mock_post, parser):
        """Test API request construction."""
        mock_response = Mock()
        mock_response.json.return_value = {"test": "data"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        result = parser._make_api_request("Test prompt")
        
        assert result == {"test": "data"}
        
        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        assert call_args[0][0] == parser.api_endpoint
        assert call_args[1]["timeout"] == parser.timeout_seconds
        assert "Authorization" in call_args[1]["headers"]
        assert f"Bearer {parser.api_key}" in call_args[1]["headers"]["Authorization"]
        
        payload = call_args[1]["json"]
        assert payload["model"] == "gpt-4"
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"] == "Test prompt"

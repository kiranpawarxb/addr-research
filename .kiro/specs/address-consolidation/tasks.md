# Implementation Plan

- [x] 1. Set up project structure and dependencies





  - Create Python project with proper directory structure (src/, tests/, config/)
  - Set up virtual environment and install dependencies (pandas, requests, hypothesis, pytest, pyyaml, rapidfuzz, tqdm)
  - Create requirements.txt file
  - Set up pytest configuration
  - _Requirements: All_

- [x] 2. Implement data models





  - Create dataclasses for AddressRecord, ParsedAddress, ConsolidatedGroup, and ConsolidationStats
  - Add type hints and validation
  - _Requirements: 1.1, 2.2, 3.1_

- [ ]* 2.1 Write property test for data models
  - **Property 3: LLM parser extracts all fields**
  - **Validates: Requirements 2.2**

- [x] 3. Implement CSV Reader component









  - Create CSVReader class with streaming file reading
  - Implement column validation logic
  - Add malformed row detection and logging
  - Handle file not found and empty file errors
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ]* 3.1 Write property test for CSV loading
  - **Property 1: CSV loading preserves record count**
  - **Validates: Requirements 1.1, 1.2, 1.5**

- [ ]* 3.2 Write property test for malformed row handling
  - **Property 2: Malformed rows are skipped gracefully**
  - **Validates: Requirements 1.4**

- [x] 4. Implement LLM Parser component





  - Create LLMParser class with API client
  - Implement prompt building with Indian address examples
  - Add JSON parsing and field extraction logic
  - Implement retry logic with exponential backoff (3 retries)
  - Handle timeout, invalid JSON, and API errors
  - Add batch processing support for parallel API calls
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 7.1, 7.2_

- [ ]* 4.1 Write property test for field extraction
  - **Property 3: LLM parser extracts all fields**
  - **Validates: Requirements 2.2**

- [ ]* 4.2 Write property test for default values
  - **Property 4: Missing fields get default values**
  - **Validates: Requirements 2.3**

- [ ]* 4.3 Write property test for invalid JSON handling
  - **Property 5: Invalid JSON is handled gracefully**
  - **Validates: Requirements 2.5**

- [ ]* 4.4 Write property test for retry logic
  - **Property 15: Retry logic executes correctly**
  - **Validates: Requirements 7.1**

- [x] 5. Checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Consolidation Engine component





  - Create ConsolidationEngine class
  - Implement society name normalization (lowercase, trim, remove special chars)
  - Add grouping logic based on Society Name and PIN code
  - Implement fuzzy matching with configurable similarity threshold using rapidfuzz
  - Handle empty/missing Society Names (unmatched group)
  - Assign unique group IDs using UUID
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.2, 6.3_

- [ ]* 6.1 Write property test for grouping logic
  - **Property 6: Grouping requires both Society Name and PIN match**
  - **Validates: Requirements 3.1, 3.2**

- [ ]* 6.2 Write property test for unmatched group
  - **Property 7: Empty Society Names go to unmatched group**
  - **Validates: Requirements 3.5**

- [ ]* 6.3 Write property test for group count
  - **Property 10: Group count matches unique (SN, PIN) pairs**
  - **Validates: Requirements 3.4, 5.1**

- [ ]* 6.4 Write property test for fuzzy matching
  - **Property 14: Fuzzy matching respects threshold**
  - **Validates: Requirements 6.2, 6.3**

- [x] 7. Implement Output Writer component





  - Create OutputWriter class
  - Implement CSV writing with all original columns plus parsed fields
  - Add group_id and location_identifier columns
  - Handle file write errors (permission denied, disk full)
  - Escape invalid characters in output data
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ]* 7.1 Write property test for data preservation
  - **Property 8: Data preservation through pipeline**
  - **Validates: Requirements 3.3, 4.4**

- [ ]* 7.2 Write property test for output columns
  - **Property 9: Output contains all required columns**
  - **Validates: Requirements 4.2, 4.3**

- [x] 8. Implement Statistics Reporter component




  - Create StatisticsReporter class
  - Implement calculations for total groups, average per group, largest group
  - Calculate match percentage and parse success/failure rates
  - Create readable display format with formatting
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 8.1 Write property test for average calculation
  - **Property 11: Average calculation is accurate**
  - **Validates: Requirements 5.2**

- [ ]* 8.2 Write property test for largest group identification
  - **Property 12: Largest group identification is correct**
  - **Validates: Requirements 5.3**

- [ ]* 8.3 Write property test for match percentage
  - **Property 13: Match percentage calculation is accurate**
  - **Validates: Requirements 5.4**

- [x] 9. Implement configuration system




  - Create configuration loader using PyYAML
  - Define configuration schema with validation
  - Support environment variable substitution (e.g., ${OPENAI_API_KEY})
  - Handle invalid configuration with error reporting and defaults
  - _Requirements: 6.1, 6.5_

- [x] 10. Implement main pipeline orchestrator





  - Create main application class that coordinates all components
  - Implement end-to-end pipeline: Read → Parse → Consolidate → Write → Report
  - Add progress tracking with tqdm
  - Handle errors gracefully and continue processing
  - Ensure unparseable addresses are included in output with empty fields
  - _Requirements: 7.3, 7.4, 7.5_

- [ ]* 10.1 Write property test for error recovery
  - **Property 16: Processing continues after errors**
  - **Validates: Requirements 7.2, 7.3, 7.4**

- [ ]* 10.2 Write property test for parse counts
  - **Property 17: Parse success/failure counts are accurate**
  - **Validates: Requirements 7.5**

- [x] 11. Checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Create command-line interface





  - Implement CLI using argparse
  - Add arguments for config file path, input file, output file
  - Add verbose/debug logging options
  - Display progress and statistics during execution
  - _Requirements: All_

- [x] 13. Add logging infrastructure





  - Set up Python logging with file and console handlers
  - Configure log levels (INFO, DEBUG, WARNING, ERROR)
  - Add structured logging for key events (parsing start/end, errors, statistics)
  - _Requirements: 1.4, 2.5, 7.2_

- [x] 14. Create example configuration file





  - Write config.yaml with all configuration options documented
  - Include comments explaining each setting
  - Provide sensible defaults
  - _Requirements: 6.1, 6.2, 6.3, 6.5_

- [x] 15. Write README documentation





  - Document installation instructions
  - Explain configuration options
  - Provide usage examples
  - Document expected input/output formats
  - Include troubleshooting section
  - _Requirements: All_

- [x] 16. Create integration tests





  - Write end-to-end test with sample CSV file
  - Test with mocked LLM API responses
  - Verify complete pipeline execution
  - Test error scenarios (missing columns, API failures)
  - _Requirements: All_

- [x] 17. Final checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.

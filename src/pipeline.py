"""Main pipeline orchestrator for the Address Consolidation System.

This module coordinates all components to execute the end-to-end pipeline:
Read → Parse → Consolidate → Write → Report
"""

import logging
from typing import List, Tuple
from tqdm import tqdm

from src.models import AddressRecord, ParsedAddress, ConsolidatedGroup
from src.csv_reader import CSVReader
from src.llm_parser import LLMParser
from src.local_llm_parser import LocalLLMParser
from src.indicbert_parser import IndicBERTParser
from src.shiprocket_parser import ShiprocketParser
from src.consolidation_engine import ConsolidationEngine
from src.output_writer import OutputWriter
from src.statistics_reporter import StatisticsReporter
from src.config_loader import Config


logger = logging.getLogger(__name__)


class AddressConsolidationPipeline:
    """Main application class that coordinates all components.
    
    Implements the end-to-end pipeline with progress tracking and error handling.
    Ensures unparseable addresses are included in output with empty fields.
    
    Validates: Requirements 7.3, 7.4, 7.5
    """
    
    def __init__(self, config: Config):
        """Initialize pipeline with configuration.
        
        Args:
            config: Configuration object with all settings
        """
        self.config = config
        
        # Initialize components
        self.csv_reader = CSVReader(
            file_path=config.input.file_path,
            required_columns=config.input.required_columns
        )
        
        # Initialize parser based on configuration
        parser_type = getattr(config.llm, 'parser_type', 'openai')
        
        if parser_type == 'local':
            logger.info("Using local rule-based parser")
            local_model = getattr(config.llm, 'local_model', 'ai4bharat/indic-bert')
            self.llm_parser = LocalLLMParser(
                model_name=local_model,
                batch_size=config.llm.batch_size
            )
        elif parser_type == 'indicbert':
            logger.info("Using IndicBERT transformer parser")
            local_model = getattr(config.llm, 'local_model', 'ai4bharat/indic-bert')
            use_gpu = getattr(config.llm, 'use_gpu', False)
            self.llm_parser = IndicBERTParser(
                model_name=local_model,
                batch_size=config.llm.batch_size,
                use_gpu=use_gpu
            )
        elif parser_type == 'shiprocket':
            logger.info("Using Shiprocket fine-tuned IndicBERT parser for Indian addresses")
            use_gpu = getattr(config.llm, 'use_gpu', False)
            self.llm_parser = ShiprocketParser(
                batch_size=config.llm.batch_size,
                use_gpu=use_gpu
            )
        else:
            logger.info("Using OpenAI API parser")
            self.llm_parser = LLMParser(
                api_endpoint=config.llm.api_endpoint,
                api_key=config.llm.api_key,
                model=config.llm.model,
                max_retries=config.llm.max_retries,
                timeout_seconds=config.llm.timeout_seconds,
                batch_size=config.llm.batch_size
            )
        
        self.consolidation_engine = ConsolidationEngine(
            fuzzy_matching=config.consolidation.fuzzy_matching,
            similarity_threshold=config.consolidation.similarity_threshold,
            normalize_society_names=config.consolidation.normalize_society_names
        )
        
        self.output_writer = OutputWriter(
            output_path=config.output.file_path
        )
        
        self.statistics_reporter = StatisticsReporter()
        
        # Statistics tracking
        self._total_records = 0
        self._failed_parses = 0
    
    def run(self) -> None:
        """Execute the complete pipeline.
        
        Runs the end-to-end process:
        1. Read CSV file
        2. Parse addresses with LLM
        3. Consolidate by Society Name and PIN
        4. Write output CSV
        5. Display statistics
        
        Handles errors gracefully and continues processing.
        Ensures unparseable addresses are included in output.
        """
        logger.info("=" * 70)
        logger.info("Starting Address Consolidation Pipeline")
        logger.info("=" * 70)
        
        try:
            # Stage 1: Read CSV file
            logger.info("STAGE 1: Reading CSV file")
            logger.info("-" * 70)
            address_records = self._read_csv()
            
            if not address_records:
                logger.warning("No records to process. Exiting pipeline.")
                return
            
            logger.info(f"Stage 1 complete: {len(address_records)} records loaded")
            
            # Stage 2: Parse addresses with LLM
            logger.info("")
            logger.info("STAGE 2: Parsing addresses with LLM")
            logger.info("-" * 70)
            records_with_parsed = self._parse_addresses(address_records)
            logger.info(f"Stage 2 complete: {len(records_with_parsed)} addresses processed")
            
            # Stage 3: Consolidate addresses
            logger.info("")
            logger.info("STAGE 3: Consolidating addresses")
            logger.info("-" * 70)
            consolidated_groups = self._consolidate_addresses(records_with_parsed)
            logger.info(f"Stage 3 complete: {len(consolidated_groups)} groups created")
            
            # Stage 4: Write output
            logger.info("")
            logger.info("STAGE 4: Writing output CSV")
            logger.info("-" * 70)
            self._write_output(consolidated_groups)
            logger.info("Stage 4 complete: Output file written")
            
            # Stage 5: Display statistics
            if self.config.output.include_statistics:
                logger.info("")
                logger.info("STAGE 5: Generating statistics")
                logger.info("-" * 70)
                self._display_statistics(consolidated_groups)
                logger.info("Stage 5 complete: Statistics generated")
            
            logger.info("")
            logger.info("=" * 70)
            logger.info("Pipeline completed successfully!")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error("=" * 70)
            logger.error("Pipeline failed with error")
            logger.error("=" * 70)
            logger.error(f"Error: {e}", exc_info=True)
            raise
    
    def _read_csv(self) -> List[AddressRecord]:
        """Read and validate CSV file.
        
        Returns:
            List of AddressRecord objects
            
        Raises:
            ValueError: If CSV validation fails
            FileNotFoundError: If CSV file doesn't exist
        """
        try:
            # Validate columns first
            is_valid, missing_columns = self.csv_reader.validate_columns()
            if not is_valid:
                raise ValueError(
                    f"CSV file is missing required columns: {', '.join(missing_columns)}"
                )
            
            # Read all records with progress bar
            address_records = []
            
            print("\nReading CSV file...")
            for record in tqdm(
                self.csv_reader.read(),
                desc="Loading records",
                unit="records"
            ):
                address_records.append(record)
            
            self._total_records = len(address_records)
            
            logger.info(
                f"Successfully loaded {self._total_records} records "
                f"(skipped {self.csv_reader.get_malformed_count()} malformed rows)"
            )
            
            return address_records
            
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
    
    def _parse_addresses(
        self,
        address_records: List[AddressRecord]
    ) -> List[Tuple[AddressRecord, ParsedAddress]]:
        """Parse addresses using LLM.
        
        Handles errors gracefully and continues processing.
        Unparseable addresses get empty parsed fields.
        
        Args:
            address_records: List of AddressRecord objects to parse
            
        Returns:
            List of tuples (AddressRecord, ParsedAddress)
            
        Validates: Requirements 7.2, 7.3, 7.4
        """
        records_with_parsed = []
        
        print("\nParsing addresses with LLM...")
        
        # Process in batches for efficiency
        batch_size = self.config.llm.batch_size
        
        for i in tqdm(
            range(0, len(address_records), batch_size),
            desc="Parsing batches",
            unit="batch"
        ):
            batch = address_records[i:i + batch_size]
            
            try:
                # Extract address texts
                address_texts = [record.addr_text for record in batch]
                
                # Parse batch
                parsed_addresses = self.llm_parser.parse_batch(address_texts)
                
                # Pair records with parsed addresses
                for record, parsed in zip(batch, parsed_addresses):
                    # Track failed parses (Requirement 7.5)
                    if not parsed.parse_success:
                        self._failed_parses += 1
                        logger.debug(
                            f"Failed to parse address: {record.addr_text[:50]}... "
                            f"Error: {parsed.parse_error}"
                        )
                    
                    # Include all addresses in output, even unparseable ones (Requirement 7.3)
                    records_with_parsed.append((record, parsed))
                    
            except Exception as e:
                # Handle batch processing errors gracefully (Requirement 7.4)
                logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
                
                # Create empty parsed addresses for failed batch
                for record in batch:
                    empty_parsed = ParsedAddress(
                        parse_success=False,
                        parse_error=f"Batch processing error: {str(e)}"
                    )
                    records_with_parsed.append((record, empty_parsed))
                    self._failed_parses += 1
                
                # Continue processing remaining batches (Requirement 7.4)
                continue
        
        logger.info(
            f"Parsing complete. Success: {self._total_records - self._failed_parses}, "
            f"Failed: {self._failed_parses}"
        )
        
        return records_with_parsed
    
    def _consolidate_addresses(
        self,
        records_with_parsed: List[Tuple[AddressRecord, ParsedAddress]]
    ) -> List[ConsolidatedGroup]:
        """Consolidate addresses by Society Name and PIN code.
        
        Args:
            records_with_parsed: List of tuples (AddressRecord, ParsedAddress)
            
        Returns:
            List of ConsolidatedGroup objects
        """
        print("\nConsolidating addresses...")
        
        try:
            consolidated_groups = self.consolidation_engine.consolidate(
                records_with_parsed
            )
            
            logger.info(f"Created {len(consolidated_groups)} consolidated groups")
            
            return consolidated_groups
            
        except Exception as e:
            logger.error(f"Error during consolidation: {e}")
            raise
    
    def _write_output(self, consolidated_groups: List[ConsolidatedGroup]) -> None:
        """Write consolidated results to output CSV.
        
        Args:
            consolidated_groups: List of ConsolidatedGroup objects
            
        Raises:
            PermissionError: If output file cannot be written
            OSError: If disk is full or other I/O error
        """
        print("\nWriting output CSV...")
        
        try:
            records_written = self.output_writer.write(consolidated_groups)
            
            logger.info(
                f"Successfully wrote {records_written} records to "
                f"{self.config.output.file_path}"
            )
            
            print(f"\nOutput written to: {self.config.output.file_path}")
            
        except Exception as e:
            logger.error(f"Error writing output: {e}")
            raise
    
    def _display_statistics(
        self,
        consolidated_groups: List[ConsolidatedGroup]
    ) -> None:
        """Generate and display consolidation statistics.
        
        Args:
            consolidated_groups: List of ConsolidatedGroup objects
        """
        try:
            stats = self.statistics_reporter.generate_stats(
                consolidated_groups=consolidated_groups,
                total_records=self._total_records,
                failed_parses=self._failed_parses
            )
            
            self.statistics_reporter.display(stats)
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            # Don't raise - statistics are informational only

"""Example demonstrating how to use the configuration loader.

This example shows how to load configuration from a YAML file
and access the various configuration sections.
"""

from src.config_loader import load_config, ConfigurationError


def main():
    """Load and display configuration."""
    try:
        # Load configuration from file
        config = load_config('config/config.yaml')
        
        print("Configuration loaded successfully!\n")
        
        # Access input configuration
        print("Input Configuration:")
        print(f"  File path: {config.input.file_path}")
        print(f"  Required columns: {', '.join(config.input.required_columns)}\n")
        
        # Access LLM configuration
        print("LLM Configuration:")
        print(f"  API endpoint: {config.llm.api_endpoint}")
        print(f"  Model: {config.llm.model}")
        print(f"  Max retries: {config.llm.max_retries}")
        print(f"  Timeout: {config.llm.timeout_seconds}s")
        print(f"  Batch size: {config.llm.batch_size}")
        print(f"  API key: {'*' * len(config.llm.api_key) if config.llm.api_key else '(not set)'}\n")
        
        # Access consolidation configuration
        print("Consolidation Configuration:")
        print(f"  Fuzzy matching: {config.consolidation.fuzzy_matching}")
        print(f"  Similarity threshold: {config.consolidation.similarity_threshold}")
        print(f"  Normalize society names: {config.consolidation.normalize_society_names}\n")
        
        # Access output configuration
        print("Output Configuration:")
        print(f"  File path: {config.output.file_path}")
        print(f"  Include statistics: {config.output.include_statistics}\n")
        
        # Access logging configuration
        print("Logging Configuration:")
        print(f"  Level: {config.logging.level}")
        print(f"  File path: {config.logging.file_path}\n")
        
    except ConfigurationError as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

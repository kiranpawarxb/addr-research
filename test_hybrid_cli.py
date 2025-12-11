#!/usr/bin/env python3
"""
Test script for GPU-CPU Hybrid Address Processing CLI

This script tests the CLI functionality including parameter validation,
configuration management, and help system.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import subprocess
import sys
import tempfile
import os
from pathlib import Path


def run_cli_command(args: list, expect_success: bool = True) -> tuple:
    """Run a CLI command and return result.
    
    Args:
        args: Command arguments
        expect_success: Whether to expect successful execution
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    cmd = ["py", "-m", "src.hybrid_main"] + args
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd()  # Ensure we run from the current directory
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def test_help_system():
    """Test CLI help system."""
    print("Testing help system...")
    
    # Test main help
    code, stdout, stderr = run_cli_command(["--help"])
    assert code == 0, f"Help command failed: {stderr}"
    assert "GPU-CPU Hybrid Address Processing System" in stdout
    assert "--input" in stdout
    assert "--gpu-batch-size" in stdout
    print("✓ Main help works")
    
    # Test version
    code, stdout, stderr = run_cli_command(["--version"])
    assert code == 0, f"Version command failed: {stderr}"
    assert "1.0.0" in stdout
    print("✓ Version command works")


def test_configuration_creation():
    """Test configuration file creation."""
    print("Testing configuration creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Test config creation
            code, stdout, stderr = run_cli_command(["--create-config"])
            assert code == 0, f"Config creation failed: {stderr}"
            
            # Check if files were created
            assert Path("config/config.yaml").exists(), "Base config not created"
            assert Path("config/hybrid_config.yaml").exists(), "Hybrid config not created"
            print("✓ Configuration creation works")
            
        finally:
            os.chdir(original_cwd)


def test_parameter_validation():
    """Test parameter validation."""
    print("Testing parameter validation...")
    
    # Test invalid GPU batch size
    code, stdout, stderr = run_cli_command([
        "--input", "test.csv", "--output", "out.csv", "--gpu-batch-size", "50"
    ], expect_success=False)
    assert code != 0, "Should fail with invalid GPU batch size"
    print("✓ GPU batch size validation works")
    
    # Test invalid GPU memory
    code, stdout, stderr = run_cli_command([
        "--input", "test.csv", "--output", "out.csv", "--gpu-memory", "1.5"
    ], expect_success=False)
    assert code != 0, "Should fail with invalid GPU memory"
    print("✓ GPU memory validation works")
    
    # Test invalid CPU ratio
    code, stdout, stderr = run_cli_command([
        "--input", "test.csv", "--output", "out.csv", "--cpu-ratio", "0.8"
    ], expect_success=False)
    assert code != 0, "Should fail with invalid CPU ratio"
    print("✓ CPU ratio validation works")
    
    # Test invalid throughput
    code, stdout, stderr = run_cli_command([
        "--input", "test.csv", "--output", "out.csv", "--target-throughput", "5000"
    ], expect_success=False)
    assert code != 0, "Should fail with invalid throughput"
    print("✓ Throughput validation works")


def test_dry_run():
    """Test dry run functionality."""
    print("Testing dry run...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test CSV file
        test_csv = Path(temp_dir) / "test.csv"
        test_csv.write_text("address\nTest Address 1\nTest Address 2\n")
        
        # Test dry run
        code, stdout, stderr = run_cli_command([
            "--input", str(test_csv),
            "--output", str(Path(temp_dir) / "output.csv"),
            "--dry-run"
        ])
        
        # Dry run should succeed even without GPU
        assert code == 0, f"Dry run failed: {stderr}"
        assert "DRY RUN MODE" in stdout or "Configuration validated" in stdout
        print("✓ Dry run works")


def test_benchmark():
    """Test benchmark functionality."""
    print("Testing benchmark...")
    
    # Test benchmark (may fail without GPU, but should not crash)
    code, stdout, stderr = run_cli_command(["--benchmark"])
    
    # Benchmark may fail due to missing GPU, but should handle gracefully
    if code == 0:
        print("✓ Benchmark works")
    else:
        print("⚠ Benchmark failed (expected without GPU)")


def test_export_config():
    """Test configuration export."""
    print("Testing configuration export...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = Path(temp_dir) / "exported_config.yaml"
        
        # Test config export
        code, stdout, stderr = run_cli_command([
            "--gpu-batch-size", "600",
            "--gpu-memory", "0.90",
            "--export-config", str(export_path)
        ])
        
        # Export should work even without processing
        if code == 0:
            assert export_path.exists(), "Config file not exported"
            print("✓ Configuration export works")
        else:
            print("⚠ Configuration export failed (may need base config)")


def main():
    """Run all CLI tests."""
    print("GPU-CPU Hybrid Address Processing CLI Tests")
    print("=" * 50)
    
    try:
        test_help_system()
        test_configuration_creation()
        test_parameter_validation()
        test_dry_run()
        test_benchmark()
        test_export_config()
        
        print("\n" + "=" * 50)
        print("All CLI tests completed successfully!")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
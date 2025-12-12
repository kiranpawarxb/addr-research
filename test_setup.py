#!/usr/bin/env python3
"""
Test script to verify the Sustained GPU Maximizer setup.
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✅ torch imported successfully (version: {torch.__version__})")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.device_count()} GPU(s) detected")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {gpu_name}")
        else:
            print("⚠️ CUDA not available - will use CPU only")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("✅ transformers imported successfully")
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
        return False
    
    try:
        # Test our custom modules
        sys.path.insert(0, 'src')
        from src.models import ParsedAddress
        from src.ultimate_multi_device_parser import UltimateMultiDeviceParser
        print("✅ Custom modules imported successfully")
    except ImportError as e:
        print(f"❌ Custom module import failed: {e}")
        return False
    
    return True

def test_gpu_setup():
    """Test GPU setup and model loading."""
    print("\n🔧 Testing GPU setup...")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available - skipping GPU tests")
            return True
        
        model_name = "shiprocket-ai/open-indicbert-indian-address-ner"
        print(f"📥 Testing model download: {model_name}")
        
        # Test tokenizer download
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer loaded successfully")
        
        # Test model download (this might take a while on first run)
        print("📥 Loading model (this may take a few minutes on first run)...")
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        print("✅ Model loaded successfully")
        
        # Test GPU loading
        if torch.cuda.is_available():
            model = model.cuda()
            print("✅ Model moved to GPU successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU setup test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic parsing functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        sys.path.insert(0, 'src')
        from src.ultimate_multi_device_parser import UltimateMultiDeviceParser
        
        # Initialize parser
        parser = UltimateMultiDeviceParser(
            batch_size=2,  # Small batch for testing
            use_nvidia_gpu=True,
            use_intel_gpu=False,
            use_all_cpu_cores=False
        )
        
        # Test with sample addresses
        test_addresses = [
            "Flat 301, Sunrise Apartments, MG Road, Bangalore 560034",
            "House 45, Green Valley, Noida 201301"
        ]
        
        print(f"🔄 Processing {len(test_addresses)} test addresses...")
        results = parser.parse_ultimate_multi_device(test_addresses)
        
        if results and len(results) == len(test_addresses):
            success_count = sum(1 for r in results if r.parse_success)
            print(f"✅ Basic functionality test passed")
            print(f"   Processed: {len(results)} addresses")
            print(f"   Success: {success_count}/{len(results)}")
            return True
        else:
            print("❌ Basic functionality test failed - unexpected results")
            return False
            
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 SUSTAINED GPU MAXIMIZER - SETUP TEST")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test GPU setup
    if not test_gpu_setup():
        all_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Setup is ready.")
        print("\n💡 Next steps:")
        print("   1. Run: python run_example.py")
        print("   2. Or place your CSV files here and run: python sustained_gpu_maximizer.py")
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
        print("\n🔧 Common fixes:")
        print("   1. Install requirements: pip install -r requirements.txt")
        print("   2. Check NVIDIA drivers: nvidia-smi")
        print("   3. Verify CUDA installation: nvcc --version")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Check multi-GPU configuration including Intel and NVIDIA GPUs.
"""

import torch
import subprocess
import sys

def check_all_gpus():
    """Check all available GPU configurations."""
    
    print("🔧 Multi-GPU System Analysis")
    print("=" * 60)
    
    # Check PyTorch CUDA (NVIDIA)
    print(f"📊 NVIDIA CUDA STATUS:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   NVIDIA GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"\n   NVIDIA GPU {i}: {gpu_props.name}")
            print(f"      Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
            print(f"      Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            print(f"      Multiprocessors: {gpu_props.multi_processor_count}")
    
    # Check Intel GPU (if available)
    print(f"\n📊 INTEL GPU STATUS:")
    try:
        # Check if Intel Extension for PyTorch is available
        import intel_extension_for_pytorch as ipex
        print(f"   Intel Extension Available: True")
        print(f"   Intel Extension Version: {ipex.__version__}")
        
        # Check XPU devices
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            print(f"   Intel XPU Available: True")
            print(f"   Intel XPU Count: {torch.xpu.device_count()}")
        else:
            print(f"   Intel XPU Available: False")
    except ImportError:
        print(f"   Intel Extension Available: False")
        print(f"   Note: Install with 'pip install intel_extension_for_pytorch'")
    
    # Check system GPU info
    print(f"\n💻 SYSTEM GPU DETECTION:")
    print("-" * 60)
    try:
        # Try nvidia-smi for NVIDIA GPUs
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("NVIDIA GPUs detected:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"   {line}")
        
        # Try to detect Intel GPUs
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("\nAll GPUs detected by system:")
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # Skip header
                if line.strip() and 'Name' not in line:
                    print(f"   {line.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"   Could not detect system GPUs: {e}")
    
    return torch.cuda.device_count() if torch.cuda.is_available() else 0

if __name__ == "__main__":
    gpu_count = check_all_gpus()
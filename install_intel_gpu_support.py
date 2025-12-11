#!/usr/bin/env python3
"""
Install Intel GPU support for multi-GPU processing.
"""

import subprocess
import sys

def install_intel_gpu_support():
    """Install Intel Extension for PyTorch to enable Intel GPU processing."""
    
    print("🔧 Installing Intel GPU Support")
    print("=" * 60)
    
    commands = [
        # Install Intel Extension for PyTorch
        [sys.executable, "-m", "pip", "install", "intel-extension-for-pytorch"],
        
        # Install Intel Neural Compressor (optional, for optimization)
        [sys.executable, "-m", "pip", "install", "neural-compressor"],
    ]
    
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Success: {cmd[-1]} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {cmd[-1]}: {e}")
            print(f"Error output: {e.stderr}")
    
    print(f"\n🎯 Next Steps:")
    print("1. Restart your Python environment")
    print("2. Run the test again with Multi-GPU Optimized configuration")
    print("3. You should see both 'cuda:0' and 'xpu:0' in devices used")

if __name__ == "__main__":
    install_intel_gpu_support()
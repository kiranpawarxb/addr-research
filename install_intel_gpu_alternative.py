#!/usr/bin/env python3
"""
Alternative approach to install Intel GPU support with compatibility fixes.
"""

import subprocess
import sys
import platform

def check_system_compatibility():
    """Check system compatibility for Intel GPU support."""
    
    print("🔍 System Compatibility Check")
    print("=" * 60)
    
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    
    # Check if we have Intel GPU
    try:
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("\nDetected GPUs:")
            lines = result.stdout.strip().split('\n')
            intel_gpu_found = False
            for line in lines[1:]:
                if line.strip() and 'Name' not in line:
                    print(f"   {line.strip()}")
                    if 'Intel' in line:
                        intel_gpu_found = True
            
            if intel_gpu_found:
                print("✅ Intel GPU detected")
            else:
                print("❌ No Intel GPU detected")
                return False
    except Exception as e:
        print(f"❌ Could not detect GPUs: {e}")
        return False
    
    return True

def try_intel_gpu_installation():
    """Try different approaches to install Intel GPU support."""
    
    print(f"\n🔧 Attempting Intel GPU Support Installation")
    print("=" * 60)
    
    # Approach 1: Try with specific Python version compatibility
    approaches = [
        {
            "name": "Intel Extension for PyTorch (Latest)",
            "commands": [
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                [sys.executable, "-m", "pip", "install", "intel-extension-for-pytorch", "--no-deps"],
            ]
        },
        {
            "name": "Intel Extension for PyTorch (CPU only)",
            "commands": [
                [sys.executable, "-m", "pip", "install", "intel-extension-for-pytorch-cpu"],
            ]
        },
        {
            "name": "OpenVINO Runtime (Alternative)",
            "commands": [
                [sys.executable, "-m", "pip", "install", "openvino"],
            ]
        }
    ]
    
    successful_installs = []
    
    for approach in approaches:
        print(f"\n🔄 Trying: {approach['name']}")
        print("-" * 40)
        
        success = True
        for cmd in approach['commands']:
            print(f"Running: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
                print(f"✅ Success")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed: {e}")
                print(f"Error: {e.stderr[:200]}...")
                success = False
                break
            except subprocess.TimeoutExpired:
                print(f"❌ Timeout")
                success = False
                break
        
        if success:
            successful_installs.append(approach['name'])
            print(f"✅ {approach['name']} installed successfully")
        else:
            print(f"❌ {approach['name']} installation failed")
    
    return successful_installs

def create_fallback_multi_gpu_solution():
    """Create a fallback solution for multi-GPU processing without Intel GPU."""
    
    print(f"\n🔧 Creating Fallback Multi-GPU Solution")
    print("=" * 60)
    
    fallback_code = '''"""
Fallback Multi-GPU Solution using CUDA Multi-Processing.

Since Intel GPU support is not available, this creates multiple CUDA contexts
to better utilize the NVIDIA GPU's multiple streaming multiprocessors.
"""

import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import logging

class FallbackMultiGPUProcessor:
    """Fallback processor that uses multiple CUDA streams for parallel processing."""
    
    def __init__(self, num_streams: int = 4):
        self.num_streams = num_streams
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    def create_cuda_streams(self):
        """Create multiple CUDA streams for parallel processing."""
        if self.device == "cuda:0":
            streams = []
            for i in range(self.num_streams):
                stream = torch.cuda.Stream()
                streams.append(stream)
            return streams
        return [None] * self.num_streams
    
    def process_with_streams(self, addresses, parser):
        """Process addresses using multiple CUDA streams."""
        streams = self.create_cuda_streams()
        chunk_size = len(addresses) // self.num_streams
        
        results = []
        
        # Process chunks in parallel using different streams
        for i in range(self.num_streams):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.num_streams - 1 else len(addresses)
            chunk = addresses[start_idx:end_idx]
            
            if self.device == "cuda:0" and streams[i]:
                with torch.cuda.stream(streams[i]):
                    chunk_results = parser._process_batch_on_device(chunk, self.device)
            else:
                chunk_results = parser._process_batch_on_device(chunk, self.device)
            
            results.extend(chunk_results)
        
        # Synchronize all streams
        if self.device == "cuda:0":
            for stream in streams:
                if stream:
                    stream.synchronize()
        
        return results
'''
    
    with open('src/fallback_multi_gpu.py', 'w') as f:
        f.write(fallback_code)
    
    print("✅ Created fallback multi-GPU solution: src/fallback_multi_gpu.py")
    print("   This uses multiple CUDA streams to better utilize your NVIDIA GPU")

def main():
    """Main installation function."""
    
    print("🚀 Intel GPU Support Installation")
    print("=" * 60)
    
    # Check compatibility
    if not check_system_compatibility():
        print("\n❌ System not compatible with Intel GPU processing")
        create_fallback_multi_gpu_solution()
        return
    
    # Try installation approaches
    successful = try_intel_gpu_installation()
    
    if successful:
        print(f"\n🎉 SUCCESS!")
        print(f"Successfully installed: {', '.join(successful)}")
        print(f"\n🔄 Please restart your Python environment and run:")
        print(f"   py test_optimized_10k.py")
    else:
        print(f"\n⚠️  Intel GPU installation failed")
        print(f"Creating fallback solution...")
        create_fallback_multi_gpu_solution()
        
        print(f"\n🎯 Alternative Solutions:")
        print(f"1. Use the fallback multi-stream CUDA processing")
        print(f"2. Install Visual Studio Build Tools for C++ compilation")
        print(f"3. Use Python 3.11 or 3.10 (better Intel Extension compatibility)")
        print(f"4. Focus on optimizing single GPU performance")

if __name__ == "__main__":
    main()
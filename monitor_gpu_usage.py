#!/usr/bin/env python3
"""
Real-time GPU usage monitoring script.
Shows NVIDIA GPU utilization, memory usage, and temperature.
"""

import time
import subprocess
import json
import logging
from datetime import datetime

def get_nvidia_gpu_stats():
    """Get NVIDIA GPU statistics using nvidia-smi."""
    try:
        # Run nvidia-smi with JSON output
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_stats = []
            
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpu_stats.append({
                        'gpu_id': i,
                        'name': parts[4],
                        'utilization_percent': float(parts[0]) if parts[0] != 'N/A' else 0,
                        'memory_used_mb': float(parts[1]) if parts[1] != 'N/A' else 0,
                        'memory_total_mb': float(parts[2]) if parts[2] != 'N/A' else 0,
                        'temperature_c': float(parts[3]) if parts[3] != 'N/A' else 0
                    })
            
            return gpu_stats
        else:
            print(f"nvidia-smi error: {result.stderr}")
            return []
            
    except subprocess.TimeoutExpired:
        print("nvidia-smi timeout")
        return []
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return []

def get_pytorch_gpu_stats():
    """Get PyTorch GPU memory statistics."""
    try:
        import torch
        if torch.cuda.is_available():
            stats = []
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**2   # MB
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**2  # MB
                
                stats.append({
                    'gpu_id': i,
                    'pytorch_allocated_mb': memory_allocated,
                    'pytorch_reserved_mb': memory_reserved,
                    'pytorch_total_mb': memory_total,
                    'pytorch_utilization_percent': (memory_allocated / memory_total) * 100
                })
            return stats
        return []
    except ImportError:
        return []
    except Exception as e:
        print(f"Error getting PyTorch GPU stats: {e}")
        return []

def monitor_gpu_usage(duration_minutes=60, interval_seconds=5):
    """Monitor GPU usage for specified duration."""
    
    print("🔍 GPU USAGE MONITOR")
    print("=" * 80)
    print(f"Monitoring for {duration_minutes} minutes, updating every {interval_seconds} seconds")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    max_utilization = 0
    avg_utilization = 0
    sample_count = 0
    
    try:
        while time.time() < end_time:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Get NVIDIA GPU stats
            nvidia_stats = get_nvidia_gpu_stats()
            pytorch_stats = get_pytorch_gpu_stats()
            
            if nvidia_stats:
                for gpu in nvidia_stats:
                    util = gpu['utilization_percent']
                    memory_used = gpu['memory_used_mb']
                    memory_total = gpu['memory_total_mb']
                    memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
                    temp = gpu['temperature_c']
                    
                    # Update statistics
                    max_utilization = max(max_utilization, util)
                    avg_utilization = ((avg_utilization * sample_count) + util) / (sample_count + 1)
                    sample_count += 1
                    
                    print(f"[{timestamp}] GPU {gpu['gpu_id']} ({gpu['name']}):")
                    print(f"  🎯 Utilization: {util:5.1f}% (Max: {max_utilization:5.1f}%, Avg: {avg_utilization:5.1f}%)")
                    print(f"  💾 Memory: {memory_used:7.0f}/{memory_total:7.0f} MB ({memory_percent:5.1f}%)")
                    print(f"  🌡️  Temperature: {temp:4.0f}°C")
                    
                    # PyTorch memory info if available
                    if pytorch_stats and gpu['gpu_id'] < len(pytorch_stats):
                        pt_stats = pytorch_stats[gpu['gpu_id']]
                        print(f"  🔥 PyTorch Allocated: {pt_stats['pytorch_allocated_mb']:7.0f} MB "
                              f"({pt_stats['pytorch_utilization_percent']:5.1f}%)")
                    
                    # Performance indicators
                    if util >= 90:
                        print("  ✅ EXCELLENT GPU utilization!")
                    elif util >= 70:
                        print("  🟡 Good GPU utilization")
                    elif util >= 50:
                        print("  🟠 Moderate GPU utilization")
                    else:
                        print("  🔴 Low GPU utilization - consider optimization")
                    
                    print()
            else:
                print(f"[{timestamp}] ❌ No GPU statistics available")
                print()
            
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    
    print("\n📊 MONITORING SUMMARY:")
    print(f"   Duration: {(time.time() - start_time)/60:.1f} minutes")
    print(f"   Samples: {sample_count}")
    print(f"   Max GPU Utilization: {max_utilization:.1f}%")
    print(f"   Average GPU Utilization: {avg_utilization:.1f}%")
    
    if max_utilization >= 90:
        print("   🎉 Excellent GPU utilization achieved!")
    elif max_utilization >= 70:
        print("   👍 Good GPU utilization")
    else:
        print("   💡 Consider GPU optimization strategies")

def check_current_gpu_status():
    """Quick check of current GPU status."""
    print("🔍 CURRENT GPU STATUS")
    print("=" * 50)
    
    nvidia_stats = get_nvidia_gpu_stats()
    pytorch_stats = get_pytorch_gpu_stats()
    
    if nvidia_stats:
        for gpu in nvidia_stats:
            print(f"GPU {gpu['gpu_id']}: {gpu['name']}")
            print(f"  Utilization: {gpu['utilization_percent']:.1f}%")
            print(f"  Memory: {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB")
            print(f"  Temperature: {gpu['temperature_c']:.0f}°C")
            
            if pytorch_stats and gpu['gpu_id'] < len(pytorch_stats):
                pt_stats = pytorch_stats[gpu['gpu_id']]
                print(f"  PyTorch Memory: {pt_stats['pytorch_allocated_mb']:.0f} MB allocated")
            print()
    else:
        print("❌ No GPU information available")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "status":
            check_current_gpu_status()
        elif sys.argv[1] == "monitor":
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            interval = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            monitor_gpu_usage(duration, interval)
        else:
            print("Usage:")
            print("  py monitor_gpu_usage.py status          - Quick status check")
            print("  py monitor_gpu_usage.py monitor [min] [sec] - Monitor for duration")
    else:
        check_current_gpu_status()
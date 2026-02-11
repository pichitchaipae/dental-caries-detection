"""
CUDA Verification Script for RTX 4080
=====================================
Run this script to verify your GPU setup before training.
"""

import sys

def verify_cuda():
    """Verify CUDA installation and RTX 4080 detection."""
    print("=" * 60)
    print("CUDA VERIFICATION FOR RTX 4080")
    print("=" * 60)
    
    # Check PyTorch import
    try:
        import torch
        print(f"✓ PyTorch Version: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed!")
        print("  Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        return False
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("✗ CUDA is NOT available!")
        print("\n  Possible causes:")
        print("  1. CPU-only PyTorch installed")
        print("  2. NVIDIA drivers not installed")
        print("  3. CUDA toolkit missing")
        print("\n  Fix: Run these commands:")
        print("  pip uninstall torch torchvision -y")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        return False
    
    print(f"✓ CUDA Available: True")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    
    # Check GPU
    gpu_count = torch.cuda.device_count()
    print(f"✓ GPU Count: {gpu_count}")
    
    if gpu_count == 0:
        print("✗ No GPU detected!")
        return False
    
    # GPU details
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\n  GPU {i}: {props.name}")
        print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
        print(f"    Multi-Processors: {props.multi_processor_count}")
    
    # Verify CUDA operations work
    print("\n  Testing CUDA operations...")
    try:
        # Create tensor on GPU
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        print(f"    ✓ Matrix multiplication on GPU: Success")
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated(0) / 1e6
        cached = torch.cuda.memory_reserved(0) / 1e6
        print(f"    ✓ Memory allocated: {allocated:.1f} MB")
        print(f"    ✓ Memory cached: {cached:.1f} MB")
        
        # Cleanup
        del x, y, z
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"    ✗ CUDA operation failed: {e}")
        return False
    
    # Check ultralytics
    print("\n  Checking Ultralytics...")
    try:
        from ultralytics import YOLO
        print(f"    ✓ Ultralytics imported successfully")
    except ImportError as e:
        print(f"    ✗ Ultralytics import failed: {e}")
        print("    Run: pip install ultralytics")
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL CHECKS PASSED - RTX 4080 READY FOR TRAINING!")
    print("=" * 60)
    
    # Recommendations
    print("\n📋 Recommended Training Settings for RTX 4080 (12GB):")
    print("   • batch=4 (safe) or batch=6 (aggressive)")
    print("   • workers=16 (for Ryzen 9)")
    print("   • amp=True (Mixed Precision)")
    print("   • imgsz=1536")
    print("   • cache='ram' (with 32GB RAM)")
    
    return True


def estimate_memory_usage(batch_size=4, imgsz=1536):
    """Estimate GPU memory usage for given batch size."""
    import torch
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print(f"\n📊 Memory Estimation (batch={batch_size}, imgsz={imgsz}):")
    
    # Rough estimation for YOLO11m-seg
    # Model: ~2.5GB, Per image: ~2GB with gradients
    model_memory = 2.5  # GB
    per_image_memory = 2.0  # GB (with gradients at 1536px)
    
    estimated_total = model_memory + (batch_size * per_image_memory)
    available = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"   Model memory: ~{model_memory:.1f} GB")
    print(f"   Per image: ~{per_image_memory:.1f} GB")
    print(f"   Estimated total: ~{estimated_total:.1f} GB")
    print(f"   Available VRAM: {available:.1f} GB")
    
    if estimated_total > available * 0.9:
        print(f"   ⚠️  WARNING: May cause OOM! Try batch={batch_size-2}")
    else:
        headroom = available - estimated_total
        print(f"   ✓ Headroom: {headroom:.1f} GB")


if __name__ == "__main__":
    success = verify_cuda()
    
    if success:
        estimate_memory_usage(batch_size=4, imgsz=1536)
        print("\n")
        estimate_memory_usage(batch_size=6, imgsz=1536)
    
    sys.exit(0 if success else 1)

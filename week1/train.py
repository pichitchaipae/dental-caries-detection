"""
Dental X-ray Tooth Segmentation Training Script
================================================
YOLO11-seg Instance Segmentation with FDI Notation

Author: Senior Computer Vision Engineer
Project: Phase 1 - Dental Analysis
Model: YOLOv11m-seg (Medium variant for detailed segmentation)

Hardware Optimized For:
- GPU: NVIDIA GeForce RTX 4080 (12GB VRAM)
- CPU: AMD Ryzen 9 7945HX (16 cores)
- RAM: 32GB DDR5
- Storage: 1TB NVMe SSD

Technical Notes:
- Input: Panoramic dental X-rays (3036x1536 px, ~2:1 aspect ratio)
- Output: Instance segmentation masks with 32 FDI tooth classes
- Critical: Horizontal flip DISABLED (left/right teeth have different IDs)
"""

import os
import torch
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


def get_device_info():
    """Display available compute device information and verify CUDA setup."""
    print("\n" + "=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ CUDA Available: {gpu_count} GPU(s) detected")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ PyTorch Version: {torch.__version__}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            gpu_memory = props.total_memory / 1e9
            print(f"  GPU {i}: {gpu_name}")
            print(f"         Memory: {gpu_memory:.1f} GB")
            print(f"         Compute Capability: {props.major}.{props.minor}")
        device = 0  # Use first GPU (RTX 4080)
        
        # Verify CUDA is working
        test_tensor = torch.zeros(1).cuda()
        print(f"✓ CUDA Test: Tensor created on {test_tensor.device}")
    else:
        print("✗ CUDA not available!")
        print("  Please run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        device = "cpu"
    
    print("=" * 60 + "\n")
    return device


# =============================================================================
# HARDWARE-OPTIMIZED CONFIGURATION FOR RTX 4080 + RYZEN 9
# =============================================================================

class HardwareConfig:
    """
    Optimized settings for:
    - GPU: NVIDIA GeForce RTX 4080 (12GB VRAM)
    - CPU: AMD Ryzen 9 7945HX (16 cores / 32 threads)
    - RAM: 32GB DDR5
    """
    # GPU Settings
    DEVICE = 0                    # Force GPU 0 (RTX 4080)
    BATCH_SIZE = 4                # Safe for 12GB VRAM at 1536px
                                  # Can try 6 if no OOM errors
    
    # CPU/DataLoader Settings
    # NOTE: Reduced to 4 to avoid Windows paging file errors
    # Windows multiprocessing spawns full Python processes per worker
    WORKERS = 4                   # Safe for Windows + large images
    
    # Memory Optimization
    AMP = True                    # Mixed Precision (FP16) - 2x speedup
    CACHE = False                 # Disabled to avoid memory issues
    
    # Performance Flags
    PIN_MEMORY = True             # Faster CPU->GPU transfer
    PERSISTENT_WORKERS = True     # Keep workers alive between epochs


def calculate_batch_size(imgsz: int = 1536, gpu_memory_gb: float = None):
    """
    Calculate optimal batch size based on image size and GPU memory.
    
    RTX 4080 (12GB) at 1536px images:
    - YOLO11m-seg: batch=4 (safe), batch=6 (aggressive)
    - With AMP enabled, can push slightly higher
    
    Memory estimation for YOLO11m-seg at 1536px:
    - ~2.5GB base model
    - ~2GB per image in batch (with gradients)
    - batch=4: ~10GB used, ~2GB headroom ✓
    - batch=6: ~14GB needed, may OOM ✗
    """
    if torch.cuda.is_available() and gpu_memory_gb is None:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    if gpu_memory_gb is None:
        return 2  # Conservative default for CPU
    
    # RTX 4080 specific (12-13GB)
    if 12 <= gpu_memory_gb < 16:
        return 4  # Safe and stable
    # RTX 4090 / A6000 (24GB)
    elif gpu_memory_gb >= 24:
        return 8
    # RTX 4080 16GB variant
    elif gpu_memory_gb >= 16:
        return 6
    # RTX 3070/3080 (8-10GB)
    elif gpu_memory_gb >= 8:
        return 2
    else:
        return 1


def train_dental_segmentation():
    """
    Main training function for dental tooth segmentation.
    
    Training Configuration Rationale:
    ---------------------------------
    1. imgsz=1536: Maintains detail for small tooth structures
    2. rect=True: Optimizes for 2:1 panoramic aspect ratio
    3. fliplr=0.0: CRITICAL - prevents left/right tooth ID confusion
    4. degrees=10.0: Allows natural head tilt variation
    5. scale=0.5: Handles zoom variation in X-rays
    """
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_YAML = PROJECT_ROOT / "data.yaml"
    OUTPUT_DIR = PROJECT_ROOT / "runs"
    
    # Model
    BASE_MODEL = "yolo11m-seg.pt"  # Medium model for better detail
    
    # Training Parameters
    EPOCHS = 150
    IMGSZ = 1536  # High resolution for dental detail
    PATIENCE = 30  # Early stopping patience
    
    # =======================================================================
    # HARDWARE-OPTIMIZED SETTINGS (RTX 4080 + Ryzen 9)
    # =======================================================================
    device = get_device_info()
    
    # Override with hardware-specific config
    DEVICE = HardwareConfig.DEVICE          # Force GPU 0
    BATCH_SIZE = HardwareConfig.BATCH_SIZE  # 4 for 12GB VRAM
    WORKERS = HardwareConfig.WORKERS        # 16 for Ryzen 9
    AMP = HardwareConfig.AMP                # Mixed precision ON
    CACHE = HardwareConfig.CACHE            # RAM caching
    
    batch_size = BATCH_SIZE
    
    # Experiment naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"dental_yolo11m_seg_{timestamp}"
    
    print("=" * 60)
    print("DENTAL TOOTH SEGMENTATION - TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Base Model:      {BASE_MODEL}")
    print(f"Data Config:     {DATA_YAML}")
    print(f"Output Dir:      {OUTPUT_DIR}")
    print(f"Experiment:      {experiment_name}")
    print(f"Image Size:      {IMGSZ}px")
    print(f"Batch Size:      {batch_size}")
    print(f"Epochs:          {EPOCHS}")
    print(f"Device:          GPU {DEVICE} (RTX 4080)")
    print(f"Workers:         {WORKERS} (Ryzen 9 optimized)")
    print(f"Mixed Precision: {AMP}")
    print(f"Cache Mode:      {CACHE}")
    print("=" * 60 + "\n")
    
    # =========================================================================
    # VERIFY DATA CONFIGURATION
    # =========================================================================
    
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Data configuration not found: {DATA_YAML}\n"
            "Please ensure data.yaml exists with correct paths."
        )
    
    # =========================================================================
    # INITIALIZE MODEL
    # =========================================================================
    
    print(f"Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)
    
    # =========================================================================
    # TRAIN MODEL
    # =========================================================================
    
    print("\nStarting training...\n")
    
    results = model.train(
        # === Data Configuration ===
        data=str(DATA_YAML),
        
        # === Training Parameters ===
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=batch_size,
        patience=PATIENCE,
        
        # === Rectangular Training ===
        # CRITICAL: Enable for panoramic images (2:1 aspect ratio)
        rect=True,
        
        # === Augmentation Settings ===
        # CRITICAL: Disable horizontal flip - left/right teeth have different FDI IDs
        fliplr=0.0,      # NO horizontal flip (18 ≠ 28, 11 ≠ 21, etc.)
        flipud=0.0,      # NO vertical flip (anatomically incorrect)
        
        # Geometric augmentations (safe for dental X-rays)
        degrees=10.0,    # Slight rotation (natural head positioning variation)
        scale=0.5,       # Scale augmentation (zoom variation)
        translate=0.1,   # Minor translation
        shear=0.0,       # No shear (distorts tooth shapes)
        perspective=0.0, # No perspective (distorts anatomy)
        
        # Color augmentations (conservative for X-rays)
        hsv_h=0.0,       # No hue shift (grayscale X-rays)
        hsv_s=0.0,       # No saturation shift
        hsv_v=0.2,       # Slight brightness variation (exposure differences)
        
        # Mosaic and mixup (disabled for medical imaging)
        mosaic=0.0,      # Disabled - maintains anatomical context
        mixup=0.0,       # Disabled - prevents unrealistic overlays
        copy_paste=0.0,  # Disabled - maintains anatomical integrity
        
        # === Optimization ===
        optimizer="AdamW",
        lr0=0.001,       # Initial learning rate
        lrf=0.01,        # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # === Loss Weights ===
        box=7.5,         # Box loss weight
        cls=0.5,         # Classification loss weight
        dfl=1.5,         # Distribution focal loss weight
        
        # === Output Configuration ===
        project=str(OUTPUT_DIR),
        name=experiment_name,
        exist_ok=False,
        
        # === Logging & Saving ===
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,      # Generate training plots
        
        # === Performance (RTX 4080 + Ryzen 9 Optimized) ===
        device=DEVICE,   # Force GPU 0 (RTX 4080)
        workers=WORKERS, # 16 workers for Ryzen 9 (prevents data bottleneck)
        amp=AMP,         # Mixed Precision FP16 (2x speedup on RTX 4080)
        cache=CACHE,     # Cache in RAM (32GB available)
        
        # === Validation ===
        val=True,
        split="val",
        
        # === Reproducibility ===
        seed=42,
        deterministic=True,
        
        # === Verbosity ===
        verbose=True,
    )
    
    # =========================================================================
    # POST-TRAINING SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Best model path
    best_model_path = OUTPUT_DIR / experiment_name / "weights" / "best.pt"
    last_model_path = OUTPUT_DIR / experiment_name / "weights" / "last.pt"
    
    print(f"Best Model:  {best_model_path}")
    print(f"Last Model:  {last_model_path}")
    print(f"Results:     {OUTPUT_DIR / experiment_name}")
    
    # =========================================================================
    # VALIDATION ON BEST MODEL
    # =========================================================================
    
    print("\nRunning final validation with best weights...")
    
    best_model = YOLO(str(best_model_path))
    val_results = best_model.val(
        data=str(DATA_YAML),
        imgsz=IMGSZ,
        batch=batch_size,
        split="val",
        plots=True,
        save_json=True,  # Save COCO format results
    )
    
    print("\n" + "=" * 60)
    print("VALIDATION METRICS")
    print("=" * 60)
    print(f"mAP50 (Box):     {val_results.box.map50:.4f}")
    print(f"mAP50-95 (Box):  {val_results.box.map:.4f}")
    print(f"mAP50 (Mask):    {val_results.seg.map50:.4f}")
    print(f"mAP50-95 (Mask): {val_results.seg.map:.4f}")
    print("=" * 60)
    
    return results, val_results


def export_model(model_path: str, formats: list = None):
    """
    Export trained model to various formats for deployment.
    
    Args:
        model_path: Path to the trained .pt model
        formats: List of export formats (default: ONNX, TensorRT)
    """
    if formats is None:
        formats = ["onnx", "engine"]  # ONNX and TensorRT
    
    model = YOLO(model_path)
    
    for fmt in formats:
        print(f"\nExporting to {fmt.upper()}...")
        model.export(
            format=fmt,
            imgsz=1536,
            half=True,  # FP16 for efficiency
            simplify=True,  # Simplify ONNX graph
        )
        print(f"✓ {fmt.upper()} export complete")


if __name__ == "__main__":
    # Run training
    train_results, val_results = train_dental_segmentation()
    
    # Optional: Export best model (uncomment when ready for deployment)
    # export_model("runs/dental_yolo11m_seg_*/weights/best.pt", formats=["onnx"])

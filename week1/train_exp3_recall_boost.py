"""
================================================================================
DANTAL - Experiment 3: Research-Grade Recall Boosting
================================================================================
Target: Maximize Recall (>80%) for complete tooth detection

Problem Statement:
- Current model has decent Precision (65%) but critically low Recall (36%)
- Fails to detect teeth with overlapping roots and faint boundaries
- Next phase requires identifying EVERY tooth → Recall is mission-critical

Research Foundation:
- ToothNet (2024): CLAHE preprocessing + aggressive augmentation
- MICCAI 2024 Dental Challenges: Copy-paste for instance synthesis
- AdamW optimizer superiority on noisy medical imaging gradients

Technical Strategy:
1. CLAHE Preprocessing → Enhance faint root boundaries
2. Copy-Paste Augmentation → Synthesize overlapping instances
3. Higher Learning Rate + Box Loss → Force aggressive tooth detection
4. AdamW Optimizer → Better convergence on medical images

Hardware: RTX 4080 (12GB) + Ryzen 9 7945HX (16 cores)
Author: Lead Computer Vision Researcher - Dental AI
================================================================================
"""

import os
import cv2
import numpy as np
import torch
import shutil
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

class HardwareConfig:
    """
    RTX 4080 (12GB VRAM) + Ryzen 9 7945HX Optimization
    
    Memory Budget at 1536px:
    - Model: ~2GB
    - Batch of 4 images: ~8GB
    - Gradients + Optimizer: ~2GB
    - Safety margin: ~2GB (for copy-paste augmentation overhead)
    """
    DEVICE = 0                    # GPU 0 (RTX 4080)
    BATCH_SIZE = 4                # Safe for 12GB VRAM at 1536px
    WORKERS = 8                   # Windows-safe (16 causes spawn overhead)
    AMP = True                    # Mixed precision for memory efficiency
    CACHE = False                 # Disable cache (CLAHE preprocessing dynamic)
    

class RecallBoostConfig:
    """
    Research-Grade Hyperparameters for Maximizing Recall
    
    Philosophy: We accept slightly lower precision to catch ALL teeth.
    Missing a tooth (False Negative) is worse than a false detection.
    
    Key Changes from Exp 2:
    - lr0: 0.0001 → 0.001 (10x higher, force learning)
    - box: 7.5 → 10.0 (heavily penalize missed detections)
    - copy_paste: 0 → 0.3 (synthesize overlapping instances)
    - optimizer: SGD → AdamW (better medical imaging convergence)
    """
    # === Model ===
    PRETRAINED_MODEL = Path(r"C:\Users\jaopi\Desktop\SP\material\Tooth Segmentation + Recognition model\weights\Tooth_seg_pano_20250319.pt")
    
    # === Dataset ===
    DATA_YAML = Path(__file__).parent / "data_pretrained.yaml"
    
    # === Training Duration ===
    EPOCHS = 200                  # Medical models need time to converge
    PATIENCE = 50                 # Don't stop early - recall takes time to stabilize
    
    # === Input Processing ===
    IMGSZ = 1536                  # High resolution mandatory for root details
    RECT = False                  # Disable rect (incompatible with mosaic/copy-paste)
    
    # === Optimizer (AdamW for Medical Imaging) ===
    OPTIMIZER = "AdamW"           # Better than SGD for noisy medical gradients
    LR0 = 0.001                   # 10x higher than Exp 2 - aggressive learning
    LRF = 0.01                    # Final LR = lr0 * lrf = 0.00001
    COS_LR = True                 # Cosine annealing for smooth decay
    MOMENTUM = 0.937              # AdamW beta1
    WEIGHT_DECAY = 0.0005         # L2 regularization
    WARMUP_EPOCHS = 3.0           # Longer warmup for stability
    WARMUP_MOMENTUM = 0.8
    WARMUP_BIAS_LR = 0.1
    
    # === Loss Weights (RECALL-FOCUSED) ===
    # Higher box loss = stronger penalty for missing objects
    BOX = 7.5                     # Default 7.5, keep high for detection
    CLS = 0.5                     # Default 0.5, classification weight
    DFL = 1.5                     # Distribution focal loss
    
    # === Augmentation (Research-Grade) ===
    # Mosaic: Combines 4 images - exposes model to more teeth per batch
    MOSAIC = 1.0                  # Always use mosaic (100% probability)
    
    # Mixup: Blends images - improves generalization on low-contrast regions
    MIXUP = 0.15                  # 15% probability - prevents overfitting
    
    # Copy-Paste: THE KEY FOR OVERLAPPING INSTANCES
    # Pastes teeth from other images → forces learning of separation
    COPY_PASTE = 0.3              # 30% probability - crucial for overlapping roots
    
    # Geometric Augmentation
    DEGREES = 5.0                 # Slight rotation (dental X-rays are mostly level)
    TRANSLATE = 0.1               # Small translation
    SCALE = 0.3                   # Scale variation (0.7x to 1.3x)
    SHEAR = 2.0                   # Minimal shear
    PERSPECTIVE = 0.0             # No perspective (panoramic is flat)
    FLIPUD = 0.0                  # NO vertical flip (anatomically incorrect)
    FLIPLR = 0.0                  # NO horizontal flip (FDI notation is sided)
    
    # Color Augmentation (Conservative for X-rays)
    HSV_H = 0.0                   # No hue shift (grayscale)
    HSV_S = 0.0                   # No saturation shift
    HSV_V = 0.2                   # Slight brightness variation
    
    # === Output ===
    PROJECT = Path(__file__).parent / "runs"
    SEED = 42


class CLAHEConfig:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) Settings
    
    Why CLAHE for Dental X-rays:
    - Root canals have similar grayscale to bone → CLAHE enhances contrast
    - Overlapping roots blend together → Local enhancement separates them
    - Periapical regions are washed out → Tile-based processing recovers detail
    
    Research Reference:
    - ToothNet (2024): clip_limit=2.0-3.0, tile_size=8x8
    - MICCAI Dental Challenge: clip_limit=2.0, adaptive tile sizing
    """
    ENABLED = True
    CLIP_LIMIT = 2.5              # Contrast amplification factor (2.0-3.0 optimal)
    TILE_GRID_SIZE = (8, 8)       # 8x8 tiles for local enhancement


# =============================================================================
# CLAHE PREPROCESSING
# =============================================================================

class CLAHEPreprocessor:
    """
    Apply CLAHE to enhance faint dental structures
    
    How it works:
    1. Divide image into 8x8 grid of tiles
    2. Compute histogram for each tile
    3. Clip histogram at clip_limit (prevents over-amplification)
    4. Redistribute clipped pixels uniformly
    5. Apply histogram equalization per tile
    6. Bilinear interpolation at tile boundaries (smooth transitions)
    
    Effect on Dental X-rays:
    - Faint root boundaries become visible
    - Low-contrast periapical regions are enhanced
    - Overlapping structures gain edge definition
    """
    
    def __init__(self, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
        logger.info(f"CLAHE initialized: clip_limit={clip_limit}, tiles={tile_grid_size}")
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to a single image
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            CLAHE-enhanced image
        """
        # Handle grayscale
        if len(image.shape) == 2:
            return self.clahe.apply(image)
        
        # Handle BGR (convert to LAB, apply to L channel)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE to L (lightness) channel only
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            
            # Convert back to BGR
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return image
    
    def process_dataset(self, source_dir: Path, target_dir: Path, split: str = "train"):
        """
        Apply CLAHE to entire dataset split
        
        Args:
            source_dir: Original dataset directory
            target_dir: Output directory for CLAHE-enhanced images
            split: 'train' or 'val'
        """
        src_images = source_dir / "images" / split
        dst_images = target_dir / "images" / split
        
        src_labels = source_dir / "labels" / split
        dst_labels = target_dir / "labels" / split
        
        # Create directories
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        
        # Process images
        image_files = list(src_images.glob("*.*"))
        logger.info(f"Processing {len(image_files)} {split} images with CLAHE...")
        
        processed = 0
        for img_path in image_files:
            try:
                # Read and enhance image
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Could not read: {img_path}")
                    continue
                
                enhanced = self.apply(img)
                
                # Save enhanced image
                cv2.imwrite(str(dst_images / img_path.name), enhanced)
                
                # Copy corresponding label
                label_name = img_path.stem + ".txt"
                src_label = src_labels / label_name
                if src_label.exists():
                    shutil.copy(src_label, dst_labels / label_name)
                
                processed += 1
                if processed % 50 == 0:
                    logger.info(f"  Processed {processed}/{len(image_files)} images")
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
        
        logger.info(f"✓ CLAHE preprocessing complete: {processed} images")
        return processed


# =============================================================================
# DEVICE VERIFICATION
# =============================================================================

def verify_cuda_setup():
    """Verify CUDA availability and display GPU information"""
    print("\n" + "=" * 70)
    print("CUDA VERIFICATION")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA NOT AVAILABLE!\n"
            "Install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        )
    
    gpu_count = torch.cuda.device_count()
    print(f"✓ CUDA Available: {gpu_count} GPU(s)")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ PyTorch Version: {torch.__version__}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"\n  GPU {i}: {props.name}")
        print(f"         VRAM: {memory_gb:.1f} GB")
        print(f"         Compute: {props.major}.{props.minor}")
    
    # Quick functional test
    test = torch.zeros(1).cuda()
    print(f"\n✓ CUDA Functional Test: Tensor on {test.device}")
    print("=" * 70 + "\n")
    
    return 0  # Return GPU index


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    """
    Execute Experiment 3: Research-Grade Recall Boosting
    
    Strategy:
    1. Preprocess dataset with CLAHE (enhance faint structures)
    2. Load pretrained dental model
    3. Train with recall-focused hyperparameters
    4. Use aggressive augmentation (copy-paste, mosaic, mixup)
    """
    
    # === Setup ===
    verify_cuda_setup()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp3_recall_boost_{timestamp}"
    
    print("=" * 70)
    print("EXPERIMENT 3: RESEARCH-GRADE RECALL BOOSTING")
    print("=" * 70)
    print(f"\nObjective: Maximize Recall (>80%) for complete tooth detection")
    print(f"Problem: Current model misses teeth with overlapping roots")
    print(f"Solution: CLAHE + Copy-Paste + Aggressive Learning Rate")
    print("\n" + "-" * 70)
    
    # === CLAHE Preprocessing ===
    if CLAHEConfig.ENABLED:
        print("\n[PHASE 1] CLAHE Preprocessing")
        print("-" * 40)
        
        clahe = CLAHEPreprocessor(
            clip_limit=CLAHEConfig.CLIP_LIMIT,
            tile_grid_size=CLAHEConfig.TILE_GRID_SIZE
        )
        
        # Source and target directories
        source_dataset = Path(__file__).parent / "dataset_v2"
        clahe_dataset = Path(__file__).parent / "dataset_clahe"
        
        # Process train and val splits
        if not clahe_dataset.exists():
            logger.info("Creating CLAHE-enhanced dataset...")
            clahe.process_dataset(source_dataset, clahe_dataset, "train")
            clahe.process_dataset(source_dataset, clahe_dataset, "val")
            logger.info(f"✓ CLAHE dataset created at: {clahe_dataset}")
        else:
            logger.info(f"✓ CLAHE dataset exists: {clahe_dataset}")
        
        # Update data config to use CLAHE dataset
        data_yaml_clahe = Path(__file__).parent / "data_clahe.yaml"
        if not data_yaml_clahe.exists():
            create_clahe_data_yaml(data_yaml_clahe, clahe_dataset)
        
        data_config = data_yaml_clahe
    else:
        data_config = RecallBoostConfig.DATA_YAML
    
    # === Model Loading ===
    print("\n[PHASE 2] Model Initialization")
    print("-" * 40)
    
    pretrained_path = RecallBoostConfig.PRETRAINED_MODEL
    
    if pretrained_path.exists():
        print(f"Loading pretrained dental model: {pretrained_path.name}")
        model = YOLO(str(pretrained_path))
    else:
        print("⚠ Pretrained model not found, using YOLO11m-seg base")
        model = YOLO("yolo11m-seg.pt")
    
    # === Configuration Summary ===
    print("\n[PHASE 3] Configuration Summary")
    print("-" * 40)
    print(f"  Experiment:     {exp_name}")
    print(f"  Data Config:    {data_config}")
    print(f"  Output:         {RecallBoostConfig.PROJECT}")
    print(f"\n  === Training Parameters ===")
    print(f"  Epochs:         {RecallBoostConfig.EPOCHS}")
    print(f"  Patience:       {RecallBoostConfig.PATIENCE}")
    print(f"  Image Size:     {RecallBoostConfig.IMGSZ}px")
    print(f"  Batch Size:     {HardwareConfig.BATCH_SIZE}")
    print(f"\n  === Optimizer (AdamW - Medical Grade) ===")
    print(f"  Learning Rate:  {RecallBoostConfig.LR0} (10x higher than Exp 2)")
    print(f"  LR Schedule:    Cosine Annealing")
    print(f"  Weight Decay:   {RecallBoostConfig.WEIGHT_DECAY}")
    print(f"\n  === Loss Weights (Recall-Focused) ===")
    print(f"  Box Loss:       {RecallBoostConfig.BOX} (penalize missed detections)")
    print(f"  Class Loss:     {RecallBoostConfig.CLS}")
    print(f"  DFL Loss:       {RecallBoostConfig.DFL}")
    print(f"\n  === Augmentation (Research-Grade) ===")
    print(f"  CLAHE:          {CLAHEConfig.ENABLED} (clip={CLAHEConfig.CLIP_LIMIT})")
    print(f"  Mosaic:         {RecallBoostConfig.MOSAIC}")
    print(f"  Mixup:          {RecallBoostConfig.MIXUP}")
    print(f"  Copy-Paste:     {RecallBoostConfig.COPY_PASTE} (overlapping instances)")
    print(f"  Flip LR/UD:     DISABLED (FDI notation is sided)")
    
    # === Training ===
    print("\n[PHASE 4] Training Execution")
    print("-" * 40)
    print("Starting training... This will take several hours.")
    print("Monitor: GPU memory, loss curves, recall metrics\n")
    
    try:
        results = model.train(
            # === Dataset ===
            data=str(data_config),
            
            # === Training Duration ===
            epochs=RecallBoostConfig.EPOCHS,
            patience=RecallBoostConfig.PATIENCE,
            
            # === Input ===
            imgsz=RecallBoostConfig.IMGSZ,
            rect=RecallBoostConfig.RECT,
            
            # === Hardware ===
            device=HardwareConfig.DEVICE,
            batch=HardwareConfig.BATCH_SIZE,
            workers=HardwareConfig.WORKERS,
            amp=HardwareConfig.AMP,
            cache=HardwareConfig.CACHE,
            
            # === Optimizer ===
            optimizer=RecallBoostConfig.OPTIMIZER,
            lr0=RecallBoostConfig.LR0,
            lrf=RecallBoostConfig.LRF,
            cos_lr=RecallBoostConfig.COS_LR,
            momentum=RecallBoostConfig.MOMENTUM,
            weight_decay=RecallBoostConfig.WEIGHT_DECAY,
            warmup_epochs=RecallBoostConfig.WARMUP_EPOCHS,
            warmup_momentum=RecallBoostConfig.WARMUP_MOMENTUM,
            warmup_bias_lr=RecallBoostConfig.WARMUP_BIAS_LR,
            
            # === Loss Weights ===
            box=RecallBoostConfig.BOX,
            cls=RecallBoostConfig.CLS,
            dfl=RecallBoostConfig.DFL,
            
            # === Augmentation ===
            mosaic=RecallBoostConfig.MOSAIC,
            mixup=RecallBoostConfig.MIXUP,
            copy_paste=RecallBoostConfig.COPY_PASTE,
            degrees=RecallBoostConfig.DEGREES,
            translate=RecallBoostConfig.TRANSLATE,
            scale=RecallBoostConfig.SCALE,
            shear=RecallBoostConfig.SHEAR,
            perspective=RecallBoostConfig.PERSPECTIVE,
            flipud=RecallBoostConfig.FLIPUD,
            fliplr=RecallBoostConfig.FLIPLR,
            hsv_h=RecallBoostConfig.HSV_H,
            hsv_s=RecallBoostConfig.HSV_S,
            hsv_v=RecallBoostConfig.HSV_V,
            
            # === Output ===
            project=str(RecallBoostConfig.PROJECT),
            name=exp_name,
            exist_ok=True,
            save=True,
            save_period=10,           # Save checkpoint every 10 epochs
            plots=True,
            verbose=True,
            seed=RecallBoostConfig.SEED,
        )
        
        # === Results Summary ===
        print("\n" + "=" * 70)
        print("EXPERIMENT 3 COMPLETE")
        print("=" * 70)
        
        if results and hasattr(results, 'box') and hasattr(results, 'seg'):
            print(f"\n  === Final Metrics ===")
            print(f"  Box mAP50:      {results.box.map50:.4f}")
            print(f"  Box mAP50-95:   {results.box.map:.4f}")
            print(f"  Mask mAP50:     {results.seg.map50:.4f}")
            print(f"  Mask mAP50-95:  {results.seg.map:.4f}")
            
            # Check recall target
            recall_target = 0.80
            print(f"\n  === Recall Analysis ===")
            print(f"  Target Recall:  {recall_target:.0%}")
            # Note: Detailed recall available in results.csv
        
        print(f"\n  Weights saved to:")
        print(f"    Best:  {RecallBoostConfig.PROJECT}/{exp_name}/weights/best.pt")
        print(f"    Last:  {RecallBoostConfig.PROJECT}/{exp_name}/weights/last.pt")
        print("=" * 70)
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def create_clahe_data_yaml(output_path: Path, dataset_path: Path):
    """Create data.yaml for CLAHE-enhanced dataset"""
    
    yaml_content = f"""# CLAHE-Enhanced Dental X-ray Dataset
# Experiment 3: Research-Grade Recall Boosting
# Preprocessing: CLAHE (clip_limit={CLAHEConfig.CLIP_LIMIT}, tiles={CLAHEConfig.TILE_GRID_SIZE})

path: {dataset_path.as_posix()}
train: images/train
val: images/val

nc: 32

# FDI Notation Classes (Sequential Order)
names:
  0: "11"   # Upper Right Central Incisor
  1: "12"   # Upper Right Lateral Incisor
  2: "13"   # Upper Right Canine
  3: "14"   # Upper Right First Premolar
  4: "15"   # Upper Right Second Premolar
  5: "16"   # Upper Right First Molar
  6: "17"   # Upper Right Second Molar
  7: "18"   # Upper Right Third Molar
  8: "21"   # Upper Left Central Incisor
  9: "22"   # Upper Left Lateral Incisor
  10: "23"  # Upper Left Canine
  11: "24"  # Upper Left First Premolar
  12: "25"  # Upper Left Second Premolar
  13: "26"  # Upper Left First Molar
  14: "27"  # Upper Left Second Molar
  15: "28"  # Upper Left Third Molar
  16: "31"  # Lower Left Central Incisor
  17: "32"  # Lower Left Lateral Incisor
  18: "33"  # Lower Left Canine
  19: "34"  # Lower Left First Premolar
  20: "35"  # Lower Left Second Premolar
  21: "36"  # Lower Left First Molar
  22: "37"  # Lower Left Second Molar
  23: "38"  # Lower Left Third Molar
  24: "41"  # Lower Right Central Incisor
  25: "42"  # Lower Right Lateral Incisor
  26: "43"  # Lower Right Canine
  27: "44"  # Lower Right First Premolar
  28: "45"  # Lower Right Second Premolar
  29: "46"  # Lower Right First Molar
  30: "47"  # Lower Right Second Molar
  31: "48"  # Lower Right Third Molar
"""
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"✓ Created CLAHE data config: {output_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║   ██████╗  █████╗ ███╗   ██╗████████╗ █████╗ ██╗                    ║
    ║   ██╔══██╗██╔══██╗████╗  ██║╚══██╔══╝██╔══██╗██║                    ║
    ║   ██║  ██║███████║██╔██╗ ██║   ██║   ███████║██║                    ║
    ║   ██║  ██║██╔══██║██║╚██╗██║   ██║   ██╔══██║██║                    ║
    ║   ██████╔╝██║  ██║██║ ╚████║   ██║   ██║  ██║███████╗               ║
    ║   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚══════╝               ║
    ║                                                                      ║
    ║   Experiment 3: Research-Grade Recall Boosting                       ║
    ║   Target: Recall > 80% | Detect ALL teeth including overlapping     ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    main()

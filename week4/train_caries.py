"""
Dental Caries Detection Model Training Script
==============================================
Trains a YOLOv8s model for 3-class caries surface detection.

Classes:
    0: Occlusal  - Caries on the chewing surface (minority class)
    1: Proximal  - Caries on side surfaces (dominant class)
    2: Lingual   - Caries on tongue-facing surface (empty in dataset)

Notes on Class Imbalance:
    - Proximal (86%) dominates the dataset
    - Occlusal (14%) is the minority class
    - Lingual (0%) has no samples
    
    YOLO handles this through:
    1. Built-in augmentation (mosaic, mixup, hsv shifts)
    2. Focal loss variant in classification head
    3. We use lower conf threshold during validation for better recall

Author: Lead CV Engineer
Date: 2026-01-27
"""

import os
from pathlib import Path
from ultralytics import YOLO


def train_caries_model():
    """Train the 3-class caries detection model."""
    
    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================
    
    # Paths
    SCRIPT_DIR = Path(__file__).parent.resolve()
    DATA_YAML = SCRIPT_DIR / "dataset_3class" / "data.yaml"
    PROJECT_DIR = SCRIPT_DIR / "runs" / "caries_train"
    
    # Model
    BASE_MODEL = "yolov8s.pt"  # Small model - better for medical anomaly detection
    
    # Training hyperparameters
    EPOCHS = 50
    IMAGE_SIZE = 640
    BATCH_SIZE = 16
    
    # Validation settings (lower conf for minority class recall)
    CONF_THRESHOLD = 0.25  # Default is 0.25, good for imbalanced data
    
    # ==========================================================================
    # TRAINING
    # ==========================================================================
    
    print("=" * 60)
    print("DENTAL CARIES DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Base Model:    {BASE_MODEL}")
    print(f"  Data Config:   {DATA_YAML}")
    print(f"  Epochs:        {EPOCHS}")
    print(f"  Image Size:    {IMAGE_SIZE}")
    print(f"  Batch Size:    {BATCH_SIZE}")
    print(f"  Output Dir:    {PROJECT_DIR}")
    print("=" * 60)
    
    # Verify data.yaml exists
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Data config not found: {DATA_YAML}")
    
    # Load model
    print(f"\nLoading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)
    
    # Start training
    print("\nStarting training...")
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=str(PROJECT_DIR),
        name="caries_3class",
        
        # Optimizer settings
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        
        # Augmentation (helps with class imbalance)
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,  # Don't flip vertically for dental images
        
        # Validation
        val=True,
        conf=CONF_THRESHOLD,
        
        # Device
        device=0,  # Use GPU 0
        
        # Saving
        save=True,
        save_period=-1,  # Save only best and last
        
        # Logging
        verbose=True,
        plots=True,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # ==========================================================================
    # EXPORT BEST MODEL
    # ==========================================================================
    
    # Find the best model
    best_model_path = PROJECT_DIR / "caries_3class" / "weights" / "best.pt"
    
    if best_model_path.exists():
        print(f"\nBest model saved at: {best_model_path}")
        
        # Load best model for export
        best_model = YOLO(str(best_model_path))
        
        # Export to ONNX for deployment (optional)
        print("\nExporting model to ONNX format...")
        try:
            export_path = best_model.export(format="onnx", imgsz=IMAGE_SIZE)
            print(f"ONNX model exported to: {export_path}")
        except Exception as e:
            print(f"ONNX export failed (non-critical): {e}")
        
        print("\n" + "=" * 60)
        print("MODEL READY FOR INFERENCE")
        print("=" * 60)
        print(f"\nUse this model in inference.py:")
        print(f"  --model_path {best_model_path}")
        
    else:
        print(f"\nWarning: Best model not found at expected path: {best_model_path}")
        print("Check the runs directory for the actual output location.")
    
    return results


if __name__ == "__main__":
    train_caries_model()

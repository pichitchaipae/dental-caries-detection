"""
Fine-tune Pre-trained Dental Tooth Segmentation Model
Uses Tooth_seg_pano_20250319.pt as starting weights for transfer learning
Optimized for RTX 4080 + Ryzen 9
"""

import os
import torch
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO


class HardwareConfig:
    """Hardware-optimized settings for RTX 4080 + Ryzen 9"""
    DEVICE = 0
    BATCH_SIZE = 4
    WORKERS = 4
    AMP = True
    CACHE = False


class TrainingConfig:
    """Training configuration for fine-tuning"""
    # Pre-trained model path
    PRETRAINED_MODEL = r"C:\Users\jaopi\Desktop\SP\material\Tooth Segmentation + Recognition model\weights\Tooth_seg_pano_20250319.pt"
    
    # Dataset
    DATA_YAML = Path(__file__).parent / "data_pretrained.yaml"
    
    # Training parameters
    EPOCHS = 100  # Fewer epochs needed for fine-tuning
    IMGSZ = 1536
    PATIENCE = 30
    SEED = 42
    
    # Output
    PROJECT = Path(__file__).parent / "runs"
    

def get_device_info():
    """Print device information"""
    print("\n" + "=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA Available: {torch.cuda.device_count()} GPU(s) detected")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ PyTorch Version: {torch.__version__}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"         Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"         Compute Capability: {props.major}.{props.minor}")
        
        test_tensor = torch.zeros(1).cuda()
        print(f"✓ CUDA Test: Tensor created on {test_tensor.device}")
    else:
        print("✗ CUDA NOT AVAILABLE - Training will be slow!")
    
    print("=" * 60)


def main():
    get_device_info()
    
    # Generate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"dental_finetune_{timestamp}"
    
    print("\n" + "=" * 60)
    print("FINE-TUNING PRE-TRAINED DENTAL MODEL")
    print("=" * 60)
    print(f"Pre-trained:  {TrainingConfig.PRETRAINED_MODEL}")
    print(f"Data Config:  {TrainingConfig.DATA_YAML}")
    print(f"Output Dir:   {TrainingConfig.PROJECT}")
    print(f"Experiment:   {exp_name}")
    print(f"Image Size:   {TrainingConfig.IMGSZ}px")
    print(f"Batch Size:   {HardwareConfig.BATCH_SIZE}")
    print(f"Epochs:       {TrainingConfig.EPOCHS}")
    print(f"Device:       GPU {HardwareConfig.DEVICE}")
    print(f"Workers:      {HardwareConfig.WORKERS}")
    print(f"Mixed Prec:   {HardwareConfig.AMP}")
    print("=" * 60)
    
    # Load pre-trained model
    print(f"\nLoading pre-trained model: {TrainingConfig.PRETRAINED_MODEL}")
    model = YOLO(TrainingConfig.PRETRAINED_MODEL)
    
    print("\nStarting fine-tuning...")
    
    # Train with transfer learning
    results = model.train(
        # Dataset
        data=str(TrainingConfig.DATA_YAML),
        
        # Training parameters
        epochs=TrainingConfig.EPOCHS,
        imgsz=TrainingConfig.IMGSZ,
        patience=TrainingConfig.PATIENCE,
        
        # Hardware optimization
        device=HardwareConfig.DEVICE,
        batch=HardwareConfig.BATCH_SIZE,
        workers=HardwareConfig.WORKERS,
        amp=HardwareConfig.AMP,
        cache=HardwareConfig.CACHE,
        
        # Output
        project=str(TrainingConfig.PROJECT),
        name=exp_name,
        exist_ok=False,
        
        # Deterministic training
        seed=TrainingConfig.SEED,
        deterministic=True,
        
        # Fine-tuning specific - use lower learning rate
        lr0=0.0001,  # Lower LR for fine-tuning (default is 0.01)
        lrf=0.01,
        warmup_epochs=1.0,  # Shorter warmup
        
        # Optimizer
        optimizer="AdamW",
        weight_decay=0.0005,
        
        # Image handling
        rect=True,  # Rectangular training for panoramic X-rays
        
        # Augmentation - CRITICAL for dental X-rays
        fliplr=0.0,  # NO horizontal flip (left/right teeth are different!)
        flipud=0.0,  # No vertical flip
        mosaic=0.0,  # Disable mosaic (disrupts panoramic structure)
        mixup=0.0,
        copy_paste=0.0,
        
        # Conservative augmentation
        degrees=5.0,  # Reduced rotation for fine-tuning
        translate=0.05,
        scale=0.3,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.1,
        
        # Validation and checkpoints
        val=True,
        plots=True,
        save=True,
        save_period=10,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Best Model:  {TrainingConfig.PROJECT / exp_name / 'weights' / 'best.pt'}")
    print(f"Last Model:  {TrainingConfig.PROJECT / exp_name / 'weights' / 'last.pt'}")
    print(f"Results:     {TrainingConfig.PROJECT / exp_name}")
    
    # Validate best model
    print("\nRunning final validation with best weights...")
    best_model = YOLO(TrainingConfig.PROJECT / exp_name / 'weights' / 'best.pt')
    val_results = best_model.val(
        data=str(TrainingConfig.DATA_YAML),
        imgsz=TrainingConfig.IMGSZ,
        batch=HardwareConfig.BATCH_SIZE,
        device=HardwareConfig.DEVICE,
        workers=HardwareConfig.WORKERS,
        split='val'
    )
    
    print("\n" + "=" * 60)
    print("VALIDATION METRICS")
    print("=" * 60)
    print(f"mAP50 (Box):     {val_results.box.map50:.4f}")
    print(f"mAP50-95 (Box):  {val_results.box.map:.4f}")
    print(f"mAP50 (Mask):    {val_results.seg.map50:.4f}")
    print(f"mAP50-95 (Mask): {val_results.seg.map:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

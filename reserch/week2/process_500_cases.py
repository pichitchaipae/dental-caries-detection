"""
Process 500 dental cases with Tooth Segmentation + Recognition model.
Outputs bounding box visualizations, mask overlays, and JSON results for each case.
"""
import os
import sys
import warnings

# ============================================================================
# CRITICAL: Force non-interactive matplotlib backend BEFORE any other imports
# This prevents Tkinter thread crashes on Windows when saving many figures
# ============================================================================
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import ML libraries
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Import matplotlib.pyplot AFTER setting backend
import matplotlib.pyplot as plt


def compute_iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)

    return interArea / unionArea if unionArea > 0 else 0


def initialize_models():
    """Initialize both tooth recognition models."""
    import torch
    print("Initializing models...")
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Tooth Recognition (Pano-Seg) - YOLO model
    model_path = "../material/Tooth Segmentation + Recognition model/weights/Tooth_seg_pano_20250319.pt"
    yolo_model = YOLO(model_path)
    yolo_model.to(device)  # Explicitly move to GPU
    
    # Tooth Recognition (Crop-Seg) - Detectron2 model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = "../material/Tooth Segmentation + Recognition model/weights/Tooth_seg_crop_20250424.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = device
    detectron_predictor = DefaultPredictor(cfg)
    
    print("Models initialized successfully!")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    return yolo_model, detectron_predictor


def process_panoramic_image(pano_img, yolo_model):
    """Process panoramic image to detect and segment teeth."""
    # Run YOLO with verbose=False to suppress per-image output
    results = yolo_model(pano_img, verbose=False)
    
    list_of_masks = []
    original_height, original_width = pano_img.shape[:2]

    for result in results:
        if result.masks is None:
            continue
            
        num_masks = len(result.masks.xy) if hasattr(result.masks, 'xy') else 0

        temp_masks = []
        for i in range(num_masks):
            if not hasattr(result.masks, 'data'):
                continue

            # Get mask matrix and resize
            mask_matrix = result.masks.data[i].cpu().numpy()
            resized_mask = cv2.resize(
                mask_matrix.astype(np.uint8),
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            # Get bounding box
            box = result.boxes.xyxy[i].cpu().numpy()

            mask_data = {
                'id': i,
                'polygon': result.masks.xy[i].tolist(),
                'normalized': result.masks.xyn[i].tolist(),
                'matrix_resized': resized_mask,
                'confidence': float(result.boxes.conf[i].item()),
                'class_id': int(result.boxes.cls[i].item()),
                'class_name': result.names[int(result.boxes.cls[i].item())],
                'bbox': box.tolist()
            }
            temp_masks.append(mask_data)

        # Filter duplicates by IoU
        kept = []
        for m in temp_masks:
            duplicate = False
            for k in kept:
                iou = compute_iou(m['bbox'], k['bbox'])
                if iou > 0.5:
                    if m['confidence'] > k['confidence']:
                        kept.remove(k)
                        kept.append(m)
                    duplicate = True
                    break
            if not duplicate:
                kept.append(m)

        list_of_masks.extend(kept)

    return list_of_masks


def crop_and_segment_teeth(pano_img, mask_data_list, detectron_predictor, pad=20):
    """Crop each tooth and perform detailed segmentation with pixel coordinates."""
    results = []
    for data in mask_data_list:
        mask = data['matrix_resized'].astype(np.uint8) * 255
        x, y, w, h = cv2.boundingRect(mask)
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, pano_img.shape[1])
        y2 = min(y + h + pad, pano_img.shape[0])
        
        cropped_img = pano_img[y1:y2, x1:x2]
        
        # Run Detectron2 segmentation
        detectron_output = detectron_predictor(cropped_img)
        
        # Extract segmentation masks and pixel coordinates
        instances = detectron_output["instances"].to("cpu")
        seg_masks = []
        all_pixel_coordinates = []  # All pixel coordinates for this tooth
        
        if len(instances) > 0:
            masks = instances.pred_masks.numpy()
            scores = instances.scores.numpy()
            for mask_idx, (seg_mask, score) in enumerate(zip(masks, scores)):
                # Extract pixel coordinates for this mask segment
                # np.where returns (row_indices, col_indices) where mask is True
                # row = y-axis (local), col = x-axis (local)
                rows, cols = np.where(seg_mask)
                
                # Convert local crop coordinates to global panoramic coordinates
                # x_global = x1 + col, y_global = y1 + row
                pixel_coords = [(int(x1 + col), int(y1 + row)) for row, col in zip(rows, cols)]
                all_pixel_coordinates.extend(pixel_coords)
                
                seg_masks.append({
                    'mask_id': mask_idx,
                    'score': float(score),
                    'mask_shape': list(seg_mask.shape),
                    'num_pixels': len(pixel_coords),
                    'pixel_coordinates': pixel_coords  # List of (x, y) tuples
                })
        
        results.append({
            'tooth_id': data['class_name'],
            'crop_coords': [x1, y1, x2, y2],
            'num_segments': len(seg_masks),
            'segments': seg_masks,
            'pixel_coordinates': all_pixel_coordinates,  # Combined pixel coords for the tooth
            'total_pixels': len(all_pixel_coordinates),
            'detectron_output': detectron_output  # Keep for visualization
        })
    return results


def create_bounding_box_visualization(pano_img, list_of_masks, output_path):
    """Create visualization with bounding boxes and tooth IDs."""
    vis_image = pano_img.copy()
    
    # Draw bounding boxes and tooth IDs
    for mask_data in list_of_masks:
        x1, y1, x2, y2 = map(int, mask_data['bbox'])
        class_name = mask_data['class_name']
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_image, str(class_name), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    
    # Save using matplotlib to match the notebook style
    fig = plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title("Bounding Boxes with Class ID")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    plt.clf()


def create_mask_overlay_visualization(pano_img, list_seg_tooth, output_path, alpha=0.5):
    """Create mask overlay visualization on panoramic image."""
    import random
    
    pano_vis = pano_img.copy()

    for result in list_seg_tooth:
        x1, y1, x2, y2 = result['crop_coords']
        detectron_output = result['detectron_output']
        
        instances = detectron_output["instances"].to("cpu")

        # Get binary masks and loop through each instance
        if len(instances) > 0:
            masks = instances.pred_masks.numpy()
            for mask in masks:
                h, w = mask.shape
                color = [random.randint(0, 255) for _ in range(3)]

                # Create a colored mask image
                color_mask = np.zeros((h, w, 3), dtype=np.uint8)
                color_mask[mask] = color

                # Extract region from pano
                pano_crop = pano_vis[y1:y2, x1:x2]

                # Blend the color mask with pano_crop
                blended_crop = np.where(mask[:, :, None], 
                                        cv2.addWeighted(pano_crop, 1 - alpha, color_mask, alpha, 0),
                                        pano_crop)

                # Paste back to pano
                pano_vis[y1:y2, x1:x2] = blended_crop

    # Save result
    fig = plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(pano_vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Mask Overlay on Panoramic Image (No Labels, No Boxes)")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    plt.clf()


def save_results(case_num, list_of_masks, list_seg_tooth, output_dir):
    """Save processing results to JSON file including pixel coordinates."""
    results = {
        'case_number': case_num,
        'num_teeth_detected': len(list_of_masks),
        'teeth_data': []
    }
    
    for mask_data, seg_data in zip(list_of_masks, list_seg_tooth):
        tooth_info = {
            'tooth_id': mask_data['class_name'],
            'confidence': mask_data['confidence'],
            'bbox': mask_data['bbox'],
            'crop_coords': seg_data['crop_coords'],
            'num_segments': seg_data['num_segments'],
            'total_pixels': seg_data['total_pixels'],
            'pixel_coordinates': seg_data['pixel_coordinates'],  # All (x, y) coordinates
            'segments_detail': [{
                'mask_id': seg['mask_id'],
                'score': seg['score'],
                'num_pixels': seg['num_pixels'],
                'pixel_coordinates': seg['pixel_coordinates']
            } for seg in seg_data['segments']]
        }
        results['teeth_data'].append(tooth_info)
    
    # Save to JSON with UTF-8 encoding
    json_path = os.path.join(output_dir, f'case_{case_num}_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


def process_single_case(case_dir, case_num, yolo_model, detectron_predictor, output_dir, overwrite=False):
    """Process a single case."""
    # Find the PNG file
    png_files = list(Path(case_dir).glob('*.png'))
    if not png_files:
        return False, "No PNG file found"
    
    img_path = str(png_files[0])
    
    # Create case-specific output directory
    case_output_dir = os.path.join(output_dir, f'case {case_num}')
    os.makedirs(case_output_dir, exist_ok=True)

    bbox_output_path = os.path.join(case_output_dir, f'case_{case_num}_bounding_boxes.png')
    mask_overlay_path = os.path.join(case_output_dir, f'case_{case_num}_mask_overlay.png')
    json_output_path = os.path.join(case_output_dir, f'case_{case_num}_results.json')

    # Skip if already processed (unless overwrite)
    if (not overwrite
        and os.path.exists(bbox_output_path)
        and os.path.exists(mask_overlay_path)
        and os.path.exists(json_output_path)):
        return True, "Skipped (already exists)"
    
    # Read image
    pano_img = cv2.imread(img_path)
    if pano_img is None:
        return False, "Failed to read image"
    
    # Convert BGR to RGB for processing
    pano_img_rgb = cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB)
    
    # Process panoramic image
    list_of_masks = process_panoramic_image(pano_img_rgb, yolo_model)
    
    if len(list_of_masks) == 0:
        return False, "No teeth detected"
    
    # Crop and segment teeth
    list_seg_tooth = crop_and_segment_teeth(pano_img_rgb, list_of_masks, detectron_predictor)
    
    # Create bounding box visualization
    create_bounding_box_visualization(pano_img_rgb, list_of_masks, bbox_output_path)
    
    # Create mask overlay visualization (reuses detectron output from list_seg_tooth)
    create_mask_overlay_visualization(pano_img_rgb, list_seg_tooth, mask_overlay_path)
    
    # Save results JSON
    save_results(case_num, list_of_masks, list_seg_tooth, case_output_dir)
    
    return True, f"OK ({len(list_of_masks)} teeth)"


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description="Process 500 cases with tooth segmentation+recognition")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess cases even if outputs already exist")
    args = parser.parse_args()

    # Setup paths
    base_dir = Path("../material/500 cases with annotation")
    output_dir = Path("500-segmentation+recognition")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize models
    yolo_model, detectron_predictor = initialize_models()
    
    # Get all case directories and sort numerically
    case_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('case')]
    case_dirs = sorted(case_dirs, key=lambda x: int(x.name.replace('case ', '').replace('case', '')))
    
    print(f"\nFound {len(case_dirs)} cases to process")
    print(f"Output directory: {output_dir.absolute()}\n")
    
    # Process all cases
    successful = 0
    skipped = 0
    failed = 0
    
    pbar = tqdm(case_dirs, desc="Processing cases", unit="case")
    for case_dir in pbar:
        case_name = case_dir.name
        case_num = case_name.replace('case ', '').replace('case', '')
        
        try:
            success, msg = process_single_case(
                case_dir, case_num, yolo_model, detectron_predictor, output_dir, 
                overwrite=args.overwrite
            )
            if success:
                if "Skipped" in msg:
                    skipped += 1
                else:
                    successful += 1
                pbar.set_postfix_str(f"case {case_num}: {msg}")
            else:
                failed += 1
                pbar.set_postfix_str(f"case {case_num}: FAIL - {msg}")
        except Exception as e:
            failed += 1
            pbar.set_postfix_str(f"case {case_num}: ERROR - {str(e)[:30]}")
    
    # Print summary
    print("\n" + "="*50)
    print("Processing Complete!")
    print("="*50)
    print(f"Total cases: {len(case_dirs)}")
    print(f"Newly processed: {successful}")
    print(f"Skipped (existed): {skipped}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()

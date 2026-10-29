"""Read-only model/data audit; writes diagnostics, never changes serving settings.

Run in the ML image with the repository mounted, for example:
  python scripts/audit_inference_parity.py --images-root /data \
    --dataset datasets/caries_crop_v1 --weights-dir weights \
    --cases 10,20,42 --device cpu --out /tmp/parity.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.pipeline.crop_utils import suppress_other_teeth
from app.pipeline.detection import run_stage1
from app.pipeline.surface_heuristic import classify_lesion_surface
from app.pipeline.xml_annotations import parse_case_dir
from evaluate_caries_pipeline import _bbox, _counts, _match
from models.registry import load_models_in_child


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images-root', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--weights-dir', required=True)
    p.add_argument('--cases', default='10,20,42')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    import torch
    import ultralytics

    torch.set_num_threads(4)
    started = time.perf_counter()
    models = load_models_in_child(args.weights_dir, args.device)
    # Serving currently passes ML_DEVICE only to Detectron2. Make this audit's
    # device explicit for both YOLOs so CPU/GPU comparisons are reproducible.
    for key in ('pano', 'caries'):
        models[key].to(args.device)
    payload = {'args': vars(args), 'versions': {
        'torch': torch.__version__, 'ultralytics': ultralytics.__version__},
        'crop_input_format': models['crop']['cfg'].INPUT.FORMAT,
        'cases': []}
    dataset = Path(args.dataset)
    for cid in map(int, args.cases.split(',')):
        case_dir = Path(args.images_root) / f'case {cid}'
        image_path = case_dir / f'case_{cid}.png'
        detections, image = run_stage1(str(image_path), models)
        gt, issues = parse_case_dir(case_dir)
        gt_by_fdi, pred_by_fdi = defaultdict(list), defaultdict(list)
        for lesion in gt:
            poly = np.asarray(lesion.polygon)
            gt_by_fdi[lesion.fdi].append({'polygon': poly, 'bbox': _bbox(poly),
                                         'surface': lesion.surface})
        row = {'case': cid, 'xml_issues': len(issues), 'teeth': len(detections),
               'refined': sum(d.mask_source == 'crop_seg' for d in detections),
               'grayscale_channels_equal': bool(np.array_equal(image[:,:,0], image[:,:,2])),
               'crop_comparisons': [], 'predictions': []}
        for det in detections:
            for lesion in det.caries_lesions:
                poly = np.asarray(lesion.polygon)
                surface = classify_lesion_surface(det.fdi, det.tooth_polygon,
                                                  lesion.centroid)['name']
                pred_by_fdi[det.fdi].append({'polygon': poly, 'bbox': _bbox(poly),
                                            'surface': surface})
                row['predictions'].append({'fdi': det.fdi, 'confidence': lesion.confidence,
                                           'surface': surface, 'centroid': lesion.centroid})
            paths = list((dataset/'images'/'val').glob(f'case{cid:03d}_fdi{det.fdi}.png'))
            if not paths:
                continue
            x1,y1,x2,y2 = det.crop_bbox_xyxy
            crop = suppress_other_teeth(image[y1:y2,x1:x2], det.crop_bbox_xyxy,
                      det.tooth_polygon, [d.tooth_polygon for d in detections if d is not det])
            saved = cv2.imread(str(paths[0]))
            same_shape = crop.shape == saved.shape
            equal = same_shape and np.array_equal(crop,saved)
            comparison = {'fdi': det.fdi, 'saved_shape': list(saved.shape),
                          'live_shape': list(crop.shape), 'pixels_equal': bool(equal)}
            if same_shape:
                comparison['different_pixel_fraction'] = float(np.any(crop != saved, axis=2).mean())
            row['crop_comparisons'].append(comparison)
        row['lesion_metrics'] = {}
        for iou in (0.3, 0.5):
            pairs = _match(gt_by_fdi, pred_by_fdi, iou)
            tp,fp,fn = _counts(gt_by_fdi, pred_by_fdi, pairs)
            exact_surface = sum(g['surface'] == pr['surface'] for g,pr,_ in pairs)
            row['lesion_metrics'][str(iou)] = {'tp':tp,'fp':fp,'fn':fn,
                'matched_correct_surface': exact_surface,
                'all_gt_correct_surface_fraction': exact_surface / len(gt) if gt else None}
        print(json.dumps(row), flush=True)
        payload['cases'].append(row)
    payload['predictor_args'] = {k: {a:getattr(models[k].predictor.args,a) for a in
        ['imgsz','conf','iou','rect','retina_masks','device']} for k in ('pano','caries')}
    payload['elapsed_seconds'] = time.perf_counter() - started
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    print('audit saved:', args.out, flush=True)


if __name__ == '__main__':
    main()

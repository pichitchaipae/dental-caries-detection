"""
Pinned model version identifiers and verified SHA-256 checksums.

All checksums were computed from the actual weight files in ml_model/models/.
Do NOT update these values without re-verifying the files.

The old ``caries_detect.pt`` was a Week-4 artifact and is quarantined in
``_non-using/deprecated-serving-models/``.  It was a 3-class pano-space model,
not a 1-class crop-space model and must never be served.

``caries_detector`` below is the crop-space replacement: YOLOv8s-seg trained on
tooth crops with a single ``caries`` class.  The surface (mesial / occlusal /
distal) is NOT predicted by the network; it comes from the deterministic
PCA-aligned geometry heuristic in ``app/pipeline/surface_heuristic.py``.
"""

MODEL_VERSIONS: dict[str, dict] = {
    "pano_detector": {
        "version": "20250319",
        "filename": "Tooth_seg_pano_20250319.pt",
        "sha256": "502bd58099e50d5c5bb16245aa5a135943fbd87debe9f66928a5b938798ca579",
        "architecture": "yolo_segment",
        "num_classes": 32,
        "description": "Ultralytics YOLO segmentation — 32 FDI tooth classes on panoramic radiograph",
    },
    "crop_segmenter": {
        "version": "20250424",
        "filename": "Tooth_seg_crop_20250424.pth",
        "sha256": "b39fef8288634e6e756d3aace628d145d0f927a76d3f7d9bfa17f919a8c9b4a9",
        "architecture": "mask_rcnn_R_50_FPN_3x",
        "num_classes": 1,
        "score_thresh_test": 0.5,
        # Detectron2 models do NOT resize internally.  The reference loader uses
        # DefaultPredictor, whose ResizeShortestEdge is built from these values.
        # Serving must apply the same transform or scores collapse to ~0.
        "min_size_test": 800,
        "max_size_test": 1333,
        "description": "Detectron2 Mask R-CNN ResNet-50 FPN — 1-class tooth crop segmentation",
    },
    "caries_detector": {
        "version": "caries-crop-seg-20261028",
        "filename": "caries_crop_seg_20261028.pt",
        "sha256": "c0d4768fdbd90567feb28c5df34fa760321e79a08acbb47e11ba84a5c2c67d74",
        "architecture": "yolo_segment",
        "num_classes": 1,
        "classes": ["caries"],
        "trained_on": "caries_crop_v1 (500 cases / 3,243 crops, 400 train - 100 val cases)",
        "imgsz": 640,
        "description": "YOLOv8s-seg — 1-class caries mask on tooth crops (PAD 20, neighbours suppressed)",
    },
}

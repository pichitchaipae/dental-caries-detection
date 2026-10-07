"""
Pinned model version identifiers and verified SHA-256 checksums.

All checksums were computed from the actual weight files in ml_model/models/.
Do NOT update these values without re-verifying the files.
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
        "description": "Detectron2 Mask R-CNN ResNet-50 FPN — 1-class tooth crop segmentation",
    },
    "caries_detector": {
        "version": "caries-yolo-proximal-occlusal-v1",
        "filename": "caries_detect.pt",
        "sha256": "515e09fe1a11c683dfaa94156a679bdbee7ab177848c6558a63f159cbf8dc1fd",
        "architecture": "yolo_detect",
        "description": "YOLOv8 lesion detector; runtime confidence is configurable and defaults to 0.02",
    },
    "surface_classifier": {
        "version": "run3-rf-14features",
        "filename": "rf_classify_ml.pkl",
        "sha256": "201459b7a54e8b32b4ac4ed7ca77a198e622d9237e98b50c3426ca048d0e2686",
        "architecture": "random_forest",
        "n_features": 14,
        "feature_cols": [
            "is_upper", "x_mean", "y_mean", "x_std", "y_std",
            "x_min", "x_max", "y_min", "y_max", "x_range",
            "y_range", "x_centroid_dist", "aspect_ratio", "coverage",
        ],
        "classes": ["Distal", "Mesial", "Occlusal"],
        "description": "Run 3 scikit-learn RandomForestClassifier (200 estimators, 14 geometric features) for surface classification; legacy artifact filename retained for deployment compatibility",
    },
}

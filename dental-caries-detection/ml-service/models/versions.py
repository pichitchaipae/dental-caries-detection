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
        "version": "caries_detect",
        "filename": "caries_detect.pt",
        "sha256": "515e09fe1a11c683dfaa94156a679bdbee7ab177848c6558a63f159cbf8dc1fd",
        "architecture": "yolo_detect",
        "description": "YOLOv8 caries lesion detector on panoramic radiograph (conf=0.005)",
    },
    "surface_classifier": {
        "version": "20260806",
        "filename": "rf_classify_ml.pkl",
        "sha256": "2d1091b7842e7ddfeeebadea96895226c48b27ce05db0b97824fde7079ad9470",
        "architecture": "random_forest",
        "n_features": 14,
        "feature_cols": [
            "is_upper", "x_mean", "y_mean", "x_std", "y_std",
            "x_min", "x_max", "y_min", "y_max", "x_range",
            "y_range", "x_centroid_dist", "aspect_ratio", "coverage",
        ],
        "classes": ["Distal", "Mesial", "Occlusal"],
        "description": "scikit-learn RandomForestClassifier (200 estimators) — surface classification",
    },
}

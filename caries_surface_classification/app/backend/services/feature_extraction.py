import math
import numpy as np
import cv2

FEATURE_COLS = [
    'is_upper',
    'x_mean', 'y_mean',
    'x_std', 'y_std',
    'x_min', 'x_max',
    'y_min', 'y_max',
    'x_range',
    'y_range',
    'x_centroid_dist',
    'aspect_ratio',
    'coverage',
]

MIN_CLUSTER_SIZE = 15
MAX_TILT_DEG = 45.0

class FeatureExtractor:
    @staticmethod
    def is_upper_jaw(tooth_id: str) -> bool:
        try:
            return int(str(tooth_id)[0]) in [1, 2]
        except (ValueError, IndexError):
            return False

    @staticmethod
    def get_quadrant(tooth_id: str) -> int:
        try:
            return int(str(tooth_id)[0])
        except (ValueError, IndexError):
            return 4 # Default fallback

    @staticmethod
    def get_bbox(pts):
        p = np.array(pts, dtype=np.float64)
        bbox_min, bbox_max = np.min(p, 0), np.max(p, 0)
        return bbox_min[0], bbox_min[1], bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1]

    @staticmethod
    def rotate(pts, center, angle):
        p = np.array(pts, dtype=np.float64) - center
        c, s = np.cos(angle), np.sin(angle)
        return np.dot(p, np.array([[c, -s], [s, c]]).T) + center

    @staticmethod
    def remove_small_clusters(caries_pts, min_cluster=MIN_CLUSTER_SIZE):
        if len(caries_pts) < min_cluster:
            return caries_pts
        pts = np.array(caries_pts, dtype=np.int32)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        pad = 2
        w = int(x_max - x_min + 1 + 2 * pad)
        h = int(y_max - y_min + 1 + 2 * pad)
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = pts - np.array([x_min - pad, y_min - pad])
        mask[shifted[:, 1], shifted[:, 0]] = 255
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_cluster:
                keep[labels == lbl] = 255
        ys, xs = np.where(keep > 0)
        if len(xs) == 0:
            return caries_pts
        return np.column_stack([xs + x_min - pad, ys + y_min - pad]).astype(np.float64)

    @classmethod
    def perform_pca(cls, points, tooth_id):
        pts = np.array(points, dtype=np.float64).reshape(-1, 2)
        mean = np.mean(pts, axis=0)
        centered = pts - mean

        _, eigvecs = cv2.PCACompute(centered.astype(np.float32), mean=None)
        primary_eigenvector = eigvecs[0].astype(np.float64)
        secondary_eigenvector = eigvecs[1].astype(np.float64)

        if abs(primary_eigenvector[1]) >= abs(secondary_eigenvector[1]):
            vertical_axis = primary_eigenvector.copy()
            horizontal_axis = secondary_eigenvector.copy()
        else:
            vertical_axis = secondary_eigenvector.copy()
            horizontal_axis = primary_eigenvector.copy()

        upper = cls.is_upper_jaw(tooth_id)
        if upper:
            if vertical_axis[1] < 0:
                vertical_axis = -vertical_axis
        else:
            if vertical_axis[1] > 0:
                vertical_axis = -vertical_axis

        quadrant = cls.get_quadrant(tooth_id)
        if quadrant in [1, 4]:
            if horizontal_axis[0] < 0:
                horizontal_axis = -horizontal_axis
        else:
            if horizontal_axis[0] > 0:
                horizontal_axis = -horizontal_axis

        angle_from_x = math.atan2(vertical_axis[1], vertical_axis[0])
        target_angle = math.pi / 2 if upper else -math.pi / 2
        rotation_angle = target_angle - angle_from_x

        while rotation_angle > math.pi:
            rotation_angle -= 2 * math.pi
        while rotation_angle < -math.pi:
            rotation_angle += 2 * math.pi

        clamped = False
        if abs(math.degrees(rotation_angle)) > MAX_TILT_DEG:
            rotation_angle = 0.0
            clamped = True

        return mean, rotation_angle, clamped

    @classmethod
    def extract_features(cls, tooth_id: str, tooth_pts: list, caries_pts: list) -> dict:
        caries_clean = cls.remove_small_clusters(caries_pts)
        if len(caries_clean) == 0:
            return None

        center, angle, _ = cls.perform_pca(tooth_pts, tooth_id)
        tooth_rot = cls.rotate(tooth_pts, center, angle)
        caries_rot = cls.rotate(caries_clean, center, angle)

        bbox_x, bbox_y, w, h = cls.get_bbox(tooth_rot)
        if w <= 0 or h <= 0:
            return None

        x_rel = np.clip((caries_rot[:, 0] - bbox_x) / w, 0.0, 1.0)
        y_rel = np.clip((caries_rot[:, 1] - bbox_y) / h, 0.0, 1.0)

        return {
            "is_upper": 1 if cls.is_upper_jaw(tooth_id) else 0,
            "x_mean": float(np.mean(x_rel)),
            "y_mean": float(np.mean(y_rel)),
            "x_std": float(np.std(x_rel)),
            "y_std": float(np.std(y_rel)),
            "x_min": float(np.min(x_rel)),
            "x_max": float(np.max(x_rel)),
            "y_min": float(np.min(y_rel)),
            "y_max": float(np.max(y_rel)),
            "x_range": float(np.max(x_rel) - np.min(x_rel)),
            "y_range": float(np.max(y_rel) - np.min(y_rel)),
            "x_centroid_dist": float(abs(np.mean(x_rel) - 0.5)),
            "aspect_ratio": float(w / h),
            "coverage": float(len(caries_clean) / (len(tooth_pts) + 1e-6)),
        }

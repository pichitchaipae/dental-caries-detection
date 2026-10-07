import os
import math
import warnings
import logging
import joblib
import numpy as np
import pandas as pd

from schemas.inference import RFClassificationResult, PipelineError
from services.feature_extraction import FEATURE_COLS, FeatureExtractor

logger = logging.getLogger(__name__)
RF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rf_classify_ml.pkl")

# X-Thirds zone boundaries (from reference pipeline constants)
LEFT_BOUND = 0.40
RIGHT_BOUND = 0.60


class RFClassifierService:
    def __init__(self):
        self.rf_model = None

    def load_model(self):
        if not os.path.exists(RF_MODEL_PATH):
            raise FileNotFoundError(f"Missing model: {RF_MODEL_PATH}")
        if self.rf_model is None:
            self.rf_model = joblib.load(RF_MODEL_PATH)

    # ------------------------------------------------------------------
    # X-Thirds Baseline Classifier (Smart Fallback)
    # Ported verbatim from reference: rf-model-optimized-pipeline.py
    # classify_xthird() lines 1057-1099
    # ------------------------------------------------------------------
    @staticmethod
    def classify_xthird(tooth_id: str, tooth_pts: list, caries_pts: list) -> RFClassificationResult:
        """
        Baseline X-Thirds surface classifier (v4.5 dominant-zone voting).

        Divides the PCA-aligned, normalised tooth width into three zones:
          • Left  zone  (x < 0.40) → Distal  (Q1/Q4) or Mesial (Q2/Q3)
          • Centre zone (0.40–0.60) → Occlusal
          • Right zone  (x > 0.60) → Mesial  (Q1/Q4) or Distal (Q2/Q3)

        Args:
            tooth_id:   FDI two-digit tooth identifier (str).
            tooth_pts:  Tooth mask pixel coordinates (list of [x, y]).
            caries_pts: Caries region pixel coordinates (list of [x, y]).

        Returns:
            RFClassificationResult with method="XThirds_Fallback", fallback_used=True.
        """
        caries_clean = FeatureExtractor.remove_small_clusters(caries_pts)
        if len(caries_clean) == 0:
            return RFClassificationResult(
                predicted_surface="Occlusal",
                rf_confidence=0.0,
                method="XThirds_Fallback",
                fallback_used=True,
                fallback_reason="Empty caries region after noise removal",
                probabilities={}
            )

        center, angle, clamped = FeatureExtractor.perform_pca(tooth_pts, tooth_id)
        tooth_rot = FeatureExtractor.rotate(tooth_pts, center, angle)
        caries_rot = FeatureExtractor.rotate(caries_clean, center, angle)

        bbox_x, bbox_y, w, h = FeatureExtractor.get_bbox(tooth_rot)
        if w <= 0 or h <= 0:
            return RFClassificationResult(
                predicted_surface="Occlusal",
                rf_confidence=0.0,
                method="XThirds_Fallback",
                fallback_used=True,
                fallback_reason="Invalid tooth bounding box (zero width/height)",
                probabilities={}
            )

        rel_xs = np.clip((caries_rot[:, 0] - bbox_x) / w, 0.0, 1.0)
        n_pts = len(rel_xs)

        # Quadrant-aware zone assignment (Rule 3 from reference)
        quadrant = FeatureExtractor.get_quadrant(tooth_id)
        if quadrant in [1, 4]:
            d_mask = rel_xs < LEFT_BOUND
            c_mask = (rel_xs >= LEFT_BOUND) & (rel_xs <= RIGHT_BOUND)
            m_mask = rel_xs > RIGHT_BOUND
        else:
            m_mask = rel_xs < LEFT_BOUND
            c_mask = (rel_xs >= LEFT_BOUND) & (rel_xs <= RIGHT_BOUND)
            d_mask = rel_xs > RIGHT_BOUND

        vote_map = {
            "Mesial":   int(np.sum(m_mask)),
            "Occlusal": int(np.sum(c_mask)),
            "Distal":   int(np.sum(d_mask)),
        }
        winner = max(vote_map, key=vote_map.get)
        vote_fractions = {k: round(v / max(n_pts, 1), 4) for k, v in vote_map.items()}
        winner_confidence = vote_fractions[winner]

        return RFClassificationResult(
            predicted_surface=winner,
            rf_confidence=winner_confidence,
            method="XThirds_Fallback",
            fallback_used=True,
            fallback_reason="RF unavailable or feature extraction failed",
            probabilities=vote_fractions
        )

    # ------------------------------------------------------------------
    # Primary Classifier: RF predict_proba with Smart Fallback
    # Mirrors reference: classify_ml() lines 1113-1156
    # ------------------------------------------------------------------
    def classify_surface(self, features: dict, tooth_id: str = "46",
                         tooth_pts: list = None, caries_pts: list = None) -> RFClassificationResult:
        """
        Classify caries surface via RF predict_proba.
        Falls back to X-Thirds classifier (Smart Fallback) when:
          • features dict is None / missing columns
          • RF model is not loaded
          • surface_scores dict is empty
          • Any exception occurs

        Args:
            features:   Pre-computed feature dict keyed by FEATURE_COLS.
            tooth_id:   FDI identifier (needed for X-Thirds fallback).
            tooth_pts:  Tooth mask points (needed for X-Thirds fallback).
            caries_pts: Caries points (needed for X-Thirds fallback).

        Returns:
            RFClassificationResult
        """
        tooth_pts = tooth_pts or []
        caries_pts = caries_pts or []

        try:
            # --- Guard: features missing ---
            if features is None:
                logger.warning("RF classify: features is None → X-Thirds fallback")
                return self.classify_xthird(tooth_id, tooth_pts, caries_pts)

            # --- Load model ---
            try:
                self.load_model()
            except FileNotFoundError as e:
                logger.warning(f"RF model not found: {e} → X-Thirds fallback")
                return self.classify_xthird(tooth_id, tooth_pts, caries_pts)

            # --- Guard: model still None ---
            if self.rf_model is None:
                logger.warning("RF model is None after load → X-Thirds fallback")
                return self.classify_xthird(tooth_id, tooth_pts, caries_pts)

            # --- Validate feature schema ---
            missing_features = [col for col in FEATURE_COLS if col not in features]
            if missing_features:
                logger.warning(f"RF classify: missing features {missing_features} → X-Thirds fallback")
                return self.classify_xthird(tooth_id, tooth_pts, caries_pts)

            prediction_input_df = pd.DataFrame(
                [[features[col] for col in FEATURE_COLS]],
                columns=FEATURE_COLS,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                class_probabilities = self.rf_model.predict_proba(prediction_input_df)[0]

            model_classes = list(self.rf_model.classes_)
            valid_surface_classes = ["Occlusal", "Mesial", "Distal"]
            surface_scores = {
                cls: float(class_probabilities[model_classes.index(cls)])
                for cls in valid_surface_classes
                if cls in model_classes
            }

            # --- Guard: empty surface scores → X-Thirds fallback ---
            if not surface_scores:
                logger.warning("RF classify: no valid surface scores → X-Thirds fallback")
                return self.classify_xthird(tooth_id, tooth_pts, caries_pts)

            prediction = max(surface_scores, key=surface_scores.get)
            confidence = surface_scores[prediction]

            return RFClassificationResult(
                predicted_surface=prediction,
                rf_confidence=confidence,
                method="RandomForest_Proba",
                fallback_used=False,
                probabilities=surface_scores
            )

        except Exception as e:
            logger.warning(f"RF classify exception: {e} → X-Thirds fallback")
            try:
                return self.classify_xthird(tooth_id, tooth_pts, caries_pts)
            except Exception as e2:
                logger.error(f"X-Thirds fallback also failed: {e2}")
                return RFClassificationResult(
                    predicted_surface="Occlusal",
                    rf_confidence=0.0,
                    method="DefaultFallback",
                    fallback_used=True,
                    fallback_reason=f"Both RF and X-Thirds failed: {e2}",
                    probabilities={}
                )

import pytest
from services.feature_extraction import FeatureExtractor, FEATURE_COLS

def test_feature_extraction_parity():
    # Synthetic test data for parity check
    # A simple square tooth and a smaller square caries in the center
    tooth_pts = [[10, 10], [110, 10], [110, 110], [10, 110]]
    caries_pts = [[50, 50], [70, 50], [70, 70], [50, 70]]
    
    # We use a tooth_id that maps to Q4 (e.g., "46")
    tooth_id = "46"
    
    features = FeatureExtractor.extract_features(tooth_id, tooth_pts, caries_pts)
    
    assert features is not None
    assert "is_upper" in features
    assert features["is_upper"] == 0 # 46 is lower jaw
    
    # The bounding box of tooth is 100x100
    # Caries is 20x20 in the middle
    # With PCA alignment (square will have arbitrary axes, but let's check basic sanity)
    assert features["aspect_ratio"] > 0
    assert 0 <= features["x_mean"] <= 1
    assert 0 <= features["y_mean"] <= 1
    
    # Ensure all feature cols are present
    for col in FEATURE_COLS:
        assert col in features

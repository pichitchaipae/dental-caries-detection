"""
Dental Caries Surface Classification using PCA + OBB (Oriented Bounding Box)
=============================================================================

This module classifies the surface location of dental caries on 2D X-ray images
using Principal Component Analysis (PCA) for orientation normalization and
pure geometric logic for classification.

Algorithm:
1. PCA Alignment: Use PCA to find the major axis of the tooth and rotate it to vertical
2. Apply the same rotation to caries polygon/points
3. Calculate bounding box of rotated tooth
4. Classify caries based on relative position within the bounding box

Surface Classes:
- Class 0 (Occlusal): Top/bottom surface (chewing surface)
- Class 1 (Proximal): Left or right sides (contact surfaces between teeth)
- Class 2 (Lingual/Other): Other positions

FDI Notation (ISO 3950):
- Upper Jaw (11-28): Quadrants 1 and 2, roots at top (low Y), occlusal at bottom (high Y)
- Lower Jaw (31-48): Quadrants 3 and 4, roots at bottom (high Y), occlusal at top (low Y)

Author: Computer Vision Engineer
Date: 2026
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import math


# =============================================================================
# Constants for Classification Thresholds
# =============================================================================

# Zone thresholds (as fraction of tooth dimension)
OCCLUSAL_ZONE_THRESHOLD = 0.20  # Top/bottom 20% for occlusal classification
PROXIMAL_ZONE_THRESHOLD = 0.20  # Left/right 20% for proximal classification


# =============================================================================
# Helper Functions
# =============================================================================

def is_upper_jaw(tooth_id: str) -> bool:
    """
    Determine if tooth is in upper jaw based on FDI notation (ISO 3950).
    
    FDI System:
    - First digit indicates quadrant:
        - 1x, 2x = Upper jaw (maxilla)
        - 3x, 4x = Lower jaw (mandible)
    
    Args:
        tooth_id: Tooth identifier in FDI notation (e.g., "11", "26", "45")
        
    Returns:
        True if upper jaw (quadrants 1-2), False if lower jaw (quadrants 3-4)
    """
    if not tooth_id or len(tooth_id) < 2:
        return True  # Default to upper jaw if invalid
    
    try:
        quadrant = int(tooth_id[0])
        return quadrant in [1, 2]  # Quadrants 1 and 2 are upper jaw
    except (ValueError, IndexError):
        return True  # Default to upper jaw if parsing fails


def compute_centroid(points: np.ndarray) -> Tuple[float, float]:
    """
    Compute the centroid (center of mass) of a set of points.
    
    Uses arithmetic mean of all coordinates.
    
    Args:
        points: Nx2 array of [x, y] coordinates
        
    Returns:
        Tuple of (centroid_x, centroid_y)
    """
    if points is None or len(points) == 0:
        return (0.0, 0.0)
    
    points = np.array(points, dtype=np.float64)
    centroid_x = np.mean(points[:, 0])
    centroid_y = np.mean(points[:, 1])
    
    return (centroid_x, centroid_y)


def perform_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform Principal Component Analysis on 2D points.
    
    PCA finds the principal axes of variation in the point cloud.
    The first eigenvector (major axis) indicates the primary direction of elongation.
    
    Math Background:
    - Compute covariance matrix of centered points
    - Find eigenvectors (directions) and eigenvalues (variance along each direction)
    - Major axis = eigenvector with largest eigenvalue
    
    Args:
        points: Nx2 array of [x, y] coordinates
        
    Returns:
        Tuple of:
        - mean: Center of the point distribution (2,)
        - eigenvectors: 2x2 matrix where rows are principal axes
        - angle: Rotation angle (radians) to make major axis vertical
    """
    # Convert to float64 for numerical stability
    points = np.array(points, dtype=np.float64)
    
    # Compute mean (centroid)
    mean = np.mean(points, axis=0)
    
    # Center the data (subtract mean)
    centered = points - mean
    
    # Use OpenCV's PCACompute for robust computation
    # This handles edge cases better than manual covariance computation
    mean_out, eigenvectors = cv2.PCACompute(centered, mean=None)
    
    # eigenvectors[0] is the major axis (largest variance direction)
    # eigenvectors[1] is the minor axis (perpendicular)
    major_axis = eigenvectors[0]
    
    # Calculate angle to rotate major axis to vertical (Y-axis)
    # We want the major axis to align with [0, 1] (vertical)
    # arctan2(y, x) gives angle from positive X-axis
    # For vertical alignment, we need to rotate so major_axis becomes [0, ±1]
    
    # Angle of major axis from positive X-axis
    angle_from_x = math.atan2(major_axis[1], major_axis[0])
    
    # We want to rotate to vertical (90° or -90° from X-axis)
    # Choose the rotation that keeps the tooth mostly upright
    # Target angle is π/2 (90°) for vertical
    target_angle = math.pi / 2
    rotation_angle = target_angle - angle_from_x
    
    # Normalize angle to [-π, π]
    while rotation_angle > math.pi:
        rotation_angle -= 2 * math.pi
    while rotation_angle < -math.pi:
        rotation_angle += 2 * math.pi
    
    return mean, eigenvectors, rotation_angle


def create_rotation_matrix(angle: float, center: Tuple[float, float]) -> np.ndarray:
    """
    Create a 2x3 affine rotation matrix for rotating points around a center.
    
    The rotation matrix performs:
    1. Translate point to origin (subtract center)
    2. Rotate by angle
    3. Translate back (add center)
    
    Rotation formula:
    x' = (x - cx) * cos(θ) - (y - cy) * sin(θ) + cx
    y' = (x - cx) * sin(θ) + (y - cy) * cos(θ) + cy
    
    Args:
        angle: Rotation angle in radians (positive = counter-clockwise)
        center: (cx, cy) pivot point for rotation
        
    Returns:
        2x3 affine transformation matrix
    """
    cx, cy = center
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # 2x3 affine matrix: [[cos, -sin, tx], [sin, cos, ty]]
    # where tx and ty account for the pivot point
    rotation_matrix = np.array([
        [cos_a, -sin_a, cx - cx * cos_a + cy * sin_a],
        [sin_a, cos_a, cy - cx * sin_a - cy * cos_a]
    ], dtype=np.float64)
    
    return rotation_matrix


def rotate_points(points: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Apply affine transformation to rotate points.
    
    Uses OpenCV's transform function for efficient batch processing.
    
    Args:
        points: Nx2 array of [x, y] coordinates
        rotation_matrix: 2x3 affine transformation matrix
        
    Returns:
        Nx2 array of rotated coordinates
    """
    if points is None or len(points) == 0:
        return np.array([])
    
    points = np.array(points, dtype=np.float64)
    
    # OpenCV transform expects points in shape (N, 1, 2)
    points_reshaped = points.reshape(-1, 1, 2)
    
    # Apply transformation
    rotated = cv2.transform(points_reshaped, rotation_matrix)
    
    # Reshape back to Nx2
    return rotated.reshape(-1, 2)


def get_bounding_box(points: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calculate the axis-aligned bounding box of a set of points.
    
    Args:
        points: Nx2 array of [x, y] coordinates
        
    Returns:
        Tuple of (x, y, width, height) where (x, y) is top-left corner
    """
    if points is None or len(points) == 0:
        return (0, 0, 0, 0)
    
    points = np.array(points, dtype=np.float64)
    
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    
    width = x_max - x_min
    height = y_max - y_min
    
    return (x_min, y_min, width, height)


def polygon_from_points(points: List[List[int]]) -> np.ndarray:
    """
    Convert list of pixel coordinates to a boundary polygon.
    
    For scattered pixel coordinates, compute the convex hull to get the boundary.
    This is useful when caries_coordinates is a list of all pixels, not a polygon.
    
    Args:
        points: List of [x, y] pixel coordinates
        
    Returns:
        Nx2 array of polygon boundary points
    """
    if not points or len(points) < 3:
        return np.array(points) if points else np.array([])
    
    points_array = np.array(points, dtype=np.float32)
    
    # If we have enough points, compute convex hull for boundary
    if len(points_array) >= 3:
        hull = cv2.convexHull(points_array)
        return hull.reshape(-1, 2)
    
    return points_array


# =============================================================================
# Main Classification Function
# =============================================================================

def classify_caries_surface(
    tooth_id: str,
    tooth_poly: List[List[float]],
    caries_poly: List[List[float]],
    occlusal_threshold: float = OCCLUSAL_ZONE_THRESHOLD,
    proximal_threshold: float = PROXIMAL_ZONE_THRESHOLD
) -> int:
    """
    Classify the surface location of dental caries using PCA-based orientation.
    
    This function uses geometric analysis to determine which surface of the tooth
    the caries lesion is located on, based on its position relative to the
    orientation-normalized tooth.
    
    Algorithm Steps:
    1. Convert tooth polygon to numpy array
    2. Perform PCA to find the major axis (elongation direction)
    3. Calculate rotation angle to make major axis vertical
    4. Rotate both tooth and caries polygons using tooth centroid as pivot
    5. Calculate bounding box of rotated tooth
    6. Determine relative position of caries centroid within bounding box
    7. Apply classification rules based on jaw type (upper/lower)
    
    Classification Rules:
    - Upper Jaw (11-28): Occlusal surface is at bottom (high Y values)
    - Lower Jaw (31-48): Occlusal surface is at top (low Y values)
    
    - Class 0 (Occlusal): Caries in occlusal zone (biting surface)
    - Class 1 (Proximal): Caries in left/right zones (inter-dental contact)
    - Class 2 (Lingual/Other): Caries in other positions
    
    Args:
        tooth_id: FDI tooth notation (e.g., "11", "26", "45")
        tooth_poly: List of [x, y] coordinates for tooth boundary polygon
        caries_poly: List of [x, y] coordinates for caries boundary/pixels
        occlusal_threshold: Fraction of height for occlusal zone (default 0.20)
        proximal_threshold: Fraction of width for proximal zone (default 0.20)
        
    Returns:
        int: Classification result
            - 0: Occlusal (biting/chewing surface)
            - 1: Proximal (mesial/distal - between teeth)
            - 2: Lingual/Other (tongue-side or other)
            
    Raises:
        ValueError: If input polygons are invalid or empty
    """
    # ==========================================================================
    # Step 0: Input Validation
    # ==========================================================================
    
    if not tooth_poly or len(tooth_poly) < 3:
        raise ValueError("Tooth polygon must have at least 3 points")
    
    if not caries_poly or len(caries_poly) < 1:
        raise ValueError("Caries polygon must have at least 1 point")
    
    # Convert to numpy arrays
    tooth_points = np.array(tooth_poly, dtype=np.float64)
    caries_points = np.array(caries_poly, dtype=np.float64)
    
    # ==========================================================================
    # Step 1: PCA Alignment
    # ==========================================================================
    # Find the principal axes of the tooth to determine its orientation.
    # The major axis indicates the direction of maximum elongation (root-to-crown).
    
    tooth_centroid = compute_centroid(tooth_points)
    _, _, rotation_angle = perform_pca(tooth_points)
    
    # ==========================================================================
    # Step 2: Create Rotation Matrix
    # ==========================================================================
    # Build an affine transformation matrix to rotate points around the tooth centroid.
    # This normalizes the tooth orientation to vertical (major axis parallel to Y-axis).
    
    rotation_matrix = create_rotation_matrix(rotation_angle, tooth_centroid)
    
    # ==========================================================================
    # Step 3: Apply Rotation to Both Polygons
    # ==========================================================================
    # Rotate tooth polygon - this gives us the orientation-normalized tooth
    # Rotate caries polygon - maintains relative position to tooth
    
    rotated_tooth = rotate_points(tooth_points, rotation_matrix)
    rotated_caries = rotate_points(caries_points, rotation_matrix)
    
    # ==========================================================================
    # Step 4: Calculate Bounding Box of Rotated Tooth
    # ==========================================================================
    # After rotation, the axis-aligned bounding box represents the oriented
    # extent of the tooth in normalized coordinates.
    
    bbox_x, bbox_y, bbox_w, bbox_h = get_bounding_box(rotated_tooth)
    
    # Handle edge case: zero dimensions
    if bbox_w <= 0 or bbox_h <= 0:
        return 2  # Default to Lingual/Other for degenerate cases
    
    # ==========================================================================
    # Step 5: Calculate Caries Centroid Position
    # ==========================================================================
    # Find the center of the caries lesion in rotated coordinates
    
    caries_centroid_x, caries_centroid_y = compute_centroid(rotated_caries)
    
    # ==========================================================================
    # Step 6: Calculate Relative Position
    # ==========================================================================
    # Express caries position as fraction of tooth dimensions (0.0 to 1.0)
    # This normalizes the position regardless of tooth size
    
    # Relative X: 0 = left edge, 1 = right edge
    rel_x = (caries_centroid_x - bbox_x) / bbox_w
    
    # Relative Y: 0 = top edge, 1 = bottom edge
    rel_y = (caries_centroid_y - bbox_y) / bbox_h
    
    # Clamp to [0, 1] range to handle caries slightly outside tooth boundary
    rel_x = max(0.0, min(1.0, rel_x))
    rel_y = max(0.0, min(1.0, rel_y))
    
    # ==========================================================================
    # Step 7: Apply Classification Logic Based on Jaw Type
    # ==========================================================================
    # 
    # In dental X-rays with standard orientation:
    # - Upper Jaw (Maxilla): Roots point UP (low Y), Crown/Occlusal points DOWN (high Y)
    # - Lower Jaw (Mandible): Roots point DOWN (high Y), Crown/Occlusal points UP (low Y)
    #
    # After PCA alignment, the tooth is vertical with its major axis along Y.
    # We classify based on which zone the caries centroid falls into.
    
    upper_jaw = is_upper_jaw(tooth_id)
    
    # Determine if caries is in occlusal zone
    if upper_jaw:
        # Upper Jaw: Occlusal surface is at BOTTOM (high Y, rel_y close to 1.0)
        # Occlusal zone: bottom 20% of tooth height
        is_occlusal = rel_y >= (1.0 - occlusal_threshold)
    else:
        # Lower Jaw: Occlusal surface is at TOP (low Y, rel_y close to 0.0)
        # Occlusal zone: top 20% of tooth height
        is_occlusal = rel_y <= occlusal_threshold
    
    # Check for Occlusal first (Class 0)
    if is_occlusal:
        return 0  # Class 0: Occlusal
    
    # Check for Proximal (Class 1)
    # Proximal surfaces are on left (mesial) or right (distal) sides
    is_proximal_left = rel_x <= proximal_threshold
    is_proximal_right = rel_x >= (1.0 - proximal_threshold)
    
    if is_proximal_left or is_proximal_right:
        return 1  # Class 1: Proximal (Mesial/Distal)
    
    # Default: Lingual/Other (Class 2)
    return 2  # Class 2: Lingual/Buccal/Other


# =============================================================================
# Extended Classification with Details
# =============================================================================

def classify_caries_surface_detailed(
    tooth_id: str,
    tooth_poly: List[List[float]],
    caries_poly: List[List[float]],
    occlusal_threshold: float = OCCLUSAL_ZONE_THRESHOLD,
    proximal_threshold: float = PROXIMAL_ZONE_THRESHOLD
) -> Dict:
    """
    Classify caries surface with detailed diagnostic information.
    
    This extended version returns additional information useful for debugging
    and visualization, including:
    - Rotation angle applied
    - Centroid positions
    - Relative coordinates
    - Zone determinations
    
    Args:
        tooth_id: FDI tooth notation
        tooth_poly: Tooth boundary polygon coordinates
        caries_poly: Caries boundary/pixel coordinates
        occlusal_threshold: Fraction of height for occlusal zone
        proximal_threshold: Fraction of width for proximal zone
        
    Returns:
        Dictionary containing:
        - classification: int (0, 1, or 2)
        - surface_name: str ("Occlusal", "Proximal", or "Lingual/Other")
        - tooth_id: str
        - jaw_type: str ("upper" or "lower")
        - rotation_angle_deg: float
        - tooth_centroid: tuple
        - caries_centroid_rotated: tuple
        - relative_position: dict with rel_x, rel_y
        - bounding_box: dict with x, y, width, height
        - rotated_tooth_polygon: list
        - rotated_caries_polygon: list
    """
    # Input validation
    if not tooth_poly or len(tooth_poly) < 3:
        return {
            'classification': -1,
            'surface_name': 'Invalid',
            'error': 'Tooth polygon must have at least 3 points'
        }
    
    if not caries_poly or len(caries_poly) < 1:
        return {
            'classification': -1,
            'surface_name': 'Invalid',
            'error': 'Caries polygon must have at least 1 point'
        }
    
    # Convert to numpy
    tooth_points = np.array(tooth_poly, dtype=np.float64)
    caries_points = np.array(caries_poly, dtype=np.float64)
    
    # PCA analysis
    tooth_centroid = compute_centroid(tooth_points)
    _, eigenvectors, rotation_angle = perform_pca(tooth_points)
    
    # Create rotation matrix and rotate
    rotation_matrix = create_rotation_matrix(rotation_angle, tooth_centroid)
    rotated_tooth = rotate_points(tooth_points, rotation_matrix)
    rotated_caries = rotate_points(caries_points, rotation_matrix)
    
    # Bounding box
    bbox_x, bbox_y, bbox_w, bbox_h = get_bounding_box(rotated_tooth)
    
    # Handle degenerate cases
    if bbox_w <= 0 or bbox_h <= 0:
        return {
            'classification': 2,
            'surface_name': 'Lingual/Other',
            'tooth_id': tooth_id,
            'warning': 'Degenerate bounding box'
        }
    
    # Caries centroid in rotated space
    caries_cx, caries_cy = compute_centroid(rotated_caries)
    
    # Relative positions
    rel_x = max(0.0, min(1.0, (caries_cx - bbox_x) / bbox_w))
    rel_y = max(0.0, min(1.0, (caries_cy - bbox_y) / bbox_h))
    
    # Jaw type and classification
    upper_jaw = is_upper_jaw(tooth_id)
    
    if upper_jaw:
        is_occlusal = rel_y >= (1.0 - occlusal_threshold)
    else:
        is_occlusal = rel_y <= occlusal_threshold
    
    is_proximal_left = rel_x <= proximal_threshold
    is_proximal_right = rel_x >= (1.0 - proximal_threshold)
    
    # Determine classification
    if is_occlusal:
        classification = 0
        surface_name = "Occlusal"
    elif is_proximal_left or is_proximal_right:
        classification = 1
        surface_name = "Proximal"
        if is_proximal_left:
            surface_name = "Proximal (Left/Mesial)"
        else:
            surface_name = "Proximal (Right/Distal)"
    else:
        classification = 2
        surface_name = "Lingual/Other"
    
    return {
        'classification': classification,
        'surface_name': surface_name,
        'tooth_id': tooth_id,
        'jaw_type': 'upper' if upper_jaw else 'lower',
        'rotation_angle_deg': math.degrees(rotation_angle),
        'tooth_centroid': tooth_centroid,
        'caries_centroid_rotated': (caries_cx, caries_cy),
        'relative_position': {
            'rel_x': rel_x,
            'rel_y': rel_y
        },
        'bounding_box': {
            'x': bbox_x,
            'y': bbox_y,
            'width': bbox_w,
            'height': bbox_h
        },
        'rotated_tooth_polygon': rotated_tooth.tolist(),
        'rotated_caries_polygon': rotated_caries.tolist(),
        'zone_checks': {
            'is_occlusal': is_occlusal,
            'is_proximal_left': is_proximal_left,
            'is_proximal_right': is_proximal_right,
            'occlusal_threshold': occlusal_threshold,
            'proximal_threshold': proximal_threshold
        }
    }


# =============================================================================
# Surface Name Lookup
# =============================================================================

SURFACE_NAMES = {
    0: "Occlusal",
    1: "Proximal",
    2: "Lingual/Other"
}


def get_surface_name(classification: int) -> str:
    """
    Get human-readable surface name from classification code.
    
    Args:
        classification: Integer classification (0, 1, or 2)
        
    Returns:
        String name of the surface
    """
    return SURFACE_NAMES.get(classification, "Unknown")


# =============================================================================
# Demo / Test Function
# =============================================================================

def demo():
    """
    Demonstrate the classification function with sample data.
    """
    print("=" * 60)
    print("Dental Caries Surface Classifier - Demo")
    print("=" * 60)
    
    # Sample tooth polygon (simplified rectangle-ish shape representing a molar)
    # This simulates a tooth that is taller than wide (root-to-crown orientation)
    tooth_polygon = [
        [100, 50],   # Top-left (root area)
        [150, 50],   # Top-right (root area)
        [160, 100],  # Upper-right
        [160, 200],  # Lower-right (crown area)
        [100, 200],  # Lower-left (crown area)
        [90, 100],   # Upper-left
    ]
    
    # Test cases with different caries positions
    test_cases = [
        {
            'name': 'Upper Molar - Occlusal Caries',
            'tooth_id': '16',  # Upper right first molar
            'caries': [[120, 180], [130, 185], [125, 190], [140, 188]]  # Bottom area
        },
        {
            'name': 'Upper Molar - Proximal Left Caries',
            'tooth_id': '16',
            'caries': [[95, 120], [100, 125], [98, 130]]  # Left side
        },
        {
            'name': 'Upper Molar - Lingual Caries',
            'tooth_id': '16',
            'caries': [[125, 120], [130, 125], [128, 130]]  # Center area
        },
        {
            'name': 'Lower Molar - Occlusal Caries',
            'tooth_id': '46',  # Lower right first molar
            'caries': [[120, 60], [130, 65], [125, 55]]  # Top area (occlusal for lower)
        },
        {
            'name': 'Lower Molar - Proximal Right Caries',
            'tooth_id': '46',
            'caries': [[155, 120], [158, 125], [156, 130]]  # Right side
        },
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print("-" * 40)
        
        result = classify_caries_surface_detailed(
            test['tooth_id'],
            tooth_polygon,
            test['caries']
        )
        
        print(f"  Tooth ID: {result['tooth_id']} ({result['jaw_type']} jaw)")
        print(f"  Classification: {result['classification']} - {result['surface_name']}")
        print(f"  Rotation: {result['rotation_angle_deg']:.2f}°")
        print(f"  Relative Position: X={result['relative_position']['rel_x']:.3f}, "
              f"Y={result['relative_position']['rel_y']:.3f}")
    
    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    demo()

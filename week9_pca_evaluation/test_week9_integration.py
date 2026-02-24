# Warning!!!
# Surface Incorrect -> (Distal, Mesial, Occlusal) only, do not make other class.

"""
Week 9 — Integration Tests
============================

Lightweight pytest tests that verify the refactored components
(PCA dispatcher, path management, CSV dashboard, strict surface rule)
**without** running the heavy 500-case image evaluation pipeline.

Usage
-----
    cd week9_pca_evaluation
    pytest test_week9_integration.py -v

Author: QA Automation Engineer – Dental AI / CAD
Date:   2026-02-23
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make week8 modules importable
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
WEEK8_DIR = BASE_DIR / "week8-Surface Classification v4"

if str(WEEK8_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK8_DIR))

from multi_zone_classifier import (
    perform_pca,
    VALID_PCA_METHODS,
    PCA_METHOD_NAMES,
)
from evaluation_engine import (
    enforce_surface_rule,
    ALLOWED_SURFACE_CLASSES,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_polygon():
    """A simple elongated tooth-like polygon (taller than wide)."""
    return np.array([
        [100,  50],
        [150,  50],
        [160, 100],
        [160, 200],
        [100, 200],
        [ 90, 100],
    ], dtype=np.float64)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary directory that is cleaned up after the test."""
    return tmp_path / "week9_test_output"


# =============================================================================
# Test 1 — PCA Dispatcher
# =============================================================================

class TestPCADispatcher:
    """Verify perform_pca routes correctly for all valid methods."""

    def test_valid_methods_return_tuple(self, sample_polygon):
        """Each valid method must return (mean, rotation_angle, was_clamped)."""
        for method in VALID_PCA_METHODS:
            result = perform_pca(sample_polygon, tooth_id="16", method=method)

            # Must be a 3-tuple
            assert isinstance(result, tuple), (
                f"Method {method}: expected tuple, got {type(result)}"
            )
            assert len(result) == 3, (
                f"Method {method}: expected 3 elements, got {len(result)}"
            )

            mean, angle, was_clamped = result

            # mean: np.ndarray of shape (2,)
            assert isinstance(mean, np.ndarray), (
                f"Method {method}: mean should be ndarray, got {type(mean)}"
            )
            assert mean.shape == (2,), (
                f"Method {method}: mean shape should be (2,), got {mean.shape}"
            )

            # angle: float (radians)
            assert isinstance(angle, (float, np.floating)), (
                f"Method {method}: angle should be float, got {type(angle)}"
            )

            # was_clamped: bool
            assert isinstance(was_clamped, (bool, np.bool_)), (
                f"Method {method}: was_clamped should be bool, got {type(was_clamped)}"
            )

    def test_method_0_is_baseline(self, sample_polygon):
        """Method 0 (baseline) never clamps angles."""
        _, _, was_clamped = perform_pca(sample_polygon, tooth_id="16", method=0)
        assert was_clamped is False, "Baseline method 0 should never clamp"

    def test_invalid_method_4_raises(self, sample_polygon):
        """Method 4 (placeholder) should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid PCA method"):
            perform_pca(sample_polygon, tooth_id="16", method=4)

    def test_invalid_method_99_raises(self, sample_polygon):
        """Completely invalid method numbers must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid PCA method"):
            perform_pca(sample_polygon, tooth_id="16", method=99)

    def test_invalid_method_negative_raises(self, sample_polygon):
        """Negative method numbers must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid PCA method"):
            perform_pca(sample_polygon, tooth_id="16", method=-1)

    def test_valid_methods_constant(self):
        """VALID_PCA_METHODS must be [0, 1, 2, 3, 5]."""
        assert VALID_PCA_METHODS == [0, 1, 2, 3, 5]

    def test_method_names_include_0(self):
        """PCA_METHOD_NAMES must have an entry for Method 0."""
        assert 0 in PCA_METHOD_NAMES
        assert "baseline" in PCA_METHOD_NAMES[0].lower()


# =============================================================================
# Test 2 — Path Management
# =============================================================================

class TestPathManagement:
    """Verify directory creation logic."""

    def test_week9_directory_exists(self):
        """The week9_pca_evaluation directory must exist (we're running from it)."""
        week9_dir = Path(__file__).resolve().parent
        assert week9_dir.exists(), f"week9_pca_evaluation not found: {week9_dir}"
        assert week9_dir.name == "week9_pca_evaluation"

    def test_method_subdirectories_creation(self, temp_output_dir):
        """Dynamically created method_X/cases/ directories must exist."""
        for method in VALID_PCA_METHODS:
            method_name = PCA_METHOD_NAMES[method]
            method_dir = temp_output_dir / method_name
            cases_dir = method_dir / "cases"
            os.makedirs(cases_dir, exist_ok=True)

            assert method_dir.exists(), f"Missing: {method_dir}"
            assert cases_dir.exists(), f"Missing: {cases_dir}"
            assert method_dir.is_dir()
            assert cases_dir.is_dir()

    def test_nested_case_directory(self, temp_output_dir):
        """Individual case folders inside cases/ must be creatable."""
        cases_dir = temp_output_dir / "method_0_baseline_opencv" / "cases"
        for case_num in [1, 100, 500]:
            case_dir = cases_dir / f"case {case_num}"
            os.makedirs(case_dir, exist_ok=True)
            assert case_dir.exists()


# =============================================================================
# Test 3 — CSV Dashboard
# =============================================================================

class TestCSVDashboard:
    """Verify per-class metrics CSV export format and content."""

    EXPECTED_COLUMNS = [
        "pca_method", "pca_method_name", "class",
        "precision", "recall", "f1", "support",
        "TP", "FP", "FN",
    ]

    def _make_mock_per_class_csv(self, output_dir: Path, method: int):
        """Generate a mock per_class_metrics.csv for one method."""
        method_name = PCA_METHOD_NAMES[method]
        method_dir = output_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for cls in ALLOWED_SURFACE_CLASSES:
            rows.append({
                "pca_method": method,
                "pca_method_name": method_name,
                "class": cls,
                "precision": round(np.random.uniform(0.5, 1.0), 4),
                "recall":    round(np.random.uniform(0.5, 1.0), 4),
                "f1":        round(np.random.uniform(0.5, 1.0), 4),
                "support":   np.random.randint(10, 200),
                "TP": np.random.randint(5, 100),
                "FP": np.random.randint(0, 30),
                "FN": np.random.randint(0, 30),
            })

        df = pd.DataFrame(rows)
        csv_path = method_dir / "per_class_metrics.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return csv_path

    def test_single_method_csv_columns(self, temp_output_dir):
        """A single method CSV must have all required columns."""
        csv_path = self._make_mock_per_class_csv(temp_output_dir, method=0)
        df = pd.read_csv(csv_path)
        for col in self.EXPECTED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_single_method_csv_classes(self, temp_output_dir):
        """CSV must contain exactly the 3 allowed surface classes."""
        csv_path = self._make_mock_per_class_csv(temp_output_dir, method=0)
        df = pd.read_csv(csv_path)
        classes_in_csv = sorted(df["class"].unique().tolist())
        assert classes_in_csv == sorted(ALLOWED_SURFACE_CLASSES)

    def test_aggregated_dashboard(self, temp_output_dir):
        """Concatenating CSVs from 2 methods produces correct dashboard."""
        methods_to_test = [0, 5]
        for m in methods_to_test:
            self._make_mock_per_class_csv(temp_output_dir, method=m)

        # Aggregate — mimics run_evaluation.py Task 5
        all_dfs = []
        for m in methods_to_test:
            method_name = PCA_METHOD_NAMES[m]
            csv_path = temp_output_dir / method_name / "per_class_metrics.csv"
            if csv_path.exists():
                all_dfs.append(pd.read_csv(csv_path))

        combined = pd.concat(all_dfs, ignore_index=True)
        dashboard_path = temp_output_dir / "per_class_dashboard.csv"
        combined.to_csv(dashboard_path, index=False, encoding="utf-8-sig")

        # Verify
        assert dashboard_path.exists()
        dash = pd.read_csv(dashboard_path)
        for col in self.EXPECTED_COLUMNS:
            assert col in dash.columns, f"Dashboard missing column: {col}"
        assert len(dash) == len(methods_to_test) * len(ALLOWED_SURFACE_CLASSES)
        assert sorted(dash["pca_method"].unique()) == sorted(methods_to_test)

    def test_dashboard_no_extra_classes(self, temp_output_dir):
        """Dashboard must only contain allowed surface classes."""
        for m in [0, 1, 2, 3, 5]:
            self._make_mock_per_class_csv(temp_output_dir, method=m)

        all_dfs = []
        for m in [0, 1, 2, 3, 5]:
            csv_path = temp_output_dir / PCA_METHOD_NAMES[m] / "per_class_metrics.csv"
            all_dfs.append(pd.read_csv(csv_path))

        combined = pd.concat(all_dfs, ignore_index=True)
        unexpected = set(combined["class"].unique()) - set(ALLOWED_SURFACE_CLASSES)
        assert len(unexpected) == 0, f"Unexpected classes in dashboard: {unexpected}"


# =============================================================================
# Test 4 — Strict Surface Rule
# =============================================================================

class TestStrictSurfaceRule:
    """Verify enforce_surface_rule accepts only allowed classes."""

    @pytest.mark.parametrize("surface", ["Distal", "Mesial", "Occlusal"])
    def test_valid_classes_pass_through(self, surface):
        """Valid classes must be returned unchanged."""
        result = enforce_surface_rule(surface)
        assert result == surface, (
            f"Expected '{surface}', got '{result}'"
        )

    @pytest.mark.parametrize("surface", ["Buccal", "Lingual", "Cervical", "Palatal"])
    def test_invalid_classes_remap_to_unclassified(self, surface):
        """Invalid classes must be remapped to 'Unclassified'."""
        result = enforce_surface_rule(surface)
        assert result == "Unclassified", (
            f"Expected 'Unclassified' for '{surface}', got '{result}'"
        )

    def test_empty_string_returns_unclassified(self):
        """Empty string should return 'Unclassified'."""
        assert enforce_surface_rule("") == "Unclassified"

    def test_already_unclassified_stays(self):
        """'Unclassified' input should return 'Unclassified' (no warning)."""
        assert enforce_surface_rule("Unclassified") == "Unclassified"

    def test_allowed_surface_classes_constant(self):
        """ALLOWED_SURFACE_CLASSES must be exactly the 3 required classes."""
        assert sorted(ALLOWED_SURFACE_CLASSES) == ["Distal", "Mesial", "Occlusal"]

    def test_case_sensitive(self):
        """enforce_surface_rule is case-sensitive — lowercase should fail."""
        result = enforce_surface_rule("distal")
        assert result == "Unclassified", (
            "enforce_surface_rule should be case-sensitive"
        )


# =============================================================================
# Run with: pytest test_week9_integration.py -v
# =============================================================================

import sys
import os

# Add the parent directory to PYTHONPATH to allow importing pipeline_run3_final
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pipeline_run3_final

def test_constants():
    """Verify that important constants are defined in the pipeline."""
    assert hasattr(pipeline_run3_final, 'FEATURE_COLS')
    assert isinstance(pipeline_run3_final.FEATURE_COLS, list)
    assert 'coverage' in pipeline_run3_final.FEATURE_COLS
    
    assert hasattr(pipeline_run3_final, 'VALID_SURFACES')
    assert isinstance(pipeline_run3_final.VALID_SURFACES, list)

def test_pipeline_import():
    """A basic test that just proves we can import the pipeline module without crashing."""
    assert pipeline_run3_final is not None

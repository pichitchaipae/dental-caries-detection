"""
FDI tooth numbering helpers.

The panoramic YOLO model (Tooth_seg_pano_20250319.pt) has 32 named classes
where each class name IS the FDI number: {0: '11', 1: '12', ..., 31: '48'}.

No positional derivation is needed — we read directly from model.names.
"""

from __future__ import annotations


def class_id_to_fdi(class_id: int, model_names: dict[int, str]) -> int:
    """
    Convert a YOLO class index to an FDI tooth number (integer).

    Args:
        class_id:    Integer class index from YOLO prediction.
        model_names: The model's .names dict, e.g. {0: '11', 1: '12', ...}

    Returns:
        FDI number as int, e.g. 36.

    Raises:
        KeyError:  if class_id is not in model_names.
        ValueError: if the name cannot be parsed as an integer.
    """
    name = model_names[class_id]
    try:
        return int(name)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Cannot parse FDI from model class name '{name}' (class_id={class_id})"
        ) from exc


def fdi_to_str(fdi: int) -> str:
    """Return FDI as zero-padded two-digit string (matches existing pipeline convention)."""
    return str(fdi)

"""Tests for the crop-based Stage 1 pipeline (no real weights required)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from app.pipeline.detection import (
    ToothDetection,
    dedupe_detections,
    refine_with_crop_segmenter,
    run_caries_on_crops,
    run_stage1,
)


# ---------------------------------------------------------------------------
# Fakes that mimic the small part of the Ultralytics API we use
# ---------------------------------------------------------------------------

class FakeBox:
    def __init__(self, cls: int, conf: float, xyxy):
        self.cls = np.array([cls], dtype=np.float32)
        self.conf = np.array([conf], dtype=np.float32)
        self.xyxy = np.array([xyxy], dtype=np.float32)


class FakeMasks:
    def __init__(self, polygons):
        self.xy = [np.asarray(p, dtype=np.float32) for p in polygons]


class FakeResult:
    def __init__(self, boxes, polygons, shape):
        self.boxes = boxes
        self.masks = FakeMasks(polygons) if polygons is not None else None
        self.orig_shape = shape


class FakeModel:
    def __init__(self, names: dict[int, str], result: FakeResult):
        self.names = names
        self._result = result
        self.calls = []

    def predict(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return [self._result]


def _tooth(fdi=11, bbox=(100, 50, 60, 80), polygon=None, conf=0.9):
    return ToothDetection(
        fdi=fdi,
        bbox_xywh=bbox,
        tooth_polygon=polygon or [[100, 50], [160, 50], [160, 130], [100, 130]],
        pano_confidence=conf,
    )


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------

def test_dedupe_drops_overlapping_same_fdi():
    dets = [
        _tooth(fdi=11, bbox=(0, 0, 100, 100), conf=0.9),
        _tooth(fdi=11, bbox=(10, 10, 100, 100), conf=0.8),
        _tooth(fdi=12, bbox=(200, 0, 100, 100), conf=0.7),
    ]
    out = dedupe_detections(dets, iou_thresh=0.5)
    assert len(out) == 2
    assert [d.fdi for d in out] == [11, 12]
    assert out[0].pano_confidence == 0.9


def test_dedupe_keeps_one_per_fdi_even_without_overlap():
    dets = [
        _tooth(fdi=11, bbox=(0, 0, 100, 100), conf=0.5),
        _tooth(fdi=11, bbox=(500, 500, 100, 100), conf=0.9),
    ]
    out = dedupe_detections(dets)
    assert len(out) == 1
    assert out[0].pano_confidence == 0.9


# ---------------------------------------------------------------------------
# Caries on crops
# ---------------------------------------------------------------------------

def test_caries_on_crop_is_transformed_back_to_pano():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    det = _tooth()
    det.crop_bbox_xyxy = (80, 30, 180, 150)

    caries = FakeModel(
        {0: "caries"},
        FakeResult(
            boxes=[FakeBox(0, 0.8, [10, 10, 30, 30])],
            polygons=[[[10, 10], [30, 10], [30, 30], [10, 30]]],
            shape=(120, 100),
        ),
    )

    run_caries_on_crops(image, [det], caries, caries_conf=0.25, caries_iou=0.5)
    assert len(det.caries_lesions) == 1
    lesion = det.caries_lesions[0]
    # crop offset is (80, 30)
    assert lesion.bbox_xyxy == (90.0, 40.0, 110.0, 60.0)
    assert lesion.centroid == pytest.approx((100.0, 50.0), abs=1e-3)
    assert lesion.polygon[0] == [90.0, 40.0]
    assert det.caries_points is not None
    assert det.caries_points.shape[1] == 2


def test_caries_on_crop_bbox_fallback_when_no_mask():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    det = _tooth()
    det.crop_bbox_xyxy = (80, 30, 180, 150)
    caries = FakeModel(
        {0: "caries"},
        FakeResult(boxes=[FakeBox(0, 0.6, [10, 10, 30, 30])], polygons=None, shape=(120, 100)),
    )
    run_caries_on_crops(image, [det], caries)
    assert len(det.caries_lesions) == 1
    assert det.caries_lesions[0].bbox_xyxy == (90.0, 40.0, 110.0, 60.0)


def test_caries_rejects_multi_class_model():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    det = _tooth()
    det.crop_bbox_xyxy = (80, 30, 180, 150)
    bad = FakeModel(
        {0: "occlusal", 1: "proximal", 2: "lingual"},
        FakeResult(boxes=[], polygons=None, shape=(120, 100)),
    )
    with pytest.raises(ValueError, match="1-class"):
        run_caries_on_crops(image, [det], bad)


def test_caries_rejects_wrong_single_class_name():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    det = _tooth()
    det.crop_bbox_xyxy = (80, 30, 180, 150)
    bad = FakeModel(
        {0: "decay"},
        FakeResult(boxes=[], polygons=None, shape=(120, 100)),
    )
    with pytest.raises(ValueError, match="'caries'"):
        run_caries_on_crops(image, [det], bad)


def test_caries_missing_model_is_a_noop():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    det = _tooth()
    det.crop_bbox_xyxy = (80, 30, 180, 150)
    run_caries_on_crops(image, [det], None)
    assert det.caries_lesions == []


# ---------------------------------------------------------------------------
# Crop-Seg preprocessing
# ---------------------------------------------------------------------------

class _RecordingResize:
    """Stand-in for detectron2 ResizeShortestEdge."""

    def __init__(self, new_w: int, new_h: int):
        self.new_w = new_w
        self.new_h = new_h
        self.seen = None

    def get_transform(self, image):
        self.seen = image
        return self

    def apply_image(self, image):
        return cv2.resize(image, (self.new_w, self.new_h))


class _FakeInstances:
    def __init__(self, mask):
        self.scores = torch.tensor([0.9])
        self.pred_masks = torch.tensor(mask[None, ...])

    def __len__(self):
        return 1


class _FakeCropModel:
    def __init__(self, instances):
        self._instances = instances
        self.inputs = []

    def __call__(self, inputs):
        self.inputs.append(inputs[0])
        return [{"instances": self._instances}]


def test_crop_seg_applies_rgb_and_resize_then_maps_mask_back():
    # Distinct channel values prove the BGR->RGB swap.
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)  # B, G, R
    det = _tooth()
    det.tooth_polygon = [[100, 50], [160, 50], [160, 130], [100, 130]]

    # Model-space mask lives at the resized resolution (40x60 here), not at
    # the original crop size (100x120).
    resized_mask = np.zeros((40, 60), dtype=np.uint8)
    resized_mask[10:30, 15:45] = 1
    resize = _RecordingResize(new_w=60, new_h=40)
    model = _FakeCropModel(_FakeInstances(resized_mask))

    out = refine_with_crop_segmenter(
        image,
        [det],
        {"model": model, "resize": resize},
    )

    crop = image[30:150, 80:180]
    # DefaultPredictor receives RGB...
    assert resize.seen is not None
    assert np.array_equal(resize.seen[0, 0], crop[0, 0][::-1])
    # ...resized to the model input, with the original h/w passed as metadata.
    inp = model.inputs[0]
    assert tuple(inp["image"].shape) == (3, 40, 60)
    assert (inp["height"], inp["width"]) == (120, 100)

    assert out[0].mask_source == "crop_seg"
    # The mask was mapped back to crop space, then offset into pano space.
    xs = [p[0] for p in out[0].tooth_polygon]
    ys = [p[1] for p in out[0].tooth_polygon]
    assert 80 < min(xs) and max(xs) < 180
    assert 30 < min(ys) and max(ys) < 150


class _EmptyInstances:
    def __init__(self):
        self.scores = torch.tensor([])
        self.pred_masks = torch.zeros((0, 10, 10))

    def __len__(self):
        return 0


def test_crop_seg_without_instances_keeps_pano_mask():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    det = _tooth()
    det.tooth_polygon = [[100, 50], [160, 50], [160, 130], [100, 130]]
    original_polygon = [list(p) for p in det.tooth_polygon]

    model = _FakeCropModel(_EmptyInstances())

    refine_with_crop_segmenter(
        image,
        [det],
        {"model": model, "resize": _RecordingResize(60, 40)},
    )
    assert det.mask_source == "pano"
    assert det.tooth_polygon == original_polygon


# ---------------------------------------------------------------------------
# Full Stage 1 (pano -> crop -> caries -> pano)
# ---------------------------------------------------------------------------

def test_run_stage1_orders_crop_before_caries(tmp_path):
    img_path = tmp_path / "case.png"
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    assert cv2.imwrite(str(img_path), image)

    pano = FakeModel(
        {0: "11"},
        FakeResult(
            boxes=[FakeBox(0, 0.9, [100, 50, 160, 130])],
            polygons=[[[100, 50], [160, 50], [160, 130], [100, 130]]],
            shape=(200, 300),
        ),
    )
    caries = FakeModel(
        {0: "caries"},
        FakeResult(
            boxes=[FakeBox(0, 0.7, [10, 10, 30, 30])],
            polygons=[[[10, 10], [30, 10], [30, 30], [10, 30]]],
            shape=(120, 100),
        ),
    )

    detections, loaded = run_stage1(
        str(img_path),
        {"pano": pano, "crop": None, "caries": caries},
        detection_threshold=0.25,
    )

    assert loaded.shape == image.shape
    assert len(detections) == 1
    det = detections[0]
    assert det.fdi == 11
    assert det.crop_bbox_xyxy == (80, 30, 180, 150)
    assert det.mask_source == "pano"          # Crop-Seg disabled in this test
    assert len(det.caries_lesions) == 1
    assert det.caries_lesions[0].bbox_xyxy == (90.0, 40.0, 110.0, 60.0)
    # The caries model was called on the crop, not on the pano path.
    assert caries.calls[0][1]["source"].shape == (120, 100, 3)


def test_run_stage1_suppresses_neighbour_teeth_in_the_crop(tmp_path):
    img_path = tmp_path / "case.png"
    image = np.full((200, 400, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(img_path), image)

    pano = FakeModel(
        {0: "11", 1: "12"},
        FakeResult(
            boxes=[
                FakeBox(0, 0.9, [100, 50, 160, 130]),
                FakeBox(1, 0.9, [170, 50, 230, 130]),
            ],
            polygons=[
                [[100, 50], [160, 50], [160, 130], [100, 130]],
                [[170, 50], [230, 50], [230, 130], [170, 130]],
            ],
            shape=(200, 400),
        ),
    )
    caries = FakeModel(
        {0: "caries"},
        FakeResult(boxes=[], polygons=None, shape=(120, 100)),
    )

    run_stage1(
        str(img_path),
        {"pano": pano, "crop": None, "caries": caries},
        detection_threshold=0.25,
        suppress_neighbours=True,
    )

    # First call is FDI 11, crop bbox (80, 30, 180, 150).  The neighbour starts
    # at pano x=170; the target mask is dilated 15 px, so only x≈176..180 of
    # the neighbour is removed.  (178, 100) -> crop (98, 70).
    first_crop = caries.calls[0][1]["source"]
    assert first_crop[70, 98].tolist() == [0, 0, 0]
    # Pixels inside the target tooth stay untouched.
    assert first_crop[70, 50].tolist() == [255, 255, 255]

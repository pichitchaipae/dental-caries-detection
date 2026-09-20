"""Evaluate caries confidence thresholds against the annotated case summary."""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path


def load_ground_truth(summary_path: Path) -> dict[str, set[str]]:
    ground_truth = {}
    with summary_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            case_id = row["case_id"].strip().lower()
            ground_truth[case_id] = {
                tooth.strip()
                for tooth in row["Teeth_with_Caries"].split(",")
                if tooth.strip()
            }
    return ground_truth


def find_case_image(dataset_dir: Path, case_id: str) -> Path | None:
    case_dir = dataset_dir / case_id
    images = sorted(
        path for path in case_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ) if case_dir.is_dir() else []
    return images[0] if images else None


def evaluate_thresholds(
    root: Path,
    thresholds: list[float],
    min_ownership_score: float,
    max_cases: int | None,
    device: str | int | None = None,
) -> dict:
    backend_dir = root / "caries_surface_classification" / "app" / "backend"
    sys.path.insert(0, str(backend_dir))

    from services.caries_detection import CariesDetectionService
    from services.tooth_detection import ToothDetectionService

    summary_path = root / "data" / "caries_per_case_summary.csv"
    dataset_dir = root / "data" / "500 cases with annotation"
    ground_truth = load_ground_truth(summary_path)
    case_ids = sorted(
        ground_truth,
        key=lambda value: int(value.replace("case ", "")),
    )
    if max_cases is not None:
        case_ids = case_ids[:max_cases]

    tooth_detector = ToothDetectionService()
    caries_detector = CariesDetectionService(device=device)
    prepared_cases = []
    for case_id in case_ids:
        image_path = find_case_image(dataset_dir, case_id)
        if image_path is None:
            continue
        detections = tooth_detector.detect_teeth(str(image_path))
        raw_detections = caries_detector.detect_caries_raw(
            str(image_path), confidence=min(thresholds)
        )
        prepared_cases.append((case_id, image_path, detections, raw_detections))

    results = {}
    for threshold in thresholds:
        threshold_key = f"{threshold:g}"
        aggregate = {
            "cases_evaluated": len(prepared_cases),
            "exact_case_matches": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "predicted_teeth": 0,
            "expected_teeth": 0,
            "case_results": [],
        }
        for case_id, image_path, detections, raw_detections in prepared_cases:
            caries_map = caries_detector.detect_caries(
                str(image_path),
                detections,
                confidence=threshold,
                min_ownership_score=min_ownership_score,
                raw_detections=raw_detections,
            )
            predicted = {
                detection.fdi
                for detection in detections
                if detection.detection_id in caries_map
            }
            expected = ground_truth[case_id]
            true_positives = len(predicted & expected)
            false_positives = len(predicted - expected)
            false_negatives = len(expected - predicted)
            exact_match = predicted == expected

            aggregate["exact_case_matches"] += int(exact_match)
            aggregate["true_positives"] += true_positives
            aggregate["false_positives"] += false_positives
            aggregate["false_negatives"] += false_negatives
            aggregate["predicted_teeth"] += len(predicted)
            aggregate["expected_teeth"] += len(expected)
            aggregate["case_results"].append({
                "case_id": case_id,
                "expected": sorted(expected),
                "predicted": sorted(predicted),
                "exact_match": exact_match,
                "false_positives": sorted(predicted - expected),
                "false_negatives": sorted(expected - predicted),
            })

        tp = aggregate["true_positives"]
        fp = aggregate["false_positives"]
        fn = aggregate["false_negatives"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        aggregate["precision"] = precision
        aggregate["recall"] = recall
        aggregate["f1"] = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        aggregate["exact_match_rate"] = (
            aggregate["exact_case_matches"] / aggregate["cases_evaluated"]
            if aggregate["cases_evaluated"] else 0.0
        )
        results[threshold_key] = aggregate
        print(
            f"threshold={threshold_key} cases={aggregate['cases_evaluated']} "
            f"exact={aggregate['exact_match_rate']:.3f} "
            f"precision={precision:.3f} recall={recall:.3f} f1={aggregate['f1']:.3f} "
            f"fp={fp} fn={fn}"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--thresholds", default="0.03,0.02,0.01,0.005")
    parser.add_argument("--min-ownership-score", type=float, default=0.25)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. cpu, 0, or cuda:0. Defaults to Ultralytics auto-selection.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    thresholds = [float(value.strip()) for value in args.thresholds.split(",")]
    results = evaluate_thresholds(
        args.root.resolve(),
        thresholds,
        args.min_ownership_score,
        args.max_cases,
        device=None if args.device in {None, "auto"} else args.device,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
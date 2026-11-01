"""Exploratory audit of model behavior on unlabeled out-of-distribution images."""

import argparse
from collections import Counter
from pathlib import Path

from artifact_io import atomic_write_text
from paths import OOD_RESULTS_PATH, SUPPORTED_IMAGE_EXTENSIONS
from predict import predict_with_uncertainty


def collect_images(source_dir: Path) -> list[Path]:
    """Collect supported images recursively from an OOD source directory."""
    source_dir = Path(source_dir).expanduser().resolve()
    if not source_dir.exists():
        raise ValueError(f"OOD source path does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"OOD source path is not a directory: {source_dir}")

    images = sorted(
        image
        for image in source_dir.rglob("*")
        if image.is_file() and image.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"OOD source contains no supported images: {source_dir}")
    return images


def evaluate_ood(source_dir: Path, output_path: Path = OOD_RESULTS_PATH) -> str:
    """Run the existing inference engine without treating scores as OOD proof."""
    source_dir = Path(source_dir).expanduser().resolve()
    images = collect_images(source_dir)
    top_scores = []
    screen_passed = 0
    empty_sets = 0
    set_sizes = Counter()

    for image_path in images:
        result = predict_with_uncertainty(image_path)
        top_scores.append(float(result["top_confidence"]))
        screen_passed += int(bool(result["leaf_screen_passed"]))
        empty_sets += int(result["prediction_set_status"] == "empty")
        set_sizes[result["set_size"]] += 1

    total = len(images)
    report = [
        "=" * 70,
        "  EXPLORATORY OOD BEHAVIOR AUDIT",
        "=" * 70,
        f"Source Directory       : {source_dir}",
        f"Images Audited         : {total}",
        f"Top Score Mean         : {sum(top_scores) / total:.4f}",
        f"Top Score Minimum      : {min(top_scores):.4f}",
        f"Top Score Maximum      : {max(top_scores):.4f}",
        f"20% Heuristic Passed   : {screen_passed}/{total} ({screen_passed / total * 100:.2f}%)",
        f"Empty Prediction Sets  : {empty_sets}/{total} ({empty_sets / total * 100:.2f}%)",
        "Set Size Distribution:",
    ]
    for set_size in sorted(set_sizes):
        report.append(f"  {set_size}: {set_sizes[set_size]}")
    report.extend(
        [
            "-" * 70,
            "This is an exploratory audit of softmax and conformal-set behavior.",
            "It does not validate OOD detection and does not provide coverage guarantees.",
            "=" * 70,
        ]
    )
    output = "\n".join(report) + "\n"
    atomic_write_text(output_path, output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit model behavior on unlabeled out-of-distribution images."
    )
    parser.add_argument("--src", required=True, type=Path, help="OOD image directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=OOD_RESULTS_PATH,
        help=f"Audit report path (default: {OOD_RESULTS_PATH})",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        print(evaluate_ood(args.src, args.output))
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

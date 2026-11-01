"""Validation helpers shared by dataset preparation and ML pipeline stages."""

from pathlib import Path

from class_mapping import EXPECTED_CLASSES, validate_class_names
from paths import SUPPORTED_IMAGE_EXTENSIONS


def supported_images(directory: Path) -> list[Path]:
    """Return supported image files directly inside a class directory."""
    return sorted(
        image
        for image in directory.iterdir()
        if image.is_file() and image.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def validate_image_split(split_dir: Path, source: str) -> None:
    """Require every expected class and at least one supported image per class."""
    split_dir = Path(split_dir)
    if not split_dir.exists() or not split_dir.is_dir():
        raise ValueError(f"Dataset split '{source}' does not exist: {split_dir}")

    class_dirs = [directory for directory in split_dir.iterdir() if directory.is_dir()]
    validate_class_names((directory.name for directory in class_dirs), source)

    empty_classes = [
        class_name
        for class_name in EXPECTED_CLASSES
        if not supported_images(split_dir / class_name)
    ]
    if empty_classes:
        raise ValueError(
            f"Dataset split '{source}' has no supported images for: "
            f"{', '.join(empty_classes)}"
        )

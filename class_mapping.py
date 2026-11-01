"""Shared validation for the project's fixed tomato disease label set."""

from collections.abc import Iterable, Mapping


EXPECTED_CLASSES = (
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "Tomato_healthy",
)


def validate_class_names(class_names: Iterable[str], source: str) -> None:
    """Require exactly the supported tomato classes, with a useful error."""
    names = set(class_names)
    expected = set(EXPECTED_CLASSES)
    missing = sorted(expected - names)
    unexpected = sorted(names - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing expected tomato classes: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected classes: {', '.join(unexpected)}")
        raise ValueError(f"Invalid classes in {source}: {'; '.join(details)}")


def validate_class_mapping(mapping: Mapping[str, int], source: str) -> dict[str, int]:
    """Validate class names and contiguous ImageFolder-compatible indices."""
    validate_class_names(mapping.keys(), source)
    indices = list(mapping.values())
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise ValueError(f"Class indices in {source} must be integers.")
    if set(indices) != set(range(len(mapping))):
        raise ValueError(f"Class indices in {source} must be unique and start at 0.")
    return dict(mapping)


def ensure_same_class_mapping(
    expected: Mapping[str, int], actual: Mapping[str, int], source: str
) -> None:
    """Raise when a dataset or artifact uses a different label-to-index mapping."""
    validate_class_mapping(expected, "reference class mapping")
    validate_class_mapping(actual, source)
    if dict(expected) != dict(actual):
        raise ValueError(f"Class mapping in {source} does not match the reference mapping.")

"""Deterministic class-stratified sampling for calibration and evaluation."""

from collections import defaultdict
from collections.abc import Sequence
from random import Random


def stratified_sample_indices(
    targets: Sequence[int], max_samples: int, seed: int = 42
) -> list[int]:
    """Select at most ``max_samples`` indices while distributing them by class."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if not targets:
        return []

    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        indices_by_class[int(target)].append(index)

    classes = sorted(indices_by_class)
    rng = Random(seed)
    for indices in indices_by_class.values():
        rng.shuffle(indices)

    sample_count = min(len(targets), max_samples)
    base_quota, _ = divmod(sample_count, len(classes))
    quotas = {
        class_index: min(base_quota, len(indices_by_class[class_index]))
        for class_index in classes
    }

    remaining = sample_count - sum(quotas.values())
    while remaining:
        progressed = False
        for class_index in classes:
            if quotas[class_index] < len(indices_by_class[class_index]):
                quotas[class_index] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            break

    selected = []
    for class_index in classes:
        selected.extend(indices_by_class[class_index][: quotas[class_index]])
    return selected

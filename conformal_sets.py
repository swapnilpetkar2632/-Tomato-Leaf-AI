"""Shared split-conformal prediction-set construction."""

from collections.abc import Sequence


def prediction_set_indices(probabilities: Sequence[float], q_hat: float) -> list[int]:
    """Return classes meeting the calibrated score threshold; an empty set stays empty."""
    return [
        index
        for index, probability in enumerate(probabilities)
        if 1.0 - float(probability) <= q_hat
    ]

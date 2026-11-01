"""
calibrate.py
------------
STAGE 4 — Conformal Calibration

Conformal Prediction Theory:
-----------------------------
1. Nonconformity Score:
   For each calibration sample (x_i, y_i), we compute the softmax probability vector s(x_i).
   The nonconformity score s_i measuring how "surprising" the true label y_i is to the model:
       s_i = 1 - s(x_i)_{y_i}

2. Finite-Sample Calibration Quantile (q_hat):
   With target coverage (1 - alpha) = 0.75 (alpha = 0.25) and n calibration samples:
       level = min(1.0, ceil((n + 1) * (1 - alpha)) / n)
   q_hat is computed as the `level`-quantile of sorted calibration scores using
   higher interpolation: np.quantile(scores, level, method='higher').
"""

import math
import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from class_mapping import ensure_same_class_mapping, validate_class_mapping
from artifact_io import atomic_write_json
from dataset_validation import validate_image_split
from paths import CLASS_MAP_PATH, DATA_DIR, MODEL_PATH, THRESH_PATH
from sampling import stratified_sample_indices

DATA_CALIB  = DATA_DIR / "calib"
CLASS_MAP   = CLASS_MAP_PATH
THRESH_OUT  = THRESH_PATH

ALPHA       = 0.25        # Target coverage = 75%
BATCH_SIZE  = 64
MAX_SAMPLES = 500         # Stratified sample cap for fast CPU calibration

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

calib_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_trained_model(num_classes: int, device: torch.device) -> nn.Module:
    """Load model architecture and trained weights."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(  # type: ignore[assignment]
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def main():
    print("=" * 70)
    print("  STAGE 4 -- Conformal Calibration (q_hat Threshold Computation)")
    print("=" * 70)

    cpus = os.cpu_count() or 4
    torch.set_num_threads(cpus)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in Use : {device} (CPU threads: {cpus})")

    if not CLASS_MAP.exists():
        raise FileNotFoundError(f"Class mapping file '{CLASS_MAP}' not found. Run Stage 3 first.")

    with open(CLASS_MAP, "r") as f:
        class_to_idx = validate_class_mapping(json.load(f), str(CLASS_MAP))

    validate_image_split(DATA_CALIB, str(DATA_CALIB))
    full_calib_dataset = datasets.ImageFolder(DATA_CALIB, transform=calib_transform)
    ensure_same_class_mapping(class_to_idx, full_calib_dataset.class_to_idx, str(DATA_CALIB))
    num_classes = len(full_calib_dataset.classes)

    # Subsample deterministically and stratified by class for fast CPU calibration.
    sampled_indices = stratified_sample_indices(full_calib_dataset.targets, MAX_SAMPLES)

    calib_subset = Subset(full_calib_dataset, sampled_indices)
    calib_loader = DataLoader(calib_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    n_samples = len(calib_subset)
    print(f"Calibration Samples : {n_samples}")
    print(f"Target Coverage     : {(1.0 - ALPHA) * 100:.1f}% (alpha = {ALPHA})")

    model = load_trained_model(num_classes, device)

    scores = []
    softmax = nn.Softmax(dim=1)

    print("\nComputing calibration nonconformity scores...")
    with torch.no_grad():
        for images, labels in calib_loader:
            images = images.to(device)
            logits = model(images)
            probs  = softmax(logits).cpu().numpy()

            for prob_vec, true_cls in zip(probs, labels.numpy()):
                score = 1.0 - prob_vec[true_cls]
                scores.append(score)

    scores = np.array(scores)
    sorted_scores = np.sort(scores)

    n = len(sorted_scores)
    level = min(1.0, math.ceil((n + 1) * (1.0 - ALPHA)) / n)

    # Conformal finite-sample quantile (higher interpolation)
    q_index = min(n - 1, max(0, math.ceil((n + 1) * (1.0 - ALPHA)) - 1))
    q_hat   = float(sorted_scores[q_index])

    score_min    = float(np.min(scores))
    score_max    = float(np.max(scores))
    score_mean   = float(np.mean(scores))
    score_median = float(np.median(scores))

    print("\n" + "-" * 70)
    print("Calibration Nonconformity Score Distribution Stats:")
    print(f"  * Min    : {score_min:.4f}")
    print(f"  * Max    : {score_max:.4f}")
    print(f"  * Mean   : {score_mean:.4f}")
    print(f"  * Median : {score_median:.4f}")
    print(f"  * Quantile Level (adjusted): {level:.6f}")
    print(f"  * Computed q_hat Threshold : {q_hat:.4f}")
    print("-" * 70)

    threshold_data = {
        "alpha": ALPHA,
        "target_coverage": 1.0 - ALPHA,
        "q_hat": q_hat,
        "n_calibration_samples": n,
        "quantile_level": level,
        "score_stats": {
            "min": score_min,
            "max": score_max,
            "mean": score_mean,
            "median": score_median,
        },
        "class_mapping": class_to_idx,
    }

    atomic_write_json(THRESH_OUT, threshold_data)

    print(f"\n[OK] Saved threshold data -> {THRESH_OUT}")
    print("[OK] STAGE 4 Complete: Calibration finished successfully.\n")


if __name__ == "__main__":
    main()

"""
evaluate.py
-----------
STAGE 5 — Conformal Evaluation on Test Set

Conformal Evaluation Logic:
---------------------------
For each test sample (x_i, y_i):
1. Compute softmax probability vector s(x_i).
2. Construct Prediction Set:
       C(x_i) = { c : (1 - s(x_i)_c) <= q_hat }
3. Check Coverage:
       Covered = (y_i in C(x_i))
4. Compute Set Size:
       Set Size = |C(x_i)|
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from class_mapping import ensure_same_class_mapping, validate_class_mapping
from conformal_sets import prediction_set_indices
from artifact_io import atomic_write_text
from dataset_validation import validate_image_split
from paths import CLASS_MAP_PATH, DATA_DIR, MODEL_PATH, RESULTS_PATH, THRESH_PATH
from sampling import stratified_sample_indices

DATA_TEST   = DATA_DIR / "test"
CLASS_MAP   = CLASS_MAP_PATH
RESULTS_OUT = RESULTS_PATH

BATCH_SIZE  = 64
MAX_SAMPLES = 500         # Stratified sample cap for fast CPU evaluation

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model(num_classes: int, device: torch.device) -> nn.Module:
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
    print("  STAGE 5 -- Conformal Prediction Evaluation")
    print("=" * 70)

    cpus = os.cpu_count() or 4
    torch.set_num_threads(cpus)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in Use : {device} (CPU threads: {cpus})")

    if not THRESH_PATH.exists():
        raise FileNotFoundError(f"Threshold file '{THRESH_PATH}' not found. Run Stage 4 first.")

    with open(THRESH_PATH, "r") as f:
        thresh_data = json.load(f)

    q_hat           = thresh_data["q_hat"]
    target_coverage = thresh_data["target_coverage"]
    class_to_idx    = validate_class_mapping(thresh_data["class_mapping"], str(THRESH_PATH))
    if not CLASS_MAP.exists():
        raise FileNotFoundError(f"Class mapping file '{CLASS_MAP}' not found. Run Stage 3 first.")
    with open(CLASS_MAP, "r") as f:
        saved_class_mapping = validate_class_mapping(json.load(f), str(CLASS_MAP))
    ensure_same_class_mapping(class_to_idx, saved_class_mapping, str(CLASS_MAP))
    idx_to_class    = {idx: name for name, idx in class_to_idx.items()}

    print(f"Loaded q_hat Threshold : {q_hat:.4f}")
    print(f"Target Coverage        : {target_coverage * 100:.1f}%\n")

    validate_image_split(DATA_TEST, str(DATA_TEST))
    full_test_dataset = datasets.ImageFolder(DATA_TEST, transform=eval_transform)
    ensure_same_class_mapping(class_to_idx, full_test_dataset.class_to_idx, str(DATA_TEST))
    num_classes = len(full_test_dataset.classes)

    sampled_indices = stratified_sample_indices(full_test_dataset.targets, MAX_SAMPLES)

    test_subset = Subset(full_test_dataset, sampled_indices)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    n_test      = len(test_subset)

    print(f"Test Set Samples       : {n_test}")

    model = load_model(num_classes, device)
    softmax = nn.Softmax(dim=1)

    covered_count = 0
    set_sizes     = []
    size_counts   = defaultdict(int)
    per_class_stats = defaultdict(lambda: [0, 0])

    print("Evaluating test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs  = softmax(logits).cpu().numpy()

            for prob_vec, true_label in zip(probs, labels.numpy()):
                prediction_set = prediction_set_indices(prob_vec, q_hat)
                set_size = len(prediction_set)

                is_covered = true_label in prediction_set
                if is_covered:
                    covered_count += 1

                set_sizes.append(set_size)

                if set_size >= 4:
                    size_counts["4+"] += 1
                else:
                    size_counts[str(set_size)] += 1

                per_class_stats[true_label][0] += 1
                if is_covered:
                    per_class_stats[true_label][1] += 1

    empirical_coverage = covered_count / n_test
    avg_set_size       = float(np.mean(set_sizes))
    diff_from_target   = (empirical_coverage - target_coverage) * 100

    results_str = []
    results_str.append("=" * 70)
    results_str.append("  CONFORMAL PREDICTION EVALUATION RESULTS")
    results_str.append("=" * 70)
    results_str.append(f"Total Test Images      : {n_test}")
    results_str.append(f"Target Coverage        : {target_coverage * 100:.2f}%")
    results_str.append(f"Empirical Coverage     : {empirical_coverage * 100:.2f}% ({covered_count}/{n_test})")
    results_str.append(f"Coverage Difference    : {diff_from_target:+.2f}% vs target")
    results_str.append(f"Average Set Size       : {avg_set_size:.2f} classes")
    results_str.append("-" * 70)

    results_str.append("\nSet Size Distribution:")
    results_str.append(f"{'Set Size':<10} | {'Count':<10} | {'Percentage':<10}")
    results_str.append("-" * 35)
    for sz_key in ["0", "1", "2", "3", "4+"]:
        cnt = size_counts[sz_key]
        pct = (cnt / n_test) * 100
        results_str.append(f"{sz_key:<10} | {cnt:<10} | {pct:>9.2f}%")

    results_str.append("\n" + "-" * 70)
    results_str.append("Per-Class Empirical Coverage Breakdown:")
    col_w = 12
    cls_w = 32
    results_str.append(f"{'Class Name':<{cls_w}} | {'Test Count':>{col_w}} | {'Covered':>{col_w}} | {'Coverage':>{col_w}}")
    results_str.append("-" * 70)

    for c_idx in sorted(per_class_stats.keys()):
        c_name = idx_to_class[c_idx]
        tot, cov = per_class_stats[c_idx]
        c_cov_pct = (cov / tot * 100) if tot > 0 else 0.0
        results_str.append(f"{c_name:<{cls_w}} | {tot:>{col_w}} | {cov:>{col_w}} | {c_cov_pct:>{col_w}.2f}%")

    results_str.append("=" * 70)

    full_output = "\n".join(results_str)
    print("\n" + full_output)

    atomic_write_text(RESULTS_OUT, full_output + "\n")

    print(f"\n[OK] Results saved to -> {RESULTS_OUT}")
    print("[OK] STAGE 5 Complete: Evaluation finished successfully.\n")


if __name__ == "__main__":
    main()

"""
train.py
--------
STAGE 3 — Model Training (ResNet18 Fine-Tuning)

- Pretrained ResNet18 (ImageNet weights)
- Derives the output layer size from `ImageFolder` after validating the expected tomato classes
- Standard image transforms (Resize 224x224, RandomHorizontalFlip, Normalize)
- Per-class held-out validation subset
- CrossEntropyLoss + Adam optimizer + ReduceLROnPlateau scheduler
- Saves the lowest-validation-loss weights from the configured four epochs
- Saves best model to model.pth and class mapping to class_mapping.json
"""

import os
import random
import time
from copy import deepcopy
from typing import cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm
from class_mapping import ensure_same_class_mapping, validate_class_mapping
from artifact_io import atomic_torch_save, atomic_write_json
from dataset_validation import validate_image_split
from paths import CLASS_MAP_PATH, DATA_DIR, MODEL_PATH

DATA_TRAIN  = DATA_DIR / "train"
MODEL_OUT   = MODEL_PATH
CLASS_MAP   = CLASS_MAP_PATH

BATCH_SIZE          = 128
NUM_EPOCHS          = 4
LR                  = 5e-4
VAL_SPLIT           = 0.10
SAMPLES_PER_CLASS   = 250      # Balanced subsample (2,500 images total for ~1-2 min fast CPU training)
RANDOM_SEED         = 42


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_model(num_classes: int) -> nn.Module:
    """Load pretrained ResNet18, unfreeze layer3 and layer4 for fine-tuning, and replace FC layer."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze early layers (conv1, bn1, layer1, layer2)
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer3 and layer4 for robust feature adaptation
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(  # type: ignore[assignment]
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model



def main():
    print("=" * 70)
    print("  STAGE 3 -- Model Training (ResNet18 Fine-Tuning)")
    print("=" * 70)

    # Optimize CPU threading (use physical cores to avoid thread thrashing)
    num_cpus = os.cpu_count() or 4
    torch.set_num_threads(max(1, num_cpus // 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in Use : {device} (Available CPUs: {num_cpus})")

    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    validate_image_split(DATA_TRAIN, str(DATA_TRAIN))
    full_train_dataset = datasets.ImageFolder(DATA_TRAIN, transform=train_transform)
    full_val_dataset   = datasets.ImageFolder(DATA_TRAIN, transform=val_transform)

    class_to_idx = validate_class_mapping(full_train_dataset.class_to_idx, str(DATA_TRAIN))
    ensure_same_class_mapping(class_to_idx, full_val_dataset.class_to_idx, "training validation dataset")
    num_classes = len(full_train_dataset.classes)
    # Collect sample indices per class tqdm
    class_indices = {i: [] for i in range(num_classes)}
    for idx, sample in enumerate(full_train_dataset.samples):
        label = sample[1]
        class_indices[label].append(idx)

    selected_train_idx = []
    selected_val_idx   = []

    for label, idxs in class_indices.items():
        random.shuffle(idxs)
        sampled = idxs[:SAMPLES_PER_CLASS] if SAMPLES_PER_CLASS else idxs
        n_v = max(1, int(len(sampled) * VAL_SPLIT))
        selected_val_idx.extend(sampled[:n_v])
        selected_train_idx.extend(sampled[n_v:])

    train_subset = Subset(full_train_dataset, selected_train_idx)
    val_subset   = Subset(full_val_dataset, selected_val_idx)

    num_workers = min(4, num_cpus) if device.type == "cuda" else min(2, num_cpus)
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0)
    )
    val_loader   = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0)
    )

    print(f"Train Samples : {len(train_subset)} | Val Samples: {len(val_subset)}\n")

    model     = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss  = float("inf")
    best_weights   = None

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | {'Val Acc':>8} | {'LR':>10} | {'Time':>6}")
    print("-" * 75)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]", leave=False)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            cast(torch.Tensor, scaler.scale(loss)).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            train_loss += batch_loss * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)
            train_bar.set_postfix({"loss": f"{batch_loss:.4f}"})

        epoch_train_loss = train_loss / train_total
        epoch_train_acc  = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]  ", leave=False)
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                batch_loss = loss.item()
                val_loss += batch_loss * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)
                val_bar.set_postfix({"loss": f"{batch_loss:.4f}"})

        epoch_val_loss = val_loss / val_total
        epoch_val_acc  = val_correct / val_total

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"{epoch:>6} | {epoch_train_loss:>10.4f} | {epoch_train_acc*100:>8.2f}% | "
              f"{epoch_val_loss:>9.4f} | {epoch_val_acc*100:>7.2f}% | {current_lr:>10.2e} | {elapsed:>5.1f}s")

        if epoch_val_loss < best_val_loss or best_weights is None:
            best_val_loss = epoch_val_loss
            best_weights  = deepcopy(model.state_dict())

    if best_weights is not None:
        atomic_torch_save(best_weights, MODEL_OUT)
        atomic_write_json(CLASS_MAP, class_to_idx)
        print(f"Saved class mapping ({len(class_to_idx)} classes) -> {CLASS_MAP}")
        print(f"\n[OK] Saved best model weights -> {MODEL_OUT} (Best Val Loss: {best_val_loss:.4f})")

    print("\n[OK] STAGE 3 Complete: Model training finished successfully.\n")


if __name__ == "__main__":
    main()

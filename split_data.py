"""
split_data.py
-------------
STAGE 2 — Data Organization and Splitting

- Inspects the dataset directory.
- Maps folder names case-insensitively to the 10 standard tomato class names:
  1. Tomato_Bacterial_spot
  2. Tomato_Early_blight
  3. Tomato_Late_blight
  4. Tomato_Leaf_Mold
  5. Tomato_Septoria_leaf_spot
  6. Tomato_Spider_mites
  7. Tomato_Target_Spot
  8. Tomato_Yellow_Leaf_Curl_Virus
  9. Tomato_mosaic_virus
  10. Tomato_healthy
- With the default source, or an explicit --src dataset root, verifies an existing complete split.
- If raw un-split class folders exist, performs a 70/15/15 stratified split with seed 42,
  copying images into train, calib, and test subfolders without altering the originals.
- Prints a clear per-class count table and summary stats.
"""

import argparse
import shutil
import random
from pathlib import Path
from collections import defaultdict
from class_mapping import EXPECTED_CLASSES, validate_class_names
from dataset_validation import validate_image_split, supported_images
from paths import DATA_DIR, SUPPORTED_IMAGE_EXTENSIONS

TARGET_CLASSES = list(EXPECTED_CLASSES)

VALID_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
CALIB_RATIO = 0.15
TEST_RATIO  = 0.15


def normalize_name(name: str) -> str:
    """Normalize a folder name for case-insensitive matching."""
    cleaned = name.lower().replace("___", "_").replace(" ", "_").replace("-", "_")
    return cleaned


def get_class_mapping(folder_names: list[str]) -> dict[str, str]:
    """Map external/variant folder names to standard target class names."""
    mapping = {}
    target_norm_map = {normalize_name(cls): cls for cls in TARGET_CLASSES}

    for folder in folder_names:
        norm_folder = normalize_name(folder)
        # Check direct normalized match
        if norm_folder in target_norm_map:
            mapping[folder] = target_norm_map[norm_folder]
        else:
            # Check partial match (e.g. "bacterial_spot" in "tomato_bacterial_spot")
            for t_norm, t_std in target_norm_map.items():
                short_t = t_norm.replace("tomato_", "")
                if short_t in norm_folder:
                    mapping[folder] = t_std
                    break

    return mapping


def inspect_and_sample(base_dir: Path):
    """List subfolders and a sample of file names in each."""
    print("\n--- Folder Structure & File Sampling ---")
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"No subdirectories found in {base_dir}")
        return

    for d in sorted(subdirs):
        files = [f.name for f in d.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
        sample = files[:3]
        print(f"  * Folder: '{d.name}' (Total files: {len(files)})")
        if sample:
            print(f"    Sample files: {sample}")
        else:
            child_subdirs = [c.name for c in d.iterdir() if c.is_dir()]
            if child_subdirs:
                print(f"    Sub-class folders ({len(child_subdirs)}): {child_subdirs[:3]}...")


def validate_source_path(source_dir: Path) -> Path:
    """Resolve a dataset root and reject missing or partially-created splits."""
    source_dir = Path(source_dir).expanduser()
    if not source_dir.exists():
        raise ValueError(f"Dataset source path does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"Dataset source path is not a directory: {source_dir}")

    source_dir = source_dir.resolve()
    split_dirs = [source_dir / name for name in ("train", "calib", "test")]
    invalid_splits = [path.name for path in split_dirs if path.exists() and not path.is_dir()]
    if invalid_splits:
        raise ValueError(f"Split paths must be directories: {', '.join(invalid_splits)}")
    existing_splits = [path.exists() for path in split_dirs]
    if any(existing_splits) and not all(existing_splits):
        raise ValueError(
            f"Dataset source has only some split directories; expected train, calib, and test together: {source_dir}"
        )
    return source_dir


def validate_raw_source(raw_folders: list[Path], source_dir: Path):
    """Validate all expected class folders and images before copying any files."""
    folder_map = get_class_mapping([folder.name for folder in raw_folders])
    unmapped = sorted(folder.name for folder in raw_folders if folder.name not in folder_map)
    if unmapped:
        raise ValueError(f"Unrecognized class folders in {source_dir}: {', '.join(unmapped)}")

    sources_by_class = defaultdict(list)
    for folder_name, class_name in folder_map.items():
        sources_by_class[class_name].append(folder_name)
    duplicates = sorted(name for name, sources in sources_by_class.items() if len(sources) > 1)
    if duplicates:
        raise ValueError(f"Multiple source folders map to the same class: {', '.join(duplicates)}")

    validate_class_names(folder_map.values(), str(source_dir))
    images_by_folder = {folder.name: supported_images(folder) for folder in raw_folders}
    empty_folders = sorted(name for name, images in images_by_folder.items() if not images)
    if empty_folders:
        raise ValueError(f"Class folders contain no supported images: {', '.join(empty_folders)}")

    too_small = []
    for folder_name, images in images_by_folder.items():
        n_train = int(len(images) * TRAIN_RATIO)
        n_calib = int(len(images) * CALIB_RATIO)
        n_test = len(images) - n_train - n_calib
        if min(n_train, n_calib, n_test) < 1:
            too_small.append(folder_name)
    if too_small:
        raise ValueError(
            "Each class needs enough images for non-empty train/calib/test splits: "
            f"{', '.join(sorted(too_small))}"
        )
    return folder_map, images_by_folder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split tomato class folders into train/calib/test sets.")
    parser.add_argument(
        "--src",
        type=Path,
        default=DATA_DIR,
        help=f"Dataset root to inspect or split (default: {DATA_DIR})",
    )
    return parser


def organize_and_split(source_dir: Path = DATA_DIR):
    source_dir = validate_source_path(source_dir)
    print("=" * 70)
    print("  STAGE 2 -- Data Organization and Splitting")
    print("=" * 70)
    print(f"Data Path: {source_dir}\n")

    # Step 1: Inspect folder structure
    inspect_and_sample(source_dir)

    train_dir = source_dir / "train"
    calib_dir = source_dir / "calib"
    test_dir  = source_dir / "test"

    already_split = train_dir.exists() and calib_dir.exists() and test_dir.exists()

    if already_split:
        print("\nDetected existing train/calib/test split in source directory.")
        splits = {"train": train_dir, "calib": calib_dir, "test": test_dir}
    else:
        print("\nCreating new train/calib/test stratified split...")
        raw_folders = [d for d in source_dir.iterdir() if d.is_dir()]
        folder_map, images_by_folder = validate_raw_source(raw_folders, source_dir)
        rng = random.Random(RANDOM_SEED)

        for raw_folder in raw_folders:
            std_cls = folder_map.get(raw_folder.name)
            images = images_by_folder[raw_folder.name]
            rng.shuffle(images)

            n_total = len(images)
            n_train = int(n_total * TRAIN_RATIO)
            n_calib = int(n_total * CALIB_RATIO)
            
            train_imgs = images[:n_train]
            calib_imgs = images[n_train:n_train + n_calib]
            test_imgs  = images[n_train + n_calib:]

            for img, split_name in [(train_imgs, "train"), (calib_imgs, "calib"), (test_imgs, "test")]:
                dest_dir = source_dir / split_name / std_cls
                dest_dir.mkdir(parents=True, exist_ok=True)
                for src_file in img:
                    dest_file = dest_dir / src_file.name
                    if not dest_file.exists():
                        shutil.copy2(src_file, dest_file)

        splits = {"train": train_dir, "calib": calib_dir, "test": test_dir}

    for split_name, split_path in splits.items():
        validate_image_split(split_path, f"{source_dir / split_name}")

    # Step 2: Tabulate image counts per class per split
    counts = defaultdict(lambda: {"train": 0, "calib": 0, "test": 0})

    for split_name, split_path in splits.items():
        if not split_path.exists():
            continue
        for cls_dir in split_path.iterdir():
            if cls_dir.is_dir():
                c_name = cls_dir.name
                n_files = sum(1 for f in cls_dir.glob("*") if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS)
                counts[c_name][split_name] += n_files

    # Print summary table
    col_w = 12
    cls_w = 32

    print("\n" + "=" * 70)
    print(f"{'Class Name':<{cls_w}} | {'Train (70%)':>{col_w}} | {'Calib (15%)':>{col_w}} | {'Test (15%)':>{col_w}} | {'Total':>{col_w}}")
    print("-" * 70)

    total_train = 0
    total_calib = 0
    total_test  = 0

    for std_cls in TARGET_CLASSES:
        tr = counts[std_cls]["train"]
        ca = counts[std_cls]["calib"]
        te = counts[std_cls]["test"]
        tot = tr + ca + te

        total_train += tr
        total_calib += ca
        total_test  += te

        print(f"{std_cls:<{cls_w}} | {tr:>{col_w}} | {ca:>{col_w}} | {te:>{col_w}} | {tot:>{col_w}}")

        # Warnings for missing or very low counts
        if tot == 0:
            print(f"  [WARNING] Class '{std_cls}' is completely missing from the dataset!")
        elif tr < 10 or ca < 5 or te < 5:
            print(f"  [WARNING] Class '{std_cls}' has low sample count (train={tr}, calib={ca}, test={te})")

    print("-" * 70)
    grand_total = total_train + total_calib + total_test
    print(f"{'TOTAL':<{cls_w}} | {total_train:>{col_w}} | {total_calib:>{col_w}} | {total_test:>{col_w}} | {grand_total:>{col_w}}")
    print("=" * 70)

    if grand_total > 0:
        print(f"Actual split percentages: Train={total_train/grand_total*100:.1f}%, "
              f"Calib={total_calib/grand_total*100:.1f}%, "
              f"Test={total_test/grand_total*100:.1f}%")

    print("\n[OK] STAGE 2 Complete: Data splitting & verification finalized.\n")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    try:
        organize_and_split(args.src)
    except ValueError as error:
        parser.error(str(error))

"""Create small local disease-library images from the read-only training split."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "data" / "train"
OUTPUT_ROOT = ROOT / "static" / "img" / "diseases"
DISEASES_PATH = ROOT / "data" / "diseases.json"


def make_placeholder(path: Path, label: str) -> None:
    image = Image.new("RGB", (800, 520), "#e8f1e4")
    draw = ImageDraw.Draw(image)
    draw.ellipse((250, 90, 550, 390), fill="#79a96a", outline="#2e5b2e", width=5)
    draw.line((400, 150, 400, 430), fill="#2e5b2e", width=8)
    draw.text((40, 450), f"Sample image unavailable: {label}", fill="#173b1b")
    image.save(path, format="JPEG", quality=86, optimize=True)


def convert(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        if width > 800:
            image = image.resize((800, max(1, round(height * 800 / width))), Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=86, optimize=True)


def main() -> None:
    import json

    catalog = json.loads(DISEASES_PATH.read_text(encoding="utf-8"))
    for class_name, disease in catalog.items():
        source_dir = TRAIN_ROOT / class_name
        output_dir = OUTPUT_ROOT / disease["slug"]
        output_dir.mkdir(parents=True, exist_ok=True)
        sources = sorted(
            path for path in source_dir.iterdir() if path.is_file()
        ) if source_dir.is_dir() else []
        for index in range(2):
            destination = output_dir / f"sample-{index + 1}.jpg"
            if sources:
                convert(sources[index % len(sources)], destination)
            else:
                make_placeholder(destination, disease["name"])
        print(f"Prepared {disease['name']}")


if __name__ == "__main__":
    main()

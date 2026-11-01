"""Load and normalize the local tomato disease library."""

import json
import re
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DISEASES_PATH = PROJECT_ROOT / "data" / "diseases.json"
DISEASE_IMAGE_ROOT = PROJECT_ROOT / "static" / "img" / "diseases"


def clean_class_name(class_name: str) -> str:
    """Return the display-safe class label used by the prediction API."""
    return class_name.replace("Tomato_", "").replace("_", " ").strip()


def slugify(value: str) -> str:
    """Return a stable library slug for a raw or display class name."""
    normalized = clean_class_name(value).lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return normalized or "unknown"


@lru_cache(maxsize=1)
def load_diseases() -> dict[str, dict]:
    with DISEASES_PATH.open(encoding="utf-8") as file:
        data = json.load(file)
    return data


def disease_items() -> list[dict]:
    """Return catalog objects with local image URLs ready for templates."""
    items = []
    for class_name, disease in load_diseases().items():
        item = dict(disease)
        item["class_name"] = class_name
        item["display_name"] = disease["name"]
        item["images"] = [
            f"img/diseases/{disease['slug']}/{filename}"
            for filename in ("sample-1.jpg", "sample-2.jpg")
            if (DISEASE_IMAGE_ROOT / disease["slug"] / filename).exists()
        ]
        items.append(item)
    return sorted(items, key=lambda item: item["name"].lower())


def disease_lookup() -> dict[str, dict]:
    lookup = {}
    for class_name, disease in load_diseases().items():
        item = dict(disease)
        item["class_name"] = class_name
        for key in (class_name, clean_class_name(class_name), disease["name"], disease["slug"]):
            lookup[key.lower()] = item
    return lookup


def find_disease(slug: str) -> dict | None:
    normalized = slugify(slug)
    for disease in load_diseases().values():
        if disease["slug"] == normalized:
            item = dict(disease)
            item["images"] = [
                f"img/diseases/{disease['slug']}/{filename}"
                for filename in ("sample-1.jpg", "sample-2.jpg")
                if (DISEASE_IMAGE_ROOT / disease["slug"] / filename).exists()
            ]
            return item
    return None


def library_link(value: str | None) -> str | None:
    if not value:
        return None
    disease = disease_lookup().get(value.lower())
    return disease["slug"] if disease else slugify(value)

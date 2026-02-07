from pathlib import Path

from PIL import Image

from app.services.models import ModelStore, extract_features, classify_feature


def classify_crops(store: ModelStore, objects: list[dict]) -> list[dict]:
    for obj in objects:
        crop_path = Path(obj.get("crop", ""))
        if not crop_path.exists():
            obj["pred_label"] = None
            obj["pred_confidence"] = 0.0
            continue
        with Image.open(crop_path) as img:
            img = img.convert("RGB")
            feat = extract_features(store, img)
        if feat is None:
            obj["pred_label"] = None
            obj["pred_confidence"] = 0.0
            continue
        label, conf = classify_feature(store, feat)
        obj["pred_label"] = label
        obj["pred_confidence"] = conf
    return objects

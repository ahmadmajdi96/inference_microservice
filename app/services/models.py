import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import timm
import clip
from ultralytics import YOLO

from app.core.config import settings


@dataclass
class ModelStore:
    yolo: Any
    clf: Any
    label_encoder: Any
    dino_model: Any
    clip_model: Any
    clip_preprocess: Any
    dino_transform: Any
    device: str
    expected_dim: int | None = None


def _resolve_device() -> str:
    if settings.device:
        return settings.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_models() -> ModelStore:
    device = _resolve_device()

    yolo = YOLO(settings.yolov8_weights)

    clf = joblib.load(settings.logreg_path)
    label_encoder = joblib.load(settings.label_encoder_path)

    dino_model = timm.create_model(settings.dino_model, pretrained=True)
    dino_model.eval().to(device)

    clip_model, clip_preprocess = clip.load(settings.clip_model, device=device)
    clip_model.eval()

    dino_cfg = dino_model.default_cfg
    input_size = dino_cfg.get("input_size", (3, 224, 224))
    if isinstance(input_size, tuple) and len(input_size) == 3:
        _, height, width = input_size
    else:
        height, width = 224, 224

    dino_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    store = ModelStore(
        yolo=yolo,
        clf=clf,
        label_encoder=label_encoder,
        dino_model=dino_model,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        dino_transform=dino_transform,
        device=device,
    )
    expected_dim = getattr(store.clf, "coef_", None)
    if expected_dim is not None:
        expected = int(expected_dim.shape[1])
        sample = np.zeros((1, 1), dtype=np.float32)
        try:
            # quick probe: 1x1 dummy to force shape check
            store.clf.predict(sample)
        except Exception:
            pass
        store.expected_dim = expected
    return store


def extract_features(store: ModelStore, image: Image.Image) -> np.ndarray | None:
    try:
        dino_tensor = store.dino_transform(image).unsqueeze(0).to(store.device)
        with torch.no_grad():
            feats = store.dino_model.forward_features(dino_tensor)
            dino_feat = feats[:, 0, :].cpu().numpy().flatten()

        clip_tensor = store.clip_preprocess(image).unsqueeze(0).to(store.device)
        with torch.no_grad():
            clip_feat = store.clip_model.encode_image(clip_tensor).cpu().numpy().flatten()

        combined = np.concatenate([dino_feat, clip_feat])
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined
    except Exception as exc:
        print(f"feature extraction failed: {exc}")
        return None


def classify_feature(store: ModelStore, feature: np.ndarray) -> tuple[str, float]:
    if feature.ndim == 1:
        feature = feature.reshape(1, -1)
    if store.expected_dim and feature.shape[1] != store.expected_dim:
        raise RuntimeError(
            f"Feature dim mismatch: got {feature.shape[1]}, expected {store.expected_dim}. "
            f"Ensure DINO/CLIP models match training (e.g., clip_model={settings.clip_model})."
        )
    if hasattr(store.clf, "predict_proba"):
        probs = store.clf.predict_proba(feature)[0]
        idx = int(np.argmax(probs))
        label = store.label_encoder.inverse_transform([idx])[0]
        return str(label), float(probs[idx])
    pred = store.clf.predict(feature)[0]
    label = store.label_encoder.inverse_transform([pred])[0]
    return str(label), 1.0

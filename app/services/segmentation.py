from pathlib import Path

from PIL import Image

from app.core.config import settings
from app.services.io import rel_or_abs
from app.services.models import ModelStore


def _sanitize_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)


def _clamp_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1i = max(0, min(int(x1), w - 1))
    y1i = max(0, min(int(y1), h - 1))
    x2i = max(0, min(int(x2), w - 1))
    y2i = max(0, min(int(y2), h - 1))
    if x2i <= x1i:
        x2i = min(w - 1, x1i + 1)
    if y2i <= y1i:
        y2i = min(h - 1, y1i + 1)
    return x1i, y1i, x2i, y2i


def segment_image(store: ModelStore, image_path: Path, crops_dir: Path, job_root: Path) -> tuple[list[dict], dict]:
    results = store.yolo.predict(
        source=str(image_path),
        conf=settings.conf,
        iou=settings.iou,
        device=store.device,
        imgsz=settings.imgsz,
        max_det=settings.max_det,
        verbose=False,
    )
    res = results[0]
    names = res.names or {}

    annotated_dir = job_root / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = annotated_dir / f"{image_path.stem}_annotated.jpg"
    try:
        plotted = res.plot()  # numpy array (BGR)
        annotated_img = Image.fromarray(plotted[..., ::-1])  # BGR -> RGB
        annotated_img.save(annotated_path, quality=92, optimize=True)
    except Exception as exc:
        # Non-fatal if plotting fails
        print(f"annotation save failed for {image_path}: {exc}")

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        objects = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes
            for obj_idx in range(len(boxes)):
                box = boxes[obj_idx]
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = _clamp_box(xyxy[0], xyxy[1], xyxy[2], xyxy[3], w, h)
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                label = names.get(cls_id, str(cls_id))

                crop = img.crop((x1, y1, x2, y2))
                label_safe = _sanitize_label(str(label))
                crop_name = f"{image_path.stem}_obj{obj_idx + 1:03d}_{label_safe}.jpg"
                crop_path = crops_dir / crop_name
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path, quality=92, optimize=True)

                objects.append({
                    "label": str(label),
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "crop": str(crop_path),
                    "crop_rel": rel_or_abs(crop_path, job_root),
                })

    annotated_info = {
        "annotated": str(annotated_path),
        "annotated_rel": rel_or_abs(annotated_path, job_root),
    }
    return objects, annotated_info

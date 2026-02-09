import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PIL import Image

from app.services.io import list_images, rel_or_abs
from app.services.segmentation import segment_image
from app.services.classification import classify_crops
from app.services.models import ModelStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(jobs_dir: Path) -> tuple[str, Path]:
    job_id = str(uuid.uuid4())
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_id, job_dir


def write_status(job_dir: Path, status: str, stage: str = "", error: str | None = None) -> None:
    data = {
        "job_id": job_dir.name,
        "status": status,
        "stage": stage,
        "updated_at": _now(),
    }
    if error:
        data["error"] = error
    (job_dir / "status.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_status(job_dir: Path) -> dict:
    path = job_dir / "status.json"
    if not path.exists():
        return {"job_id": job_dir.name, "status": "UNKNOWN"}
    return json.loads(path.read_text(encoding="utf-8") or "{}")


def process_job(store: ModelStore, job_dir: Path, input_dir: Path) -> None:
    write_status(job_dir, "RUNNING", stage="segmentation")

    crops_dir = job_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if not images:
        write_status(job_dir, "FAILED", stage="segmentation", error="no images found")
        return

    results = {
        "job_id": job_dir.name,
        "status": "SUCCEEDED",
        "total_images": len(images),
        "images": [],
    }

    try:
        def _shelf_summary(objects: list[dict], image_h: int) -> dict:
            if not objects:
                return {"shelves": [], "total_known": 0, "total_unknown": 0, "total_objects": 0}

            centers = []
            heights = []
            for obj in objects:
                x1, y1, x2, y2 = obj.get("bbox", [0, 0, 0, 0])
                centers.append(((y1 + y2) / 2.0, obj))
                heights.append(max(1.0, (y2 - y1)))

            centers.sort(key=lambda x: x[0])
            median_h = sorted(heights)[len(heights) // 2]
            gap_thresh = max(image_h * 0.08, median_h * 1.2)

            shelves = []
            current = []
            last_y = None
            for y, obj in centers:
                if last_y is None or (y - last_y) <= gap_thresh:
                    current.append((y, obj))
                else:
                    shelves.append(current)
                    current = [(y, obj)]
                last_y = y
            if current:
                shelves.append(current)

            shelf_rows = []
            total_known = 0
            total_unknown = 0
            total_objects = 0
            for idx, shelf in enumerate(shelves, start=1):
                ys = [y for y, _ in shelf]
                objs = [o for _, o in shelf]
                known = [o for o in objs if (o.get("pred_label") or "").upper() != "UNKNOWN"]
                unknown = [o for o in objs if (o.get("pred_label") or "").upper() == "UNKNOWN"]
                class_counts = {}
                for o in known:
                    label = str(o.get("pred_label") or "UNKNOWN")
                    class_counts[label] = class_counts.get(label, 0) + 1
                shelf_rows.append({
                    "shelf_index": idx,
                    "y_center_min": min(ys),
                    "y_center_max": max(ys),
                    "total_objects": len(objs),
                    "known_count": len(known),
                    "unknown_count": len(unknown),
                    "class_counts": class_counts,
                })
                total_known += len(known)
                total_unknown += len(unknown)
                total_objects += len(objs)

            return {
                "shelves": shelf_rows,
                "total_known": total_known,
                "total_unknown": total_unknown,
                "total_objects": total_objects,
            }

        for image_path in images:
            objects, annotated = segment_image(store, image_path, crops_dir, job_dir)
            objects = classify_crops(store, objects)
            with Image.open(image_path) as img:
                _, image_h = img.size
            shelves = _shelf_summary(objects, image_h)
            results["images"].append({
                "image": str(image_path),
                "image_rel": rel_or_abs(image_path, job_dir),
                "annotated": annotated.get("annotated"),
                "annotated_rel": annotated.get("annotated_rel"),
                "objects": objects,
                "shelves": shelves,
            })

        results_path = job_dir / "results.json"
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        write_status(job_dir, "SUCCEEDED", stage="done")
    except Exception as exc:
        write_status(job_dir, "FAILED", stage="processing", error=str(exc))

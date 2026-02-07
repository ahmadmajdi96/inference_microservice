import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

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
        for image_path in images:
            objects, annotated = segment_image(store, image_path, crops_dir, job_dir)
            objects = classify_crops(store, objects)
            results["images"].append({
                "image": str(image_path),
                "image_rel": rel_or_abs(image_path, job_dir),
                "annotated": annotated.get("annotated"),
                "annotated_rel": annotated.get("annotated_rel"),
                "objects": objects,
            })

        results_path = job_dir / "results.json"
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        write_status(job_dir, "SUCCEEDED", stage="done")
    except Exception as exc:
        write_status(job_dir, "FAILED", stage="processing", error=str(exc))

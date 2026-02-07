from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse

from app.core.config import settings
from app.services.io import extract_zip
from app.services.jobs import create_job, process_job, load_status, write_status

router = APIRouter(prefix="/v1", tags=["inference"])


def _save_upload(file: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _is_within(path: Path, base: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(base.resolve(strict=False))
        return True
    except ValueError:
        return False


@router.post("/infer/image")
async def infer_image(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    job_id, job_dir = create_job(Path(settings.jobs_dir))
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / (file.filename or "image")

    _save_upload(file, input_path)
    write_status(job_dir, "QUEUED", stage="upload")

    background_tasks.add_task(process_job, request.app.state.models, job_dir, input_dir)
    return {"job_id": job_id, "status": "QUEUED"}


@router.post("/infer/zip")
async def infer_zip(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    job_id, job_dir = create_job(Path(settings.jobs_dir))
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    zip_path = job_dir / "input.zip"

    _save_upload(file, zip_path)
    try:
        extract_zip(zip_path, input_dir)
    except Exception as exc:
        write_status(job_dir, "FAILED", stage="extract_zip", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))

    write_status(job_dir, "QUEUED", stage="extract_zip")
    background_tasks.add_task(process_job, request.app.state.models, job_dir, input_dir)
    return {"job_id": job_id, "status": "QUEUED"}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job_dir = Path(settings.jobs_dir) / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    return load_status(job_dir)


@router.get("/jobs/{job_id}/results")
async def get_results(job_id: str):
    job_dir = Path(settings.jobs_dir) / job_id
    results = job_dir / "results.json"
    if not results.exists():
        raise HTTPException(status_code=404, detail="results not found")
    return FileResponse(results)


@router.get("/jobs/{job_id}/files/{rel_path:path}")
async def get_file(job_id: str, rel_path: str):
    base = Path(settings.jobs_dir) / job_id
    file_path = (base / rel_path).resolve()
    if not _is_within(file_path, base) or not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(file_path)

import zipfile
from pathlib import Path, PurePosixPath

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_within_base(path: Path, base: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(base.resolve(strict=False))
        return True
    except ValueError:
        return False


def extract_zip(zip_path: Path, out_dir: Path) -> int:
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError("uploaded file is not a zip")

    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        safe_members = []
        for member in zf.infolist():
            if member.is_dir():
                continue
            member_path = PurePosixPath(member.filename.replace("\\", "/"))
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"unsafe path in zip: {member.filename}")
            if member_path.parts and member_path.parts[0] == "__MACOSX":
                continue
            dest = out_dir / Path(*member_path.parts)
            if not _is_within_base(dest, out_dir):
                raise RuntimeError(f"unsafe extract target: {member.filename}")
            safe_members.append(member)

        zf.extractall(out_dir, members=safe_members)
        return len(safe_members)


def list_images(root: Path) -> list[Path]:
    images = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
    return sorted(images)


def rel_or_abs(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)

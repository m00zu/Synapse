"""File upload + server-browse + preview routes."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

router = APIRouter(prefix="/api/files", tags=["files"])


def _uploads_dir() -> Path:
    """The content-addressable uploads dir lives under $HOME/.synapse/uploads.

    HOME is resolved fresh on every call so tests can monkeypatch it.
    """
    d = Path(os.environ.get("HOME", str(Path.home()))) / ".synapse" / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    (d / "by-name").mkdir(exist_ok=True)
    return d


def _allowed_root(request: Request) -> Path:
    """Where the user is allowed to browse from. Default $HOME; widened by
    --allow-path CLI flag."""
    return Path(
        request.app.state.session.allow_path
        or os.environ.get("HOME", str(Path.home()))
    ).resolve()


@router.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    """SHA-256 content-addressable upload. Returns absolute server path."""
    data = await file.read()
    h = hashlib.sha256(data).hexdigest()
    # Preserve all suffixes (e.g. ".tar.gz") so file-type sniffing still works.
    ext = "".join(Path(file.filename or "").suffixes) or ""
    target = _uploads_dir() / f"{h}{ext}"
    if not target.exists():
        target.write_bytes(data)
    if file.filename:
        link = _uploads_dir() / "by-name" / file.filename
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
        except OSError:
            # On filesystems that don't support symlinks, just skip the alias.
            pass
    return {"server_path": str(target)}


@router.get("/browse")
async def browse(request: Request, path: str) -> dict:
    """Directory listing, rooted at $HOME (or --allow-path). Path-traversal guarded."""
    root = _allowed_root(request)
    p = Path(path).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=403, detail="path outside allowed root")
    if not p.exists():
        raise HTTPException(status_code=404, detail="path not found")
    if not p.is_dir():
        raise HTTPException(status_code=400, detail="path is not a directory")
    entries = []
    for child in sorted(p.iterdir()):
        entries.append({
            "name": child.name,
            "is_dir": child.is_dir(),
            "path": str(child),
        })
    return {"root": str(p), "entries": entries}


@router.get("/preview/{node_id}/{port}")
async def preview(request: Request, node_id: str, port: str) -> dict:
    """Phase 1b stub. Real preview emission ships in Phase 1c alongside
    React display-node rendering."""
    raise HTTPException(status_code=404, detail="preview not yet implemented (Phase 1c)")

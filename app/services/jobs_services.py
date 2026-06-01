import json
from collections.abc import Iterator
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.helpers.fs_helpers import relative_paths_with_suffix
from app.helpers.jobs_helpers import save_uploaded_files, write_job_descriptor
from config.paths import INBOX_DIR, PENDING_DIR, TEX_DIR
from latex import compile as latex
from llm import OpenRouterClient

_TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"
_DASHBOARD_HTML = _TEMPLATES_DIR / "jobs_dashboard.html"
_EDITOR_HTML = _TEMPLATES_DIR / "editor.html"

_EDIT_SPAN_SYSTEM = (
    "You edit a span of body-only LaTeX from a larger math document "
    "(amsbook class, custom theorem environments and macros are already "
    "defined in the preamble). Apply the user's instruction to the SELECTION, "
    "using the surrounding context only to stay consistent. Output ONLY the "
    "replacement LaTeX for the selection — no preamble, no \\documentclass, no "
    "markdown code fences, no explanation."
)


def serve_dashboard() -> str:
    return _DASHBOARD_HTML.read_text()


def serve_editor() -> str:
    return _EDITOR_HTML.read_text()


def _note_tex_path(path: str) -> Path:
    if ".." in Path(path).parts or Path(path).is_absolute():
        raise HTTPException(status_code=422, detail="Invalid path")
    return TEX_DIR / f"{path}.tex"


def save_note_tex(path: str, body: str) -> dict:
    tex_path = _note_tex_path(path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(body, encoding="utf-8")
    return {"id": path, "status": "done"}


def delete_note(path: str) -> dict:
    tex_path = _note_tex_path(path)
    if not tex_path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    tex_path.unlink()
    return {"id": path, "deleted": True}


def rename_note(path: str, new_path: str) -> dict:
    src = _note_tex_path(path)
    dst = _note_tex_path(new_path)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Not found")
    if dst.exists():
        raise HTTPException(status_code=409, detail=f"Already exists: {new_path}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    return {"id": new_path, "renamed_from": path}


def compile_note_body(path: str, body: str) -> bytes:
    _note_tex_path(path)  # validate path
    result = latex.compile_body(body, Path(path).parent.name)
    if not result.success or result.pdf_bytes is None:
        raise HTTPException(status_code=422, detail=result.stderr)
    return result.pdf_bytes


def stream_span_edit(
    model: str,
    instruction: str,
    selection: str,
    context_before: str,
    context_after: str,
) -> Iterator[str]:
    user = (
        f"CONTEXT BEFORE:\n{context_before}\n\n"
        f"SELECTION TO EDIT:\n{selection}\n\n"
        f"CONTEXT AFTER:\n{context_after}\n\n"
        f"INSTRUCTION: {instruction}"
    )
    client = OpenRouterClient()
    for chunk in client.send_prompt_stream(model, _EDIT_SPAN_SYSTEM, user):
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


def list_jobs() -> list[dict]:
    jobs: dict[str, str] = {}
    for p in relative_paths_with_suffix(TEX_DIR, ".tex"):
        jobs[p] = "done"
    for p in relative_paths_with_suffix(PENDING_DIR, ".tex"):
        jobs.setdefault(p, "compiling")
    for p in relative_paths_with_suffix(PENDING_DIR, ".error"):
        jobs.setdefault(p, "error")
    for p in relative_paths_with_suffix(INBOX_DIR, ".job"):
        jobs.setdefault(p, "pending")
    return [{"id": k, "status": v} for k, v in sorted(jobs.items())]


async def create_job(
    path: str, model: str, fidelity: str, files: list[UploadFile]
) -> dict:
    inbox_dir = INBOX_DIR / Path(path).parent
    stem = Path(path).name
    inbox_dir.mkdir(parents=True, exist_ok=True)
    saved = await save_uploaded_files(files, inbox_dir, stem)
    write_job_descriptor(inbox_dir, stem, model, fidelity, saved)
    return {"id": path, "status": "pending", "files": saved}

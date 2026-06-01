from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from app.helpers.jobs_helpers import ALLOWED_SUFFIXES, FIDELITY_VALUES
from app.services import jobs_services
from config.paths import INBOX_DIR, TEX_DIR
from llm import list_models

jobs_router = APIRouter(prefix="/jobs")


@jobs_router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(jobs_services.serve_dashboard())


@jobs_router.get("/editor", response_class=HTMLResponse)
async def editor():
    return HTMLResponse(jobs_services.serve_editor())


@jobs_router.put("/note/{path:path}")
async def save_note(path: str, body: Annotated[str, Body(media_type="text/plain")]):
    return jobs_services.save_note_tex(path, body)


@jobs_router.delete("/note/{path:path}")
async def delete_note(path: str):
    return jobs_services.delete_note(path)


@jobs_router.post("/rename/{path:path}")
async def rename_note(path: str, to: Annotated[str, Body(embed=True)]):
    return jobs_services.rename_note(path, to)


@jobs_router.post("/compile/{path:path}")
async def compile_note(path: str, body: Annotated[str, Body(media_type="text/plain")]):
    pdf = jobs_services.compile_note_body(path, body)
    return Response(content=pdf, media_type="application/pdf")


@jobs_router.post("/ai/edit-span")
async def edit_span(request: Request):
    data = await request.json()
    if data.get("model") not in list_models():
        raise HTTPException(status_code=422, detail="Unknown model")
    stream = jobs_services.stream_span_edit(
        model=data["model"],
        instruction=data.get("instruction", ""),
        selection=data.get("selection", ""),
        context_before=data.get("context_before", ""),
        context_after=data.get("context_after", ""),
    )
    return StreamingResponse(stream, media_type="text/event-stream")


@jobs_router.get("")
async def list_jobs():
    return jobs_services.list_jobs()


@jobs_router.get("/models")
async def models():
    return list_models()


@jobs_router.post("")
async def create_job(
    path: str = Form(...),
    model: str = Form(...),
    fidelity: str = Form("standard"),
    files: Annotated[list[UploadFile], File(...)] = ...,
):
    if ".." in Path(path).parts or Path(path).is_absolute():
        raise HTTPException(status_code=422, detail="Invalid path")
    if fidelity not in FIDELITY_VALUES:
        raise HTTPException(status_code=422, detail=f"Invalid fidelity: {fidelity}")
    if model not in list_models():
        raise HTTPException(status_code=422, detail=f"Unknown model: {model}")
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=415, detail=f"Unsupported file: {f.filename}"
            )
    if (TEX_DIR / f"{path}.tex").exists():
        raise HTTPException(status_code=409, detail=f"Output already exists for {path}")
    if (INBOX_DIR / Path(path).parent / f"{Path(path).name}.job").exists():
        raise HTTPException(status_code=409, detail=f"Job already pending for {path}")
    return await jobs_services.create_job(path, model, fidelity, files)

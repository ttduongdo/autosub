"""FastAPI backend for the AutoSub frontend.

Single-user, local-only, in-memory job tracking (no queue/DB — matches the
project's local-first scope). Jobs run in a background thread per request;
the frontend polls /status/{job_id} for progress and final results.
"""
import threading
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.align import align_segments
from pipeline.export import write_srt, write_vtt
from pipeline.ingest import resolve_input
from pipeline.separate import separate_vocals
from pipeline.transcribe import transcribe

OUTPUT_DIR = Path(__file__).parents[1] / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"

BASELINE_MODEL = "openai/whisper-base"

app = FastAPI(title="AutoSub")

JobStatus = Literal["pending", "downloading", "separating", "transcribing", "aligning", "done", "error"]


class Job:
    def __init__(self, job_id: str):
        self.id = job_id
        self.status: JobStatus = "pending"
        self.error: str | None = None
        self.segments: list[dict] | None = None
        self.words: list[dict] | None = None
        self.vocals_path: Path | None = None
        self.instrumental_path: Path | None = None
        self.mix_path: Path | None = None
        self.srt_path: Path | None = None
        self.vtt_path: Path | None = None


JOBS: dict[str, Job] = {}

STEM_PATH_ATTR = {
    "vocals": "vocals_path",
    "instrumental": "instrumental_path",
    "mix": "mix_path",
}


def _run_pipeline(job: Job, source: str):
    try:
        job.status = "downloading"
        normalized_path = resolve_input(source)

        job.status = "separating"
        vocals_path, instrumental_path, mix_path = separate_vocals(normalized_path)
        job.vocals_path = vocals_path
        job.instrumental_path = instrumental_path
        job.mix_path = mix_path

        job.status = "transcribing"
        segments = transcribe(vocals_path, model_id=BASELINE_MODEL)
        job.segments = segments

        job.status = "aligning"
        words = align_segments(vocals_path, segments)
        job.words = words

        job.srt_path = write_srt(segments, OUTPUT_DIR / f"{job.id}.srt")
        job.vtt_path = write_vtt(segments, OUTPUT_DIR / f"{job.id}.vtt")

        job.status = "done"
    except Exception as e:
        job.status = "error"
        job.error = str(e)


class ProcessRequest(BaseModel):
    url: str


@app.post("/api/process")
def process(req: ProcessRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(400, "No URL provided.")
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id)
    JOBS[job_id] = job
    thread = threading.Thread(target=_run_pipeline, args=(job, req.url.strip()), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.post("/api/upload")
async def upload(file: UploadFile):
    job_id = uuid.uuid4().hex[:12]
    raw_dir = Path(__file__).parents[1] / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{job_id}_{file.filename}"
    dest.write_bytes(await file.read())

    job = Job(job_id)
    JOBS[job_id] = job
    thread = threading.Thread(target=_run_pipeline, args=(job, str(dest)), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")

    response = {"status": job.status}
    if job.status == "error":
        response["error"] = job.error
    if job.status == "done":
        response["segments"] = job.segments
        response["words"] = job.words
        response["audio_urls"] = {
            stem: f"/api/audio/{job_id}/{stem}" for stem in STEM_PATH_ATTR
        }
        response["srt_url"] = f"/api/download/{job_id}/srt"
        response["vtt_url"] = f"/api/download/{job_id}/vtt"
    return response


@app.get("/api/audio/{job_id}/{stem}")
def get_audio(job_id: str, stem: Literal["vocals", "instrumental", "mix"]):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    path = getattr(job, STEM_PATH_ATTR[stem])
    if path is None:
        raise HTTPException(404, f"'{stem}' stem not available for this job.")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/download/{job_id}/{fmt}")
def download(job_id: str, fmt: Literal["srt", "vtt"]):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    path = job.srt_path if fmt == "srt" else job.vtt_path
    if path is None:
        raise HTTPException(404, f"{fmt} not available for this job.")
    return FileResponse(path, media_type="text/plain", filename=path.name)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

import os
import shutil
import uuid

from celery.result import AsyncResult
from fastapi import APIRouter, UploadFile, File, Form

from services.celery_app import celery_app

router = APIRouter()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
SEGMENT_DIR = os.path.join(DATA_DIR, "segments")
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SEGMENT_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

@router.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    target_language: str = Form("en"),
    selected_speakers: str | None = Form(None),
    user_voice_id: str | None = Form(None),
):
    job_id = str(uuid.uuid4())
    filename = os.path.basename(file.filename)
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}_{filename}")
    segment_dir = os.path.join(SEGMENT_DIR, job_id)
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    transcript_dir = os.path.join(TRANSCRIPT_DIR, job_id)

    os.makedirs(segment_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(transcript_dir, exist_ok=True)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    parsed_speakers = None
    if selected_speakers:
        parsed_speakers = [
            speaker.strip()
            for speaker in selected_speakers.split(",")
            if speaker.strip()
        ]

    pipeline_options = {
        "target_language": target_language,
        "job_id": job_id,
        "transcripts_dir": TRANSCRIPT_DIR,
        "transcript_dir": transcript_dir,
    }
    if parsed_speakers:
        pipeline_options["selected_speakers"] = parsed_speakers
    if user_voice_id:
        pipeline_options["user_voice_id"] = user_voice_id

    task = celery_app.send_task(
        "process_video",
        args=[video_path, segment_dir, output_dir, pipeline_options],
    )

    return {
        "message": "Video enqueued for processing",
        "job_id": job_id,
        "task_id": task.id,
        "transcript_dir": transcript_dir,
        "pipeline_options": pipeline_options,
    }

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": result.status}
    if result.successful():
        response["result"] = result.result
    elif result.failed():
        response["error"] = str(result.result)
    return response

import os
import shutil
import uuid

from celery.result import AsyncResult
from fastapi import APIRouter, UploadFile, File, Form

from services.celery_app import celery_app

router = APIRouter()

UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "data/outputs"
SEGMENT_DIR = "data/segments"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SEGMENT_DIR, exist_ok=True)

@router.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    target_language: str = Form("en"),
    selected_speakers: str | None = Form(None),
    user_voice_id: str | None = Form(None),
):
    job_id = str(uuid.uuid4())
    filename = os.path.basename(file.filename)
    video_path = f"{UPLOAD_DIR}/{job_id}_{filename}"
    segment_dir = f"{SEGMENT_DIR}/{job_id}"
    output_dir = f"{OUTPUT_DIR}/{job_id}"

    os.makedirs(segment_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    parsed_speakers = None
    if selected_speakers:
        parsed_speakers = [
            speaker.strip()
            for speaker in selected_speakers.split(",")
            if speaker.strip()
        ]

    pipeline_options = {"target_language": target_language}
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

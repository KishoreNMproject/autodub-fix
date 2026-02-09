import os
import shutil
import uuid

from celery.result import AsyncResult
from fastapi import APIRouter, UploadFile, File

from services.celery_app import celery_app

router = APIRouter()

UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "data/outputs"
SEGMENT_DIR = "data/segments"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SEGMENT_DIR, exist_ok=True)

@router.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    filename = os.path.basename(file.filename)
    video_path = f"{UPLOAD_DIR}/{job_id}_{filename}"
    segment_dir = f"{SEGMENT_DIR}/{job_id}"
    output_dir = f"{OUTPUT_DIR}/{job_id}"

    os.makedirs(segment_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task = celery_app.send_task(
        "process_video",
        args=[video_path, segment_dir, output_dir],
    )

    return {
        "message": "Video enqueued for processing",
        "job_id": job_id,
        "task_id": task.id,
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

import os
import shutil

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
    filename = os.path.basename(file.filename)
    video_path = f"{UPLOAD_DIR}/{filename}"

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task = celery_app.send_task(
        "process_video",
        args=[video_path, SEGMENT_DIR, OUTPUT_DIR],
    )

    return {
        "message": "Video enqueued for processing",
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

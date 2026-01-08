from fastapi import APIRouter, UploadFile, File
import shutil
import os
from services.audio_extract import extract_audio
from services.vad import split_speech

router = APIRouter()

UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "data/outputs"
SEGMENT_DIR = "data/segments"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SEGMENT_DIR, exist_ok=True)

@router.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    video_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Step 1: Extract audio
    audio_path = extract_audio(video_path)

    # Step 2: VAD segmentation
    segments = split_speech(audio_path, SEGMENT_DIR)

    return {
        "message": "Video processed successfully",
        "audio_file": audio_path,
        "num_speech_segments": len(segments),
        "segments": segments
    }

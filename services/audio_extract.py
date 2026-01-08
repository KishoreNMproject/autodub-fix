import subprocess
import os

def extract_audio(video_path: str) -> str:
    audio_path = video_path.replace(".mp4", ".wav")
    audio_path = audio_path.replace("uploads", "outputs")

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return audio_path

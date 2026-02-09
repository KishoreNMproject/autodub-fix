from services.audio_extract import extract_audio
from services.celery_app import celery_app
from services.vad import split_speech


@celery_app.task(name="process_video")
def process_video(video_path: str, segment_dir: str, output_dir: str):
    audio_path = extract_audio(video_path, output_dir=output_dir)
    segments = split_speech(audio_path, segment_dir)
    return {
        "audio_file": audio_path,
        "num_speech_segments": len(segments),
        "segments": segments,
        "output_dir": output_dir,
    }

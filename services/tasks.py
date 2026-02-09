from services.audio_extract import extract_audio
from services.celery_app import celery_app
from services.ssvtu import build_translation_units, select_speakers
from services.uvivd import identify_voice_segments
from services.vad import split_speech


@celery_app.task(name="process_video")
def process_video(
    video_path: str,
    segment_dir: str,
    output_dir: str,
    pipeline_options: dict | None = None,
):
    options = pipeline_options or {}
    target_language = options.get("target_language", "en")
    selected_speakers = options.get("selected_speakers")
    user_voice_id = options.get("user_voice_id")

    audio_path = extract_audio(video_path, output_dir=output_dir)
    timestamped_segments = split_speech(audio_path, segment_dir)
    uvivd_segments = identify_voice_segments(timestamped_segments)
    speaker_filtered_segments = select_speakers(uvivd_segments, selected_speakers)
    translation_units = build_translation_units(
        speaker_filtered_segments,
        target_language=target_language,
        user_voice_id=user_voice_id,
    )

    return {
        "audio_file": audio_path,
        "num_speech_segments": len(timestamped_segments),
        "num_uvivd_segments": len(uvivd_segments),
        "num_selected_segments": len(speaker_filtered_segments),
        "segments": uvivd_segments,
        "translation_units": translation_units,
        "pipeline_options": options,
        "output_dir": output_dir,
    }

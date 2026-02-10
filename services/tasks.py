import json
import os

from services.audio_extract import extract_audio
from services.asr import transcribe_segments
from services.celery_app import celery_app
from services.ssvtu import build_translation_units, select_speakers
from services.uvivd import identify_voice_segments
from services.vad import split_speech

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _format_ms(total_ms: int) -> str:
    total_ms = max(0, int(total_ms))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _save_transcripts(
    transcribed_segments: list[dict],
    transcript_dir: str,
) -> dict:
    os.makedirs(transcript_dir, exist_ok=True)

    transcript_txt_path = os.path.join(transcript_dir, "transcript.txt")
    transcript_json_path = os.path.join(transcript_dir, "transcript.json")

    lines = []
    for segment in transcribed_segments:
        start_ms = segment.get("start_ms", 0)
        end_ms = segment.get("end_ms", 0)
        speaker_id = segment.get("speaker_id", "unknown_speaker")
        transcript_text = (segment.get("transcript_text") or segment.get("text") or "").strip()
        lines.append(
            f"[{_format_ms(start_ms)} --> {_format_ms(end_ms)}] "
            f"{speaker_id}: {transcript_text}"
        )

    if not lines:
        lines.append("[no transcribed segments]")

    with open(transcript_txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write("\n".join(lines))

    transcript_payload = {
        "num_segments": len(transcribed_segments),
        "segments": transcribed_segments,
    }
    with open(transcript_json_path, "w", encoding="utf-8") as json_file:
        json.dump(transcript_payload, json_file, ensure_ascii=False, indent=2)

    return {
        "transcript_dir": transcript_dir,
        "transcript_txt": transcript_txt_path,
        "transcript_json": transcript_json_path,
    }


@celery_app.task(name="process_video")
def process_video(
    video_path: str,
    segment_dir: str,
    output_dir: str,
    pipeline_options: dict | None = None,
):
    options = pipeline_options or {}
    video_path = _resolve_project_path(video_path)
    segment_dir = _resolve_project_path(segment_dir)
    output_dir = _resolve_project_path(output_dir)

    target_language = options.get("target_language", "en")
    source_language = options.get("source_language")
    selected_speakers = options.get("selected_speakers")
    user_voice_id = options.get("user_voice_id")
    job_id = options.get("job_id") or os.path.basename(output_dir.rstrip("/"))
    transcripts_root = _resolve_project_path(options.get("transcripts_dir", "data/transcripts"))
    transcript_dir = options.get("transcript_dir") or os.path.join(transcripts_root, job_id)
    transcript_dir = _resolve_project_path(transcript_dir)

    audio_path = extract_audio(video_path, output_dir=output_dir)
    timestamped_segments = split_speech(audio_path, segment_dir)
    uvivd_segments = identify_voice_segments(timestamped_segments)
    transcribed_segments = transcribe_segments(
        uvivd_segments,
        source_language=source_language,
    )
    speaker_filtered_segments = select_speakers(transcribed_segments, selected_speakers)
    translation_units = build_translation_units(
        speaker_filtered_segments,
        target_language=target_language,
        user_voice_id=user_voice_id,
    )
    transcript_files = _save_transcripts(transcribed_segments, transcript_dir)

    return {
        "audio_file": audio_path,
        "num_speech_segments": len(timestamped_segments),
        "num_uvivd_segments": len(uvivd_segments),
        "num_selected_segments": len(speaker_filtered_segments),
        "num_transcribed_segments": len(transcribed_segments),
        "segments": uvivd_segments,
        "transcribed_segments": transcribed_segments,
        "translation_units": translation_units,
        **transcript_files,
        "pipeline_options": options,
        "output_dir": output_dir,
    }

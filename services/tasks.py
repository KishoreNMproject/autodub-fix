import json
import os
from collections import Counter

from services.audio_extract import extract_audio
from services.asr import transcribe_segments
from services.celery_app import celery_app
from services.media_output import build_dubbed_track, mux_multitrack_video
from services.ssvtu import build_translation_units, select_speakers
from services.translation import translate_segments
from services.uvivd import identify_voice_segments
from services.vad import split_speech
from services.vits_service import preload_xtts_model_cache, run_vits_pipeline
from services.voice_layers import build_speaker_voice_layers

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


def _save_translation_units(
    translation_units: list[dict],
    translation_dir: str,
    target_language: str,
) -> dict:
    os.makedirs(translation_dir, exist_ok=True)
    units_json_path = os.path.join(translation_dir, f"translation_units_{target_language}.json")

    payload = {
        "target_language": target_language,
        "num_units": len(translation_units),
        "units": translation_units,
    }
    with open(units_json_path, "w", encoding="utf-8") as units_file:
        json.dump(payload, units_file, ensure_ascii=False, indent=2)

    return {
        "translation_dir": translation_dir,
        "translation_units_json": units_json_path,
    }


def _save_translated_segments(
    translated_segments: list[dict],
    translation_dir: str,
    target_language: str,
) -> dict:
    os.makedirs(translation_dir, exist_ok=True)
    translated_json_path = os.path.join(
        translation_dir,
        f"translated_segments_{target_language}.json",
    )

    payload = {
        "target_language": target_language,
        "num_segments": len(translated_segments),
        "segments": translated_segments,
    }
    with open(translated_json_path, "w", encoding="utf-8") as translated_file:
        json.dump(payload, translated_file, ensure_ascii=False, indent=2)

    return {"translated_segments_json": translated_json_path}


def _resolve_task_id() -> str:
    task_request = getattr(process_video, "request", None)
    task_id = getattr(task_request, "id", None)
    if task_id:
        return str(task_id)
    return "manual_task"


def _infer_source_language(
    transcribed_segments: list[dict],
    requested_source_language: str | None,
) -> str | None:
    normalized_requested = str(requested_source_language or "").strip().lower()
    if normalized_requested:
        return normalized_requested

    detected = []
    for segment in transcribed_segments:
        language = str(segment.get("detected_source_language") or "").strip().lower()
        if language and language != "unknown":
            detected.append(language)

    if not detected:
        return None
    return Counter(detected).most_common(1)[0][0]


def _count_by_status(items: list[dict], key: str, expected: str) -> int:
    return sum(1 for item in items if str(item.get(key) or "").strip() == expected)


@celery_app.task(name="preload_xtts_model")
def preload_xtts_model_task(load_model: bool = False):
    task_request = getattr(preload_xtts_model_task, "request", None)
    task_id = getattr(task_request, "id", None)
    result = preload_xtts_model_cache(load_model=bool(load_model))
    result["task_id"] = str(task_id) if task_id else "manual_task"
    return result


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
    requested_source_language = options.get("source_language")
    selected_speakers = options.get("selected_speakers")
    user_voice_id = options.get("user_voice_id")
    job_id = options.get("job_id") or os.path.basename(output_dir.rstrip("/"))
    task_id = _resolve_task_id()
    transcripts_root = _resolve_project_path(options.get("transcripts_dir", "data/transcripts"))
    transcript_dir = options.get("transcript_dir") or os.path.join(transcripts_root, job_id)
    transcript_dir = _resolve_project_path(transcript_dir)
    translations_root = _resolve_project_path(options.get("translations_dir", "data/translations/segments"))
    translation_dir = options.get("translation_dir") or os.path.join(translations_root, job_id)
    translation_dir = _resolve_project_path(translation_dir)
    final_output_root = _resolve_project_path(options.get("final_output_root", "data/outputs"))
    final_output_dir = os.path.join(final_output_root, job_id, task_id)
    os.makedirs(final_output_dir, exist_ok=True)

    audio_path = extract_audio(video_path, output_dir=output_dir)
    timestamped_segments = split_speech(audio_path, segment_dir)
    uvivd_segments = identify_voice_segments(timestamped_segments)
    transcribed_segments = transcribe_segments(
        uvivd_segments,
        source_language=requested_source_language,
    )
    detected_source_language = _infer_source_language(
        transcribed_segments=transcribed_segments,
        requested_source_language=requested_source_language,
    )
    translated_segments = translate_segments(
        transcribed_segments=transcribed_segments,
        target_language=target_language,
        source_language=detected_source_language,
    )
    speaker_filtered_segments = select_speakers(translated_segments, selected_speakers)
    translation_units = build_translation_units(
        speaker_filtered_segments,
        target_language=target_language,
        user_voice_id=user_voice_id,
    )
    transcript_files = _save_transcripts(transcribed_segments, transcript_dir)
    translation_files = _save_translation_units(
        translation_units=translation_units,
        translation_dir=translation_dir,
        target_language=target_language,
    )
    translated_segments_file = _save_translated_segments(
        translated_segments=translated_segments,
        translation_dir=translation_dir,
        target_language=target_language,
    )
    speaker_voice_layers = build_speaker_voice_layers(
        transcribed_segments=transcribed_segments,
        translation_dir=translation_dir,
    )
    vits_result = run_vits_pipeline(
        translation_units=translation_units,
        speaker_voice_layers=speaker_voice_layers,
        translation_dir=translation_dir,
        target_language=target_language,
    )
    dubbed_track_output = os.path.join(final_output_dir, f"dubbed_track_{target_language}.wav")
    dubbed_track = build_dubbed_track(
        dubbed_segments=vits_result.get("dubbed_segments", []),
        output_path=dubbed_track_output,
    )
    final_video_output = os.path.join(final_output_dir, f"{job_id}_{target_language}_multitrack.mp4")
    multitrack_video = mux_multitrack_video(
        video_path=video_path,
        dubbed_track_path=dubbed_track.get("dubbed_track_audio", ""),
        output_video_path=final_video_output,
        target_language=target_language,
    )
    translated_success_count = _count_by_status(
        items=translated_segments,
        key="translation_status",
        expected="translated",
    )
    finetune_results = dict(vits_result.get("finetune_results") or {})
    finetune_success_count = sum(
        1
        for speaker_result in finetune_results.values()
        if str(speaker_result.get("status") or "").strip()
        in {"finetune_command_completed", "finetune_internal_xtts_adaptation"}
    )
    generated_dubbed_segments_count = _count_by_status(
        items=list(vits_result.get("dubbed_segments") or []),
        key="status",
        expected="generated_with_vits",
    )

    return {
        "job_id": job_id,
        "task_id": task_id,
        "audio_file": audio_path,
        "num_speech_segments": len(timestamped_segments),
        "num_uvivd_segments": len(uvivd_segments),
        "num_selected_segments": len(speaker_filtered_segments),
        "num_transcribed_segments": len(transcribed_segments),
        "num_translated_segments": len(translated_segments),
        "num_successfully_translated_segments": translated_success_count,
        "num_finetuned_speakers": finetune_success_count,
        "num_generated_dubbed_segments": generated_dubbed_segments_count,
        "detected_source_language": detected_source_language,
        "segments": uvivd_segments,
        "transcribed_segments": transcribed_segments,
        "translated_segments": translated_segments,
        "translation_units": translation_units,
        **transcript_files,
        **translation_files,
        **translated_segments_file,
        "speaker_voice_layers": speaker_voice_layers,
        "vits": vits_result,
        "dubbed_track": dubbed_track,
        "final_output_dir": final_output_dir,
        "final_video": multitrack_video,
        "pipeline_options": options,
        "output_dir": output_dir,
    }

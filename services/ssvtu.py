from __future__ import annotations

from typing import Any


def _normalize_speaker_token(value: Any) -> str:
    return str(value or "").strip().lower()


def select_speakers(
    segments: list[dict[str, Any]],
    selected_speakers: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    SSVTU stage 1: speaker-selective filtering.
    """
    if not selected_speakers:
        return segments

    selected = {
        _normalize_speaker_token(speaker)
        for speaker in selected_speakers
        if _normalize_speaker_token(speaker)
    }
    if not selected or "all" in selected or "*" in selected:
        return segments

    filtered = [
        segment
        for segment in segments
        if _normalize_speaker_token(segment.get("speaker_id")) in selected
        or _normalize_speaker_token(segment.get("voice_id")) in selected
    ]
    if filtered:
        return filtered

    if "ai" in selected:
        unique_speakers = {
            _normalize_speaker_token(segment.get("speaker_id"))
            for segment in segments
            if _normalize_speaker_token(segment.get("speaker_id"))
        }
        if len(unique_speakers) == 1:
            return segments

    return filtered


def build_translation_units(
    segments: list[dict[str, Any]],
    target_language: str = "en",
    user_voice_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    SSVTU stage 2: prepare speaker-aware translation units.
    """
    units: list[dict[str, Any]] = []
    for segment in segments:
        target_voice_id = user_voice_id or segment.get("voice_id")
        translated_text = (
            segment.get("translated_text")
            or segment.get("target_text")
            or segment.get("transcript_text", "")
        )
        units.append(
            {
                "segment_index": segment.get("segment_index"),
                "segment_path": segment.get("segment_path"),
                "speaker_id": segment.get("speaker_id"),
                "source_voice_id": segment.get("voice_id"),
                "target_voice_id": target_voice_id,
                "start_ms": segment.get("start_ms"),
                "end_ms": segment.get("end_ms"),
                "duration_ms": segment.get("duration_ms"),
                "target_language": target_language,
                "transcript_text": segment.get("transcript_text", ""),
                "translated_text": translated_text,
                "translation_status": segment.get("translation_status"),
                "status": "ready_for_translation",
            }
        )
    return units

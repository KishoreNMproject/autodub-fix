from __future__ import annotations

from typing import Any


def select_speakers(
    segments: list[dict[str, Any]],
    selected_speakers: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    SSVTU stage 1: speaker-selective filtering.
    """
    if not selected_speakers:
        return segments

    selected = set(selected_speakers)
    return [
        segment
        for segment in segments
        if segment.get("speaker_id") in selected
    ]


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
                "status": "ready_for_translation",
            }
        )
    return units

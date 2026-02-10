from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from services.device import get_device

_ASR_MODEL = None
_ASR_INIT_ERROR = None


def _resolve_model_path() -> str:
    configured = os.getenv("ASR_MODEL_PATH", "models/faster-whisper-small")
    return str(Path(configured))


def _compute_type(device_name: str) -> str:
    if device_name == "cuda":
        return "float16"
    return "int8"


def _get_model():
    global _ASR_MODEL, _ASR_INIT_ERROR
    if _ASR_MODEL is not None:
        return _ASR_MODEL
    if _ASR_INIT_ERROR is not None:
        return None

    try:
        from faster_whisper import WhisperModel

        device, _ = get_device()
        device_name = str(device)
        model_path = _resolve_model_path()
        _ASR_MODEL = WhisperModel(
            model_size_or_path=model_path,
            device=device_name,
            compute_type=_compute_type(device_name),
        )
        return _ASR_MODEL
    except Exception as exc:
        _ASR_INIT_ERROR = exc
        return None


def transcribe_segments(
    segments: list[dict[str, Any]],
    source_language: str | None = None,
) -> list[dict[str, Any]]:
    """
    Adds transcript_text per segment.
    If ASR is unavailable, uses a non-empty fallback marker.
    """
    if not segments:
        return []

    model = _get_model()
    output: list[dict[str, Any]] = []

    for segment in segments:
        item = dict(segment)
        text = ""

        if model is not None:
            try:
                result_iter, _ = model.transcribe(
                    audio=item["segment_path"],
                    language=source_language,
                    beam_size=5,
                    condition_on_previous_text=False,
                    vad_filter=False,
                )
                text = " ".join(part.text.strip() for part in result_iter).strip()
            except Exception:
                text = ""

        if not text:
            text = "[transcription unavailable]"

        item["transcript_text"] = text
        output.append(item)

    return output

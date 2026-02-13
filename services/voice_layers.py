from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from typing import Any

import torch
import torchaudio

TARGET_SAMPLE_RATE = 16000
DEFAULT_REFERENCE_SECONDS = 90


def _safe_speaker_id(value: Any) -> str:
    text = str(value or "unknown_speaker").strip().lower()
    return re.sub(r"[^a-z0-9_-]+", "_", text).strip("_") or "unknown_speaker"


def _load_segment_as_mono_16k(path: str) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE,
        )
    return waveform


def _average_pitch_hz(waveform: torch.Tensor) -> float:
    try:
        pitches = torchaudio.functional.detect_pitch_frequency(
            waveform,
            sample_rate=TARGET_SAMPLE_RATE,
        )
        if pitches.numel() == 0:
            return 0.0
        non_zero = pitches[pitches > 0]
        if non_zero.numel() == 0:
            return 0.0
        return float(non_zero.mean().item())
    except Exception:
        return 0.0


def _emotion_label(avg_pitch_hz: float, rms_energy: float, zcr: float) -> str:
    if rms_energy > 0.08 and avg_pitch_hz > 170:
        return "excited"
    if rms_energy < 0.02 and avg_pitch_hz < 140:
        return "calm"
    if zcr > 0.12 and avg_pitch_hz > 180:
        return "tense"
    return "neutral"


def _segment_voice_characteristics(
    segment_path: str,
    waveform: torch.Tensor,
    duration_ms: int | None = None,
) -> dict[str, Any]:
    mono = waveform.squeeze(0)
    if mono.numel() == 0:
        return {
            "duration_ms": int(duration_ms or 0),
            "avg_pitch_hz": 0.0,
            "rms_energy": 0.0,
            "zcr": 0.0,
            "emotion_label": "neutral",
        }

    if duration_ms is None:
        duration_ms = int((mono.numel() * 1000) / TARGET_SAMPLE_RATE)

    rms_energy = float(torch.sqrt(torch.mean(mono * mono)).item())
    avg_pitch_hz = _average_pitch_hz(waveform)
    zero_crossings = torch.sum((mono[:-1] * mono[1:]) < 0).item() if mono.numel() > 1 else 0
    zcr = float(zero_crossings / max(1, mono.numel()))
    emotion_label = _emotion_label(avg_pitch_hz=avg_pitch_hz, rms_energy=rms_energy, zcr=zcr)

    return {
        "segment_path": segment_path,
        "duration_ms": int(duration_ms),
        "avg_pitch_hz": round(avg_pitch_hz, 3),
        "rms_energy": round(rms_energy, 6),
        "zcr": round(zcr, 6),
        "emotion_label": emotion_label,
    }


def _speaker_characterization_summary(characteristics: list[dict[str, Any]]) -> dict[str, Any]:
    if not characteristics:
        return {
            "num_segments": 0,
            "avg_pitch_hz": 0.0,
            "avg_rms_energy": 0.0,
            "avg_zcr": 0.0,
            "dominant_emotion": "neutral",
        }

    num_segments = len(characteristics)
    avg_pitch = sum(float(item.get("avg_pitch_hz", 0.0)) for item in characteristics) / num_segments
    avg_energy = sum(float(item.get("rms_energy", 0.0)) for item in characteristics) / num_segments
    avg_zcr = sum(float(item.get("zcr", 0.0)) for item in characteristics) / num_segments

    emotion_counts: dict[str, int] = {}
    for item in characteristics:
        label = str(item.get("emotion_label") or "neutral")
        emotion_counts[label] = emotion_counts.get(label, 0) + 1
    dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"

    return {
        "num_segments": num_segments,
        "avg_pitch_hz": round(avg_pitch, 3),
        "avg_rms_energy": round(avg_energy, 6),
        "avg_zcr": round(avg_zcr, 6),
        "dominant_emotion": dominant_emotion,
        "emotion_distribution": emotion_counts,
    }


def _build_reference_wav(segment_paths: list[str], output_path: str) -> str:
    if not segment_paths:
        return ""

    max_seconds = int(os.getenv("VOICE_LAYER_REFERENCE_SECONDS", str(DEFAULT_REFERENCE_SECONDS)))
    max_samples = max_seconds * TARGET_SAMPLE_RATE

    pieces: list[torch.Tensor] = []
    used_samples = 0

    for segment_path in segment_paths:
        if not os.path.exists(segment_path):
            continue

        try:
            waveform = _load_segment_as_mono_16k(segment_path)
        except Exception:
            continue

        remaining = max_samples - used_samples
        if remaining <= 0:
            break

        if waveform.size(1) > remaining:
            waveform = waveform[:, :remaining]

        pieces.append(waveform)
        used_samples += waveform.size(1)

    if not pieces:
        return ""

    merged = torch.cat(pieces, dim=1)
    torchaudio.save(output_path, merged, TARGET_SAMPLE_RATE)
    return output_path


def build_speaker_voice_layers(
    transcribed_segments: list[dict[str, Any]],
    translation_dir: str,
) -> dict[str, Any]:
    os.makedirs(translation_dir, exist_ok=True)
    speakers_root = os.path.join(translation_dir, "speakers")
    os.makedirs(speakers_root, exist_ok=True)

    grouped_segments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in transcribed_segments:
        speaker_id = _safe_speaker_id(segment.get("speaker_id") or segment.get("voice_id"))
        grouped_segments[speaker_id].append(segment)

    speaker_layers: dict[str, dict[str, Any]] = {}
    voice_characterizations: dict[str, Any] = {}

    for speaker_id, speaker_segments in grouped_segments.items():
        speaker_dir = os.path.join(speakers_root, speaker_id)
        segments_dir = os.path.join(speaker_dir, "segments")
        os.makedirs(segments_dir, exist_ok=True)

        copied_segment_paths: list[str] = []
        manifest_segments: list[dict[str, Any]] = []
        segment_characteristics: list[dict[str, Any]] = []

        for segment in speaker_segments:
            source_path = str(segment.get("segment_path") or "")
            if not source_path or not os.path.exists(source_path):
                continue

            segment_filename = os.path.basename(source_path)
            local_segment_path = os.path.join(segments_dir, segment_filename)
            shutil.copy2(source_path, local_segment_path)
            copied_segment_paths.append(local_segment_path)

            try:
                local_waveform = _load_segment_as_mono_16k(local_segment_path)
                segment_characterization = _segment_voice_characteristics(
                    segment_path=local_segment_path,
                    waveform=local_waveform,
                    duration_ms=segment.get("duration_ms"),
                )
            except Exception:
                segment_characterization = {
                    "segment_path": local_segment_path,
                    "duration_ms": int(segment.get("duration_ms") or 0),
                    "avg_pitch_hz": 0.0,
                    "rms_energy": 0.0,
                    "zcr": 0.0,
                    "emotion_label": "neutral",
                }
            segment_characteristics.append(segment_characterization)

            manifest_segments.append(
                {
                    "segment_index": segment.get("segment_index"),
                    "speaker_id": segment.get("speaker_id"),
                    "voice_id": segment.get("voice_id"),
                    "start_ms": segment.get("start_ms"),
                    "end_ms": segment.get("end_ms"),
                    "duration_ms": segment.get("duration_ms"),
                    "source_segment_path": source_path,
                    "local_segment_path": local_segment_path,
                    "transcript_text": segment.get("transcript_text", ""),
                    "voice_characterization": segment_characterization,
                }
            )

        reference_audio_path = _build_reference_wav(
            copied_segment_paths,
            os.path.join(speaker_dir, "reference.wav"),
        )

        voice_layer_manifest = {
            "speaker_id": speaker_id,
            "num_segments": len(manifest_segments),
            "segments_dir": segments_dir,
            "reference_audio": reference_audio_path,
            "segments": manifest_segments,
        }
        manifest_path = os.path.join(speaker_dir, "voice_layer.json")
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(voice_layer_manifest, manifest_file, ensure_ascii=False, indent=2)

        speaker_layers[speaker_id] = {
            "speaker_id": speaker_id,
            "num_segments": len(manifest_segments),
            "segments_dir": segments_dir,
            "reference_audio": reference_audio_path,
            "manifest_path": manifest_path,
        }
        voice_characterizations[speaker_id] = {
            "speaker_id": speaker_id,
            "summary": _speaker_characterization_summary(segment_characteristics),
            "segments": segment_characteristics,
        }

    summary = {
        "speakers_root": speakers_root,
        "num_speakers": len(speaker_layers),
        "speaker_layers": speaker_layers,
    }

    summary_path = os.path.join(translation_dir, "speaker_voice_layers.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)

    characterizations_path = os.path.join(translation_dir, "voice_characterizations.json")
    with open(characterizations_path, "w", encoding="utf-8") as characterizations_file:
        json.dump(voice_characterizations, characterizations_file, ensure_ascii=False, indent=2)

    summary["speaker_voice_layers_json"] = summary_path
    summary["voice_characterizations_json"] = characterizations_path
    summary["voice_characterizations"] = voice_characterizations
    return summary

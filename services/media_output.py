from __future__ import annotations

import os
import subprocess
from typing import Any

import torch
import torchaudio

TARGET_SAMPLE_RATE = 16000


def _load_audio_16k_mono(audio_path: str) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE,
        )
    return waveform


def normalize_segment_duration(audio_path: str, target_duration_ms: int | None) -> str:
    if not audio_path or not os.path.exists(audio_path):
        return audio_path
    if target_duration_ms is None:
        return audio_path

    waveform = _load_audio_16k_mono(audio_path)
    target_samples = int(max(1, target_duration_ms) * TARGET_SAMPLE_RATE / 1000)
    current_samples = waveform.size(1)

    if current_samples == target_samples:
        return audio_path
    if current_samples > target_samples:
        waveform = waveform[:, :target_samples]
    else:
        pad_amount = target_samples - current_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

    torchaudio.save(audio_path, waveform, TARGET_SAMPLE_RATE)
    return audio_path


def build_dubbed_track(
    dubbed_segments: list[dict[str, Any]],
    output_path: str,
) -> dict[str, Any]:
    valid_segments = [
        segment
        for segment in dubbed_segments
        if str(segment.get("output_audio") or "").strip()
        and os.path.exists(str(segment.get("output_audio")))
    ]
    if not valid_segments:
        return {
            "dubbed_track_audio": "",
            "num_segments_mixed": 0,
            "sample_rate": TARGET_SAMPLE_RATE,
        }

    end_times_ms = []
    for segment in valid_segments:
        end_ms = segment.get("end_ms")
        if end_ms is not None:
            end_times_ms.append(int(end_ms))

    if end_times_ms:
        total_samples = int(max(end_times_ms) * TARGET_SAMPLE_RATE / 1000) + TARGET_SAMPLE_RATE
        track = torch.zeros(1, max(total_samples, TARGET_SAMPLE_RATE), dtype=torch.float32)
    else:
        track = torch.zeros(1, TARGET_SAMPLE_RATE, dtype=torch.float32)

    mixed_count = 0
    for segment in valid_segments:
        segment_path = str(segment.get("output_audio"))
        start_ms = int(segment.get("start_ms") or 0)
        start_sample = max(0, int(start_ms * TARGET_SAMPLE_RATE / 1000))

        waveform = _load_audio_16k_mono(segment_path)
        end_sample = start_sample + waveform.size(1)
        if end_sample > track.size(1):
            extra = end_sample - track.size(1)
            track = torch.nn.functional.pad(track, (0, extra))

        track[:, start_sample:end_sample] += waveform
        mixed_count += 1

    track = torch.clamp(track, min=-1.0, max=1.0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, track, TARGET_SAMPLE_RATE)
    return {
        "dubbed_track_audio": output_path,
        "num_segments_mixed": mixed_count,
        "sample_rate": TARGET_SAMPLE_RATE,
    }


def mux_multitrack_video(
    video_path: str,
    dubbed_track_path: str,
    output_video_path: str,
    target_language: str,
) -> dict[str, Any]:
    if not dubbed_track_path or not os.path.exists(dubbed_track_path):
        return {
            "final_video_path": "",
            "mux_status": "missing_dubbed_track",
            "mux_stdout": "",
            "mux_stderr": "",
        }

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        dubbed_track_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-metadata:s:a:1",
        f"language={target_language}",
        "-metadata:s:a:1",
        f"title=Dubbed ({target_language})",
        "-shortest",
        output_video_path,
    ]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
    )
    if completed.returncode == 0:
        status = "muxed_multitrack_video"
    else:
        status = f"mux_failed({completed.returncode})"

    return {
        "final_video_path": output_video_path if completed.returncode == 0 else "",
        "mux_status": status,
        "mux_stdout": completed.stdout or "",
        "mux_stderr": completed.stderr or "",
    }

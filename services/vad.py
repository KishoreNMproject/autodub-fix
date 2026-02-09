import os
from pathlib import Path

import torch
import torchaudio

_MODEL = None
_UTILS = None


def _get_cached_repo_path() -> str | None:
    project_root = Path(__file__).resolve().parents[1]
    local_torch_home = project_root / "torch_cache"
    hub_path = local_torch_home / "hub" / "snakers4_silero-vad_master"

    if "TORCH_HOME" not in os.environ and local_torch_home.exists():
        os.environ["TORCH_HOME"] = str(local_torch_home)

    if hub_path.exists():
        return str(hub_path)

    return None


def _load_vad():
    global _MODEL, _UTILS
    if _MODEL is not None and _UTILS is not None:
        return _MODEL, _UTILS

    cached_repo_path = _get_cached_repo_path()
    if cached_repo_path:
        _MODEL, _UTILS = torch.hub.load(
            repo_or_dir=cached_repo_path,
            model="silero_vad",
            source="local",
            force_reload=False,
        )
    else:
        _MODEL, _UTILS = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )

    return _MODEL, _UTILS


def split_speech(audio_path: str, output_dir: str):
    """
    Takes a WAV file and splits it into speech-only segments.
    Returns list of segment file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    model, utils = _load_vad()
    get_speech_timestamps, _, read_audio, _, _ = utils

    wav = read_audio(audio_path, sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

    segments = []
    for i, ts in enumerate(speech_timestamps):
        start = int(ts["start"])
        end = int(ts["end"])
        if end <= start:
            continue

        chunk = wav[start:end]
        segment_path = os.path.join(output_dir, f"seg_{i:03d}.wav")
        torchaudio.save(segment_path, chunk.unsqueeze(0), 16000)
        segments.append(segment_path)

    return segments

from __future__ import annotations

import copy
from typing import Any

import torch
import torchaudio


def _segment_embedding(segment_path: str) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(segment_path)
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    # Lightweight speaker fingerprint using MFCC statistics.
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=20,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40},
    )(waveform)

    embedding = mfcc.mean(dim=-1).squeeze(0).float()
    norm = torch.norm(embedding, p=2)
    if norm.item() == 0:
        return embedding
    return embedding / norm


def identify_voice_segments(
    segments: list[dict[str, Any]],
    similarity_threshold: float = 0.82,
) -> list[dict[str, Any]]:
    """
    UVIVD: User Voice Identified Voice Dubbing.
    Adds speaker_id and voice_id to each timestamped segment.
    """
    if not segments:
        return []

    centroids: list[torch.Tensor] = []
    centroid_counts: list[int] = []
    annotated: list[dict[str, Any]] = []

    for segment in segments:
        record = copy.deepcopy(segment)
        embedding = _segment_embedding(record["segment_path"])

        best_idx = -1
        best_sim = -1.0
        for idx, centroid in enumerate(centroids):
            sim = float(torch.dot(embedding, centroid))
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_idx == -1 or best_sim < similarity_threshold:
            centroids.append(embedding.clone())
            centroid_counts.append(1)
            speaker_idx = len(centroids) - 1
        else:
            speaker_idx = best_idx
            count = centroid_counts[speaker_idx]
            updated = (centroids[speaker_idx] * count + embedding) / (count + 1)
            updated_norm = torch.norm(updated, p=2)
            if updated_norm.item() != 0:
                updated = updated / updated_norm
            centroids[speaker_idx] = updated
            centroid_counts[speaker_idx] = count + 1

        speaker_id = f"spk_{speaker_idx + 1:02d}"
        # voice_id mirrors speaker_id unless overridden later by user voice map.
        record["speaker_id"] = speaker_id
        record["voice_id"] = speaker_id
        annotated.append(record)

    return annotated

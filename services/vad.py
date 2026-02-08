import os

import torch
import torchaudio

# Load Silero VAD model (once)
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False
)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


def split_speech(audio_path: str, output_dir: str):
    """
    Takes a WAV file and splits it into speech-only segments.
    Returns list of segment file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    wav = read_audio(audio_path, sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

    segments = []
    for i, chunk in enumerate(collect_chunks(speech_timestamps, wav)):
        segment_path = os.path.join(output_dir, f"seg_{i:03d}.wav")
        torchaudio.save(segment_path, chunk.unsqueeze(0), 16000)
        segments.append(segment_path)

    return segments

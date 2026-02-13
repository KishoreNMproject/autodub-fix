from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch
import torchaudio

try:
    import fcntl  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - unavailable on Windows
    fcntl = None

try:
    import msvcrt  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - unavailable on POSIX
    msvcrt = None

from services.device import get_device
from services.media_output import normalize_segment_duration

DEFAULT_XTTS_REQUIRED_FILES = [
    "config.json",
    "model.pth",
    "vocab.json",
    "speakers_xtts.pth",
    "dvae.pth",
]

LOGGER = logging.getLogger(__name__)
TARGET_SAMPLE_RATE = 16000


def _safe_speaker_id(value: Any) -> str:
    text = str(value or "unknown_speaker").strip().lower()
    return re.sub(r"[^a-z0-9_-]+", "_", text).strip("_") or "unknown_speaker"


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_synthesis_text(unit: dict[str, Any]) -> str:
    return str(
        unit.get("translated_text")
        or unit.get("target_text")
        or unit.get("transcript_text")
        or unit.get("text")
        or ""
    ).strip()


def _xtts_required_files() -> list[str]:
    configured = str(os.getenv("XTTS_REQUIRED_FILES", "")).strip()
    if not configured:
        return list(DEFAULT_XTTS_REQUIRED_FILES)
    return [part.strip() for part in configured.split(",") if part.strip()]


def _missing_xtts_files(model_dir: str, required_files: list[str]) -> list[str]:
    return [
        filename
        for filename in required_files
        if not os.path.exists(os.path.join(model_dir, filename))
    ]


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name)
    try:
        parsed = int(str(raw_value).strip()) if raw_value is not None else int(default)
    except Exception:
        parsed = int(default)
    return max(minimum, parsed)


def _cache_size_mb(path: str) -> float:
    total_bytes = 0
    for root, _, files in os.walk(path):
        for filename in files:
            full_path = os.path.join(root, filename)
            try:
                total_bytes += os.path.getsize(full_path)
            except OSError:
                continue
    return total_bytes / (1024 * 1024)


def _acquire_download_lock(lock_path: str, timeout_seconds: int):
    lock_file = open(lock_path, "a+", encoding="utf-8")
    start_time = time.time()
    while True:
        try:
            _lock_file_non_blocking(lock_file)
            return lock_file
        except (BlockingIOError, OSError):
            if time.time() - start_time >= timeout_seconds:
                lock_file.close()
                raise TimeoutError(
                    f"timeout waiting for XTTS download lock ({timeout_seconds}s)"
                )
            time.sleep(1.0)


def _lock_file_non_blocking(lock_file) -> None:
    if fcntl is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    if msvcrt is not None:
        # Lock one byte at offset 0 to create a process-level mutex on Windows.
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        return
    raise RuntimeError("no file locking backend available")


def _unlock_file(lock_file) -> None:
    if fcntl is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return
    if msvcrt is not None:
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        return


def _download_repo_file(
    repo_id: str,
    filename: str,
    local_dir: str,
    timeout_seconds: int,
    etag_timeout_seconds: int,
    request_timeout_seconds: int,
) -> None:
    script = (
        "import sys\n"
        "from huggingface_hub import hf_hub_download\n"
        "hf_hub_download(repo_id=sys.argv[1], filename=sys.argv[2], local_dir=sys.argv[3])\n"
    )
    env = os.environ.copy()
    env["HF_HUB_ETAG_TIMEOUT"] = str(max(5, etag_timeout_seconds))
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = str(max(10, request_timeout_seconds))
    completed = subprocess.run(
        [sys.executable, "-c", script, repo_id, filename, local_dir],
        text=True,
        capture_output=True,
        timeout=max(30, timeout_seconds),
        env=env,
    )
    if completed.returncode != 0:
        stderr_tail = (completed.stderr or "")[-1500:]
        stdout_tail = (completed.stdout or "")[-1500:]
        message = stderr_tail or stdout_tail or "hf_hub_download failed"
        raise RuntimeError(message)


def _ensure_xtts_model_cache() -> dict[str, Any]:
    model_dir = str(os.getenv("XTTS_MODEL_DIR", "/app/models/xtts-v2"))
    repo_id = str(os.getenv("XTTS_MODEL_REPO", "coqui/XTTS-v2"))
    auto_download = _is_truthy(os.getenv("XTTS_AUTO_DOWNLOAD", "1"))
    required_files = _xtts_required_files()
    lock_timeout = _env_int("XTTS_DOWNLOAD_LOCK_TIMEOUT", 300, minimum=5)
    file_download_timeout = _env_int("XTTS_FILE_DOWNLOAD_TIMEOUT", 900, minimum=30)
    file_download_retries = _env_int("XTTS_FILE_DOWNLOAD_RETRIES", 2, minimum=1)
    max_download_seconds = _env_int("XTTS_MAX_DOWNLOAD_SECONDS", 1800, minimum=60)
    etag_timeout_seconds = _env_int("HF_HUB_ETAG_TIMEOUT", 30, minimum=5)
    request_timeout_seconds = _env_int("HF_HUB_DOWNLOAD_TIMEOUT", 120, minimum=10)

    os.makedirs(model_dir, exist_ok=True)
    missing_files = _missing_xtts_files(model_dir, required_files)
    downloaded_files: list[str] = []
    cache_error = ""

    if missing_files and auto_download:
        lock_path = os.path.join(model_dir, ".xtts_download.lock")
        lock_file = None
        try:
            lock_file = _acquire_download_lock(lock_path, timeout_seconds=lock_timeout)
            missing_files = _missing_xtts_files(model_dir, required_files)

            if missing_files:
                LOGGER.info(
                    "XTTS model cache missing %s file(s): %s",
                    len(missing_files),
                    ", ".join(missing_files),
                )
                download_started_at = time.time()
                for filename in list(missing_files):
                    elapsed = int(time.time() - download_started_at)
                    remaining_budget = max_download_seconds - elapsed
                    if remaining_budget <= 0:
                        if cache_error:
                            cache_error += "; "
                        cache_error += (
                            "XTTS download budget exceeded "
                            f"({max_download_seconds}s); remaining files skipped"
                        )
                        break

                    before_size_mb = _cache_size_mb(model_dir)
                    attempt_success = False
                    for attempt in range(1, file_download_retries + 1):
                        effective_timeout = min(file_download_timeout, max(30, remaining_budget))
                        try:
                            LOGGER.info(
                                "Downloading XTTS file: %s (attempt %s/%s, timeout=%ss)",
                                filename,
                                attempt,
                                file_download_retries,
                                effective_timeout,
                            )
                            _download_repo_file(
                                repo_id=repo_id,
                                filename=filename,
                                local_dir=model_dir,
                                timeout_seconds=effective_timeout,
                                etag_timeout_seconds=etag_timeout_seconds,
                                request_timeout_seconds=request_timeout_seconds,
                            )
                            downloaded_files.append(filename)
                            after_size_mb = _cache_size_mb(model_dir)
                            LOGGER.info(
                                "Downloaded XTTS file: %s (cache %.1f MB -> %.1f MB)",
                                filename,
                                before_size_mb,
                                after_size_mb,
                            )
                            attempt_success = True
                            break
                        except subprocess.TimeoutExpired:
                            LOGGER.warning(
                                "XTTS download timeout for %s after %ss (attempt %s/%s)",
                                filename,
                                effective_timeout,
                                attempt,
                                file_download_retries,
                            )
                            if attempt >= file_download_retries:
                                if cache_error:
                                    cache_error += "; "
                                cache_error += (
                                    f"{filename}: timeout after {effective_timeout}s "
                                    f"(attempt {attempt}/{file_download_retries})"
                                )
                        except Exception as exc:
                            LOGGER.warning(
                                "XTTS download error for %s on attempt %s/%s: %s",
                                filename,
                                attempt,
                                file_download_retries,
                                exc,
                            )
                            if attempt >= file_download_retries:
                                if cache_error:
                                    cache_error += "; "
                                cache_error += f"{filename}: {exc}"
                    if not attempt_success:
                        continue
        except Exception as exc:
            if cache_error:
                cache_error += "; "
            cache_error += str(exc)
        finally:
            if lock_file is not None:
                try:
                    _unlock_file(lock_file)
                finally:
                    lock_file.close()

    missing_files = _missing_xtts_files(model_dir, required_files)
    return {
        "cache_dir": model_dir,
        "repo_id": repo_id,
        "required_files": required_files,
        "downloaded_files": downloaded_files,
        "missing_files": missing_files,
        "auto_download": auto_download,
        "cache_error": cache_error,
        "lock_timeout_seconds": lock_timeout,
        "file_download_timeout_seconds": file_download_timeout,
        "file_download_retries": file_download_retries,
        "max_download_seconds": max_download_seconds,
        "etag_timeout_seconds": etag_timeout_seconds,
        "request_timeout_seconds": request_timeout_seconds,
    }


def _load_vits_model(model_name: str):
    cache_info = _ensure_xtts_model_cache()

    try:
        from TTS.api import TTS
    except Exception as exc:
        return None, str(exc), cache_info

    model_errors: list[str] = []
    cache_dir = str(cache_info.get("cache_dir") or "")
    model_path = os.path.join(cache_dir, "model.pth")
    config_path = os.path.join(cache_dir, "config.json")

    if os.path.exists(model_path) and os.path.exists(config_path):
        try:
            model = TTS(
                model_path=model_path,
                config_path=config_path,
                progress_bar=False,
            )
            device, _ = get_device()
            if hasattr(model, "to"):
                model = model.to(str(device))
            cache_info["load_source"] = "local_cache"
            return model, "", cache_info
        except Exception as exc:
            model_errors.append(f"local_cache_load_failed: {exc}")

    allow_model_name_fallback = _is_truthy(
        os.getenv("VITS_ALLOW_MODEL_NAME_FALLBACK", "0")
    )
    if not allow_model_name_fallback:
        if cache_info.get("missing_files"):
            model_errors.append("xtts_cache_incomplete")
        if cache_info.get("cache_error"):
            model_errors.append(f"xtts_cache_error: {cache_info.get('cache_error')}")
        if not model_errors:
            model_errors.append("model_name_fallback_disabled")
        return None, "; ".join(model_errors), cache_info

    try:
        model = TTS(model_name=model_name, progress_bar=False)
        device, _ = get_device()
        if hasattr(model, "to"):
            model = model.to(str(device))
        cache_info["load_source"] = "model_name"
        return model, "", cache_info
    except Exception as exc:
        model_errors.append(str(exc))
        return None, "; ".join(model_errors), cache_info


def preload_xtts_model_cache(load_model: bool = False) -> dict[str, Any]:
    model_name = str(
        os.getenv(
            "VITS_MODEL_NAME",
            "tts_models/multilingual/multi-dataset/xtts_v2",
        )
    )
    cache_info = _ensure_xtts_model_cache()
    cache_ready = len(cache_info.get("missing_files") or []) == 0

    payload: dict[str, Any] = {
        "status": "cache_ready" if cache_ready else "cache_incomplete",
        "model_name": model_name,
        "load_model_requested": bool(load_model),
        "model_loaded": False,
        "model_load_error": "",
        "model_cache": cache_info,
    }

    if not load_model:
        return payload

    model, model_error, resolved_cache_info = _load_vits_model(model_name)
    payload["model_cache"] = resolved_cache_info
    payload["model_loaded"] = model is not None
    payload["model_load_error"] = str(model_error or "")

    if model is not None:
        payload["status"] = "cache_ready_model_loaded"
    elif payload["status"] == "cache_ready":
        payload["status"] = "cache_ready_model_load_failed"
    else:
        payload["status"] = "cache_incomplete_model_not_loaded"

    return payload


def _segment_tag(segment_index: Any, fallback_index: int) -> str:
    try:
        return f"{int(segment_index):04d}"
    except Exception:
        return f"{fallback_index:04d}"


def _speaker_token_variants(value: Any) -> list[str]:
    normalized = _safe_speaker_id(value)
    variants: list[str] = []
    if normalized:
        variants.append(normalized)

    match = re.fullmatch(r"spk_(\d+)", normalized)
    if match:
        speaker_idx = int(match.group(1))
        variants.extend([f"spk_{speaker_idx}", f"spk_{speaker_idx:02d}"])

    unique_variants: list[str] = []
    for variant in variants:
        if variant not in unique_variants:
            unique_variants.append(variant)
    return unique_variants


def _resolve_target_voice_layer(
    unit: dict[str, Any],
    speaker_layers: dict[str, Any],
) -> tuple[str, dict[str, Any], str]:
    if not speaker_layers:
        return "", {}, "no_speaker_layers"

    normalized_layers = {
        _safe_speaker_id(speaker_id): dict(layer or {})
        for speaker_id, layer in speaker_layers.items()
    }

    lookup_order = [
        ("target_voice_id", unit.get("target_voice_id")),
        ("speaker_id", unit.get("speaker_id")),
        ("source_voice_id", unit.get("source_voice_id")),
    ]
    for label, raw_value in lookup_order:
        for variant in _speaker_token_variants(raw_value):
            if variant in normalized_layers:
                return variant, normalized_layers[variant], f"matched_{label}"

    target_token = _safe_speaker_id(unit.get("target_voice_id"))
    if target_token in {"ai", "user", "default", "primary"} and len(normalized_layers) == 1:
        only_speaker = next(iter(normalized_layers.keys()))
        return only_speaker, normalized_layers[only_speaker], "alias_to_single_speaker"

    return "", {}, "target_voice_not_found"


def _voice_registry_reference(target_voice_id: Any) -> str:
    voice_key = _safe_speaker_id(target_voice_id)
    if not voice_key:
        return ""

    registry_root = str(os.getenv("VOICE_REGISTRY_DIR", "/app/models/voice_registry")).strip()
    if not registry_root:
        return ""

    candidate = os.path.join(registry_root, voice_key, "reference.wav")
    if os.path.exists(candidate):
        return candidate
    return ""


def _is_finetune_success(result: dict[str, Any]) -> bool:
    status = str(result.get("status") or "").strip()
    return status in {
        "finetune_command_completed",
        "finetune_internal_xtts_adaptation",
    }


def _load_audio_16k_mono(path: str) -> torch.Tensor:
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


def _build_internal_adaptation_reference(
    speaker_layer: dict[str, Any],
    assets: dict[str, Any],
    speaker_runtime_dir: str,
) -> str:
    runtime_ref = os.path.join(speaker_runtime_dir, "internal_adapt_reference.wav")

    reference_audio = str(speaker_layer.get("reference_audio") or "").strip()
    if reference_audio and os.path.exists(reference_audio):
        shutil.copy2(reference_audio, runtime_ref)
        return runtime_ref

    dataset_entries = list(assets.get("dataset_entries") or [])
    max_seconds = _env_int("VITS_INTERNAL_REFERENCE_SECONDS", 60, minimum=10)
    max_samples = max_seconds * TARGET_SAMPLE_RATE
    waveforms: list[torch.Tensor] = []
    consumed = 0

    for entry in dataset_entries:
        audio_file = str(entry.get("audio_file") or "")
        if not audio_file or not os.path.exists(audio_file):
            continue
        try:
            waveform = _load_audio_16k_mono(audio_file)
        except Exception:
            continue
        remaining = max_samples - consumed
        if remaining <= 0:
            break
        if waveform.size(1) > remaining:
            waveform = waveform[:, :remaining]
        waveforms.append(waveform)
        consumed += waveform.size(1)

    if not waveforms:
        return ""

    merged = torch.cat(waveforms, dim=1)
    torchaudio.save(runtime_ref, merged, TARGET_SAMPLE_RATE)
    return runtime_ref


def _prepare_finetune_assets(
    speaker_layer: dict[str, Any],
    speaker_vits_dir: str,
    model_name: str,
) -> dict[str, Any]:
    os.makedirs(speaker_vits_dir, exist_ok=True)
    manifest_input_path = str(speaker_layer.get("manifest_path") or "")

    segments: list[dict[str, Any]] = []
    if manifest_input_path and os.path.exists(manifest_input_path):
        try:
            with open(manifest_input_path, "r", encoding="utf-8") as manifest_file:
                manifest_data = json.load(manifest_file)
            segments = list(manifest_data.get("segments") or [])
        except Exception:
            segments = []

    dataset_entries = []
    for segment in segments:
        audio_path = str(segment.get("local_segment_path") or "")
        if not audio_path or not os.path.exists(audio_path):
            continue
        dataset_entries.append(
            {
                "audio_file": audio_path,
                "text": str(segment.get("transcript_text") or "").strip(),
                "speaker_id": str(segment.get("speaker_id") or speaker_layer.get("speaker_id") or ""),
            }
        )

    dataset_path = os.path.join(speaker_vits_dir, "finetune_dataset.jsonl")
    with open(dataset_path, "w", encoding="utf-8") as dataset_file:
        for entry in dataset_entries:
            dataset_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    config_path = os.path.join(speaker_vits_dir, "finetune_config.json")
    config_payload = {
        "speaker_id": speaker_layer.get("speaker_id"),
        "model_name": model_name,
        "dataset_path": dataset_path,
        "num_examples": len(dataset_entries),
        "reference_audio": speaker_layer.get("reference_audio"),
        "segments_dir": speaker_layer.get("segments_dir"),
    }
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config_payload, config_file, ensure_ascii=False, indent=2)

    return {
        "dataset_path": dataset_path,
        "config_path": config_path,
        "num_examples": len(dataset_entries),
        "dataset_entries": dataset_entries,
    }


def _run_finetune_hook(
    speaker_layer: dict[str, Any],
    speaker_runtime_dir: str,
    model_name: str,
) -> dict[str, Any]:
    assets = _prepare_finetune_assets(
        speaker_layer=speaker_layer,
        speaker_vits_dir=speaker_runtime_dir,
        model_name=model_name,
    )
    command_template = str(os.getenv("VITS_FINETUNE_COMMAND", "")).strip()
    internal_enabled = _is_truthy(os.getenv("VITS_INTERNAL_FINETUNE_ENABLED", "1"))

    result = {
        "speaker_id": speaker_layer.get("speaker_id"),
        "status": "pending",
        "finetune_mode": "external_command",
        "num_examples": assets["num_examples"],
        "dataset_path": assets["dataset_path"],
        "config_path": assets["config_path"],
        "command": "",
        "runtime_dir": speaker_runtime_dir,
        "adapted_reference_audio": "",
        "stdout_tail": "",
        "stderr_tail": "",
    }

    if assets["num_examples"] <= 0:
        result["status"] = "finetune_dataset_empty"
        return result
    if not command_template:
        if not internal_enabled:
            result["status"] = "finetune_command_missing"
            return result
        adapted_reference = _build_internal_adaptation_reference(
            speaker_layer=speaker_layer,
            assets=assets,
            speaker_runtime_dir=speaker_runtime_dir,
        )
        if adapted_reference and os.path.exists(adapted_reference):
            result["status"] = "finetune_internal_xtts_adaptation"
            result["finetune_mode"] = "internal_xtts_voice_adaptation"
            result["adapted_reference_audio"] = adapted_reference
        else:
            result["status"] = "finetune_internal_reference_missing"
            result["finetune_mode"] = "internal_xtts_voice_adaptation"
        return result

    format_values = {
        "speaker_id": speaker_layer.get("speaker_id"),
        "speaker_dir": speaker_runtime_dir,
        "dataset_path": assets["dataset_path"],
        "config_path": assets["config_path"],
        "reference_audio": speaker_layer.get("reference_audio", ""),
        "segments_dir": speaker_layer.get("segments_dir", ""),
        "model_name": model_name,
    }
    try:
        command = command_template.format(**format_values)
    except Exception as exc:
        result["status"] = f"invalid_finetune_command_template: {exc}"
        return result

    result["command"] = command
    completed = subprocess.run(
        command,
        shell=True,
        cwd=speaker_runtime_dir,
        text=True,
        capture_output=True,
    )
    stdout_text = completed.stdout or ""
    stderr_text = completed.stderr or ""
    result["stdout_tail"] = stdout_text[-4000:]
    result["stderr_tail"] = stderr_text[-4000:]
    if completed.returncode == 0:
        result["status"] = "finetune_command_completed"
    else:
        result["status"] = f"finetune_command_failed({completed.returncode})"
        if internal_enabled:
            adapted_reference = _build_internal_adaptation_reference(
                speaker_layer=speaker_layer,
                assets=assets,
                speaker_runtime_dir=speaker_runtime_dir,
            )
            if adapted_reference and os.path.exists(adapted_reference):
                result["status"] = "finetune_internal_xtts_adaptation"
                result["finetune_mode"] = "internal_xtts_voice_adaptation_after_command_failure"
                result["adapted_reference_audio"] = adapted_reference
    return result


def _synthesize_with_vits(
    model,
    text: str,
    output_path: str,
    speaker_wav: str,
    language: str | None,
) -> None:
    kwargs = {"text": text, "file_path": output_path}
    if speaker_wav and os.path.exists(speaker_wav):
        kwargs["speaker_wav"] = speaker_wav
    if language:
        kwargs["language"] = language

    try:
        model.tts_to_file(**kwargs)
    except TypeError:
        kwargs.pop("language", None)
        model.tts_to_file(**kwargs)


def run_vits_pipeline(
    translation_units: list[dict[str, Any]],
    speaker_voice_layers: dict[str, Any],
    translation_dir: str,
    target_language: str,
) -> dict[str, Any]:
    os.makedirs(translation_dir, exist_ok=True)
    vits_root = os.path.join(translation_dir, "vits")
    os.makedirs(vits_root, exist_ok=True)
    dubbed_segments_dir = translation_dir

    speaker_layers = dict(speaker_voice_layers.get("speaker_layers") or {})
    model_name = str(
        os.getenv(
            "VITS_MODEL_NAME",
            "tts_models/multilingual/multi-dataset/xtts_v2",
        )
    )
    vits_model, model_error, model_cache = _load_vits_model(model_name)
    require_finetune = _is_truthy(os.getenv("VITS_REQUIRE_FINETUNE", "1"))
    finetune_runtime_root = tempfile.mkdtemp(prefix="vits_finetune_")
    finetune_results: dict[str, Any] = {}
    try:
        for speaker_id, speaker_layer in speaker_layers.items():
            safe_speaker_id = _safe_speaker_id(speaker_id)
            speaker_runtime_dir = os.path.join(finetune_runtime_root, safe_speaker_id)
            os.makedirs(speaker_runtime_dir, exist_ok=True)
            finetune_results[safe_speaker_id] = _run_finetune_hook(
                speaker_layer=speaker_layer,
                speaker_runtime_dir=speaker_runtime_dir,
                model_name=model_name,
            )

        speakers_without_finetune = [
            speaker_id
            for speaker_id, result in finetune_results.items()
            if not _is_finetune_success(result)
        ]

        dubbed_segments = []
        for unit_idx, unit in enumerate(translation_units):
            source_speaker_id = _safe_speaker_id(unit.get("speaker_id") or unit.get("source_voice_id"))
            requested_target_voice = _safe_speaker_id(unit.get("target_voice_id") or source_speaker_id)
            segment_tag = _segment_tag(unit.get("segment_index"), unit_idx)
            output_filename = (
                f"seg_{segment_tag}_{source_speaker_id}_to_{requested_target_voice}.wav"
            )
            output_path = os.path.join(dubbed_segments_dir, output_filename)

            text = _resolve_synthesis_text(unit)
            resolved_target_speaker_id, speaker_layer, voice_resolution = _resolve_target_voice_layer(
                unit=unit,
                speaker_layers=speaker_layers,
            )
            speaker_wav = str(speaker_layer.get("reference_audio") or "")
            speaker_wav_source = "speaker_layer" if speaker_wav else "none"
            if not speaker_wav:
                registry_wav = _voice_registry_reference(unit.get("target_voice_id"))
                if registry_wav:
                    speaker_wav = registry_wav
                    speaker_wav_source = "voice_registry"
                    voice_resolution = "target_voice_from_registry"

            start_ms = int(unit.get("start_ms") or 0)
            end_ms = int(unit.get("end_ms") or 0)
            duration_ms = int(unit.get("duration_ms") or max(0, end_ms - start_ms))

            status = "pending"
            engine = "none"
            error_message = ""

            finetune_key = resolved_target_speaker_id or requested_target_voice
            finetune_result = finetune_results.get(_safe_speaker_id(finetune_key), {})
            finetune_ready = _is_finetune_success(finetune_result)
            finetune_reference = str(finetune_result.get("adapted_reference_audio") or "").strip()
            if finetune_reference and os.path.exists(finetune_reference):
                speaker_wav = finetune_reference
                speaker_wav_source = "finetune_runtime"

            if not text:
                status = "skipped_no_text"
            elif require_finetune and not finetune_ready:
                status = "blocked_finetune_required"
                error_message = str(finetune_result.get("status") or "missing_finetune")
            elif vits_model is None:
                status = "blocked_vits_model_unavailable"
                error_message = str(model_error or "VITS model unavailable")
            elif not speaker_wav:
                status = "blocked_missing_reference_voice"
            else:
                try:
                    _synthesize_with_vits(
                        model=vits_model,
                        text=text,
                        output_path=output_path,
                        speaker_wav=speaker_wav,
                        language=target_language,
                    )
                    normalize_segment_duration(
                        audio_path=output_path,
                        target_duration_ms=duration_ms,
                    )
                    status = "generated_with_vits"
                    engine = "vits"
                except Exception as exc:
                    status = "synthesis_failed"
                    error_message = str(exc)
                    output_path = ""

            if status != "generated_with_vits":
                output_path = ""

            dubbed_segments.append(
                {
                    "segment_index": unit.get("segment_index"),
                    "speaker_id": unit.get("speaker_id"),
                    "source_voice_id": unit.get("source_voice_id"),
                    "target_voice_id": unit.get("target_voice_id"),
                    "source_speaker_key": source_speaker_id,
                    "requested_target_voice": requested_target_voice,
                    "resolved_target_speaker_id": resolved_target_speaker_id,
                    "voice_resolution": voice_resolution,
                    "speaker_wav_source": speaker_wav_source,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": duration_ms,
                    "text": text,
                    "output_audio": output_path,
                    "status": status,
                    "engine": engine,
                    "error": error_message,
                }
            )

        payload = {
            "target_language": target_language,
            "model_name": model_name,
            "model_loaded": vits_model is not None,
            "model_load_error": model_error,
            "model_cache": model_cache,
            "require_finetune": require_finetune,
            "speakers_without_finetune": speakers_without_finetune,
            "finetune_runtime_policy": "ephemeral_discard_after_dubbing",
            "vits_root": vits_root,
            "dubbed_segments_dir": dubbed_segments_dir,
            "num_dubbed_segments": len(dubbed_segments),
            "finetune_results": finetune_results,
            "dubbed_segments": dubbed_segments,
        }
    finally:
        shutil.rmtree(finetune_runtime_root, ignore_errors=True)

    vits_result_json = os.path.join(translation_dir, "vits_result.json")
    with open(vits_result_json, "w", encoding="utf-8") as result_file:
        json.dump(payload, result_file, ensure_ascii=False, indent=2)
    payload["vits_result_json"] = vits_result_json
    return payload

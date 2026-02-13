from __future__ import annotations

import os
import time
from typing import Any

_TRANSLATOR_CACHE: dict[tuple[str, str], Any] = {}


def _normalize_lang(lang: str | None) -> str:
    return str(lang or "").strip().lower()


def _translation_engine() -> str:
    return str(os.getenv("TRANSLATION_ENGINE", "deep_translator")).strip().lower()


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    try:
        value = int(str(raw).strip()) if raw is not None else int(default)
    except Exception:
        value = int(default)
    return max(minimum, value)


def _get_google_translator(source_language: str | None, target_language: str):
    source = _normalize_lang(source_language) or "auto"
    target = _normalize_lang(target_language)
    cache_key = (source, target)
    if cache_key in _TRANSLATOR_CACHE:
        return _TRANSLATOR_CACHE[cache_key]

    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source=source, target=target)
    _TRANSLATOR_CACHE[cache_key] = translator
    return translator


def _get_mymemory_translator(source_language: str | None, target_language: str):
    source = _normalize_lang(source_language) or "auto"
    target = _normalize_lang(target_language)
    cache_key = (f"mymemory:{source}", target)
    if cache_key in _TRANSLATOR_CACHE:
        return _TRANSLATOR_CACHE[cache_key]

    from deep_translator import MyMemoryTranslator

    # MyMemory can be more reliable with explicit source; auto fallback remains.
    translator = MyMemoryTranslator(source=source, target=target)
    _TRANSLATOR_CACHE[cache_key] = translator
    return translator


def _skip_translation(
    target_language: str,
    source_language: str | None,
) -> tuple[bool, str]:
    target = _normalize_lang(target_language)
    source = _normalize_lang(source_language)

    if not target:
        return True, "missing_target_language"
    if target in {"auto", "source"}:
        return True, "auto_target_language"
    if source and source == target:
        return True, "same_language"
    return False, ""


def _translator_candidates(
    source_language: str | None,
    target_language: str,
) -> list[tuple[str, Any | None, str]]:
    candidates: list[tuple[str, Any | None, str]] = []
    try:
        candidates.append(("google", _get_google_translator(source_language, target_language), ""))
    except Exception as exc:
        candidates.append(("google", None, str(exc)))

    try:
        candidates.append(("mymemory", _get_mymemory_translator(source_language, target_language), ""))
    except Exception as exc:
        candidates.append(("mymemory", None, str(exc)))
    return candidates


def _translate_text_with_fallbacks(
    source_text: str,
    candidates: list[tuple[str, Any | None, str]],
    retries: int,
) -> tuple[str, str, str]:
    engine_errors: list[str] = []
    for engine_name, translator, init_error in candidates:
        if translator is None:
            if init_error:
                engine_errors.append(f"{engine_name}: {init_error}")
            continue
        for attempt in range(1, retries + 1):
            try:
                translated = str(translator.translate(source_text) or "").strip()
                if translated:
                    return translated, f"translated_{engine_name}", ""
            except Exception as exc:
                if attempt >= retries:
                    message = str(exc)
                    if init_error:
                        message = f"{init_error}; {message}"
                    engine_errors.append(f"{engine_name}: {message}")
                time.sleep(0.4)

    if engine_errors:
        return source_text, "translation_failed", "; ".join(engine_errors)
    return source_text, "translator_unavailable", ""


def translate_segments(
    transcribed_segments: list[dict[str, Any]],
    target_language: str,
    source_language: str | None = None,
) -> list[dict[str, Any]]:
    if not transcribed_segments:
        return []

    should_skip, skip_reason = _skip_translation(
        target_language=target_language,
        source_language=source_language,
    )
    engine = _translation_engine()
    retries = _env_int("TRANSLATION_RETRIES", 2, minimum=1)
    candidates: list[tuple[str, Any | None, str]] = []

    if not should_skip and engine != "none":
        candidates = _translator_candidates(
            source_language=source_language,
            target_language=target_language,
        )

    output: list[dict[str, Any]] = []
    for segment in transcribed_segments:
        item = dict(segment)
        source_text = str(item.get("transcript_text") or "").strip()
        translated_text = source_text
        translation_status = "not_attempted"
        translation_error = ""
        translation_engine = engine

        if not source_text:
            translation_status = "no_source_text"
        elif source_text == "[transcription unavailable]":
            translation_status = "no_source_text"
        elif should_skip:
            translation_status = f"skipped_{skip_reason}"
        elif engine == "none":
            translation_status = "skipped_engine_none"
        else:
            translated_text, translation_status, translation_error = _translate_text_with_fallbacks(
                source_text=source_text,
                candidates=candidates,
                retries=retries,
            )
            if translation_status.startswith("translated_"):
                translation_engine = translation_status.replace("translated_", "", 1)
                translation_status = "translated"
            elif translation_status == "translation_failed":
                translation_engine = "fallback_failed"

        item["translated_text"] = translated_text
        item["translation_target_language"] = target_language
        item["translation_engine"] = translation_engine
        item["translation_status"] = translation_status
        if translation_error:
            item["translation_error"] = translation_error
        output.append(item)

    return output

# Repository Guidelines

## Project Structure & Module Organization
- `app/`: FastAPI entrypoints (`app/main.py`, `app/routes.py`) and public API surface.
- `services/`: Core pipeline stages (audio extract, VAD, UVIVD speaker ID, ASR, translation, VITS, media muxing).
- `data/`: Runtime artifacts (`transcripts/`, `translations/segments/`, `output/`, `uploads/`, `segments/`).
- `models/`: Persistent model/cache storage (XTTS cache, HF cache, voice registry).
- `Dockerfile`, `docker-compose.yml`: Containerized backend + worker + Redis setup.
- `requirements.txt`: base runtime deps; `requirements-vits.txt`: VITS/translation extras.

## Build, Test, and Development Commands
- `docker compose build backend worker` — build API and Celery worker images.
- `docker compose up` — run backend, worker, and Redis locally.
- `uvicorn app.main:app --reload` — run API without Docker (dev only).
- `celery -A services.celery_app.celery_app worker -l info` — run worker manually.
- `python3 -m py_compile app/routes.py services/*.py` — quick syntax validation.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, PEP 8-aligned style.
- Use `snake_case` for functions/variables/files, `UPPER_SNAKE_CASE` for constants.
- Keep pipeline stages separated by responsibility (ASR, translation, voice layers, VITS, mux).
- Prefer project-root-anchored paths and explicit output directories.
- Keep changes minimal and scoped; avoid unrelated refactors.

## Testing Guidelines
- There is no full automated test suite yet; rely on targeted smoke tests.
- Validate via `/docs`:
  1. `POST /upload-video`
  2. `GET /tasks/{task_id}`
- Verify artifacts are created in:
  - `data/transcripts/<job_id>/`
  - `data/translations/segments/<job_id>/`
  - `data/output/<job_id>/<task_id>/`
- Check result fields such as `detected_source_language`, `vits.*`, and `final_video.mux_status`.

## Commit & Pull Request Guidelines
- Follow concise, imperative commit messages (e.g., `add translated segment persistence`).
- Keep one logical change per commit.
- PRs should include:
  - What changed and why
  - Affected paths/endpoints
  - Sample task response or logs for pipeline-impacting changes

## Security & Configuration Tips
- Never commit runtime artifacts from `data/` or large model caches from `models/`.
- Use environment variables for runtime behavior (`REDIS_URL`, XTTS/VITS settings).
- Keep base dependencies in `requirements.txt`; put optional VITS extras in `requirements-vits.txt`.

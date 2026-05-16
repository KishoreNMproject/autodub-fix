# AIO Engine (Experimental MVP)

A modular, AI-ready 2D prototype engine that turns natural-language prompts into playable game prototypes.

## Workflow

User Prompt → Intent Parser → AI Planning Layer → Modular Assembly → Playable Game

## Architecture

- `ai/`: intent parsing + planning interface (future model slot)
- `generators/`: scene JSON generation
- `core/`: schema + validation
- `modules/`: runtime gameplay blocks (physics, ai, inventory, crafting, survival, weather)
- `runtime/`: rendering/runtime loop (pygame)
- `editor/`: prompt-to-scene CLI

## Quick Start

```bash
python -m aio_engine.editor.cli --prompt "Forest survival game with rain and wolves."
python -m aio_engine.runtime.runner --scene aio_engine/scenes/generated_scene.json
```

Controls:
- WASD move
- E gather resources
- C craft bandage

## Save as a new Git repository

```bash
# in project root
cd /workspace/autodub-fix
git clone --no-hardlinks . ../aio-engine-new-repo
cd ../aio-engine-new-repo
git remote remove origin || true
git branch -M main
# then connect your own remote and push
git remote add origin <YOUR_NEW_REPO_URL>
git push -u origin main
```

## Notes

- Current MVP uses rule-based planning (no external model dependency).
- Architecture is prepared for future Kaggle-trained model plug-ins.

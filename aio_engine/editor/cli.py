from __future__ import annotations

import argparse
from pathlib import Path

from aio_engine.ai.prompt_interpreter import PromptInterpreter
from aio_engine.ai.planner import AIPlanner
from aio_engine.core.validator import SceneValidator
from aio_engine.generators.scene_generator import SceneGenerator


def generate_from_prompt(prompt: str, out_scene: Path) -> Path:
    interpreter = PromptInterpreter()
    planner = AIPlanner()
    generator = SceneGenerator()
    validator = SceneValidator()

    intent = interpreter.parse(prompt)
    plan = planner.build_plan(intent)
    scene = generator.generate(prompt, plan)
    validator.validate(scene)
    return generator.save(scene, out_scene)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", type=Path, default=Path("aio_engine/scenes/generated_scene.json"))
    args = parser.parse_args()

    path = generate_from_prompt(args.prompt, args.out)
    print(f"Scene generated: {path}")

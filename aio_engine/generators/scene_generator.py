from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List


class SceneGenerator:
    def generate(self, prompt: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        entities: List[Dict[str, Any]] = [
            {"id": "player", "type": "player", "x": 100, "y": 100, "w": 24, "h": 24, "hp": 100, "speed": 3}
        ]
        for enemy in plan["spawn"]["enemies"]:
            for idx in range(enemy["count"]):
                entities.append({"id": f"{enemy['type']}_{idx}", "type": enemy["type"], "x": 260 + (idx * 70), "y": 260, "w": 24, "h": 24, "hp": 40, "speed": 2})

        return {
            "metadata": {"prompt": prompt, "version": "0.2.0"},
            "world": {"width": 960, "height": 640, **plan["environment"]},
            "entities": entities,
            "systems": plan["systems"],
            "gameplay_blocks": plan["gameplay_blocks"],
        }

    def save(self, scene: Dict[str, Any], out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
        return out_path

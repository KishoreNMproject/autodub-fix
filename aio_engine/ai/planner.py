from __future__ import annotations

from typing import Dict, Any

from aio_engine.core.schema import GameIntent


class AIPlanner:
    """Planning boundary for future Kaggle model integration."""

    def build_plan(self, intent: GameIntent) -> Dict[str, Any]:
        systems = {
            "physics": True,
            "inventory": "inventory" in intent.mechanics,
            "weather": bool(intent.weather),
            "enemy_ai": bool(intent.enemies),
            "crafting": "crafting" in intent.mechanics,
            "survival": "health" in intent.mechanics,
        }
        return {
            "systems": systems,
            "gameplay_blocks": [
                {"name": name, "enabled": enabled, "config": {}}
                for name, enabled in systems.items()
            ],
            "spawn": {
                "player": {"x": 100, "y": 100},
                "enemies": [{"type": enemy, "count": 3} for enemy in intent.enemies],
            },
            "environment": {
                "biome": intent.biome,
                "weather": intent.weather,
            },
            "theme": intent.theme,
        }

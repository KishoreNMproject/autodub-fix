from __future__ import annotations

from typing import Dict, Any


class ValidationError(Exception):
    pass


class SceneValidator:
    REQUIRED_TOP_KEYS = {"metadata", "world", "entities", "systems", "gameplay_blocks"}

    def validate(self, scene: Dict[str, Any]) -> None:
        missing = self.REQUIRED_TOP_KEYS - set(scene.keys())
        if missing:
            raise ValidationError(f"Missing keys: {sorted(missing)}")
        if not isinstance(scene["entities"], list) or not scene["entities"]:
            raise ValidationError("Scene requires at least one entity")
        for ent in scene["entities"]:
            for field in ("id", "type", "x", "y", "w", "h"):
                if field not in ent:
                    raise ValidationError(f"Entity missing '{field}': {ent}")
        if not isinstance(scene["gameplay_blocks"], list):
            raise ValidationError("gameplay_blocks must be a list")

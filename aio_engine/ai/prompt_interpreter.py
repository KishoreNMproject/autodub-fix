from __future__ import annotations

from aio_engine.core.schema import GameIntent


class PromptInterpreter:
    """Rule-based interpreter; replace with model-backed parser later."""

    def parse(self, prompt: str) -> GameIntent:
        text = prompt.lower()
        biome = "forest" if "forest" in text else "snow" if "snow" in text else "plains"
        theme = "dark_fantasy" if "dark fantasy" in text else "survival"

        mechanics = ["movement", "health", "collision"]
        if "craft" in text:
            mechanics.append("crafting")
        if "inventory" in text or "survival" in text:
            mechanics.append("inventory")

        enemies = []
        if "wolves" in text or "wolf" in text:
            enemies.append("wolf")

        weather = []
        if "rain" in text:
            weather.append("rain")
        if "snow" in text:
            weather.append("snow")

        return GameIntent(theme=theme, biome=biome, mechanics=sorted(set(mechanics)), enemies=enemies, weather=weather)

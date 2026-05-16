from __future__ import annotations

import random


class WeatherSystem:
    def __init__(self, kind: str):
        self.kind = kind
        self.particles = [{"x": random.randint(0, 960), "y": random.randint(0, 640)} for _ in range(100)]

    def tick(self):
        speed = 6 if self.kind == "rain" else 2
        for p in self.particles:
            p["y"] += speed
            if p["y"] > 640:
                p["y"] = 0

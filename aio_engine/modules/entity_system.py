from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Entity:
    id: str
    type: str
    x: float
    y: float
    w: int
    h: int
    hp: int = 100
    speed: float = 2.0

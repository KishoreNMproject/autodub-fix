from __future__ import annotations


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def move_with_bounds(entity, dx: float, dy: float, world_w: int, world_h: int) -> None:
    entity.x = clamp(entity.x + dx, 0, world_w - entity.w)
    entity.y = clamp(entity.y + dy, 0, world_h - entity.h)

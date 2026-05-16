from __future__ import annotations


def apply_survival_tick(player, elapsed_seconds: float, weather_kind: str | None) -> None:
    drain = 0.2 * elapsed_seconds
    if weather_kind == "snow":
        drain *= 1.7
    elif weather_kind == "rain":
        drain *= 1.25
    player.hp = max(0, int(player.hp - drain))

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pygame

from aio_engine.modules.entity_system import Entity
from aio_engine.modules.enemy_ai import chase_step
from aio_engine.modules.physics import move_with_bounds
from aio_engine.modules.inventory import Inventory
from aio_engine.modules.weather import WeatherSystem
from aio_engine.modules.survival import apply_survival_tick
from aio_engine.modules.crafting import craft_bandage
from aio_engine.modules.gameplay_blocks import GameplayBlockRegistry
from aio_engine.runtime.asset_loader import AssetLoader


def run(scene_path: Path) -> None:
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    world = scene["world"]
    entities = [Entity(**e) for e in scene["entities"]]
    player = next(e for e in entities if e.type == "player")
    wolves = [e for e in entities if e.type == "wolf"]

    block_registry = GameplayBlockRegistry()
    for block in scene.get("gameplay_blocks", []):
        block_registry.register(block["name"], block["enabled"], block.get("config", {}))

    pygame.init()
    screen = pygame.display.set_mode((world["width"], world["height"]))
    clock = pygame.time.Clock()
    loader = AssetLoader()
    inventory = Inventory()
    weather = WeatherSystem(world["weather"][0]) if world.get("weather") else None

    running = True
    while running:
        dt = clock.tick(60) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                inventory.add("wood", 1)
                inventory.add("meat", 1)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                craft_bandage(inventory)

        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_d] - keys[pygame.K_a]) * player.speed
        dy = (keys[pygame.K_s] - keys[pygame.K_w]) * player.speed
        move_with_bounds(player, dx, dy, world["width"], world["height"])

        if block_registry.enabled_blocks().get("enemy_ai"):
            for wolf in wolves:
                wx, wy = chase_step(wolf, player)
                move_with_bounds(wolf, wx, wy, world["width"], world["height"])

        if weather and block_registry.enabled_blocks().get("weather"):
            weather.tick()

        if block_registry.enabled_blocks().get("survival"):
            apply_survival_tick(player, dt, weather.kind if weather else None)

        screen.fill((26, 50, 34))
        if weather and block_registry.enabled_blocks().get("weather"):
            color = (120, 170, 255) if weather.kind == "rain" else (240, 240, 255)
            for p in weather.particles:
                pygame.draw.line(screen, color, (p["x"], p["y"]), (p["x"], p["y"] + 4), 1)

        for e in entities:
            pygame.draw.rect(screen, loader.color_for(e.type), pygame.Rect(e.x, e.y, e.w, e.h))

        pygame.display.set_caption(f"AIO Engine MVP | HP:{player.hp} | E gather C craft | Inventory: {inventory.items}")
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()
    run(args.scene)

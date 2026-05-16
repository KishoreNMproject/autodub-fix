from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any


@dataclass
class GameplayBlock:
    name: str
    enabled: bool
    config: Dict[str, Any]


class GameplayBlockRegistry:
    def __init__(self) -> None:
        self._blocks: Dict[str, GameplayBlock] = {}

    def register(self, name: str, enabled: bool, config: Dict[str, Any] | None = None) -> None:
        self._blocks[name] = GameplayBlock(name=name, enabled=enabled, config=config or {})

    def enabled_blocks(self) -> Dict[str, GameplayBlock]:
        return {name: block for name, block in self._blocks.items() if block.enabled}

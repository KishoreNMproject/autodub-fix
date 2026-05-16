from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class GameIntent:
    theme: str
    biome: str
    mechanics: List[str] = field(default_factory=list)
    enemies: List[str] = field(default_factory=list)
    weather: List[str] = field(default_factory=list)


@dataclass
class SceneData:
    metadata: Dict[str, Any]
    world: Dict[str, Any]
    entities: List[Dict[str, Any]]
    systems: Dict[str, Any]

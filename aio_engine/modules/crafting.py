from __future__ import annotations


def craft_bandage(inventory) -> bool:
    wood = inventory.items.get("wood", 0)
    meat = inventory.items.get("meat", 0)
    if wood >= 1 and meat >= 1:
        inventory.items["wood"] = wood - 1
        inventory.items["meat"] = meat - 1
        inventory.items["bandage"] = inventory.items.get("bandage", 0) + 1
        return True
    return False

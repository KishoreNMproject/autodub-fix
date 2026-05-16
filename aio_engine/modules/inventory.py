class Inventory:
    def __init__(self):
        self.items = {"wood": 0, "meat": 0}

    def add(self, item: str, amount: int = 1):
        self.items[item] = self.items.get(item, 0) + amount

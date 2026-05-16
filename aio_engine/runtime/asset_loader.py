class AssetLoader:
    """Placeholder for future sprite/audio pipeline and AI-generated assets."""

    def color_for(self, entity_type: str):
        palette = {
            "player": (80, 180, 255),
            "wolf": (180, 180, 180),
        }
        return palette.get(entity_type, (255, 255, 255))

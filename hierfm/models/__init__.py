"""Model registry: config name -> class."""

from .classic import Boosting, SeasonalNaive
from .pretrained import Chronos2, ChronosBolt, Moirai2, TimesFM3, TimesFM25

REGISTRY = {m.name: m for m in (SeasonalNaive, Boosting, Chronos2, ChronosBolt, TimesFM25, TimesFM3, Moirai2)}


def make(spec, device):
    if spec.name not in REGISTRY:
        raise KeyError(f"unknown model {spec.name}; known: {sorted(REGISTRY)}")
    return REGISTRY[spec.name](device=device, **spec.params)

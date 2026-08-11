"""Параметры логики генерации, вынесенные из констант в namer/*.py."""
from __future__ import annotations

import json
from pathlib import Path

from .normalize import norm

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "rules.json"


class Rules:
    def __init__(self, path: str | Path = DEFAULT_RULES_PATH):
        self.path = Path(path)
        with open(self.path, encoding="utf-8") as fh:
            self.raw = json.load(fh)

        self.length = self.raw["length"]
        self.fill_order = list(self.raw["fill_order"])

        self.slot_capacity = {
            int(k): int(v) for k, v in self.raw["slot_capacity"].items() if not k.startswith("_")
        }
        self.stop_stems = {norm(w) for w in self.raw["stop_stems"]["words"]}
        self.discouraged_penalty = float(self.raw.get("discouraged_penalty", -0.3))

        comp = self.raw["composition"]
        self.composition = {
            "main_threshold": float(comp["main_threshold"]),
            "poliviscosa_fibers": {norm(v) for v in comp["poliviscosa_fibers"]},
            "poliviscosa_divisions": {norm(v).upper() for v in comp["poliviscosa_divisions"]},
            "pair_max_gap": float(comp["pair_max_gap"]),
            "valuable_min_share": float(comp["valuable_min_share"]),
            "elastane_min_share": float(comp["elastane_min_share"]),
            "trace_fibers": {norm(k): v for k, v in comp["trace_fibers"].items()},
            "material_column": comp.get("material_column", ""),
            "fabric_priority_penalty": float(comp.get("fabric_priority_penalty", 0.0)),
        }

        self.head_special = {
            k: v for k, v in self.raw["head_special"].items() if not k.startswith("_")
        }

        fallback = self.raw["fallback_head_by_division"]
        self.fallback_head_by_division = {
            norm(k).upper(): v for k, v in fallback.items() if not k.startswith("_")
        }
        self.fallback_head_default = self.fallback_head_by_division.get("DEFAULT", "Джемпер")

        self.head_lexicon_learning = {
            "min_count": int(self.raw["head_lexicon_learning"]["min_count"]),
            "min_share": float(self.raw["head_lexicon_learning"]["min_share"]),
            "group_share": float(self.raw["head_lexicon_learning"]["group_share"]),
        }

    def fallback_head(self, division: str) -> str:
        return self.fallback_head_by_division.get(norm(division).upper(), self.fallback_head_default)

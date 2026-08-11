"""Модель: веса признаков, лексикон видов изделия, обучение (усреднённый перцептрон)."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .normalize import norm

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "weights.json"


class Model:
    """Хранит веса вида `слот|признак` глобально и отдельно по товарной группе."""

    def __init__(self, path: str | Path = DEFAULT_WEIGHTS_PATH):
        self.path = Path(path)
        self.weights: dict[str, float] = {}
        self.head_lexicon: dict[str, str] = {}
        self.meta: dict = {"epochs": 0, "rows": 0, "metrics": {}}

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path = DEFAULT_WEIGHTS_PATH) -> "Model":
        model = cls(path)
        if model.path.exists():
            with open(model.path, encoding="utf-8") as fh:
                data = json.load(fh)
            model.weights = {k: float(v) for k, v in data.get("weights", {}).items()}
            model.head_lexicon = data.get("head_lexicon", {})
            model.meta = data.get("meta", model.meta)
        return model

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path or self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self.meta,
            "head_lexicon": self.head_lexicon,
            "weights": {k: round(v, 4) for k, v in sorted(self.weights.items())},
        }
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        return target

    # ------------------------------------------------------------------
    @staticmethod
    def keys_for(candidate, group: str) -> list[str]:
        base = f"{candidate.slot}|{candidate.key}"
        return [f"g::{base}", f"{group}::{base}"]

    def score(self, candidate, group: str) -> float:
        value = candidate.prior
        for key in self.keys_for(candidate, group):
            value += self.weights.get(key, 0.0)
        return value

    def update(self, candidate, group: str, delta: float) -> None:
        for key in self.keys_for(candidate, group):
            self.weights[key] = self.weights.get(key, 0.0) + delta

    # ------------------------------------------------------------------
    def top_weights(self, limit: int = 30) -> list[tuple[str, float]]:
        return sorted(self.weights.items(), key=lambda kv: -abs(kv[1]))[:limit]


def learn_head_lexicon(samples, dicts, min_count: int | None = None, min_share: float | None = None,
                       group_share: float | None = None) -> dict[str, str]:
    """Выучивает 'группа + вид изделия -> первое слово названия' по эталонам."""
    defaults = dicts.rules.head_lexicon_learning
    min_count = defaults["min_count"] if min_count is None else min_count
    min_share = defaults["min_share"] if min_share is None else min_share
    group_share = defaults["group_share"] if group_share is None else group_share
    known_heads = _known_heads(dicts)
    stats: dict[str, Counter] = defaultdict(Counter)

    for row, reference in samples:
        head = _extract_head(reference, known_heads)
        if not head:
            continue
        group = row.get("Group NEW").strip().upper()
        view = norm(row.get("Вид изделия"))
        stats[f"{group}|{view}"][head] += 1
        stats[f"{group}|"][head] += 1

    lexicon = {}
    for key, counter in stats.items():
        head, count = counter.most_common(1)[0]
        total = sum(counter.values())
        # общий ключ группы (без вида изделия) требует более уверенного большинства
        share_needed = min_share
        if key.endswith("|"):
            share_needed = group_share
        if count >= min_count and count / total >= share_needed:
            lexicon[key] = head
    return lexicon


def _known_heads(dicts) -> set[str]:
    heads = set()
    for group in dicts.raw["gender"].values():
        heads.update(norm(h) for h in group)
    heads.update(norm(v) for v in dicts.head_translate.values())
    for value in dicts.group_head.values():
        value = value.split("|")[-1]
        if not value.startswith("@"):
            heads.add(norm(value))
    return heads


def _extract_head(reference: str, known_heads: set[str]) -> str:
    tokens = str(reference).strip().split()
    if not tokens:
        return ""
    for size in (3, 2, 1):
        if len(tokens) >= size:
            candidate = " ".join(tokens[:size])
            if norm(candidate) in known_heads:
                return candidate
    return tokens[0]

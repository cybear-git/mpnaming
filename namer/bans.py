"""Загрузка запретов и стоп-слов (раньше жили внутри dictionaries.json)."""
from __future__ import annotations

import json
from pathlib import Path

from .normalize import norm

DEFAULT_BANS_PATH = Path(__file__).resolve().parent.parent / "data" / "bans.json"


class Bans:
    def __init__(self, path: str | Path = DEFAULT_BANS_PATH):
        self.path = Path(path)
        with open(self.path, encoding="utf-8") as fh:
            self.raw = json.load(fh)

        self.stop_values = {norm(v) for v in self.raw.get("stop_values", [])}
        self.head_forbidden = {norm(v) for v in self.raw.get("head_forbidden", [])}
        self.banned_phrases = [norm(v) for v in self.raw.get("banned_phrases", [])]
        self.discouraged_words = [norm(v) for v in self.raw.get("discouraged_words", [])]

    def is_stop(self, value) -> bool:
        n = norm(value)
        return (not n) or n in self.stop_values

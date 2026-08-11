"""Загрузка словарей правил и согласование по роду/числу."""
from __future__ import annotations

import json
from pathlib import Path

from .bans import DEFAULT_BANS_PATH, Bans
from .normalize import norm
from .rules import DEFAULT_RULES_PATH, Rules

DEFAULT_DICT_PATH = Path(__file__).resolve().parent.parent / "data" / "dictionaries.json"


class Dictionaries:
    def __init__(
        self,
        path: str | Path = DEFAULT_DICT_PATH,
        bans_path: str | Path = DEFAULT_BANS_PATH,
        rules_path: str | Path = DEFAULT_RULES_PATH,
    ):
        self.path = Path(path)
        with open(self.path, encoding="utf-8") as fh:
            self.raw = json.load(fh)

        self.bans = Bans(bans_path)
        self.rules = Rules(rules_path)

        # проброс атрибутов запретов/правил под старыми именами — код,
        # написанный под единый dictionaries.json, продолжает работать
        self.stop_values = self.bans.stop_values
        self.banned_phrases = self.bans.banned_phrases
        self.discouraged_words = self.bans.discouraged_words
        self.head_forbidden = self.bans.head_forbidden
        self.length = self.rules.length
        self.fill_order = self.rules.fill_order

        self.head_translate = {norm(k): v for k, v in self.raw["head_translate"].items()}
        self.group_head = {str(k).strip().upper(): v for k, v in self.raw["group_head"].items()}
        self.fabric = {norm(k): v for k, v in self.raw["fabric"].items()}
        self.fiber_translate = {norm(k): v for k, v in self.raw["fiber_translate"].items()}
        self.fiber_main = {norm(k): v for k, v in self.raw["fiber_main"].items()}
        self.fiber_genitive = {norm(k): v for k, v in self.raw["fiber_genitive"].items()}
        self.valuable_fibers = [norm(v) for v in self.raw["valuable_fibers"]]
        self.service_fibers = [norm(v) for v in self.raw["service_fibers"]]
        self.head_implies = {norm(k): v for k, v in self.raw["head_implies"].items()}
        self.extras = self.raw.get("extras", [])
        self.head_rules = self.raw.get("head_rules", [])

        # признаки: колонка -> значение -> описание
        self.features = {
            col: {norm(val): spec for val, spec in mapping.items()}
            for col, mapping in self.raw["features"].items()
        }

        # род/число по виду изделия
        self._gender = {}
        for gender, heads in self.raw["gender"].items():
            for head in heads:
                self._gender[norm(head)] = gender

    # ------------------------------------------------------------------
    def is_stop(self, value) -> bool:
        n = norm(value)
        return (not n) or n in self.stop_values

    def gender_of(self, head: str) -> str:
        """Определяет род/число по виду изделия (m/f/n/p)."""
        n = norm(head)
        if n in self._gender:
            return self._gender[n]
        # составные виды: 'куртка-пальто', 'платье-комбинация' -> по первому слову
        first = n.split("-")[0].split(" ")[0]
        if first in self._gender:
            return self._gender[first]
        for key, gender in self._gender.items():
            if n.startswith(key + " ") or n.startswith(key + "-"):
                return gender
        # эвристика по окончанию
        if first.endswith(("ы", "и")):
            return "p"
        if first.endswith("а") or first.endswith("я"):
            return "f"
        if first.endswith("е") or first.endswith("о"):
            return "n"
        return "m"

    @staticmethod
    def agree(phrase, gender: str) -> str | None:
        """Возвращает форму фразы, согласованную с родом изделия."""
        if phrase is None:
            return None
        if isinstance(phrase, str):
            return phrase
        return phrase.get(gender) or phrase.get("m")

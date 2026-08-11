"""Нормализация текста и лёгкий стеммер для сопоставления признаков с эталоном."""
from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[«»\"'`().,!?;:/\\\[\]{}]")
_SPACE_RE = re.compile(r"\s+")

# Суффиксы для лёгкого стемминга (от длинных к коротким).
_SUFFIXES = (
    "ыми", "ими", "ого", "его", "ому", "ему", "ая", "яя", "ое", "ее", "ые", "ие",
    "ый", "ий", "ой", "ем", "ом", "ах", "ях", "ам", "ям", "ей", "ов", "ев", "ую",
    "юю", "ий", "у", "ю", "а", "я", "ы", "и", "е", "о", "ь",
)

_MIN_STEM = 4


def norm(value) -> str:
    """Приводит значение к нижнему регистру без ё, пунктуации и лишних пробелов."""
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ").strip().lower().replace("ё", "е")
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def clean_output(text: str) -> str:
    """Чистит готовое название: запрещённая пунктуация, двойные пробелы."""
    text = _PUNCT_RE.sub(" ", str(text))
    text = _SPACE_RE.sub(" ", text).strip()
    return text.strip(" -–—")


def stem(token: str) -> str:
    """Отрезает окончание, чтобы 'хлопковая' и 'хлопковый' совпали."""
    token = token.strip()
    if len(token) <= _MIN_STEM or not token.isalpha():
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
            return token[: -len(suffix)]
    return token


def stem_tokens(text: str) -> list[str]:
    return [stem(tok) for tok in norm(text).split() if tok]


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def same_word(a: str, b: str) -> bool:
    """Считает слова одинаковыми, если различие — окончание или опечатка."""
    if a == b or stem(a) == stem(b):
        return True
    if min(len(a), len(b)) >= 6 and _levenshtein(a, b) <= 2:
        return True
    return False


def contains_phrase(reference: str, phrase: str) -> bool:
    """Есть ли фраза внутри эталонного названия (с поправкой на окончания и опечатки)."""
    ref = norm(reference).split()
    ph = norm(phrase).split()
    if not ph or len(ph) > len(ref):
        return False
    for i in range(len(ref) - len(ph) + 1):
        if all(same_word(a, b) for a, b in zip(ref[i : i + len(ph)], ph)):
            return True
    return False


def token_f1(pred: str, gold: str) -> float:
    """F1 по основам слов — мягкая метрика похожести названий."""
    p, g = stem_tokens(pred), stem_tokens(gold)
    if not p or not g:
        return 0.0
    common = 0
    pool = list(g)
    for tok in p:
        if tok in pool:
            pool.remove(tok)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def compare(pred: str, gold: str) -> tuple[str, list[str], list[str]]:
    """Сравнивает названия по существу.

    Возвращает вердикт и списки лишних/недостающих слов:
      «да»      — тексты совпадают с точностью до регистра, ё/е, пунктуации и пробелов
      «опечатки»— совпадают все основы слов и порядок (разница только в окончаниях/опечатках)
      «порядок» — набор слов тот же, отличается только порядок
      «почти»   — расхождение ровно в одном слове
      «нет»     — расхождение больше
    """
    if not gold:
        return "", [], []
    if norm(pred) == norm(gold):
        return "да", [], []

    p_words, g_words = norm(pred).split(), norm(gold).split()
    if len(p_words) == len(g_words) and all(
        same_word(a, b) for a, b in zip(p_words, g_words)
    ):
        return "опечатки", [], []

    def diff(source, other):
        pool = list(other)
        rest = []
        for token in source:
            match = next((w for w in pool if same_word(token, w)), None)
            if match is None:
                rest.append(token)
            else:
                pool.remove(match)
        return rest

    extra = diff(p_words, g_words)
    missing = diff(g_words, p_words)
    if not extra and not missing:
        return "порядок", [], []
    verdict = "почти" if len(extra) + len(missing) <= 1 else "нет"
    return verdict, extra, missing

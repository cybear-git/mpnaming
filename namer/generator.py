"""Сборка итогового названия из отобранных признаков (разделы 4, 11, 12 промпта)."""
from __future__ import annotations

from dataclasses import dataclass

from .features import Row, build_candidates, determine_head
from .normalize import clean_output, norm, stem_tokens


@dataclass
class Result:
    name: str
    head: str
    used: list
    skipped: list
    length: int

    def explain(self) -> str:
        parts = [f"{c.slot}:{c.phrase} ({c.score:+.2f})" for c in self.used]
        return " | ".join(parts)


def _order_key(candidate):
    slot = 1.5 if (candidate.strong and candidate.slot == 2) else float(candidate.slot)
    return (slot, -candidate.score)


def _conflicts(candidate, used_stems: set[str], stop_stems: set[str]) -> bool:
    """Не повторяем один и тот же признак разными словами (п. 9.4)."""
    stems = {s for s in stem_tokens(candidate.phrase) if s not in stop_stems}
    return bool(stems & used_stems)


def generate(row_mapping, dicts, model, min_len=None, max_len=None, threshold=None) -> Result:
    row = row_mapping if isinstance(row_mapping, Row) else Row(row_mapping)
    min_len = min_len or dicts.length["min"]
    max_len = max_len or dicts.length["max"]
    slot_capacity = dicts.rules.slot_capacity
    stop_stems = dicts.rules.stop_stems

    group = row.get("Group NEW").strip().upper()
    head = determine_head(row, dicts, model.head_lexicon)
    candidates = build_candidates(row, dicts, head)
    for cand in candidates:
        cand.score = model.score(cand, group)

    if threshold is None:
        threshold = float(model.meta.get("threshold", 0.0))
    selected = [c for c in candidates if c.score > threshold]
    rejected = [c for c in candidates if c.score <= threshold]

    # п. 1.1: при сильном признаке слова «базовая»/«офисная» теряют приоритет
    if any(c.strong for c in selected):
        for cand in selected:
            if cand.filler:
                cand.score -= 0.25
        selected = [c for c in selected if c.score > 0] + [c for c in selected if c.score <= 0][:0]
        rejected = rejected + [c for c in candidates if c.filler and c.score <= 0 and c not in rejected]

    name = head.strip()
    used_stems = {s for s in stem_tokens(name) if s not in stop_stems}
    slot_used: dict[int, int] = {}
    used: list = []

    def try_add(cand) -> bool:
        if slot_used.get(cand.slot, 0) >= slot_capacity.get(cand.slot, 1):
            return False
        if _conflicts(cand, used_stems, stop_stems):
            return False
        nonlocal name
        candidate_name = f"{name} {cand.phrase}".strip()
        if len(candidate_name) > max_len:
            return False
        name = candidate_name
        used.append(cand)
        slot_used[cand.slot] = slot_used.get(cand.slot, 0) + 1
        used_stems.update(s for s in stem_tokens(cand.phrase) if s not in stop_stems)
        return True

    # 1) основной проход в порядке раздела 4
    for cand in sorted(selected, key=_order_key):
        try_add(cand)

    # 2) добор длины (п. 11.2). Порядок добора: сначала по весу модели,
    # при заданном fill_order — по указанным слотам.
    soft_min = int(dicts.length.get("soft_min", min_len))
    fill_floor = float(dicts.length.get("fill_floor", -1e9))
    fill_priority = {slot: i for i, slot in enumerate(dicts.fill_order)}
    pool = sorted(rejected, key=lambda c: (fill_priority.get(c.slot, 99), -c.score))

    for cand in pool:
        if len(name) >= min_len:
            break
        # слабый признак берём, только если иначе название совсем короткое:
        # лучше короткое честное название, чем лишнее «базовая» ради длины
        if cand.score < fill_floor and len(name) >= soft_min:
            continue
        try_add(cand)

    # 3) гарантия заполнения (п. 12)
    tail: list[str] = []
    if len(name) < soft_min:
        gender = dicts.gender_of(head)
        for fallback in (
            dicts.agree({"m": "прямого кроя", "f": "прямого кроя",
                         "n": "прямого кроя", "p": "прямого кроя"}, gender),
            dicts.agree({"m": "базовый", "f": "базовая", "n": "базовое", "p": "базовые"}, gender),
        ):
            candidate_name = f"{name} {fallback}".strip()
            if len(candidate_name) <= max_len and not _conflicts(
                type("C", (), {"phrase": fallback})(), used_stems, stop_stems
            ):
                name = candidate_name
                tail.append(fallback)
                used_stems.update(stem_tokens(fallback))
            if len(name) >= soft_min:
                break

    # 4) финальный порядок признаков строго по разделу 4 промпта
    name = " ".join([head.strip()] + [c.phrase for c in sorted(used, key=_order_key)] + tail)

    name = clean_output(name)
    name = _drop_banned(name, dicts)
    # порядок слов после чистки не меняем, только пробелы
    return Result(name=name, head=head, used=used,
                  skipped=[c for c in candidates if c not in used], length=len(name))


def _drop_banned(name: str, dicts) -> str:
    """Удаляет запрещённые обороты целиком, не разрывая соседние слова."""
    for phrase in dicts.banned_phrases:
        if not phrase:
            continue
        words = phrase.split()
        tokens = name.split()
        size = len(words)
        index = 0
        while index + size <= len(tokens):
            if [norm(t) for t in tokens[index : index + size]] == words:
                del tokens[index : index + size]
            else:
                index += 1
        name = " ".join(tokens)
    # название не должно заканчиваться предлогом или союзом
    tokens = name.split()
    while tokens and norm(tokens[-1]) in {"с", "со", "из", "на", "в", "и", "до", "за", "по"}:
        tokens.pop()
    return " ".join(tokens)

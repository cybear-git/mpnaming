"""Определение вида изделия и сбор кандидатов-признаков для строки таблицы."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .composition import material_from_composition, parse_composition
from .normalize import norm, stem, stem_tokens

FABRIC_COLUMNS = ("Вид материала", "Вид полотна", "Рисунок ткани")
SPLIT_RE = re.compile(r"\s*[;/,]\s*")


@dataclass
class Candidate:
    slot: int
    phrase: str
    key: str
    source: str = ""
    strong: bool = False
    filler: bool = False
    prior: float = 0.0
    score: float = 0.0
    extra: dict = field(default_factory=dict)


class Row:
    """Удобный доступ к ячейкам строки вне зависимости от регистра/пробелов."""

    def __init__(self, mapping):
        self._data = {}
        for key, value in dict(mapping).items():
            self._data[norm(key)] = value

    def get(self, column, default=""):
        value = self._data.get(norm(column), default)
        if value is None:
            return default
        text = str(value).strip()
        if text.lower() in ("nan", "none", "nat"):
            return default
        return text

    def values(self, column):
        """Значение ячейки, разбитое по разделителям (',', '/', ';')."""
        raw = self.get(column)
        if not raw:
            return []
        parts = [p.strip() for p in SPLIT_RE.split(raw) if p.strip()]
        return [raw] + [p for p in parts if norm(p) != norm(raw)]


# ----------------------------------------------------------------------
# ВИД ИЗДЕЛИЯ
# ----------------------------------------------------------------------

def _rule_77(row: Row, rules) -> str:
    """Группа 77: юбка или брюки (раздел 3 промпта)."""
    hs = rules.head_special
    silhouette = norm(row.get("Силуэт"))
    model = norm(row.get("Модельные особенности"))
    length = norm(row.get("Длина"))
    if any(k in silhouette for k in hs["group_77_skirt_silhouette"]):
        return hs["group_77_skirt_head"]
    if any(k in model for k in hs["group_77_skirt_model"]):
        return hs["group_77_skirt_head"]
    if any(k in length for k in hs["group_77_skirt_length"]):
        return hs["group_77_skirt_head"]
    return hs["group_77_default_head"]


def _special_head(view: str, row: Row, rules) -> str | None:
    """Особые случаи п. 2.3."""
    hs = rules.head_special
    sleeve = norm(row.get("Длина рукава")) or norm(row.get("Вид рукава"))
    if "футболка с модельными особенностями" in view:
        if "длинн" in sleeve:
            return hs["tshirt_with_features_long_sleeve_head"]
        if "без рукав" in sleeve or "бретел" in sleeve:
            return hs["tshirt_with_features_no_sleeve_head"]
        return hs["tshirt_with_features_default_head"]
    if "майка с модельными особенностями" in view:
        if "без рукав" in sleeve or "бретел" in sleeve:
            return hs["tank_with_features_no_sleeve_head"]
        return hs["tank_with_features_default_head"]
    if "футболка с длинным рукавом" in view:
        return hs["tshirt_long_sleeve_head"]
    return None


def determine_head(row: Row, dicts, head_lexicon: dict | None = None) -> str:
    """Первое слово названия. Пустым не остаётся никогда (п. 2)."""
    group = row.get("Group NEW").strip().upper()
    view_raw = row.get("Вид изделия")
    view = norm(view_raw)

    # 1. выученный из эталонов лексикон (group + вид изделия)
    if head_lexicon:
        for key in (f"{group}|{view}", f"{group}|"):
            learned = head_lexicon.get(key)
            if learned:
                return learned

    # 2. правила вида изделия, которые не выводятся из поля «Вид изделия»
    for rule in getattr(dicts, "head_rules", []):
        if all(
            norm(row.get(column)) in {norm(a) for a in allowed}
            for column, allowed in rule.get("when", {}).items()
        ):
            return rule["head"]

    # 3. особые случаи
    special = _special_head(view, row, dicts.rules)
    if special:
        return special

    # 3. перевод английских видов изделия
    if view in dicts.head_translate:
        return dicts.head_translate[view]

    # 4. осмысленное русское значение — берём дословно
    view_clean = re.sub(r"\s*с модельными особенностями\s*", " ", view).strip()
    if (
        view_clean
        and view_clean not in dicts.head_forbidden
        and not dicts.is_stop(view_clean)
        and re.search(r"[а-я]", view_clean)
    ):
        return view_raw.strip()[0].upper() + view_raw.strip()[1:]

    # 5. справочник товарных групп
    rule = dicts.group_head.get(group, "")
    if rule == "@правило_77":
        return _rule_77(row, dicts.rules)
    if rule.startswith("@вид_изделия|"):
        return rule.split("|", 1)[1]
    if rule:
        return rule

    # 6. родовое слово по дивизиону
    return dicts.rules.fallback_head(row.get("PG"))


# ----------------------------------------------------------------------
# КАНДИДАТЫ-ПРИЗНАКИ
# ----------------------------------------------------------------------

def build_candidates(row: Row, dicts, head: str) -> list[Candidate]:
    gender = dicts.gender_of(head)
    division = row.get("PG").strip().upper()
    group = row.get("Group NEW").strip().upper()
    out: list[Candidate] = []
    seen: set[str] = set()
    discouraged_stems = {stem(w) for w in dicts.bans.discouraged_words}

    def add(slot, phrase, key, source, strong=False, filler=False, prior=0.0, extra=None):
        if not phrase:
            return
        phrase = str(phrase).strip()
        dedup = f"{slot}|{norm(phrase)}"
        if dedup in seen:
            return
        seen.add(dedup)
        # слово из discouraged_words не запрещено, а лишь менее желательно —
        # оно получает штраф к приоритету, но не вырезается, если уже попало
        # в название другим путём (см. data/bans.json)
        if discouraged_stems & set(stem_tokens(phrase)):
            prior += dicts.rules.discouraged_penalty
        out.append(
            Candidate(
                slot=slot, phrase=phrase, key=key, source=source,
                strong=strong, filler=filler, prior=prior, extra=extra or {},
            )
        )

    # --- признаки по словарю колонок -------------------------------------
    for column, mapping in dicts.features.items():
        for raw_value in row.values(column):
            value = norm(raw_value)
            if dicts.is_stop(value):
                continue
            spec = mapping.get(value)
            if spec is None:
                continue
            phrase = dicts.agree(spec.get("phrase"), gender)
            add(
                slot=int(spec["slot"]),
                phrase=phrase,
                key=f"{column}|{value}",
                source=f"{column}={raw_value}",
                strong=bool(spec.get("strong")),
                filler=bool(spec.get("filler")),
                prior=float(spec.get("prior", 0.0)),
            )

    # --- материал: сначала ткань, потом состав (п. 5.1) -------------------
    fabric_found = False
    for column in FABRIC_COLUMNS:
        for raw_value in row.values(column):
            value = norm(raw_value)
            if dicts.is_stop(value):
                continue
            phrase = dicts.fabric.get(value)
            if phrase is None:
                continue
            text = dicts.agree(phrase, gender)
            if not text:
                continue
            slot = 8 if text.startswith(("с пайетками", "с люрексом", "с варёным", "с масляным")) else 3
            add(slot, text, f"ткань|{value}", f"{column}={raw_value}", prior=0.3)
            if slot == 3:
                fabric_found = True

    # название ткани приоритетнее словарной колонки состава (п. 5.1): если
    # ткань определилась явно, кандидаты из этой колонки получают штраф,
    # а не соревнуются с ней на равных
    material_column = dicts.rules.composition.get("material_column")
    if fabric_found and material_column:
        penalty = dicts.rules.composition["fabric_priority_penalty"]
        prefix = f"{material_column}|"
        for cand in out:
            if cand.slot == 3 and cand.key.startswith(prefix):
                cand.prior += penalty

    shares = parse_composition(row.get("Composition") or row.get("Состав"), dicts)
    for text, key, is_service in material_from_composition(shares, dicts, gender, division):
        prior = -0.2 if (is_service and fabric_found) else (0.0 if fabric_found else 0.3)
        add(3, text, key, f"Composition={row.get('Composition')}", prior=prior,
            extra={"service": is_service})

    # --- деним: базовый материал «хлопковые» (п. 5.5) ---------------------
    if division == "W71" or group in ("71", "72", "79", "5D", "5G", "5W", "8D", "86"):
        add(3, dicts.agree(dicts.fiber_main.get("хлопок"), gender), "деним|хлопок",
            "правило W71", prior=0.6)

    # --- «со стиркой» для трикотажа W30 (раздел 8) ------------------------
    if division == "W30" and norm(row.get("Эффект")) not in ("", "без эффектов"):
        add(8, "со стиркой", "эффект|стирка W30", "Эффект", prior=0.1)

    # --- стиль из Category (Basic) ---------------------------------------
    if norm(row.get("Category")) == "basic":
        add(13, dicts.agree({"m": "базовый", "f": "базовая", "n": "базовое", "p": "базовые"}, gender),
            "Стиль|basic", "Category=Basic", filler=True, prior=0.2)

    # --- обучаемые «расширения»: формулировки, которых нет в исходных полях,
    # но которые заказчик регулярно использует. Включаются только весами модели.
    for extra in getattr(dicts, "extras", []):
        conditions = extra.get("when", {})
        ok = True
        for column, allowed in conditions.items():
            if norm(row.get(column)) not in {norm(a) for a in allowed}:
                ok = False
                break
        if not ok:
            continue
        phrase = dicts.agree(extra["phrase"], gender)
        context = "|".join(norm(row.get(col)) for col in extra.get("key_by", []))
        add(int(extra["slot"]), phrase, f"extra|{norm(phrase)}|{context}",
            "правило-расширение", prior=float(extra.get("prior", 0.0)))

    # --- отсечение признаков, уже выраженных видом изделия (п. 9.3) -------
    implied = dicts.head_implies.get(norm(head), [])
    head_stems = set(norm(head).replace("-", " ").split())
    filtered = []
    for cand in out:
        phrase_norm = norm(cand.phrase)
        if any(norm(imp) == phrase_norm for imp in implied):
            continue
        if phrase_norm in head_stems:
            continue
        filtered.append(cand)
    return filtered

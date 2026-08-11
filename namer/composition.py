"""Разбор состава (п. 5.2 промпта): проценты волокон -> формулировка материала."""
from __future__ import annotations

import re

from .normalize import norm

_PAIR_RE = re.compile(r"(\d{1,3})\s*%\s*([a-zA-Zа-яА-ЯёЁ\s\-\(\)]+)")


def parse_composition(text, dicts) -> list[tuple[str, float]]:
    """['72% cotton, 25% nylon'] -> [('хлопок', 72.0), ('нейлон', 25.0)] по убыванию доли."""
    if not text:
        return []
    shares: dict[str, float] = {}
    for pct, raw_name in _PAIR_RE.findall(str(text)):
        name = norm(raw_name)
        name = re.sub(r"\(.*?\)", " ", name).strip()
        name = name.split(",")[0].strip()
        ru = None
        for key, value in dicts.fiber_translate.items():
            if name == key or name.startswith(key) or key in name.split():
                ru = value
                break
        if ru is None:
            ru = name
        ru = norm(ru)
        if not ru:
            continue
        shares[ru] = shares.get(ru, 0.0) + float(pct)
    return sorted(shares.items(), key=lambda kv: -kv[1])


def material_from_composition(shares, dicts, gender, division=None, group=None):
    """Возвращает список кандидатов-материалов: [(фраза, ключ, признак_служебности)]."""
    out = []
    if not shares:
        return out

    rules = dicts.rules.composition
    main_threshold = rules["main_threshold"]
    
    # --- ЛОГИКА ДЛЯ ДЕНИМА (W71, 71, 72, 79, 5D, 5G, 5W, 8D, 86) ---
    denim_divisions = ["W71"]
    denim_groups = ["71", "72", "79", "5D", "5G", "5W", "8D", "86"]
    is_denim = (division and division.upper() in denim_divisions) or (group and group.upper() in denim_groups)
    
    if is_denim:
        # Для денима смотрим только лен, лиоцелл, эластан. Игнорируем хлопок/полиэстер.
        for name, share in shares:
            norm_name = norm(name)
            # Лен
            if any(x in norm_name for x in ["лен", "linen"]) and share >= 3:
                out.append(("со льном", f"comp:add:{name}", False))
            # Лиоцелл
            if any(x in norm_name for x in ["лиоцелл", "lyocell", "тенсел", "tencel"]) and share >= 3:
                out.append(("с лиоцеллом", f"comp:add:{name}", False))
            # Эластан
            if any(x in norm_name for x in ["эластан", "elastane", "spandex"]) and share >= 2:
                out.append(("с эластаном", "comp:add:эластан", True))
        return out
    
    # --- ЛОГИКА ДЛЯ КУРТОК (W50, группы 51-58, 5F) ---
    jacket_divisions = ["W50"]
    jacket_groups = ["51", "52", "53", "55", "56", "57", "58", "5F"]
    is_jacket = (division and division.upper() in jacket_divisions) or (group and group.upper() in jacket_groups)
    
    if is_jacket:
        top_name, top_share = shares[0]
        norm_top = norm(top_name)
        
        # Игнорируем полиэстер в куртках
        if any(x in norm_top for x in ["полиэстер", "polyester", "пэ"]):
            # Проверяем следующее волокно если есть
            if len(shares) > 1:
                second_name, second_share = shares[1]
                norm_second = norm(second_name)
                if not any(x in norm_second for x in ["полиэстер", "polyester", "пэ"]):
                    top_name, top_share, norm_top = second_name, second_share, norm_second
                else:
                    return out  # Всё равно всё полиэстер
            else:
                return out  # Только полиэстер
        
        # Обработка кожи/замши -> искусственная
        if any(x in norm_top for x in ["кожа", "leather"]):
            out.append(("из искусственной кожи", f"comp:main:{top_name}", False))
            return out
            
        if any(x in norm_top for x in ["замша", "suede"]):
            out.append(("из искусственной замши", f"comp:main:{top_name}", False))
            return out
        
        # Общая логика процентов для остальных материалов в куртках
        if top_share >= main_threshold:
            phrase = dicts.fiber_main.get(top_name)
            text = dicts.agree(phrase, gender)
            if text:
                out.append((text, f"comp:main:{top_name}", top_name in dicts.service_fibers))
        elif top_share >= 3:
            phrase = dicts.fiber_main.get(top_name)
            text = dicts.agree(phrase, gender)
            if text:
                out.append((text, f"comp:main:{top_name}", True))
        return out
    
    # --- ОБЩАЯ ЛОГИКА (Не деним, не куртка) ---
    top_name, top_share = shares[0]

    # Полиэстер + вискоза -> поливискоза (брюки/жакеты, п. 5.3)
    names = {n for n, _ in shares}
    if rules["poliviscosa_fibers"] <= names and norm(division).upper() in rules["poliviscosa_divisions"]:
        out.append(("из поливискозы", "comp:поливискоза", False))

    if top_share >= main_threshold:
        phrase = dicts.fiber_main.get(top_name)
        text = dicts.agree(phrase, gender)
        if text:
            out.append((text, f"comp:main:{top_name}", top_name in dicts.service_fibers))
    else:
        # два ценных волокна суммарно >= порога и близки по доле -> «из X и Y»
        if len(shares) >= 2:
            second_name, second_share = shares[1]
            if (
                top_share + second_share >= main_threshold
                and abs(top_share - second_share) <= rules["pair_max_gap"]
                and top_name not in dicts.service_fibers
                and second_name not in dicts.service_fibers
            ):
                g1 = dicts.fiber_genitive.get(top_name, top_name)
                g2 = dicts.fiber_genitive.get(second_name, second_name)
                out.append((f"из {g1} и {g2}", f"comp:pair:{top_name}+{second_name}", False))

        # ценные волокна ниже порога -> «с содержанием X»
        valuable = [
            (n, s) for n, s in shares if n in dicts.valuable_fibers and s >= rules["valuable_min_share"]
        ]
        if valuable:
            gens = [dicts.fiber_genitive.get(n, n) for n, _ in valuable[:2]]
            phrase = "с содержанием " + " и ".join(gens)
            out.append((phrase, "comp:contains:" + "+".join(n for n, _ in valuable[:2]), False))

        # запасной вариант — основное волокно, даже служебное (п. 11.2)
        phrase = dicts.fiber_main.get(top_name)
        text = dicts.agree(phrase, gender)
        if text:
            out.append((text, f"comp:main:{top_name}", True))

    # эластан в дениме и не только
    for name, share in shares:
        if name == "эластан" and share >= rules["elastane_min_share"]:
            out.append(("с эластаном", "comp:add:эластан", True))
        trace = rules["trace_fibers"].get(name)
        if trace and trace["min_share"] < share < min(trace["max_share"], main_threshold):
            out.append((f"{trace['prefix']} {trace['word']}", f"comp:add:{name}", False))
    return out

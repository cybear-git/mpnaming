"""Чтение таблиц, пакетная генерация, обучение и оценка качества."""
from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from .features import Row, build_candidates, determine_head
from .generator import generate
from .model import Model, learn_head_lexicon
from .normalize import compare, contains_phrase, norm, token_f1

OUTPUT_COLUMN = "Название для МП"
REFERENCE_CANDIDATES = ("Эталонный ответ", "Наименование НОРМ", "Эталон", "Reference")


def read_table(path, sheet=None) -> pd.DataFrame:
    if str(path).lower().endswith((".csv", ".tsv")):
        sep = "\t" if str(path).lower().endswith(".tsv") else ","
        return pd.read_csv(path, sep=sep)
    xls = pd.ExcelFile(path)
    if sheet is None:
        # лист с эталонной колонкой имеет приоритет, иначе первый лист
        for name in xls.sheet_names:
            head = pd.read_excel(path, sheet_name=name, nrows=0)
            if any(col in head.columns for col in REFERENCE_CANDIDATES):
                sheet = name
                break
        else:
            sheet = xls.sheet_names[0]
    return pd.read_excel(path, sheet_name=sheet)


def find_reference_column(df: pd.DataFrame) -> str | None:
    for col in REFERENCE_CANDIDATES:
        if col in df.columns:
            return col
    return None


def iter_rows(df: pd.DataFrame):
    for _, record in df.iterrows():
        yield Row(record.to_dict())


# ----------------------------------------------------------------------
# ГЕНЕРАЦИЯ
# ----------------------------------------------------------------------

def process_dataframe(df: pd.DataFrame, dicts, model) -> pd.DataFrame:
    names = [generate(row, dicts, model).name for row in iter_rows(df)]
    out = df.copy()
    if OUTPUT_COLUMN in out.columns:
        out = out.drop(columns=[OUTPUT_COLUMN])
    out[OUTPUT_COLUMN] = names
    return out


# ----------------------------------------------------------------------
# ОБУЧЕНИЕ (усреднённый перцептрон)
# ----------------------------------------------------------------------

def train_model(df: pd.DataFrame, dicts, model: Model, ref_col: str,
                epochs: int = 8, lr: float = 0.1, margin: float = 0.25,
                verbose: bool = True) -> Model:
    samples = []
    for row, reference in zip(iter_rows(df), df[ref_col]):
        text = "" if pd.isna(reference) else str(reference).strip()
        if text:
            samples.append((row, text))
    if not samples:
        raise ValueError(f"В колонке «{ref_col}» нет эталонных значений")

    model.head_lexicon = learn_head_lexicon(samples, dicts)

    weights = model.weights
    accum: dict[str, float] = defaultdict(float)
    counter = 1

    for epoch in range(1, epochs + 1):
        mistakes = 0
        for row, reference in samples:
            group = row.get("Group NEW").strip().upper()
            head = determine_head(row, dicts, model.head_lexicon)
            for cand in build_candidates(row, dicts, head):
                score = cand.prior + sum(
                    weights.get(key, 0.0) for key in Model.keys_for(cand, group)
                )
                target = 1 if contains_phrase(reference, cand.phrase) else 0
                # обучение с зазором: вес должен уверенно уходить от нуля
                needs_update = (target == 1 and score < margin) or (target == 0 and score > -margin)
                if needs_update:
                    if (score > 0) != bool(target):
                        mistakes += 1
                    delta = lr if target else -lr
                    for key in Model.keys_for(cand, group):
                        weights[key] = weights.get(key, 0.0) + delta
                        accum[key] += counter * delta
            counter += 1
        if verbose:
            snapshot = dict(weights)
            model.weights = {
                key: value - accum.get(key, 0.0) / counter for key, value in weights.items()
            }
            metrics = evaluate(df, dicts, model, ref_col, quiet=True)
            model.weights = weights = snapshot
            print(
                f"  эпоха {epoch}/{epochs}: ошибок отбора {mistakes}, "
                f"точное совпадение {metrics['exact']:.1%}, F1 {metrics['f1']:.3f}"
            )

    # усреднение весов — заметно стабильнее «последнего» вектора
    averaged = {
        key: value - accum.get(key, 0.0) / counter for key, value in weights.items()
    }
    model.weights = {k: v for k, v in averaged.items() if abs(v) > 1e-4}
    # калибровка порога отбора признаков по эталонам
    best = (None, -1.0)
    for step in range(-8, 5):
        threshold = round(step * 0.05, 2)
        model.meta["threshold"] = threshold
        scored = evaluate(df, dicts, model, ref_col, quiet=True)
        quality = scored["f1"] + scored["exact"]
        if quality > best[1]:
            best = (threshold, quality)
    model.meta["threshold"] = best[0]
    model.meta = {
        "epochs": int(model.meta.get("epochs", 0)) + epochs,
        "rows": len(samples),
        "lr": lr,
        "threshold": best[0],
        "metrics": evaluate(df, dicts, model, ref_col, quiet=True),
    }
    if verbose:
        print(f"  калиброванный порог отбора признаков: {best[0]:+.2f}")
    return model


# ----------------------------------------------------------------------
# ОЦЕНКА
# ----------------------------------------------------------------------

def evaluate(df: pd.DataFrame, dicts, model, ref_col: str, quiet: bool = False) -> dict:
    total = exact = equivalent = 0
    f1_sum = 0.0
    out_of_range = 0
    empty = 0
    by_group: dict[str, list] = defaultdict(list)

    for row, reference in zip(iter_rows(df), df[ref_col]):
        if pd.isna(reference) or not str(reference).strip():
            continue
        gold = str(reference).strip()
        result = generate(row, dicts, model)
        total += 1
        verdict, _, _ = compare(result.name, gold)
        is_exact = verdict == "да"
        exact += int(is_exact)
        if verdict in ("да", "опечатки", "порядок"):
            equivalent += 1
        score = token_f1(result.name, gold)
        f1_sum += score
        if not (dicts.length["min"] <= result.length <= dicts.length["max"]):
            out_of_range += 1
        if not result.name.strip():
            empty += 1
        by_group[row.get("Group NEW").strip().upper()].append((is_exact, score))

    metrics = {
        "rows": total,
        "exact": exact / total if total else 0.0,
        "equivalent": equivalent / total if total else 0.0,
        "f1": f1_sum / total if total else 0.0,
        "out_of_range": out_of_range,
        "empty": empty,
    }
    if not quiet:
        metrics["by_group"] = {
            group: {
                "rows": len(items),
                "exact": sum(1 for e, _ in items if e) / len(items),
                "f1": sum(s for _, s in items) / len(items),
            }
            for group, items in sorted(by_group.items())
        }
    return metrics


def review_dataframe(df: pd.DataFrame, dicts, model, ref_col: str | None) -> pd.DataFrame:
    rows = []
    for idx, row in enumerate(iter_rows(df)):
        result = generate(row, dicts, model)
        gold = ""
        if ref_col and ref_col in df.columns:
            value = df.iloc[idx][ref_col]
            gold = "" if pd.isna(value) else str(value).strip()
        verdict, extra, missing = compare(result.name, gold)
        rows.append(
            {
                "№": idx + 1,
                "Артикул": row.get("Code") or row.get("CW") or row.get("ID"),
                "Группа": row.get("Group NEW"),
                "Эталонное название": gold,
                "Название скрипта": result.name,
                "Длина": result.length,
                "F1": round(token_f1(result.name, gold), 3) if gold else "",
                "Совпало": verdict,
                "Лишние слова": ", ".join(extra),
                "Не хватает слов": ", ".join(missing),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# ОТЧЁТ О ПРОБЕЛАХ В СЛОВАРЯХ
# ----------------------------------------------------------------------

def dictionary_gaps(df: pd.DataFrame, dicts, model, ref_col: str | None = None,
                    limit: int = 40) -> dict:
    unknown_values: dict[str, Counter] = defaultdict(Counter)
    missed_tokens: Counter = Counter()

    for column, mapping in dicts.features.items():
        if column not in df.columns:
            continue
        for value in df[column].dropna():
            key = norm(value)
            if key and key not in mapping and not dicts.is_stop(key):
                unknown_values[column][str(value).strip()] += 1

    if ref_col and ref_col in df.columns:
        for row, reference in zip(iter_rows(df), df[ref_col]):
            if pd.isna(reference) or not str(reference).strip():
                continue
            gold = str(reference).strip()
            result = generate(row, dicts, model)
            produced = set(norm(result.name).split())
            for token in norm(gold).split():
                if token not in produced and len(token) > 3:
                    missed_tokens[token] += 1

    return {
        "unknown_values": {
            col: counter.most_common(limit) for col, counter in unknown_values.items()
        },
        "missed_tokens": missed_tokens.most_common(limit),
    }

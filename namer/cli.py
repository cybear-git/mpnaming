"""Консольный интерфейс: обработка файла, обучение, наглядный контроль."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .bans import DEFAULT_BANS_PATH
from .dictionaries import DEFAULT_DICT_PATH, Dictionaries
from .generator import generate
from .model import DEFAULT_WEIGHTS_PATH, Model
from .rules import DEFAULT_RULES_PATH
from .pipeline import (
    OUTPUT_COLUMN,
    dictionary_gaps,
    evaluate,
    find_reference_column,
    iter_rows,
    process_dataframe,
    read_table,
    review_dataframe,
    train_model,
)


def _load(args):
    dicts = Dictionaries(args.dictionaries, args.bans, args.rules)
    model = Model.load(args.weights)
    return dicts, model


# ----------------------------------------------------------------------
def cmd_process(args):
    dicts, model = _load(args)
    df = read_table(args.input, args.sheet)
    result = process_dataframe(df, dicts, model)

    lengths = result[OUTPUT_COLUMN].str.len()
    out_path = Path(args.output or "результат.xlsx")
    if out_path.suffix.lower() == ".csv":
        result.to_csv(out_path, index=False)
    else:
        result.to_excel(out_path, index=False)

    print(f"Обработано строк: {len(result)}")
    print(f"Длина названий: мин {lengths.min()}, макс {lengths.max()}, средняя {lengths.mean():.1f}")
    print(f"Вне коридора 30–60: {int(((lengths < 30) | (lengths > 60)).sum())}")
    print(f"Пустых ячеек: {int((result[OUTPUT_COLUMN].str.strip() == '').sum())}")
    print(f"Файл сохранён: {out_path.resolve()}")


# ----------------------------------------------------------------------
def cmd_train(args):
    dicts = Dictionaries(args.dictionaries, args.bans, args.rules)
    model = Model() if args.reset else Model.load(args.weights)
    model.path = Path(args.weights)

    df = read_table(args.input, args.sheet)
    ref_col = args.reference or find_reference_column(df)
    if not ref_col:
        sys.exit("Не найдена колонка с эталонными названиями. Укажите её через --reference")

    print(f"Обучение по колонке «{ref_col}», строк с эталоном: {df[ref_col].notna().sum()}")

    if args.holdout > 0:
        labeled = df[df[ref_col].notna()].sample(frac=1, random_state=args.seed)
        split = int(len(labeled) * (1 - args.holdout))
        train_part, test_part = labeled.iloc[:split], labeled.iloc[split:]
        probe = Model()
        train_model(train_part, dicts, probe, ref_col, epochs=args.epochs, lr=args.lr, verbose=False)
        train_metrics = evaluate(train_part, dicts, probe, ref_col, quiet=True)
        test_metrics = evaluate(test_part, dicts, probe, ref_col, quiet=True)
        print(
            f"  проверка на отложенных строках: обучение "
            f"{train_metrics['exact']:.1%}/F1 {train_metrics['f1']:.3f}, "
            f"контроль {test_metrics['exact']:.1%}/F1 {test_metrics['f1']:.3f} "
            f"({test_metrics['rows']} строк)"
        )

    train_model(df, dicts, model, ref_col, epochs=args.epochs, lr=args.lr)
    path = model.save()
    metrics = model.meta["metrics"]
    print()
    print(f"Точное совпадение: {metrics['exact']:.1%}")
    print(f"Совпадение по существу: {metrics['equivalent']:.1%}")
    print(f"Похожесть (token F1): {metrics['f1']:.3f}")
    print(f"Вне коридора 30–60: {metrics['out_of_range']}, пустых: {metrics['empty']}")
    print(f"Веса сохранены: {path.resolve()} ({len(model.weights)} признаков)")


# ----------------------------------------------------------------------
def cmd_review(args):
    dicts, model = _load(args)
    df = read_table(args.input, args.sheet)
    ref_col = args.reference or find_reference_column(df)
    table = review_dataframe(df, dicts, model, ref_col)
    # сводка считается по полной таблице, до фильтрации --only-errors,
    # иначе «да/опечатки/порядок» всегда показывали бы 0
    graded = table[table["Совпало"] != ""] if ref_col else table.iloc[0:0]

    if args.only_errors and ref_col:
        table = table[table["Совпало"].isin(["нет", "почти"])]
    view = table if args.limit <= 0 else table.head(args.limit)

    for _, record in view.iterrows():
        print(f"[{record['№']}] группа {record['Группа']}  {record['Совпало']}")
        if record["Эталонное название"]:
            print(f"    эталон: {record['Эталонное название']}")
        print(f"    скрипт: {record['Название скрипта']}  ({record['Длина']})")
        if record["Не хватает слов"]:
            print(f"    не хватает: {record['Не хватает слов']}")
        if record["Лишние слова"]:
            print(f"    лишнее:     {record['Лишние слова']}")

    if ref_col:
        if len(graded):
            counts = graded["Совпало"].value_counts()
            f1 = pd.to_numeric(graded["F1"], errors="coerce").mean()
            print(f"\nСтрок с эталоном: {len(graded)} | средний F1: {f1:.3f}")
            for verdict in ("да", "опечатки", "порядок", "почти", "нет"):
                count = int(counts.get(verdict, 0))
                print(f"  {verdict:<9} {count:>5}  {count / len(graded):>6.1%}")

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix.lower() == ".csv":
            table.to_csv(out_path, index=False)
        else:
            table.to_excel(out_path, index=False)
        print(f"Сравнительная таблица сохранена: {out_path.resolve()}")


# ----------------------------------------------------------------------
def cmd_eval(args):
    dicts, model = _load(args)
    df = read_table(args.input, args.sheet)
    ref_col = args.reference or find_reference_column(df)
    if not ref_col:
        sys.exit("Не найдена колонка с эталонными названиями")
    metrics = evaluate(df, dicts, model, ref_col)
    print(f"Строк с эталоном: {metrics['rows']}")
    print(f"Точное совпадение: {metrics['exact']:.1%}")
    print(f"Совпадение по существу (с опечатками и порядком слов): {metrics['equivalent']:.1%}")
    print(f"Похожесть (token F1): {metrics['f1']:.3f}")
    print(f"Вне коридора 30–60: {metrics['out_of_range']} | пустых: {metrics['empty']}")
    print("\nПо товарным группам:")
    for group, stats in metrics["by_group"].items():
        print(f"  {group:>3}  строк {stats['rows']:>4}  точно {stats['exact']:>6.1%}  F1 {stats['f1']:.3f}")


# ----------------------------------------------------------------------
def cmd_report(args):
    dicts, model = _load(args)
    df = read_table(args.input, args.sheet)
    ref_col = args.reference or find_reference_column(df)
    gaps = dictionary_gaps(df, dicts, model, ref_col)

    print("ЗНАЧЕНИЯ, КОТОРЫХ НЕТ В СЛОВАРЯХ (кандидаты на добавление):")
    for column, values in gaps["unknown_values"].items():
        if not values:
            continue
        print(f"\n  {column}:")
        for value, count in values:
            print(f"    {value!r}: {count}")

    if gaps["missed_tokens"]:
        print("\nСЛОВА ИЗ ЭТАЛОНОВ, КОТОРЫЕ СКРИПТ НЕ ВОСПРОИЗВОДИТ:")
        for token, count in gaps["missed_tokens"]:
            print(f"    {token}: {count}")


# ----------------------------------------------------------------------
def cmd_explain(args):
    dicts, model = _load(args)
    df = read_table(args.input, args.sheet)
    rows = list(iter_rows(df))
    index = args.row - 1
    if not 0 <= index < len(rows):
        sys.exit(f"Строка {args.row} вне диапазона 1..{len(rows)}")
    result = generate(rows[index], dicts, model)
    print(f"Строка {args.row}, группа {rows[index].get('Group NEW')}")
    print(f"Вид изделия: {result.head}")
    print(f"Название: {result.name} ({result.length} символов)")
    print("\nВзятые признаки:")
    for cand in result.used:
        print(f"  слот {cand.slot:>2}  {cand.phrase:<32} вес {cand.score:+.2f}  ← {cand.source}")
    print("\nОтклонённые признаки:")
    for cand in sorted(result.skipped, key=lambda c: -c.score)[:15]:
        print(f"  слот {cand.slot:>2}  {cand.phrase:<32} вес {cand.score:+.2f}  ← {cand.source}")


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="namer",
        description="Генерация названий карточек товара для маркетплейсов по правилам ПРОМПТ v5",
    )
    parser.add_argument("--dictionaries", default=str(DEFAULT_DICT_PATH), help="файл словарей")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS_PATH), help="файл весов модели")
    parser.add_argument("--rules", default=str(DEFAULT_RULES_PATH), help="файл параметров логики генерации")
    parser.add_argument("--bans", default=str(DEFAULT_BANS_PATH), help="файл запретов и стоп-слов")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("input", help="путь к xlsx/csv")
        sp.add_argument("--sheet", default=None, help="имя листа")
        sp.add_argument("--reference", default=None, help="колонка с эталоном")

    p = sub.add_parser("process", help="обработать файл и записать колонку «Название для МП»")
    common(p)
    p.add_argument("-o", "--output", default="результат.xlsx")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("train", help="обучить веса на эталонных названиях")
    common(p)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--reset", action="store_true", help="обучать с нуля, игнорируя старые веса")
    p.add_argument("--holdout", type=float, default=0.0,
                   help="доля строк для честной проверки качества, например 0.2")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("review", help="наглядный контроль: эталон рядом с результатом")
    common(p)
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--only-errors", action="store_true", help="показать только несовпадения")
    p.add_argument("-o", "--output", default=None, help="сохранить сравнение в файл")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("eval", help="метрики качества по группам")
    common(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("report", help="чего не хватает в словарях")
    common(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("explain", help="разбор одной строки: какие признаки и почему")
    common(p)
    p.add_argument("--row", type=int, default=1)
    p.set_defaults(func=cmd_explain)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
